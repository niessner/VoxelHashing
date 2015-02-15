#include <cutil_inline.h>
#include <cutil_math.h>
#include <device_functions.h>

#include "cuda_SimpleMatrixUtil.h"
#include "ICPUtil.h"
#include "CameraTrackingInput.h"

/////////////////////////////////////////////////////
// Defines
/////////////////////////////////////////////////////

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 64
#endif

#ifndef ARRAY_SIZE
#define ARRAY_SIZE 30
#endif

#define MINF __int_as_float(0xff800000)

/////////////////////////////////////////////////////
// Shared Memory
/////////////////////////////////////////////////////

__shared__ float bucket[ARRAY_SIZE*BLOCK_SIZE];

/////////////////////////////////////////////////////
// Helper Functions
/////////////////////////////////////////////////////

__device__ inline void addToLocalScanElement(uint inpGTid, uint resGTid, volatile float* shared)
{
	#pragma unroll
	for (uint i = 0; i<ARRAY_SIZE; i++)
	{
		shared[ARRAY_SIZE*resGTid+i] += shared[ARRAY_SIZE*inpGTid+i];
	}
}

__device__ inline void CopyToResultScanElement(uint GID, float* output)
{
	#pragma unroll
	for (uint i = 0; i<ARRAY_SIZE; i++)
	{
		output[ARRAY_SIZE*GID+i] = bucket[0+i];
	}
}

__device__ inline void SetZeroScanElement(uint GTid)
{
	#pragma unroll
	for(uint i = 0; i<ARRAY_SIZE; i++)
	{
		bucket[GTid*ARRAY_SIZE+i] = 0.0f;
	}
}

/////////////////////////////////////////////////////
// Scan
/////////////////////////////////////////////////////

__device__ inline void warpReduce(int GTid) // See Optimizing Parallel Reduction in CUDA by Mark Harris
{
	addToLocalScanElement(GTid + 32, GTid, bucket);
	addToLocalScanElement(GTid + 16, GTid, bucket);
	addToLocalScanElement(GTid + 8 , GTid, bucket);
	addToLocalScanElement(GTid + 4 , GTid, bucket);
	addToLocalScanElement(GTid + 2 , GTid, bucket);
	addToLocalScanElement(GTid + 1 , GTid, bucket);
}

/////////////////////////////////////////////////////
// Compute Normal Equations
/////////////////////////////////////////////////////

__device__ inline  void addToLocalSystem(mat1x6& jacobianBlockRow, mat1x1& residualsBlockRow, float weight, uint threadIdx, volatile float* shared)
{
	uint linRowStart = 0;

	#pragma unroll
	for (uint i = 0; i<6; i++)
	{
		mat1x1 colI; jacobianBlockRow.getBlock(0, i, colI);

		#pragma unroll
		for (uint j = i; j<6; j++)
		{
			mat1x1 colJ; jacobianBlockRow.getBlock(0, j, colJ);

			shared[ARRAY_SIZE*threadIdx+linRowStart+j-i] += colI.getTranspose()*colJ*weight;
		}

		linRowStart += 6-i;

		shared[ARRAY_SIZE*threadIdx+21+i] -= colI.getTranspose()*residualsBlockRow*weight; // -JTF
	}

	shared[ARRAY_SIZE*threadIdx+27] += weight*residualsBlockRow.norm1DSquared(); // residual
	shared[ARRAY_SIZE*threadIdx+28] += weight;									 // weight
	
	shared[ARRAY_SIZE*threadIdx+29] += 1.0f;									 // corr number
}

__global__ void scanNormalEquationsDevice(unsigned int imageWidth, unsigned int imageHeight, float* output, CameraTrackingInput cameraTrackingInput, float3x3 intrinsics, CameraTrackingParameters cameraTrackingIParameters, float3 anglesOld, float3 translationOld, unsigned int localWindowSize)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	// Set system to zero
	SetZeroScanElement(threadIdx.x);

	//Locally sum small window
	#pragma unroll
	for (uint i = 0; i<localWindowSize; i++)
	{
		const int index1D = localWindowSize*x+i;
		const uint2 index = make_uint2(index1D%imageWidth, index1D/imageWidth);
			
		if (index.x < imageWidth && index.y < imageHeight)
		{
			mat3x1 pInput = mat3x1(make_float3(cameraTrackingInput.d_inputPos[index1D]));
			mat3x1 nInput = mat3x1(make_float3(cameraTrackingInput.d_inputNormal[index1D]));
			mat1x1 iInput = mat1x1(cameraTrackingInput.d_inputIntensity[index1D]);

			if (!pInput.checkMINF() && !nInput.checkMINF() && !iInput.checkMINF())
			{
				mat3x3 I				 = mat3x3(intrinsics);
				mat3x3 ROld				 = mat3x3(evalRMat(anglesOld));
				mat3x1 pInputTransformed = ROld*pInput+mat3x1(translationOld);
				mat3x1 nInputTransformed = ROld*nInput;

				mat3x3 Ralpha = mat3x3(evalR_dGamma(anglesOld));
				mat3x3 Rbeta  = mat3x3(evalR_dBeta (anglesOld));
				mat3x3 Rgamma = mat3x3(evalR_dAlpha(anglesOld));

				mat3x1 pProjTrans = I*pInputTransformed;

				if (pProjTrans(2) > 0.0f)
				{
					mat2x1 uvModel = mat2x1(dehomogenize(pProjTrans));
					mat3x1 pTarget = mat3x1(make_float3(getValueNearestNeighbour(uvModel(0), uvModel(1), cameraTrackingInput.d_targetPos   , imageWidth, imageHeight)));
					mat3x1 nTarget = mat3x1(make_float3(getValueNearestNeighbour(uvModel(0), uvModel(1), cameraTrackingInput.d_targetNormal, imageWidth, imageHeight)));
				
					mat3x1 iTargetAndDerivative	= mat3x1(make_float3(bilinearInterpolationFloat4(uvModel(0), uvModel(1), cameraTrackingInput.d_targetIntensityAndDerivatives, imageWidth, imageHeight)));
					mat1x1 iTarget = mat1x1(iTargetAndDerivative(0)); mat1x1 iTargetDerivUIntensity = mat1x1(iTargetAndDerivative(1)); mat1x1 iTargetDerivVIntensity = mat1x1(iTargetAndDerivative(2));

					mat2x3 PI = dehomogenizeDerivative(pProjTrans);
					mat3x1 phiAlpha = Ralpha*pInputTransformed; mat3x1 phiBeta  = Rbeta *pInputTransformed;	mat3x1 phiGamma = Rgamma*pInputTransformed;		
										
					if( !pTarget.checkMINF() && !nTarget.checkMINF() && !iTarget.checkMINF() && !iTargetDerivUIntensity.checkMINF() && !iTargetDerivVIntensity.checkMINF() )
					{
						mat3x1 diff(pTarget-pInputTransformed); float dDist = diff.norm1D();
						float dNormal = nTarget.getTranspose()*nInputTransformed;

						// Point-Plane
						if (dDist <= cameraTrackingIParameters.distThres && dNormal >= cameraTrackingIParameters.normalThres)
						{
							const float weightDepth = max(0.0, 0.5f*((1.0f-dDist/cameraTrackingIParameters.distThres)+(1.0f-pInput(2)/cameraTrackingIParameters.sensorMaxDepth)));
							mat1x3 nTargetTransposed = nTarget.getTranspose();

							mat1x6 jacobianBlockRowPointPlane;
							jacobianBlockRowPointPlane.setBlock(-nTargetTransposed*phiAlpha, 0, 0);
							jacobianBlockRowPointPlane.setBlock(-nTargetTransposed*phiBeta , 0, 1);
							jacobianBlockRowPointPlane.setBlock(-nTargetTransposed*phiGamma, 0, 2);
							jacobianBlockRowPointPlane.setBlock(-nTargetTransposed		   , 0, 3);
							addToLocalSystem(jacobianBlockRowPointPlane, nTargetTransposed*diff, cameraTrackingIParameters.weightDepth*weightDepth, threadIdx.x, bucket);
						}

						// Color
						mat1x1 diffIntensity(iTarget-iInput);
						mat1x2 DIntensity; DIntensity(0, 0) = iTargetDerivUIntensity(0); DIntensity(0, 1) = iTargetDerivVIntensity(0);
						if( dDist <= cameraTrackingIParameters.distThres && dNormal >= cameraTrackingIParameters.normalThres
								&& diffIntensity.norm1D() <= cameraTrackingIParameters.colorThres && DIntensity.norm1D() > cameraTrackingIParameters.colorGradiantMin)
						{
							const float weightColor = max(0.0f, 1.0f-diffIntensity.norm1D()/cameraTrackingIParameters.colorThres);
							mat1x3 tmp0Intensity = DIntensity*PI*I;

							mat1x6 jacobianBlockRowIntensity;
							jacobianBlockRowIntensity.setBlock(tmp0Intensity*phiAlpha, 0, 0);
							jacobianBlockRowIntensity.setBlock(tmp0Intensity*phiBeta , 0, 1);
							jacobianBlockRowIntensity.setBlock(tmp0Intensity*phiGamma, 0, 2);
							jacobianBlockRowIntensity.setBlock(tmp0Intensity		 , 0, 3);
							addToLocalSystem(jacobianBlockRowIntensity, diffIntensity, cameraTrackingIParameters.weightColor*weightColor, threadIdx.x, bucket);
						}
					}
				}
			}
		}
	}
	
	__syncthreads();

	// Up sweep 2D
	#pragma unroll
	for(unsigned int stride = BLOCK_SIZE/2; stride > 32; stride >>= 1)
	{
		if (threadIdx.x < stride) addToLocalScanElement(threadIdx.x+stride/2, threadIdx.x, bucket);

		__syncthreads();
	}

	if (threadIdx.x < 32) warpReduce(threadIdx.x);

	// Copy to output texture
	if (threadIdx.x == 0) CopyToResultScanElement(blockIdx.x, output);
}

extern "C" void computeNormalEquations(unsigned int imageWidth, unsigned int imageHeight, float* output, CameraTrackingInput cameraTrackingInput, float* intrinsics, CameraTrackingParameters cameraTrackingParameters, float3 anglesOld, float3 translationOld, unsigned int localWindowSize, unsigned int blockSizeInt)
{
	const unsigned int numElements = imageWidth*imageHeight;
	dim3 blockSize(blockSizeInt, 1, 1);
	dim3 gridSize((numElements + blockSizeInt*localWindowSize-1) / (blockSizeInt*localWindowSize), 1, 1);

	scanNormalEquationsDevice<<<gridSize, blockSize>>>(imageWidth, imageHeight, output, cameraTrackingInput, float3x3(intrinsics), cameraTrackingParameters, anglesOld, translationOld, localWindowSize);
	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}
