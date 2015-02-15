#include "stdafx.h"

#include "CUDABuildLinearSystem.h"

#include "GlobalAppState.h"
#include "GlobalCameraTrackingState.h"

#include "MatrixConversion.h"

#include <iostream>

extern "C" void buildLinearSystem(
	unsigned int imageWidth,
	unsigned int imageHeight,
	float* output,
	float4* input,
	float4* target,
	float4* targetNormals,
	float* deltaTransform, unsigned int localWindowSize, unsigned int blockSize);

CUDABuildLinearSystem::CUDABuildLinearSystem(unsigned int imageWidth, unsigned int imageHeight) 
{
	cutilSafeCall(cudaMalloc(&d_output, 30*sizeof(float)*imageWidth*imageHeight));
	h_output = new float[30*imageWidth*imageHeight];
}

CUDABuildLinearSystem::~CUDABuildLinearSystem() {
	if (d_output) {
		cutilSafeCall(cudaFree(d_output));
	}
	if (h_output) {
		SAFE_DELETE_ARRAY(h_output);
	}
}

void CUDABuildLinearSystem::applyBL(float4* input, float4* target, float4* targetNormals, float3& mean, float meanStDev, Eigen::Matrix4f& deltaTransform, unsigned int imageWidth, unsigned int imageHeight, unsigned int level, Matrix6x7f& res, LinearSystemConfidence& conf) 
{
	const unsigned int localWindowSize = 12;
	const unsigned int blockSize = 64;
	const unsigned int dimX = (unsigned int)ceil(((float)imageWidth*imageHeight)/(localWindowSize*blockSize));

	//Eigen::Matrix4f deltaTransformT = deltaTransform.transpose();
	buildLinearSystem(imageWidth, imageHeight, d_output, input, target, targetNormals, deltaTransform.data(), localWindowSize, blockSize);

	cutilSafeCall(cudaMemcpy(h_output, d_output, sizeof(float)*30*dimX, cudaMemcpyDeviceToHost));

	// Copy to CPU
	res = reductionSystemCPU(h_output, dimX, conf);
}

Matrix6x7f CUDABuildLinearSystem::reductionSystemCPU( const float* data, unsigned int nElems, LinearSystemConfidence& conf )
{
	Matrix6x7f res; res.setZero();

	conf.reset();
	float numCorrF = 0.0f;

	for(unsigned int k = 0; k<nElems; k++)
	{
		unsigned int linRowStart = 0;

		for(unsigned int i = 0; i<6; i++)
		{
			for(unsigned int j = i; j<6; j++)
			{
				res(i, j) += data[30*k+linRowStart+j-i];
			}

			linRowStart += 6-i;

			res(i, 6) += data[30*k+21+i];
		}

		conf.sumRegError += data[30*k+27];
		conf.sumRegWeight += data[30*k+28];

		numCorrF += data[30*k+29];
	}

	// Fill lower triangle
	for(unsigned int i = 0; i<6; i++)
	{
		for(unsigned int j = i; j<6; j++)
		{
			res(j, i) = res(i, j);
		}
	}

	conf.numCorr = (unsigned int)numCorrF;

	return res;
}
