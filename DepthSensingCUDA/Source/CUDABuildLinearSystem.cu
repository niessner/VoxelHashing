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

__shared__ float bucket2[ARRAY_SIZE*BLOCK_SIZE];

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
		output[ARRAY_SIZE*GID+i] = bucket2[0+i];
	}
}

__device__ inline void SetZeroScanElement(uint GTid)
{
	#pragma unroll
	for(uint i = 0; i<ARRAY_SIZE; i++)
	{
		bucket2[GTid*ARRAY_SIZE+i] = 0.0f;
	}
}

/////////////////////////////////////////////////////
// Linearized System Matrix
/////////////////////////////////////////////////////

// Matrix Struct
struct Float1x6
{
	float data[6];
};

// Arguments: q moving point, n normal target
__device__ inline  Float1x6 buildRowSystemMatrixPlane(float3 q, float3 n, float w)
{
	Float1x6 row;
	row.data[0] = n.x*q.y-n.y*q.x;
	row.data[1] = n.z*q.x-n.x*q.z;
	row.data[2] = n.y*q.z-n.z*q.y;

	row.data[3] = -n.x;
	row.data[4] = -n.y;
	row.data[5] = -n.z;

	return row;
}

// Arguments: p target point, q moving point, n normal target
__device__ inline  float buildRowRHSPlane(float3 p, float3 q, float3 n, float w)
{
	return n.x*(q.x-p.x)+n.y*(q.y-p.y)+n.z*(q.z-p.z);
}

// Arguments: p target point, q moving point, n normal target
__device__ inline  void addToLocalSystem(float3 p, float3 q, float3 n, float weight, uint GTid)
{
	const Float1x6 row = buildRowSystemMatrixPlane(q, n, weight);
	const float b = buildRowRHSPlane(p, q, n, weight);
	uint linRowStart = 0;

	#pragma unroll
	for (uint i = 0; i<6; i++) {
		#pragma unroll
		for (uint j = i; j<6; j++) {
			bucket2[ARRAY_SIZE*GTid+linRowStart+j-i] += weight*row.data[i]*row.data[j];
		}

		linRowStart += 6-i;

		bucket2[ARRAY_SIZE*GTid+21+i] += weight*row.data[i]*b;
	}

	const float dN = dot(p-q, n);
	bucket2[ARRAY_SIZE*GTid+27] += weight*dN*dN;		//residual
	bucket2[ARRAY_SIZE*GTid+28] += weight;			//corr weight
	bucket2[ARRAY_SIZE*GTid+29] += 1.0f;				//corr number
}

/////////////////////////////////////////////////////
// Scan
/////////////////////////////////////////////////////

__device__ inline void warpReduce(int GTid) // See Optimizing Parallel Reduction in CUDA by Mark Harris
{
	addToLocalScanElement(GTid + 32, GTid, bucket2);
	addToLocalScanElement(GTid + 16, GTid, bucket2);
	addToLocalScanElement(GTid + 8 , GTid, bucket2);
	addToLocalScanElement(GTid + 4 , GTid, bucket2);
	addToLocalScanElement(GTid + 2 , GTid, bucket2);
	addToLocalScanElement(GTid + 1 , GTid, bucket2);
}

__global__ void scanScanElementsCS(
	unsigned int imageWidth,
	unsigned int imageHeight,
	float* output,
	float4* input,
	float4* target,
	float4* targetNormals,
	float4x4 deltaTransform, unsigned int localWindowSize)
{
	const uint x = blockIdx.x * blockDim.x + threadIdx.x;

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
			if (target[index1D].x != MINF && input[index1D].x != MINF && targetNormals[index1D].x != MINF) {
				const float g_meanStDevInv = 1.0f;
				const float3 g_mean = make_float3(0.0f,0.0f,0.0f);

				const float3 inputT = g_meanStDevInv*((deltaTransform.getTranspose()*make_float3(input[index1D])) - g_mean);
				const float3 targetT = g_meanStDevInv*(make_float3(target[index1D])-g_mean);
				const float weight = targetNormals[index1D].w;

				// Compute Linearized System
				addToLocalSystem(targetT, inputT, make_float3(targetNormals[index1D]), weight, threadIdx.x);
			}
		}
	}

	__syncthreads();

	// Up sweep 2D
	#pragma unroll
	for(unsigned int stride = BLOCK_SIZE/2; stride > 32; stride >>= 1)
	{
		if (threadIdx.x < stride) addToLocalScanElement(threadIdx.x+stride/2, threadIdx.x, bucket2);

		__syncthreads();
	}

	if (threadIdx.x < 32) warpReduce(threadIdx.x);

	// Copy to output texture
	if (threadIdx.x == 0) CopyToResultScanElement(blockIdx.x, output);
}

extern "C" void buildLinearSystem(
	unsigned int imageWidth,
	unsigned int imageHeight,
	float* output,
	float4* input,
	float4* target,
	float4* targetNormals,
	float* deltaTransform, unsigned int localWindowSize, unsigned int blockSizeInt)
{
	const unsigned int numElements = imageWidth*imageHeight;

	dim3 blockSize(blockSizeInt, 1, 1);
	dim3 gridSize((numElements + blockSizeInt*localWindowSize-1) / (blockSizeInt*localWindowSize), 1, 1);

	scanScanElementsCS<<<gridSize, blockSize>>>(imageWidth, imageHeight, output, input, target, targetNormals, float4x4(deltaTransform), localWindowSize);

	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}
