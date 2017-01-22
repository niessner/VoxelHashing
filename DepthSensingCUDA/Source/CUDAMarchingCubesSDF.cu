
#include <cutil_inline.h>
#include <cutil_math.h>

#include "VoxelUtilHashSDF.h"
#include "RayCastSDFUtil.h"
#include "MarchingCubesSDFUtil.h"


__global__ void resetMarchingCubesKernel(MarchingCubesData data) 
{
	*data.d_numTriangles = 0;
	*data.d_numOccupiedBlocks = 0;	
}
 
__global__ void extractIsoSurfaceKernel(HashData hashData, RayCastData rayCastData, MarchingCubesData data) 
{
	uint idx = blockIdx.x;

	const HashEntry& entry = hashData.d_hash[idx];
	if (entry.ptr != FREE_ENTRY) {
		int3 pi_base = hashData.SDFBlockToVirtualVoxelPos(entry.pos);
		int3 pi = pi_base + make_int3(threadIdx);
		float3 worldPos = hashData.virtualVoxelPosToWorld(pi);

		data.extractIsoSurfaceAtPosition(worldPos, hashData, rayCastData);
	}
}

extern "C" void resetMarchingCubesCUDA(MarchingCubesData& data)
{
	const dim3 blockSize(1, 1, 1);
	const dim3 gridSize(1, 1, 1);

	resetMarchingCubesKernel<<<gridSize, blockSize>>>(data);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

extern "C" void extractIsoSurfaceCUDA(const HashData& hashData, const RayCastData& rayCastData, const MarchingCubesParams& params, MarchingCubesData& data)
{
	const dim3 gridSize(params.m_hashNumBuckets*params.m_hashBucketSize, 1, 1);
	const dim3 blockSize(params.m_sdfBlockSize, params.m_sdfBlockSize, params.m_sdfBlockSize);

	extractIsoSurfaceKernel<<<gridSize, blockSize>>>(hashData, rayCastData, data);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}




///new

#define T_PER_BLOCK 8

//tags all
__global__ void extractIsoSurfacePass1Kernel(HashData hashData, RayCastData rayCastData, MarchingCubesData data)
{
	const HashParams& hashParams = c_hashParams;
	const unsigned int bucketID = blockIdx.x*blockDim.x + threadIdx.x;


	if (bucketID < hashParams.m_hashNumBuckets*HASH_BUCKET_SIZE) {

		//HashEntry entry = getHashEntry(g_Hash, bucketID);
		HashEntry& entry = hashData.d_hash[bucketID];

		if (entry.ptr != FREE_ENTRY) {

			//float3 pos = hashData.SDFBlockToWorld(entry.pos);
			//float l = SDF_BLOCK_SIZE*hashParams.m_virtualVoxelSize;
			//float3 minCorner = data.d_params->m_minCorner - l;
			//float3 maxCorner = data.d_params->m_maxCorner;
			//
			//if (data.d_params->m_boxEnabled == 1) {
			//	if (!data.isInBoxAA(minCorner, maxCorner, pos)) return;
			//}

			uint addr = atomicAdd(&data.d_numOccupiedBlocks[0], 1);
			data.d_occupiedBlocks[addr] = bucketID;
		}
	}
}

extern "C" void extractIsoSurfacePass1CUDA(const HashData& hashData, const RayCastData& rayCastData, const MarchingCubesParams& params, MarchingCubesData& data)
{
	const dim3 gridSize((params.m_hashNumBuckets*params.m_hashBucketSize + (T_PER_BLOCK*T_PER_BLOCK) - 1) / (T_PER_BLOCK*T_PER_BLOCK), 1);
	const dim3 blockSize((T_PER_BLOCK*T_PER_BLOCK), 1);


	extractIsoSurfacePass1Kernel << <gridSize, blockSize >> >(hashData, rayCastData, data);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}



__global__ void extractIsoSurfacePass2Kernel(HashData hashData, RayCastData rayCastData, MarchingCubesData data)
{
	//const HashParams& hashParams = c_hashParams;
	uint idx = data.d_occupiedBlocks[blockIdx.x];

	//if (idx >= hashParams.m_hashNumBuckets*HASH_BUCKET_SIZE) {
	//	if (threadIdx.x == 0) {
	//		printf("%d: invalid idx! %d\n", blockIdx.x, idx);
	//	}
	//	return;
	//}
	//return;

	const HashEntry& entry = hashData.d_hash[idx];
	if (entry.ptr != FREE_ENTRY) {
		int3 pi_base = hashData.SDFBlockToVirtualVoxelPos(entry.pos);
		int3 pi = pi_base + make_int3(threadIdx);
		float3 worldPos = hashData.virtualVoxelPosToWorld(pi);

		data.extractIsoSurfaceAtPosition(worldPos, hashData, rayCastData);
	}
}


extern "C" void extractIsoSurfacePass2CUDA(const HashData& hashData, const RayCastData& rayCastData, const MarchingCubesParams& params, MarchingCubesData& data, unsigned int numOccupiedBlocks)
{
	const dim3 gridSize(numOccupiedBlocks, 1, 1);
	const dim3 blockSize(params.m_sdfBlockSize, params.m_sdfBlockSize, params.m_sdfBlockSize);

	if (numOccupiedBlocks) {
		extractIsoSurfacePass2Kernel << <gridSize, blockSize >> >(hashData, rayCastData, data);
	}
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}