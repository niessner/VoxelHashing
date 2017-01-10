#pragma once

#include <cutil_inline.h>
#include <cutil_math.h>

#include "MatrixConversion.h"
#include "VoxelUtilHashSDF.h"
#include "DepthCameraUtil.h"
#include "CUDAScan.h"

#include "GlobalAppState.h"
#include "TimingLog.h"

extern "C" void resetCUDA(HashData& hashData, const HashParams& hashParams);
extern "C" void resetHashBucketMutexCUDA(HashData& hashData, const HashParams& hashParams);
extern "C" void allocCUDA(HashData& hashData, const HashParams& hashParams, const DepthCameraData& depthCameraData, const DepthCameraParams& depthCameraParams, const unsigned int* d_bitMask);
extern "C" void fillDecisionArrayCUDA(HashData& hashData, const HashParams& hashParams, const DepthCameraData& depthCameraData);
extern "C" void compactifyHashCUDA(HashData& hashData, const HashParams& hashParams);
extern "C" void integrateDepthMapCUDA(HashData& hashData, const HashParams& hashParams, const DepthCameraData& depthCameraData, const DepthCameraParams& depthCameraParams);
extern "C" void bindInputDepthColorTextures(const DepthCameraData& depthCameraData);

extern "C" void starveVoxelsKernelCUDA(HashData& hashData, const HashParams& hashParams);
extern "C" void garbageCollectIdentifyCUDA(HashData& hashData, const HashParams& hashParams);
extern "C" void garbageCollectFreeCUDA(HashData& hashData, const HashParams& hashParams);

class CUDASceneRepHashSDF
{
public:
	CUDASceneRepHashSDF(const HashParams& params) {
		create(params);
	}
	~CUDASceneRepHashSDF() {
		destroy();
	}

	static HashParams parametersFromGlobalAppState(const GlobalAppState& gas) {
		HashParams params;
		params.m_rigidTransform.setIdentity();
		params.m_rigidTransformInverse.setIdentity();
		params.m_hashNumBuckets = gas.s_hashNumBuckets;
		params.m_hashBucketSize = HASH_BUCKET_SIZE;
		params.m_hashMaxCollisionLinkedListSize = gas.s_hashMaxCollisionLinkedListSize;
		params.m_SDFBlockSize = SDF_BLOCK_SIZE;
		params.m_numSDFBlocks = gas.s_hashNumSDFBlocks;
		params.m_virtualVoxelSize = gas.s_SDFVoxelSize;
		params.m_maxIntegrationDistance = gas.s_SDFMaxIntegrationDistance;
		params.m_truncation = gas.s_SDFTruncation;
		params.m_truncScale = gas.s_SDFTruncationScale;
		params.m_integrationWeightSample = gas.s_SDFIntegrationWeightSample;
		params.m_integrationWeightMax = gas.s_SDFIntegrationWeightMax;
		params.m_streamingVoxelExtents = MatrixConversion::toCUDA(gas.s_streamingVoxelExtents);
		params.m_streamingGridDimensions = MatrixConversion::toCUDA(gas.s_streamingGridDimensions);
		params.m_streamingMinGridPos = MatrixConversion::toCUDA(gas.s_streamingMinGridPos);
		params.m_streamingInitialChunkListSize = gas.s_streamingInitialChunkListSize;
		return params;
	}

	void bindDepthCameraTextures(const DepthCameraData& depthCameraData) {
		bindInputDepthColorTextures(depthCameraData);
	}

	void integrate(const mat4f& lastRigidTransform, const DepthCameraData& depthCameraData, const DepthCameraParams& depthCameraParams, unsigned int* d_bitMask) {
		
		setLastRigidTransform(lastRigidTransform);

		//make the rigid transform available on the GPU
		m_hashData.updateParams(m_hashParams);

		//allocate all hash blocks which are corresponding to depth map entries
		alloc(depthCameraData, depthCameraParams, d_bitMask);

		//generate a linear hash array with only occupied entries
		compactifyHashEntries(depthCameraData);

		//volumetrically integrate the depth data into the depth SDFBlocks
		integrateDepthMap(depthCameraData, depthCameraParams);

		garbageCollect(depthCameraData);

		m_numIntegratedFrames++;
	}

	void setLastRigidTransform(const mat4f& lastRigidTransform) {
		m_hashParams.m_rigidTransform = MatrixConversion::toCUDA(lastRigidTransform);
		m_hashParams.m_rigidTransformInverse = m_hashParams.m_rigidTransform.getInverse();
	}

	void setLastRigidTransformAndCompactify(const mat4f& lastRigidTransform, const DepthCameraData& depthCameraData) {
		setLastRigidTransform(lastRigidTransform);
		compactifyHashEntries(depthCameraData);
	}


	const mat4f getLastRigidTransform() const {
		return MatrixConversion::toMlib(m_hashParams.m_rigidTransform);
	}

	//! resets the hash to the initial state (i.e., clears all data)
	void reset() {
		m_numIntegratedFrames = 0;

		m_hashParams.m_rigidTransform.setIdentity();
		m_hashParams.m_rigidTransformInverse.setIdentity();
		m_hashParams.m_numOccupiedBlocks = 0;
		m_hashData.updateParams(m_hashParams);
		resetCUDA(m_hashData, m_hashParams);
	}


	HashData& getHashData() {
		return m_hashData;
	} 

	const HashParams& getHashParams() const {
		return m_hashParams;
	}


	//! debug only!
	unsigned int getHeapFreeCount() {
		unsigned int count;
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(&count, m_hashData.d_heapCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost));
		return count+1;	//there is one more free than the address suggests (0 would be also a valid address)
	}

	//! debug only!
	void debugHash() {
		HashEntry* hashCPU = new HashEntry[m_hashParams.m_hashBucketSize*m_hashParams.m_hashNumBuckets];
		unsigned int* heapCPU = new unsigned int[m_hashParams.m_numSDFBlocks];
		unsigned int heapCounterCPU;

		MLIB_CUDA_SAFE_CALL(cudaMemcpy(&heapCounterCPU, m_hashData.d_heapCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost));
		heapCounterCPU++;	//points to the first free entry: number of blocks is one more

		MLIB_CUDA_SAFE_CALL(cudaMemcpy(heapCPU, m_hashData.d_heap, sizeof(unsigned int)*m_hashParams.m_numSDFBlocks, cudaMemcpyDeviceToHost));
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(hashCPU, m_hashData.d_hash, sizeof(HashEntry)*m_hashParams.m_hashBucketSize*m_hashParams.m_hashNumBuckets, cudaMemcpyDeviceToHost));

		//Check for duplicates
		class myint3Voxel {
		public:
			myint3Voxel() {}
			~myint3Voxel() {}
			bool operator<(const myint3Voxel& other) const {
				if (x == other.x) {
					if (y == other.y) {
						return z < other.z;
					}
					return y < other.y;
				}
				return x < other.x;
			}

			bool operator==(const myint3Voxel& other) const {
				return x == other.x && y == other.y && z == other.z;
			}

			int x,y,z, i;
			int offset;
			int ptr;
		}; 


		std::unordered_set<unsigned int> pointersFreeHash;
		std::vector<unsigned int> pointersFreeVec(m_hashParams.m_numSDFBlocks, 0);
		for (unsigned int i = 0; i < heapCounterCPU; i++) {
			pointersFreeHash.insert(heapCPU[i]);
			pointersFreeVec[heapCPU[i]] = FREE_ENTRY;
		}
		if (pointersFreeHash.size() != heapCounterCPU) {
			throw MLIB_EXCEPTION("ERROR: duplicate free pointers in heap array");
		}
		 

		unsigned int numOccupied = 0;
		unsigned int numMinusOne = 0;
		unsigned int listOverallFound = 0;

		std::list<myint3Voxel> l;
		//std::vector<myint3Voxel> v;
		
		for (unsigned int i = 0; i < m_hashParams.m_hashBucketSize*m_hashParams.m_hashNumBuckets; i++) {
			if (hashCPU[i].ptr == -1) {
				numMinusOne++;
			}

			if (hashCPU[i].ptr != -2) {
				numOccupied++;	// != FREE_ENTRY
				myint3Voxel a;	
				a.x = hashCPU[i].pos.x;
				a.y = hashCPU[i].pos.y;
				a.z = hashCPU[i].pos.z;
				l.push_back(a);
				//v.push_back(a);

				unsigned int linearBlockSize = m_hashParams.m_SDFBlockSize*m_hashParams.m_SDFBlockSize*m_hashParams.m_SDFBlockSize;
				if (pointersFreeHash.find(hashCPU[i].ptr / linearBlockSize) != pointersFreeHash.end()) {
					throw MLIB_EXCEPTION("ERROR: ptr is on free heap, but also marked as an allocated entry");
				}
				pointersFreeVec[hashCPU[i].ptr / linearBlockSize] = LOCK_ENTRY;
			}
		}

		unsigned int numHeapFree = 0;
		unsigned int numHeapOccupied = 0;
		for (unsigned int i = 0; i < m_hashParams.m_numSDFBlocks; i++) {
			if		(pointersFreeVec[i] == FREE_ENTRY) numHeapFree++;
			else if (pointersFreeVec[i] == LOCK_ENTRY) numHeapOccupied++;
			else {
				throw MLIB_EXCEPTION("memory leak detected: neither free nor allocated");
			}
		}
		if (numHeapFree + numHeapOccupied == m_hashParams.m_numSDFBlocks) std::cout << "HEAP OK!" << std::endl;
		else throw MLIB_EXCEPTION("HEAP CORRUPTED");

		l.sort();
		size_t sizeBefore = l.size();
		l.unique();
		size_t sizeAfter = l.size();


		std::cout << "diff: " << sizeBefore - sizeAfter << std::endl;
		std::cout << "minOne: " << numMinusOne << std::endl;
		std::cout << "numOccupied: " << numOccupied << "\t numFree: " << getHeapFreeCount() << std::endl;
		std::cout << "numOccupied + free: " << numOccupied + getHeapFreeCount() << std::endl;
		std::cout << "numInFrustum: " << m_hashParams.m_numOccupiedBlocks << std::endl;

		SAFE_DELETE_ARRAY(heapCPU);
		SAFE_DELETE_ARRAY(hashCPU);

		//getchar();
	}
private:

	void create(const HashParams& params) {
		m_hashParams = params;
		m_hashData.allocate(m_hashParams);

		reset();
	}

	void destroy() {
		m_hashData.free();
	}

	void alloc(const DepthCameraData& depthCameraData, const DepthCameraParams& depthCameraParams, const unsigned int* d_bitMask) {
		//Start Timing
		if(GlobalAppState::get().s_timingsDetailledEnabled) { cutilSafeCall(cudaDeviceSynchronize()); m_timer.start(); }

		//resetHashBucketMutexCUDA(m_hashData, m_hashParams);
		//allocCUDA(m_hashData, m_hashParams, depthCameraData, depthCameraParams, d_bitMask);

		//allocate until all blocks are allocated
		unsigned int prevFree = getHeapFreeCount();
		while (1) {
			resetHashBucketMutexCUDA(m_hashData, m_hashParams);
			allocCUDA(m_hashData, m_hashParams, depthCameraData, depthCameraParams, d_bitMask);

			unsigned int currFree = getHeapFreeCount();

			if (prevFree != currFree) {
				prevFree = currFree;
			}
			else {
				break;
			}
		}

		// Stop Timing
		if(GlobalAppState::get().s_timingsDetailledEnabled) { cutilSafeCall(cudaDeviceSynchronize()); m_timer.stop(); TimingLog::totalTimeAlloc += m_timer.getElapsedTimeMS(); TimingLog::countTimeAlloc++; }
	}


	void compactifyHashEntries(const DepthCameraData& depthCameraData) {
		//Start Timing
		if(GlobalAppState::get().s_timingsDetailledEnabled) { cutilSafeCall(cudaDeviceSynchronize()); m_timer.start(); }

		fillDecisionArrayCUDA(m_hashData, m_hashParams, depthCameraData);
		m_hashParams.m_numOccupiedBlocks = 
			m_cudaScan.prefixSum(
				m_hashParams.m_hashNumBuckets*m_hashParams.m_hashBucketSize,
				m_hashData.d_hashDecision,
				m_hashData.d_hashDecisionPrefix);

		m_hashData.updateParams(m_hashParams);	//make sure numOccupiedBlocks is updated on the GPU

		compactifyHashCUDA(m_hashData, m_hashParams);

		// Stop Timing
		if(GlobalAppState::get().s_timingsDetailledEnabled) { cutilSafeCall(cudaDeviceSynchronize()); m_timer.stop(); TimingLog::totalTimeCompactifyHash += m_timer.getElapsedTimeMS(); TimingLog::countTimeCompactifyHash++; }

		//std::cout << "numOccupiedBlocks: " << m_hashParams.m_numOccupiedBlocks << std::endl;
	}

	void integrateDepthMap(const DepthCameraData& depthCameraData, const DepthCameraParams& depthCameraParams) {
		//Start Timing
		if(GlobalAppState::get().s_timingsDetailledEnabled) { cutilSafeCall(cudaDeviceSynchronize()); m_timer.start(); }

		integrateDepthMapCUDA(m_hashData, m_hashParams, depthCameraData, depthCameraParams);

		// Stop Timing
		if(GlobalAppState::get().s_timingsDetailledEnabled) { cutilSafeCall(cudaDeviceSynchronize()); m_timer.stop(); TimingLog::totalTimeIntegrate += m_timer.getElapsedTimeMS(); TimingLog::countTimeIntegrate++; }
	}

	void garbageCollect(const DepthCameraData& depthCameraData) {
		//only perform if enabled by global app state
		if (GlobalAppState::get().s_garbageCollectionEnabled) {
			
			if (m_numIntegratedFrames > 0 && m_numIntegratedFrames % GlobalAppState::get().s_garbageCollectionStarve == 0) {
				starveVoxelsKernelCUDA(m_hashData, m_hashParams);
			}

			garbageCollectIdentifyCUDA(m_hashData, m_hashParams);
			resetHashBucketMutexCUDA(m_hashData, m_hashParams);	//needed if linked lists are enabled -> for memeory deletion
			garbageCollectFreeCUDA(m_hashData, m_hashParams);
		} 
	}



	HashParams		m_hashParams;
	HashData		m_hashData;

	CUDAScan		m_cudaScan;
	unsigned int	m_numIntegratedFrames;	//used for garbage collect

	static Timer m_timer;
};
