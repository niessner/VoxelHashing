#pragma once

/************************************************************************/
/* The core scene data structure that implements the hashing scheme     */
/************************************************************************/

#include "stdafx.h"

#include "DX11Utils.h"
#include "DX11ScanCS.h"
#include "TimingLog.h"
#include "GlobalAppState.h"

#define THREAD_GROUP_SIZE_SCENE_REP 8
#define NUM_GROUPS_X 1024


struct CB_VOXEL_HASH_SDF {
	D3DXMATRIX		m_RigidTransform;
	D3DXMATRIX		m_RigidTransformInverse;
	unsigned int	m_HashNumBuckets;
	unsigned int	m_HashBucketSize;
	unsigned int	m_InputImageWidth;
	unsigned int	m_InputImageHeight;
	float			m_VirtualVoxelSize;
	float			m_VirtualVoxelResolutionScalar;
	unsigned int	m_NumSDFBlocks;
	unsigned int	m_NumOccupiedSDFBlocks;
};



class DX11SceneRepHashSDF	{
public:
	DX11SceneRepHashSDF();
	~DX11SceneRepHashSDF();

	//! static init and destroy (for shaders)
	static HRESULT OnD3D11CreateDevice(ID3D11Device* pd3dDevice);
	static void OnD3D11DestroyDevice();
	 
	HRESULT Init(ID3D11Device* pd3dDevice, bool justHash = false, unsigned int hashNumBuckets = 300000, unsigned int hashBucketSize = 10, unsigned int numSDFBlocks = 100000, float voxelSize = 0.005f);

	void Destroy();

	//! Surface Fusion
	void Integrate(ID3D11DeviceContext* context, ID3D11ShaderResourceView* inputDepth, ID3D11ShaderResourceView* inputColor, ID3D11ShaderResourceView* bitMask, const mat4f* rigidTransform);
		
	//! Moves hash entries around
	void RemoveAndIntegrateToOther(ID3D11DeviceContext* context, DX11SceneRepHashSDF* other, const mat4f* lastRigid, bool moveOutsideFrustum);
		
	//! Resets the Data structure
	void Reset(ID3D11DeviceContext* context);

	//! Number of free hash blocks (uses the last compactify result)
	unsigned int GetHeapFreeCount(ID3D11DeviceContext* context);


	HRESULT DumpPointCloud(const std::string &filename, ID3D11Device* pDevice, ID3D11DeviceContext* pd3dImmediateContext, unsigned int minWeight = 1, bool justOccupied = false);


	const mat4f& GetLastRigidTransform() const	{	
		return m_LastRigidTransform;
	}

	unsigned int GetNumIntegratedImages() {
		return m_NumIntegratedImages;
	}

	ID3D11ShaderResourceView*	GetHashSRV() {
		return m_HashSRV;
	}
	ID3D11UnorderedAccessView*	GetHashUAV() {
		return m_HashUAV;
	}
	ID3D11ShaderResourceView*	GetSDFBlocksSDFSRV() {
		return m_SDFBlocksSDFSRV;
	}
	ID3D11ShaderResourceView*	GetSDFBlocksRGBWSRV() {
		return m_SDFBlocksRGBWSRV;
	}
	ID3D11UnorderedAccessView*	GetSDFBlocksSDFUAV() {
		return m_SDFBlocksSDFUAV;
	}
	ID3D11UnorderedAccessView* GetSDFBlocksRGBWUAV() {
		return m_SDFBlocksRGBWUAV;
	}
	ID3D11UnorderedAccessView*	GetHeapUAV() {
		return m_HeapUAV;
	}
	ID3D11UnorderedAccessView*	GetHeapStaticUAV() {
		return m_HeapStaticUAV;
	}
	ID3D11UnorderedAccessView* GetAndClearHashBucketMutex(ID3D11DeviceContext* context) {
		UINT cleanUAV[] = {0,0,0,0};
		context->ClearUnorderedAccessViewUint(m_HashBucketMutexUAV, cleanUAV);
		return m_HashBucketMutexUAV;
	}
	unsigned int GetHashBucketSize() {
		return m_HashBucketSize;
	}
	unsigned int GetHashNumBuckets() {
		return m_HashNumBuckets;
	}
	unsigned int GetSDFBlockSize() {
		return m_SDFBlockSize;
	}
	//! returns the (virtual) voxel size in meters
	float GetVoxelSize() {
		return m_VirtualVoxelSize;
	}

	unsigned int GetNumOccupiedHashEntries() {
		return m_NumOccupiedHashEntries;
	}

	ID3D11ShaderResourceView* GetHashCompactifiedSRV() {
		return m_HashCompactifiedSRV;
	}

	void SetEnableGarbageCollect(bool b) {
		m_bEnableGarbageCollect = b;
	}

	ID3D11Buffer* MapAndGetConstantBuffer(ID3D11DeviceContext* context) {
		MapConstantBuffer(context);
		return m_SDFVoxelHashCB;
	}

	//! needs to be called if integrate is disabled (to get a valid compactified hash)
	void RunCompactifyForView(ID3D11DeviceContext* context) {
		CompactifyHashEntries(context);
	}

	void StarveVoxelWeights(ID3D11DeviceContext* context) {

		//MapConstantBuffer(context);

		context->CSSetConstantBuffers(0, 1, &m_SDFVoxelHashCB);
		ID3D11Buffer* CBGlobalAppState = GlobalAppState::getInstance().MapAndGetConstantBuffer(context);
		context->CSSetConstantBuffers(8, 1, &CBGlobalAppState);
		context->CSSetShaderResources(4, 1, &m_HashCompactifiedSRV);
		context->CSSetUnorderedAccessViews(1, 1, &m_SDFBlocksSDFUAV, NULL);
		context->CSSetUnorderedAccessViews(7, 1, &m_SDFBlocksRGBWUAV, NULL);
		context->CSSetShader(s_SDFVoxelStarve, NULL, 0);

		// Start Compute Shader
		unsigned int dimX = NUM_GROUPS_X;
		unsigned int dimY = (m_NumOccupiedHashEntries + NUM_GROUPS_X - 1) / NUM_GROUPS_X;
		assert(dimX <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);	
		assert(dimY <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);	
		
		context->Dispatch(dimX, dimY, 1);


		ID3D11ShaderResourceView* nullSRV[] = { NULL, NULL, NULL, NULL, NULL, NULL };
		ID3D11UnorderedAccessView* nullUAV[] = { NULL, NULL, NULL, NULL, NULL, NULL };
		ID3D11Buffer* nullCB[] = { NULL };

		context->CSSetShaderResources(4, 1, nullSRV);
		context->CSSetUnorderedAccessViews(1, 1, nullUAV, 0);
		context->CSSetUnorderedAccessViews(7, 1, nullUAV, 0);
		context->CSSetConstantBuffers(0, 1, nullCB);
		context->CSSetConstantBuffers(8, 1, nullCB);
		context->CSSetShader(0, 0, 0);
	}


	void DumpHashToDisk(const std::string& filename, float dumpRadius = 0.0f, vec3f dumpCenter = vec3f(0.0f,0.0f,0.0f)) {
		if (dumpRadius < 0.0f) dumpRadius = 0.0f;
		struct HashEntry
		{
			point3d<short> pos;		//hash position (lower left corner of SDFBlock))
			unsigned short offset;	//offset for collisions
			int ptr;				//pointer into heap to SDFBlock
		};
		struct Voxel
		{
			float sdf;
			vec3uc color;
			unsigned char weight;
		};
		struct VoxelBlock 
		{
			Voxel voxels[SDF_BLOCK_SIZE*SDF_BLOCK_SIZE*SDF_BLOCK_SIZE];	//typically of size 512
		};

		HashEntry* hashEntries = (HashEntry*)CreateAndCopyToDebugBuf(DXUTGetD3D11Device(), DXUTGetD3D11DeviceContext(), m_Hash, true);
		float*	voxelsSDF = (float*)CreateAndCopyToDebugBuf(DXUTGetD3D11Device(), DXUTGetD3D11DeviceContext(), m_SDFBlocksSDF, true);
		int*	voxelsRGBW = (int*)CreateAndCopyToDebugBuf(DXUTGetD3D11Device(), DXUTGetD3D11DeviceContext(), m_SDFBlocksRGBW, true);
		
		const unsigned int numHashEntries = m_HashNumBuckets*m_HashBucketSize;
		const unsigned int numVoxels = m_SDFNumBlocks*SDF_BLOCK_SIZE*SDF_BLOCK_SIZE*SDF_BLOCK_SIZE;

		unsigned int occupiedBlocks = 0;
		SparseGrid3D<VoxelBlock> grid;
		for (unsigned int i = 0; i < numHashEntries; i++) {
			const unsigned int ptr = hashEntries[i].ptr;
			if (ptr != -2) {
				VoxelBlock vBlock;
				//memcpy(vBlock.voxels, &voxels[ptr], sizeof(VoxelBlock));
				for (unsigned int j = 0; j < SDF_BLOCK_SIZE*SDF_BLOCK_SIZE*SDF_BLOCK_SIZE; j++) {
					vBlock.voxels[j].sdf = voxelsSDF[ptr+j];
					int last = voxelsRGBW[ptr+j];
					vBlock.voxels[j].weight = last & 0x000000ff;
					last >>= 0x8;
					vBlock.voxels[j].color.x = last & 0x000000ff;
					last >>= 0x8;
					vBlock.voxels[j].color.y = last & 0x000000ff;
					last >>= 0x8;
					vBlock.voxels[j].color.z = last & 0x000000ff;
				}
				vec3i coord(hashEntries[i].pos.x, hashEntries[i].pos.y, hashEntries[i].pos.z);
				//std::cout << coord << std::endl;
				if (dumpRadius == 0.0f) {
					grid(coord) = vBlock;
				} else {
					//vec3f center = GetLastRigidTransform()*dumpCenter;
					vec3f center = dumpCenter;
					vec3f coordf = vec3f(coord*SDF_BLOCK_SIZE)*m_VirtualVoxelSize;
					if (vec3f::dist(center, coordf) <= dumpRadius) {
						grid(coord) = vBlock;
					}					
				}

				occupiedBlocks++;

				//for (unsigned int j = 0; j < SDF_BLOCK_SIZE*SDF_BLOCK_SIZE*SDF_BLOCK_SIZE; j++) {
				//	if (vBlock.voxels[j].weight != 0) {
				//		std::cout << vBlock.voxels[j].sdf << std::endl;
				//	}
				//	//std::cout << (unsigned int)vBlock.voxels[j].weight << " ";
				//}
			}
		}

		std::cout << "found " << occupiedBlocks << " voxel blocks   "; 
		grid.writeBinaryDump(filename);

		//SparseGrid3D<VoxelBlock> reRead;
		//reRead.readBinaryDump(filename);
		//for (auto iter = reRead.begin(); iter != reRead.end(); iter++) {
		//	std::cout << iter->first << std::endl;
		//}

		void* hashEntriesv = (void*)hashEntries;
		void* voxelsSDFv = (void*)voxelsSDF;
		void* voxelsRGBWv = (void*)voxelsRGBW;
		SAFE_DELETE_ARRAY(hashEntriesv);
		SAFE_DELETE_ARRAY(voxelsSDFv);
		SAFE_DELETE_ARRAY(voxelsRGBWv);

	}
	
	struct HashEntry
	{
		bool operator<(const HashEntry& other) {
			return ptr < other.ptr;
		}
		int3 pos;	//hash position (lower left corner of SDFBlock))
		int offset;	//offset for collisions
		int ptr;	//pointer into heap to SDFBlock
		int i;
	};


	void DebugHash0() {
		unsigned int* cpuMemory = (unsigned int*)CreateAndCopyToDebugBuf(DXUTGetD3D11Device(), DXUTGetD3D11DeviceContext(), m_Hash, true);
		std::vector<HashEntry> ptrs;
		for(unsigned int i = 0; i< m_HashNumBuckets*m_HashBucketSize; i++)
		{
			HashEntry entry;
			int i0 = cpuMemory[3*i+0];
			entry.pos.x = i0 & 0x0000ffff;
			if (entry.pos.x & (0x1 << 15))	entry.pos.x |= 0xffff0000;
			entry.pos.y = i0 >> 16;
			if (entry.pos.y & (0x1 << 15))	entry.pos.y |= 0xffff0000;

			int i1 = cpuMemory[3*i+1];
			entry.pos.z = i1 & 0x0000ffff;
			if (entry.pos.z & (0x1 << 15))	entry.pos.z |= 0xffff0000;

			entry.offset = i1 >> 16;
			if (entry.offset & (0x1 << 15)) entry.offset |= 0xffff0000;

			entry.ptr = cpuMemory[3*i+2];

			entry.i = i;
			if(entry.ptr != -2)
			{
				//std::cout << "test" << std::endl;
				ptrs.push_back(entry);
			}
		}
		std::sort(ptrs.begin(), ptrs.end());

		SAFE_DELETE_ARRAY(cpuMemory);
	}

	struct VoxelDebug
	{
		float sdf;
		uint3 color;
		unsigned int weight;
	};

	void DebugSDFBlocks1() {	
		unsigned int* cpuMemory = (unsigned int*)CreateAndCopyToDebugBuf(DXUTGetD3D11Device(), DXUTGetD3D11DeviceContext(), m_SDFBlocksSDF, true);
		unsigned int count = 0;
		const unsigned int numVoxels = m_SDFNumBlocks*SDF_BLOCK_SIZE*SDF_BLOCK_SIZE*SDF_BLOCK_SIZE;
		for (unsigned int i = 0; i< numVoxels; i++)
		{
			VoxelDebug voxel;
			float* f = (float*)&cpuMemory[2*i+0];
			voxel.sdf = *f;
			int last = cpuMemory[2*i+1];
			voxel.weight = last & 0x000000ff;
			last >>= 0x8;
			voxel.color.x = last & 0x000000ff;
			last >>= 0x8;
			voxel.color.y = last & 0x000000ff;
			last >>= 0x8;
			voxel.color.z = last & 0x000000ff;

			if(voxel.weight != 0)
			{
				//	std::cout  << "a" << std::endl;
				count++;
			}
		}

		SAFE_DELETE_ARRAY(cpuMemory);
	}



	void DebugHash() {
		unsigned int *cpuMemory = (unsigned int*)CreateAndCopyToDebugBuf(DXUTGetD3D11Device(), DXUTGetD3D11DeviceContext(), m_Hash, true);

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

		unsigned int numOccupied = 0;
		unsigned int numMinusOne = 0;
		std::list<myint3Voxel> l;
		std::vector<myint3Voxel> v;
		unsigned int listOverallFound = 0;
		for (unsigned int i = 0; i < m_HashBucketSize * m_HashNumBuckets; i++)
		{
			if (cpuMemory[3*i+2] == -1) {
				numMinusOne++;
			}
			
			myint3Voxel a;
			a.x = cpuMemory[3*i] & 0x0000ffff;
			if (a.x & (0x1 << 15)) a.x |= 0xffff0000;
			a.y = cpuMemory[3*i] >> 16;
			if (a.y & (0x1 << 15)) a.y |= 0xffff0000;
			a.z = cpuMemory[3*i+1] & 0x0000ffff;
			if (a.z & (0x1 << 15)) a.z |= 0xffff0000;
			a.offset = cpuMemory[3*i+1] >> 16;
			a.ptr = cpuMemory[3*i+2];
			a.i = i;

			if (a.ptr != -2) {
				numOccupied++;

				l.push_back(a);
				v.push_back(a);

				//if (a.offset > 0) {
				//	listOverallFound++;
				//	int first = i;
				//	int c = 0;
				//	while (true) {
				//		int idx = first + a.offset;
				//		int currOffset = cpuMemory[3*idx+1] >> 16;
				//		a.offset = currOffset;
				//		if (a.offset == 0) break;
				//		c++;
				//		if (c > 5) {
				//			int asdf = 3;
				//		}
				//	}
				//}
			}
		}


		//std::sort(l.begin(), l.end(), std::less<int>());
		l.sort();

		//std::list<myint3>::iterator iter, prev;
		//iter = l.begin();
		//prev = iter;
		//iter++;	
		//for (; iter != l.end(); iter++) {
		//	if (*prev == *iter) {
		//		std::cout << iter->i << " " << prev->i << " " << " || " << iter->x << " " << iter->y << " " << iter->z << std::endl;
		//	}
		//	prev = iter;
		//}

		size_t sizeBefore = l.size();
		l.unique();
		size_t sizeAfter = l.size();



		std::cout << "diff: " << sizeBefore - sizeAfter << std::endl;
		std::cout << "minOne: " << numMinusOne << std::endl;
		std::cout << "numOccupied: " << numOccupied << "\t numFree: " << GetHeapFreeCount(DXUTGetD3D11DeviceContext()) << std::endl;
		std::cout << "numOccupied + free: " << numOccupied + GetHeapFreeCount(DXUTGetD3D11DeviceContext()) << std::endl;
		std::cout << "numInFrustum: " << m_NumOccupiedHashEntries << std::endl;

		SAFE_DELETE_ARRAY(cpuMemory);
	}



private:
	//! maps the constant buffer to the GPU
	void MapConstantBuffer( ID3D11DeviceContext* context );


	
	//! for a given depth map allocates SDFBlocks - note that only 1 block per bucket can be allocated per frame!
	void Alloc( ID3D11DeviceContext* context, ID3D11ShaderResourceView* inputDepth, ID3D11ShaderResourceView* inputColor, ID3D11ShaderResourceView* bitMask);

	void CompactifyHashEntries( ID3D11DeviceContext* context );

	
	void IntegrateDepthMap( ID3D11DeviceContext* context, ID3D11ShaderResourceView* inputDepth, ID3D11ShaderResourceView* inputColor );

	void GarbageCollect( ID3D11DeviceContext* context );
	
	HRESULT CreateBuffers( ID3D11Device* pd3dDevice );

	
	///////////////////////
	// Control variables
	///////////////////////
	bool m_bEnableGarbageCollect;

	/////////////////////////////////
	// Variables of the current frame (set by integrate)
	//////////////////////
	unsigned int m_NumOccupiedHashEntries;

	//////////////////////////////
	// standard member variables
	//////////////////////////////

	bool						m_JustHashAndNoSDFBlocks;	//indicates that it's a global hash and does not have its own heap
	mat4f						m_LastRigidTransform;
	unsigned int				m_HashNumBuckets;
	unsigned int				m_HashBucketSize;
	unsigned int				m_SDFBlockSize;
	unsigned int				m_SDFNumBlocks;
	float						m_VirtualVoxelSize;
	unsigned int				m_NumIntegratedImages;

	ID3D11Buffer*				m_Hash;
	ID3D11UnorderedAccessView*	m_HashUAV;
	ID3D11ShaderResourceView*	m_HashSRV;

	//! for allocation phase to lock has buckets
	ID3D11Buffer*				m_HashBucketMutex;
	ID3D11UnorderedAccessView*	m_HashBucketMutexUAV;
	ID3D11ShaderResourceView*	m_HashBucketMutexSRV;
	
	//! for hash compactification
	ID3D11Buffer*				m_HashIntegrateDecision;
	ID3D11UnorderedAccessView*	m_HashIntegrateDecisionUAV;
	ID3D11ShaderResourceView*	m_HashIntegrateDecisionSRV;

	ID3D11Buffer*				m_HashIntegrateDecisionPrefix;
	ID3D11UnorderedAccessView*	m_HashIntegrateDecisionPrefixUAV;
	ID3D11ShaderResourceView*	m_HashIntegrateDecisionPrefixSRV;
	
	ID3D11Buffer*				m_HashCompactified;
	ID3D11UnorderedAccessView*	m_HashCompactifiedUAV;
	ID3D11ShaderResourceView*	m_HashCompactifiedSRV;


	//! heap to manage free sub-blocks
	ID3D11Buffer*				m_Heap;
	ID3D11ShaderResourceView*	m_HeapSRV;
	ID3D11UnorderedAccessView*	m_HeapUAV;
	ID3D11UnorderedAccessView*	m_HeapStaticUAV;	//requires another buffer
	ID3D11Buffer*				m_HeapFreeCount;

	//! sub-blocks that contain 8x8x8 voxels (linearized); are allocated by heap
	ID3D11Buffer*				m_SDFBlocksSDF;
	ID3D11ShaderResourceView*	m_SDFBlocksSDFSRV;
	ID3D11UnorderedAccessView*	m_SDFBlocksSDFUAV;

	ID3D11Buffer*				m_SDFBlocksRGBW;
	ID3D11ShaderResourceView*	m_SDFBlocksRGBWSRV;
	ID3D11UnorderedAccessView*	m_SDFBlocksRGBWUAV;


	ID3D11Buffer*				m_SDFVoxelHashCB;


	static ID3D11ComputeShader*	s_SDFVoxelHashResetHeap;
	static ID3D11ComputeShader*	s_SDFVoxelHashResetHash;
	static ID3D11ComputeShader*	s_SDFVoxelHashAlloc;
	static ID3D11ComputeShader*	s_SDFVoxelHashIntegrate;
	static ID3D11ComputeShader*	s_SDFVoxelStarve;

	static ID3D11ComputeShader*	s_HashDecisionArrayFiller;
	static ID3D11ComputeShader*	s_HashCompactify;
	static DX11ScanCS			s_PrefixSumScan;

	static ID3D11ComputeShader*	s_GarbageCollectIdentify;
	static ID3D11ComputeShader*	s_GarbageCollectIdentifyOld;
	static ID3D11ComputeShader*	s_GarbageCollectFree;

	static ID3D11ComputeShader*	s_SDFVoxelHashRemoveAndIntegrateOutFrustum;
	static ID3D11ComputeShader*	s_SDFVoxelHashRemoveAndIntegrateInFrustum;

	static Timer s_Timer;
};

