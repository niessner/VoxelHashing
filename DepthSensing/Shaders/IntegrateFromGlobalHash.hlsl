struct SDFBlockDesc
{
	int3 pos;
	int ptr;
};

RWBuffer<int> g_Hash : register(u1);
AppendStructuredBuffer<uint> g_heapAppend : register(u2);
RWBuffer<uint> g_HashBucketMutex : register (u3);

#include "GlobalAppStateShaderBuffer.h.hlsl"
#include "SDFShaderBuffer.h.hlsl"
#include "KinectCameraUtil.h.hlsl"
#include "VoxelUtilHashSDF.h.hlsl"

cbuffer consts : register(b1)
{
	uint nSDFBlockDescs;
	float radius;
	uint start;
	uint aling1;

	float3 cameraPosition;
	uint aling2;
};

//-------------------------------------------------------
// Pass 1: Find all SDFBlocks that have to be transfered
//-------------------------------------------------------

AppendStructuredBuffer<SDFBlockDesc> g_output : register(u0);

#define NUM_GROUPS_X 1024 // has to be in sync with the other application code and the CPU !!!

#ifndef groupthreads
#define groupthreads 1
#endif

static const uint linBlockSize = SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;

[numthreads(groupthreads, 1, 1)]
void integrateFromGlobalHashPass1CS(int3 dTid : SV_DispatchThreadID, uint3 GTid : SV_GroupThreadID, uint GI: SV_GroupIndex, uint3 GID : SV_GroupID)
{
	uint bucketID = start+groupthreads*(GID.x + GID.y * NUM_GROUPS_X)+GI;
	if(bucketID < g_HashNumBuckets*g_HashBucketSize)
	{
		HashEntry entry = getHashEntry(g_Hash, bucketID);

		float3 posWorld = SDFBlockToWorld(entry.pos);
		float d = distance(posWorld, cameraPosition);

		if(entry.ptr != FREE_ENTRY && d > radius)
		{
			// Write
			SDFBlockDesc d;
			d.pos = entry.pos;
			d.ptr = entry.ptr;

			#ifndef HANDLE_COLLISIONS
				g_output.Append(d);
				g_heapAppend.Append(entry.ptr/linBlockSize);
				deleteHashEntry(g_Hash, bucketID);
			#endif
			#ifdef HANDLE_COLLISIONS
				//if there is an offset or hash doesn't belong to the bucket (linked list)
				if (entry.offset != 0 || computeHashPos(entry.pos) != bucketID / g_HashBucketSize) {
					
					if (deleteHashEntryElement(g_Hash, g_heapAppend, g_HashBucketMutex, entry.pos)) {
						g_output.Append(d);
						//deleteHashEntry(g_Hash, bucketID);
					}
				} else {
					g_output.Append(d);
					g_heapAppend.Append(entry.ptr/linBlockSize);
					deleteHashEntry(g_Hash, bucketID);
				}
			#endif
		}
	}
}

//-------------------------------------------------------
// Pass 2: Copy SDFBlocks to output buffer
//-------------------------------------------------------

StructuredBuffer<SDFBlockDesc> g_SDFBLockDescs : register(t0);
RWBuffer<float> g_SDFBlocksSDF : register(u0);
RWBuffer<int> g_outputSDFBlocks : register(u1);
RWBuffer<int> g_SDFBlocksRGBW : register(u2);

[numthreads(SDF_BLOCK_SIZE*SDF_BLOCK_SIZE*SDF_BLOCK_SIZE, 1, 1)]
void integrateFromGlobalHashPass2CS(int3 dTid : SV_DispatchThreadID, uint3 GTid : SV_GroupThreadID, uint GI: SV_GroupIndex, uint3 GID : SV_GroupID)
{
	uint idxBlock = GID.x + GID.y * NUM_GROUPS_X;

	if(idxBlock < nSDFBlockDescs)
	{
		SDFBlockDesc desc = g_SDFBLockDescs[idxBlock];

		// Copy SDF block to CPU
		g_outputSDFBlocks[2*(linBlockSize*idxBlock+GI)+0] = asint(g_SDFBlocksSDF[desc.ptr+GI]);
		g_outputSDFBlocks[2*(linBlockSize*idxBlock+GI)+1] = g_SDFBlocksRGBW[desc.ptr+GI];

		//// Reset SDF Block
		//g_SDFBlocks[2*(desc.ptr+GI)+0] = 0;
		//g_SDFBlocks[2*(desc.ptr+GI)+1] = 0;
		deleteVoxel(g_SDFBlocksSDF, g_SDFBlocksRGBW, desc.ptr+GI);
	}
}
