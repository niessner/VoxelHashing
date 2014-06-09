Buffer<int> g_Hash : register(t0);
RWBuffer<uint> g_Histogram : register(u0);

#include "GlobalAppStateShaderBuffer.h.hlsl"
#include "SDFShaderBuffer.h.hlsl"
#include "KinectCameraUtil.h.hlsl"
#include "VoxelUtilHashSDF.h.hlsl"

#define NUM_GROUPS_X 1024 // has to be in sync with the other application code and the CPU !!!

#ifndef groupthreads
#define groupthreads 1
#endif
 
[numthreads(groupthreads, 1, 1)]
void computeHistogramHashSDFCS(int3 dTid : SV_DispatchThreadID, uint3 GTid : SV_GroupThreadID, uint GI: SV_GroupIndex, uint3 GID : SV_GroupID)
{
	uint bucketID = groupthreads*(GID.x + GID.y * NUM_GROUPS_X)+GI;
	if (bucketID < g_HashNumBuckets)
	{ 
		uint h = 0;
		HashEntry entry;
		for (uint i = 0; i < g_HashBucketSize; i++)
		{
			entry = getHashEntry(g_Hash, bucketID*g_HashBucketSize+i);
			if (entry.ptr != FREE_ENTRY)
			{
				h++;
			}
		} 
		InterlockedAdd(g_Histogram[h], 1);

		#ifdef HANDLE_COLLISIONS
			uint listLen = 0;
			const uint idxLastEntryInBucket = (bucketID+1)*g_HashBucketSize - 1;
			i = idxLastEntryInBucket;	//start with the last entry of the current bucket
			int offset = 0;
			HashEntry curr;	curr.offset = 0;
			//traverse list until end: memorize idx at list end and memorize offset from last element of bucket to list end

			unsigned int maxIter = 0;
			[allow_uav_condition]
			while (true && maxIter < g_MaxLoopIterCount) {
				offset = curr.offset;
				curr = getHashEntry(g_Hash, i);

				if (curr.offset == 0) {	//we have found the end of the list
					break;
				}
				i = idxLastEntryInBucket + curr.offset;		//go to next element in the list
				i %= (g_HashBucketSize * g_HashNumBuckets);	//check for overflow
				listLen++;

				maxIter++;
			}
			listLen = min(listLen, MAX_COLLISION_LINKED_LIST_SIZE-1);
			InterlockedAdd(g_Histogram[listLen + g_HashBucketSize + 1], 1);
		#endif
	}
}

[numthreads(1, 1, 1)]
void resetHistogramHashSDFCS(int3 dTid : SV_DispatchThreadID, uint3 GTid : SV_GroupThreadID, uint GI: SV_GroupIndex, uint3 GID : SV_GroupID)
{
	g_Histogram[GID.x] = 0;
}
 