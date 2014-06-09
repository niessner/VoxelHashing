
// should be set by the application
//#ifndef BLOCKSIZE
//#define BLOCKSIZE 1024
//#endif

#define DEFAULT_SIZE 512

#ifndef BUCKET_SIZE
#define BUCKET_SIZE DEFAULT_SIZE
#endif

#ifndef BUCKET_BLOCK_SIZE
#define BUCKET_BLOCK_SIZE DEFAULT_SIZE
#endif

#ifndef NUM_BUCKET_BLOCKS
#define NUM_BUCKET_BLOCKS DEFAULT_SIZE
#endif

#define DISPATCH_THREADS_X 128

Buffer<int> input : register(t0);
RWBuffer<int> output : register(u0);
 
groupshared int bucket[BUCKET_SIZE];	//TODO change!!!!
 
int CSWarpScan(uint lane, uint i)
{
	//if(lane >= 0x01) bucket[i] += bucket[i-0x01]; // IMPLICIT MEM BARRIER
	//if(lane >= 0x02) bucket[i] += bucket[i-0x02]; // IMPLICIT MEM BARRIER
	//if(lane >= 0x04) bucket[i] += bucket[i-0x04]; // IMPLICIT MEM BARRIER
	//if(lane >= 0x08) bucket[i] += bucket[i-0x08]; // IMPLICIT MEM BARRIER
	//if(lane >= 0x10) bucket[i] += bucket[i-0x10]; // IMPLICIT MEM BARRIER

	// micro optimization
	uint4 access = i - uint4(1,2,4,8);
	if(lane > 0) bucket[i]  += bucket[access.x]; // IMPLICIT MEM BARRIER
	if(lane > 1) bucket[i]  += bucket[access.y]; // IMPLICIT MEM BARRIER
	if(lane > 3) bucket[i]  += bucket[access.z]; // IMPLICIT MEM BARRIER
	if(lane > 7) bucket[i]  += bucket[access.w]; // IMPLICIT MEM BARRIER
	if(lane > 15) bucket[i] += bucket[i-0x10];  // IMPLICIT MEM BARRIER
	return bucket[i];
}

void CSScan(uint3 DTid, uint GI, int x)
{
	bucket[GI] = x;
	
	uint lane = GI & 31u;
	uint warp = GI >> 5u;
 
	x = CSWarpScan(lane, GI);
	GroupMemoryBarrierWithGroupSync();

	if (lane == 31) bucket[warp] = x;	
	GroupMemoryBarrierWithGroupSync();

	if (warp == 0)	CSWarpScan(lane, lane);
	GroupMemoryBarrierWithGroupSync();
	

	//if (warp > 0)   // uncomment if fail
	  x += bucket[warp-1];
	output[DTid.x] = x;
}

 
// scan buckets
[numthreads(BUCKET_SIZE, 1, 1)]
void CSScanBucket(uint3 DTid: SV_DispatchThreadID, uint3 GTid : SV_GroupThreadID, uint GI : SV_GroupIndex, uint3 Gid : SV_GroupID)
{
	const uint idx = (Gid.x + Gid.y*DISPATCH_THREADS_X)*BUCKET_SIZE + GI;
	int x = input[idx];
	CSScan(idx, GI, x);	

	//int x = input[DTid.x];
	//CSScan(DTid, GI, x);	
}
  
// record and scan the sum of each bucket
[numthreads(BUCKET_BLOCK_SIZE, 1, 1)]
void CSScanBucketResults(uint3 DTid: SV_DispatchThreadID, uint GI : SV_GroupIndex)
{
	int x = input[(DTid.x+1) * BUCKET_SIZE -1];
	CSScan(DTid, GI, x);
}

// record and scan the sum of each bucket block
[numthreads(NUM_BUCKET_BLOCKS, 1, 1)]
void CSScanBucketBlockResults(uint3 DTid: SV_DispatchThreadID, uint GI : SV_GroupIndex)
{
	int x = input[GI * BUCKET_BLOCK_SIZE -1];
	CSScan(DTid, GI, x);
}
 
////////////
// APPLY RESULTS
////////////

// add the bucket block scanned result to each bucket block to get the buck block results
[numthreads(BUCKET_BLOCK_SIZE, 1, 1)]
void CSScanApplyBucketBlockResults(uint3 DTid : SV_DispatchThreadID, uint GI : SV_GroupIndex)
{
	output[DTid.x] += input[DTid.x/BUCKET_BLOCK_SIZE];
}

// add the bucket block scanned result to each bucket to get the final result
[numthreads(BUCKET_SIZE, 1, 1)]
void CSScanApplyBucketResults(uint3 DTid : SV_DispatchThreadID, uint GI : SV_GroupIndex, uint3 Gid : SV_GroupID)
{
	const uint idx = (Gid.x + Gid.y*DISPATCH_THREADS_X)*BUCKET_SIZE + GI;
	output[idx] += input[((idx)/BUCKET_SIZE)-1];

	//output[DTid.x] += input[((DTid.x)/BUCKET_SIZE)-1];
}
