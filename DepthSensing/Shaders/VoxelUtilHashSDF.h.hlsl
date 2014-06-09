#ifndef _VOXEL_UTIL_HASH_SDF_H_
#define _VOXEL_UTIL_HASH_SDF_H_

// should be set by application
#ifndef groupthreads
#define groupthreads 1
#endif

//// should be set by application
//#ifndef HASH_BUCKET_SIZE
//#define HASH_BUCKET_SIZE 10
//#endif

#ifndef MINF
#define MINF asfloat(0xff800000)
#endif

#ifndef PINF
#define PINF asfloat(0x7f800000)
#endif

//status flags for hash entries
static const int LOCK_ENTRY = -1;
static const int FREE_ENTRY = -2;
static const int NO_OFFSET = 0;

//static const uint	g_WeightSample = 3;			//weight per sample (per integration step)
//static const uint	g_WeightMax = 255;			//maximum weight per voxel
//static const float	g_Truncation = 0.05f;		//5cm world space
//static const float	g_TruncScale = 0.02f;		//how to adapt the truncation: per distance meter, increase truncation by 1 cm

//#define HANDLE_COLLISIONS 1

struct HashEntry
{
	int3 pos;	//hash position (lower left corner of SDFBlock))
	int offset;	//offset for collisions
	int ptr;	//pointer into heap to SDFBlock
};

struct Voxel
{
	float sdf;
	uint3 color;
	uint weight;
};

//! computes the (local) virtual voxel pos of an index; idx in [0;511]
int3 delinearizeVoxelIndex(uint idx)
{
	uint x = idx % SDF_BLOCK_SIZE;
	uint y = (idx % (SDF_BLOCK_SIZE * SDF_BLOCK_SIZE)) / SDF_BLOCK_SIZE;
	uint z = idx / (SDF_BLOCK_SIZE * SDF_BLOCK_SIZE);	
	return uint3(x,y,z);
}

//! computes the linearized index of a local virtual voxel pos; pos in [0;7]^3
uint linearizeVoxelPos(int3 virtualVoxelPos)
{
	return  virtualVoxelPos.z * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE +
			virtualVoxelPos.y * SDF_BLOCK_SIZE +
			virtualVoxelPos.x;
}

HashEntry getHashEntry(Buffer<int> hash, in uint id)
{
	HashEntry entry;
	int i0 = hash[3*id+0];
	entry.pos.x = i0 & 0x0000ffff;
	if (entry.pos.x & (0x1 << 15))	entry.pos.x |= 0xffff0000;
	entry.pos.y = i0 >> 16;
	if (entry.pos.y & (0x1 << 15))	entry.pos.y |= 0xffff0000;

	int i1 = hash[3*id+1];
	entry.pos.z = i1 & 0x0000ffff;
	if (entry.pos.z & (0x1 << 15))	entry.pos.z |= 0xffff0000;

	entry.offset = i1 >> 16;
	if (entry.offset & (0x1 << 15)) entry.offset |= 0xffff0000;
	
	entry.ptr = hash[3*id+2];
	return entry;
}

HashEntry getHashEntry(RWBuffer<int> hash, in uint id)
{
	HashEntry entry;
	int i0 = hash[3*id+0];
	entry.pos.x = i0 & 0x0000ffff;
	if (entry.pos.x & (0x1 << 15))	entry.pos.x |= 0xffff0000;
	entry.pos.y = i0 >> 16;
	if (entry.pos.y & (0x1 << 15))	entry.pos.y |= 0xffff0000;

	int i1 = hash[3*id+1];
	entry.pos.z = i1 & 0x0000ffff;
	if (entry.pos.z & (0x1 << 15))	entry.pos.z |= 0xffff0000;

	entry.offset = i1 >> 16;
	if (entry.offset & (0x1 << 15)) entry.offset |= 0xffff0000;

	entry.ptr = hash[3*id+2];
	return entry;
}

void setHashEntry(RWBuffer<int> hash, in uint id, const in HashEntry entry)
{
	hash[3*id+0] = (entry.pos.y << 16) | (entry.pos.x & 0x0000ffff);
	hash[3*id+1] = (entry.offset << 16) | (entry.pos.z & 0x0000ffff);
	hash[3*id+2] = entry.ptr;
}

void deleteHashEntry(RWBuffer<int> hash, in uint id) 
{
	hash[3*id+0] = 0;
	hash[3*id+1] = 0;
	hash[3*id+2] = FREE_ENTRY;
}

Voxel getVoxel(RWBuffer<float> sdfBlocksSDF, RWBuffer<int> sdfBlocksRGBW, in uint id)
{
	Voxel voxel;
	voxel.sdf = sdfBlocksSDF[id];
	int last = sdfBlocksRGBW[id];
	voxel.weight = last & 0x000000ff;
	last >>= 0x8;
	voxel.color.x = last & 0x000000ff;
	last >>= 0x8;
	voxel.color.y = last & 0x000000ff;
	last >>= 0x8;
	voxel.color.z = last & 0x000000ff;
	//voxel.color.z = last;
	return voxel;
}

Voxel getVoxel(Buffer<float> sdfBlocksSDF, Buffer<int> sdfBlocksRGBW, in uint id)
{
	Voxel voxel;
	voxel.sdf = sdfBlocksSDF[id];
	int last = sdfBlocksRGBW[id];
	voxel.weight = last & 0x000000ff;
	last >>= 0x8;
	voxel.color.x = last & 0x000000ff;
	last >>= 0x8;
	voxel.color.y = last & 0x000000ff;
	last >>= 0x8;
	voxel.color.z = last & 0x000000ff;
	//voxel.color.z = last;
	return voxel;
}

void setVoxel(RWBuffer<float> sdfBlocksSDF, RWBuffer<int> sdfBlocksRGBW, in uint id, in Voxel voxel)
{
	sdfBlocksSDF[id] = voxel.sdf;
	int last = 0;
	last |= voxel.color.z & 0x000000ff;
	last <<= 8;
	last |= voxel.color.y & 0x000000ff;
	last <<= 8;
	last |= voxel.color.x & 0x000000ff;
	last <<= 8;
	last |= voxel.weight & 0x000000ff;
	//last |= voxel.color.z;
	//last <<= 8;
	//last |= voxel.color.y;
	//last <<= 8;
	//last |= voxel.color.x;
	//last <<= 8;
	//last |= voxel.weight;
	sdfBlocksRGBW[id] = last;
}

void starveVoxel(RWBuffer<int> sdfBlocksRGBW, in uint id) {
	int last = sdfBlocksRGBW[id];
	int weight = last & 0x000000ff;
	weight = max(0, weight-1);
	last &= 0xffffff00;
	last |= weight;
	sdfBlocksRGBW[id] = last;
}

void deleteVoxel(RWBuffer<float> sdfBlocksSDF, RWBuffer<int> sdfBlocksRGBW, in uint id) {
	sdfBlocksSDF[id] = 0;
	sdfBlocksRGBW[id] = 0;
}


float3 worldToVirtualVoxelPosFloat(in float3 pos)
{
	return pos*g_VirtualVoxelResolutionScalar;
}

int3 worldToVirtualVoxelPos(in float3 pos)
{
	const float3 p = pos*g_VirtualVoxelResolutionScalar;
	return (int3)(p+sign(p)*0.5f);
}

int3 virtualVoxelPosToSDFBlock(int3 virtualVoxelPos)
{
	if (virtualVoxelPos.x < 0) virtualVoxelPos.x -= SDF_BLOCK_SIZE-1;
	if (virtualVoxelPos.y < 0) virtualVoxelPos.y -= SDF_BLOCK_SIZE-1;
	if (virtualVoxelPos.z < 0) virtualVoxelPos.z -= SDF_BLOCK_SIZE-1;

	return virtualVoxelPos/SDF_BLOCK_SIZE;
}

// Computes virtual voxel position of corner sample position
int3 SDFBlockToVirtualVoxelPos(int3 sdfBlock)
{
	return sdfBlock*SDF_BLOCK_SIZE;
}

float3 virtualVoxelPosToWorld(in int3 pos)
{
	return float3(pos)*g_VirtualVoxelSize;
}

float3 SDFBlockToWorld(int3 sdfBlock)
{
	return virtualVoxelPosToWorld(SDFBlockToVirtualVoxelPos(sdfBlock));
}

int3 worldToSDFBlock(float3 worldPos)
{
	const int3 virtualVoxelPos = worldToVirtualVoxelPos(worldPos);
	return virtualVoxelPosToSDFBlock(virtualVoxelPos);
}

int virtualVoxelPosToLocalSDFBlockIndex(int3 virtualVoxelPos)
{
	int3 localVoxelPos = virtualVoxelPos%SDF_BLOCK_SIZE;

	if (localVoxelPos.x < 0) localVoxelPos.x += SDF_BLOCK_SIZE;
	if (localVoxelPos.y < 0) localVoxelPos.y += SDF_BLOCK_SIZE;
	if (localVoxelPos.z < 0) localVoxelPos.z += SDF_BLOCK_SIZE;

	return linearizeVoxelPos(localVoxelPos);
}

int worldToLocalSDFBlockIndex(float3 world)
{
	int3 virtualVoxelPos = worldToVirtualVoxelPos(world);
	return virtualVoxelPosToLocalSDFBlockIndex(virtualVoxelPos);
}

//! see teschner et al. (but with correct prime values)
uint computeHashPos(in int3 virtualVoxelPos)
{ 
	const int p0 = 73856093;
	const int p1 = 19349669;
	const int p2 = 83492791;

	int res = ((virtualVoxelPos.x * p0) ^ (virtualVoxelPos.y * p1) ^ (virtualVoxelPos.z * p2)) % g_HashNumBuckets;
	if (res < 0) res += g_HashNumBuckets;
	return (uint)res;
}

//! see teschner et al. (but with correct prime values)
uint computeHashPosOther(in int3 virtualVoxelPos, in uint hashNumBuckets)
{ 
	const int p0 = 73856093;
	const int p1 = 19349669;
	const int p2 = 83492791;

	int res = ((virtualVoxelPos.x * p0) ^ (virtualVoxelPos.y * p1) ^ (virtualVoxelPos.z * p2)) % hashNumBuckets;
	if (res < 0) res += hashNumBuckets;
	return (uint)res;
}

//merges two voxels (v0 is the input voxel, v1 the currently stored voxel)
Voxel combineVoxel(in Voxel v0, in Voxel v1)
{
	Voxel v;
	//v.color = (10*v0.weight * v0.color + v1.weight * v1.color)/(10*v0.weight + v1.weight);	//give the currently observed color more weight
	//v.color = (v0.weight * v0.color + v1.weight * v1.color)/(v0.weight + v1.weight);
	v.color = 0.5f * (v0.color + v1.color);	//exponential running average 
	v.sdf = (v0.sdf * v0.weight + v1.sdf * v1.weight) / (v0.weight + v1.weight);
	v.weight = min(g_WeightMax, v0.weight + v1.weight);
	return v;
}

//! returns the hash entry for a given sdf block id; if there was no hash entry the returned entry will have a ptr with FREE_ENTRY set
HashEntry getHashEntryForSDFBlockPos(Buffer<int> hash, int3 sdfBlock) 
{
	uint h = computeHashPos(sdfBlock);	//hash bucket
	uint hp = h * g_HashBucketSize;		//hash position

	HashEntry entry;
	entry.pos = sdfBlock;
	entry.offset = 0;
	entry.ptr = FREE_ENTRY;

	[allow_uav_condition]
	for (uint j = 0; j < g_HashBucketSize; j++) {
		uint i = j + hp;
		HashEntry curr = getHashEntry(hash, i);
		if (curr.pos.x == entry.pos.x && curr.pos.y == entry.pos.y && curr.pos.z == entry.pos.z && curr.ptr != FREE_ENTRY) {
			return curr;
		}
	}

	#ifdef HANDLE_COLLISIONS
		const uint idxLastEntryInBucket = (h+1)*g_HashBucketSize - 1;
		int i = idxLastEntryInBucket;	//start with the last entry of the current bucket
		HashEntry curr;
		//traverse list until end: memorize idx at list end and memorize offset from last element of bucket to list end
		int k = 0;

		unsigned int maxIter = 0;
		[allow_uav_condition]
		while (true && maxIter < g_MaxLoopIterCount) {
			curr = getHashEntry(hash, i);
			
			if (curr.pos.x == entry.pos.x && curr.pos.y == entry.pos.y && curr.pos.z == entry.pos.z && curr.ptr != FREE_ENTRY) {
				return curr;
			}

			if (curr.offset == 0) {	//we have found the end of the list
				break;
			}
			i = idxLastEntryInBucket + curr.offset;		//go to next element in the list
			i %= (g_HashBucketSize * g_HashNumBuckets);	//check for overflow

			maxIter++;
		}
	#endif
	return entry;
}

//! returns the hash entry for a given worldPos; if there was no hash entry the returned entry will have a ptr with FREE_ENTRY set
HashEntry getHashEntry(Buffer<int> hash, float3 worldPos) 
{
	//int3 blockID = worldToSDFVirtualVoxelPos(worldPos)/SDF_BLOCK_SIZE;	//position of sdf block
	int3 blockID = worldToSDFBlock(worldPos);
	return getHashEntryForSDFBlockPos(hash, blockID);
}

//! returns the truncation of the SDF for a given distance value
float getTruncation(float z) {
	return g_Truncation + g_TruncScale * z;
}

bool isInCameraFrustum(in float3 pos) {
	float3 pCamera = mul(float4(pos, 1.0f), g_RigidTransformInverse).xyz;
	float3 pProj = cameraToKinectProj(pCamera);
	//pProj *= 0.75f;	//TODO THIS IS A HACK FIX IT :)
	return !(pProj.x < -1.0f || pProj.x > 1.0f || pProj.y < -1.0f || pProj.y > 1.0f || pProj.z < 0.0f || pProj.z > 1.0f);  
}

bool isInCameraFrustumApprox(in float3 pos) {
	float3 pCamera = mul(float4(pos, 1.0f), g_RigidTransformInverse).xyz;
	float3 pProj = cameraToKinectProj(pCamera);
	//pProj *= 1.5f;	//TODO THIS IS A HACK FIX IT :)
	pProj *= 0.95;
	return !(pProj.x < -1.0f || pProj.x > 1.0f || pProj.y < -1.0f || pProj.y > 1.0f || pProj.z < 0.0f || pProj.z > 1.0f);  
}

bool isSDFBlockInCameraFrustumApprox(in int3 sdfBlock) {
	float3 posWorld = virtualVoxelPosToWorld(SDFBlockToVirtualVoxelPos(sdfBlock)) + g_VirtualVoxelSize * 0.5f * (SDF_BLOCK_SIZE - 1.0f);
	return isInCameraFrustumApprox(posWorld);
}





 

//! deletes a hash entry position for a given sdfBlock index (returns true uppon successful deletion; otherwise returns false)
bool deleteHashEntryElement(RWBuffer<int> hash, AppendStructuredBuffer<uint> heapAppend, RWBuffer<uint> bucketMutex, int3 sdfBlock) {
	uint h = computeHashPos(sdfBlock);	//hash bucket
	uint hp = h * g_HashBucketSize;		//hash position

	[allow_uav_condition]
	for (uint j = 0; j < g_HashBucketSize; j++) {
		uint i = j + hp;
		HashEntry curr = getHashEntry(hash, i);
		if (curr.pos.x == sdfBlock.x && curr.pos.y == sdfBlock.y && curr.pos.z == sdfBlock.z && curr.ptr != FREE_ENTRY) {
			#ifndef HANDLE_COLLISIONS
			const uint linBlockSize = SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;
			heapAppend.Append(curr.ptr / linBlockSize);
			deleteHashEntry(hash, i);
			return true;
			#endif
			#ifdef HANDLE_COLLISIONS
			if (curr.offset != 0) {	//if there was a pointer set it to the next list element
				int prevValue = 0;
				InterlockedExchange(bucketMutex[h], LOCK_ENTRY, prevValue);	//lock the hash bucket
				if (prevValue == LOCK_ENTRY)	return false;
				if (prevValue != LOCK_ENTRY) {
					const uint linBlockSize = SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;
					heapAppend.Append(curr.ptr / linBlockSize);
					int nextIdx = (i + curr.offset) % (g_HashBucketSize * g_HashNumBuckets);
					setHashEntry(hash, i, getHashEntry(hash, nextIdx));
					deleteHashEntry(hash, nextIdx);
					return true;
				}
			} else {
				const uint linBlockSize = SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;
				heapAppend.Append(curr.ptr / linBlockSize);
				deleteHashEntry(hash, i);
				return true;
			}
			#endif
		}
	}	
	#ifdef HANDLE_COLLISIONS
		const uint idxLastEntryInBucket = (h+1)*g_HashBucketSize - 1;
		int i = idxLastEntryInBucket;
		int offset = 0;
		HashEntry curr;
		curr = getHashEntry(hash, i);
		int prevIdx = i;
		i = idxLastEntryInBucket + curr.offset;		//go to next element in the list
		i %= (g_HashBucketSize * g_HashNumBuckets);	//check for overflow

		unsigned int maxIter = 0;
		[allow_uav_condition]
		while (true && maxIter < g_MaxLoopIterCount) {
			curr = getHashEntry(hash, i);
			//found that dude that we need/want to delete
			if (curr.pos.x == sdfBlock.x && curr.pos.y == sdfBlock.y && curr.pos.z == sdfBlock.z && curr.ptr != FREE_ENTRY) {
				int prevValue = 0;
				InterlockedExchange(bucketMutex[h], LOCK_ENTRY, prevValue);	//lock the hash bucket
				if (prevValue == LOCK_ENTRY)	return false;
				if (prevValue != LOCK_ENTRY) {
					const uint linBlockSize = SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;
					heapAppend.Append(curr.ptr / linBlockSize);
					deleteHashEntry(hash, i);
					HashEntry prev = getHashEntry(hash, prevIdx);				
					prev.offset = curr.offset;
					setHashEntry(hash, prevIdx, prev);
					return true;
				}
			}

			if (curr.offset == 0) {	//we have found the end of the list
				return false;	//should actually never happen because we need to find that guy before
			}
			prevIdx = i;
			i = idxLastEntryInBucket + curr.offset;		//go to next element in the list
			i %= (g_HashBucketSize * g_HashNumBuckets);	//check for overflow

			maxIter++;
		}
	#endif
	return false;
}










#endif
