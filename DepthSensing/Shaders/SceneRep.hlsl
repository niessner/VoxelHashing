
cbuffer cbCS : register( b0 )
{
	uint g_HashNumBuckets;
	uint g_HashBucketSize;
	uint g_ImageWidth;
	uint g_ImageHeight;
	float g_VirtualVoxelResolutionScalar;
	float g_VirtualVoxelSize;
	uint g_NumOtherVoxels;
	float dummy_1337;
	float4x4 g_RigidTransform;
	float4	g_CameraPos;
}; 
 
Texture2D<float3>	inputPoints : register( t0 );
Texture2D<float3>	inputColors : register( t1 );
Texture2D<float>	prevDepth   : register( t2 );
Texture2D<float4>	prevNormals	: register( t3 );
Texture2D<float4>	prevColor   : register( t4 );	//just for include
Texture2D<float4>	prevPos	    : register( t5 );	//just for include

RWBuffer<int>		g_VoxelHash			: register( u0 );
RWBuffer<int>		g_VoxelHashOther	: register( u1 );
Buffer<int4>		g_VoxelHashOtherSRV	: register( t0 );

#include "VoxelUtil.h.hlsl"
#include "KinectCameraUtil.h.hlsl"

 
// should be set by application
#ifndef groupthreads
#define groupthreads 1
#endif


struct Voxel {
	int3 pos;
	int weight;
	int3 color;
};

Voxel getVoxel(RWBuffer<int> hash, in uint id) {
	Voxel voxel;
	voxel.pos.x = hash[4*id+0];
	voxel.pos.y = hash[4*id+1];
	voxel.pos.z = hash[4*id+2];
	int last = hash[4*id+3];
	voxel.weight = last & 0x000000ff;
	last >>= 0x8;
	voxel.color.x = last & 0x000000ff;
	last >>= 0x8;
	voxel.color.y = last & 0x000000ff;
	last >>= 0x8;
	voxel.color.z = last & 0x000000ff;
	return voxel;
}

Voxel getVoxelSRV4(Buffer<int4> hash, in uint id) {
	Voxel voxel;
	voxel.pos.x = hash[id].x;
	voxel.pos.y = hash[id].y;
	voxel.pos.z = hash[id].z;
	int last = hash[id].w;
	voxel.weight = last & 0x000000ff;
	last >>= 0x8;
	voxel.color.x = last & 0x000000ff;
	last >>= 0x8;
	voxel.color.y = last & 0x000000ff;
	last >>= 0x8;
	voxel.color.z = last & 0x000000ff;
	return voxel;
}

  
void setVoxel(RWBuffer<int> hash, in uint id, in Voxel voxel) {

	//TODO so something for sync
	hash[4*id+0] = voxel.pos.x;
	hash[4*id+1] = voxel.pos.y;
	hash[4*id+2] = voxel.pos.z;
	int last = 0;
	last |= voxel.color.z & 0x000000ff;
	last <<= 8;
	last |= voxel.color.y & 0x000000ff;
	last <<= 8;
	last |= voxel.color.x & 0x000000ff;
	last <<= 8;
	last |= voxel.weight & 0x000000ff;
	hash[4*id+3] = last;
	
	//if (hash[4*id+0] != voxel.pos.x || hash[4*id+1] != voxel.pos.y || hash[4*id+2] != voxel.pos.z)	hash[4*id+3] = 0;
}

//merges two voxels (v0 is the input voxel, v1 the currently stored voxel)
Voxel combineVoxel(in Voxel v0, in Voxel v1) {
	Voxel v;
	v.color = (10*v0.weight * v0.color + v1.weight * v1.color)/(10*v0.weight + v1.weight);	//give the currently observed color more weight
	//v.color = (v0.weight * v0.color + v1.weight * v1.color)/(v0.weight + v1.weight);
	//v.pos = (v0.weight * v0.pos + v1.weight * v1.pos)/(v0.weight + v1.weight);
	v.pos = v0.pos;	//pos must be identical anyway
	v.weight = min(g_WeightMax, v0.weight + v1.weight);
	return v;
}

int3 worldToVirtualVoxelPos(in float3 pos) {
	return int3(
		(int)((pos.x * g_VirtualVoxelResolutionScalar) + 0.5f),
		(int)((pos.y * g_VirtualVoxelResolutionScalar) + 0.5f),
		(int)((pos.z * g_VirtualVoxelResolutionScalar) + 0.5f)
	);
}


//! see teschner et al. (but with correct prime values)
uint computeHashPos(in int3 virtualVoxelPos) {
	const int p0 = 73856093;
	const int p1 = 19349669;
	const int p2 = 83492791;
	int res = ((virtualVoxelPos.x * p0) ^ (virtualVoxelPos.y * p1) ^ (virtualVoxelPos.z * p2)) % g_HashNumBuckets;
	if (res < 0) res += g_HashNumBuckets;
	return (uint)res;
}

static const int lock = 0xffffffff;		//this lock includes color
static const int wLock = 0x000000ff;	//this lock only includes the weight itself

void integrateVoxel(RWBuffer<int> hash, in Voxel voxel) {
	uint hp = computeHashPos(voxel.pos) * g_HashBucketSize;

	uint firstEmpty = (uint)-1;
	bool found = false;
	[allow_uav_condition]
	for (uint j = 0; j < g_HashBucketSize; j++) {
		uint i = j + hp;
		 
		int prevWeight = -1;
		if (firstEmpty == (uint)-1) {
			InterlockedCompareExchange(hash[4*i+3], 0, lock, prevWeight);
		}
	
		Voxel curr = getVoxel(hash, i);

		if (firstEmpty == (uint)-1 && prevWeight == 0) {
			firstEmpty = i;
		} else if (curr.pos.x == voxel.pos.x && curr.pos.y == voxel.pos.y && curr.pos.z == voxel.pos.z && curr.weight > 0 && curr.weight != wLock) {	//this is hacky: hash[4*i+3] must be read from register
			setVoxel(hash, i, combineVoxel(voxel, curr));
			found = true;
			//hash[4*firstEmpty+3] = 0;
			if (firstEmpty != (uint)-1) hash[4*firstEmpty+3] = 0;
			break;
		}
	}

	if (!found) {
		if (firstEmpty != (uint)-1) {
			setVoxel(hash, firstEmpty, voxel);
		}
	}
}

void integrateVoxelToOther(RWBuffer<int> hash, in Voxel voxel) {
	uint hp = computeHashPos(voxel.pos) * g_HashBucketSize;

	uint firstEmpty = (uint)-1;
	[unroll]
	for (uint j = 0; j < g_HashBucketSize && j < 10; j++) { // 10 ???
		uint i = j + hp;
		Voxel curr = getVoxel(hash, i);
		int prevWeight = -1;
		InterlockedCompareExchange(hash[4*i+3], 0, lock, prevWeight);
		//if (curr.weight == 0) {
		if (prevWeight == 0) {
			firstEmpty = i;
			break;
		}
	}

	if (firstEmpty != (uint)-1) {
		setVoxel(hash, firstEmpty, voxel);
	}
}

void reduceVoxelWeight(RWBuffer<int> hash, in int3 pos) {
	uint hp = computeHashPos(pos) * g_HashBucketSize;

	uint firstEmpty = (uint)-1;
	Voxel curr;
	[unroll]
	for (uint j = 0; j < g_HashBucketSize && j < 10; j++) { // 10 ???
		uint i = j + hp;
		curr = getVoxel(hash, i);
		//int prevWeight = -1;
		//InterlockedCompareExchange(hash[4*i+3], 0, lock, prevWeight);
		if (curr.pos.x == pos.x && curr.pos.y == pos.y && curr.pos.z == pos.z && curr.weight > 0 && curr.weight != wLock) {
			firstEmpty = i;
			break;
		}
	}

	if (firstEmpty != (uint)-1) {
		curr.weight = max(curr.weight - 1, 0);
		setVoxel(hash, firstEmpty, curr);
	}
}

[numthreads( groupthreads, groupthreads, 1)]
void integrateCS(uint3 DTid : SV_DispatchThreadID, uint3 GTid : SV_GroupThreadID, uint GI: SV_GroupIndex)
{
	if (DTid.x >= g_ImageWidth || DTid.y >= g_ImageHeight)	return;

	//get input pos and color
	float3 p = inputPoints[DTid.xy].xyz;
	if (p.x == MINF)	return;
	float3 c = inputColors[DTid.xy].xyz;

	p = mul(float4(p, 1.0f), g_RigidTransform).xyz;

	Voxel voxel;
	voxel.pos = worldToVirtualVoxelPos(p);
	voxel.color = (int3)(c * 255.0f);
	voxel.weight = g_WeightHit;
	integrateVoxel(g_VoxelHash, voxel);

	//float3 m = voxel.pos / voxel.pos.z;
	//reduceVoxelWeight(g_VoxelHash, voxel.pos + int3(1.0f * m + 0.5f));
	//reduceVoxelWeight(g_VoxelHash, voxel.pos + int3(2.0f * m + 0.5f));
	//reduceVoxelWeight(g_VoxelHash, voxel.pos + int3(3.0f * m + 0.5f));
	//reduceVoxelWeight(g_VoxelHash, voxel.pos + int3(4.0f * m + 0.5f));
	//reduceVoxelWeight(g_VoxelHash, voxel.pos + int3(-1.0f * m + 0.5f));
	//reduceVoxelWeight(g_VoxelHash, voxel.pos + int3(-2.0f * m + 0.5f));
	//reduceVoxelWeight(g_VoxelHash, voxel.pos + int3(-3.0f * m + 0.5f));
	//reduceVoxelWeight(g_VoxelHash, voxel.pos + int3(-4.0f * m + 0.5f));
	
}

void deleteVoxel(RWBuffer<int> hash, in uint idx) {
	//hash[4*idx+0] = 0;
	//hash[4*idx+1] = 0;
	//hash[4*idx+2] = 0;
	hash[4*idx+3] = 0;
}

bool isInCameraFrustum(in float3 pos) {
	float3 pCamera = mul(float4(pos, 1.0f), g_RigidTransform).xyz;
	float3 pProj = cameraToKinectProj(pCamera);
	return !(pProj.x < -1.0f || pProj.x > 1.0f || pProj.y < -1.0f || pProj.y > 1.0f || pProj.z < 0.0f || pProj.z > 1.0f);  
}

[numthreads( groupthreads * groupthreads * 8, 1, 1)]
void removeAndIntegrateCS(uint3 DTid : SV_DispatchThreadID, uint3 GTid : SV_GroupThreadID, uint GI: SV_GroupIndex) 
{
	const uint idx = DTid.x;
	//if (idx >= g_HashNumBuckets * g_HashBucketSize)	return;
	if (idx >= g_NumOtherVoxels)	return;	//it's actually the num voxels of the current hash (not the other)

	Voxel voxel = getVoxel(g_VoxelHash, idx);
	if (voxel.weight > 0) {
#ifdef MOVE_OUT_FRUSTUM
		if (!isInCameraFrustum(float3(voxel.pos)/g_VirtualVoxelResolutionScalar)) {
#endif
#ifdef MOVE_IN_FRUSTUM
		if (isInCameraFrustum(float3(voxel.pos)/g_VirtualVoxelResolutionScalar)) {
#endif
			if (voxel.weight > g_MinRenderWeight) {
				integrateVoxelToOther(g_VoxelHashOther, voxel);
				//integrateVoxel(g_VoxelHashOther, voxel);
			}
			deleteVoxel(g_VoxelHash, idx);
#if (defined MOVE_OUT_FRUSTUM) || (defined MOVE_IN_FRUSTUM)
		}
#endif
	}

}

[numthreads( groupthreads * groupthreads * 8, 1, 1)]
void integrateFromOtherCS(uint3 DTid : SV_DispatchThreadID, uint3 GTid : SV_GroupThreadID, uint GI: SV_GroupIndex) 
{
	const uint idx = DTid.x;
	if (idx >= g_NumOtherVoxels)	return;
	Voxel voxel = getVoxelSRV4(g_VoxelHashOtherSRV, idx);
	integrateVoxel(g_VoxelHash, voxel);
}

[numthreads( groupthreads * groupthreads * 8, 1, 1)]
void removeFromOtherCS(uint3 DTid : SV_DispatchThreadID, uint3 GTid : SV_GroupThreadID, uint GI: SV_GroupIndex) 
{
	const uint idx = DTid.x;
	if (idx >= g_NumOtherVoxels)	return;
	Voxel voxel = getVoxelSRV4(g_VoxelHashOtherSRV, idx);

	uint hp = computeHashPos(voxel.pos) * g_HashBucketSize;

	uint firstEmpty = (uint)-1;
	bool found = false;
	[unroll]
	for (uint j = 0; j < g_HashBucketSize && j < 10; j++) { // 10 ???
		uint i = j + hp;
		Voxel curr = getVoxel(g_VoxelHash, i);
		if (curr.pos.x == voxel.pos.x && curr.pos.y == voxel.pos.y && curr.pos.z == voxel.pos.z && curr.weight > 0) {
			deleteVoxel(g_VoxelHash, i);
			break;
		}
	}
}

[numthreads( groupthreads * groupthreads * 8, 1, 1)]
void resetCS(uint3 DTid : SV_DispatchThreadID, uint3 GTid : SV_GroupThreadID, uint GI: SV_GroupIndex)
{
	const uint idx = DTid.x;
	if (idx >= g_HashNumBuckets * g_HashBucketSize)	return;
	deleteVoxel(g_VoxelHash, idx);
}

void starveVoxel(uint idx, uint starveWeight) {
	//g_VoxelHash[4*idx+3] = max(0, g_VoxelHash[4*idx+3] - starveWeight);
	int last = g_VoxelHash[4*idx+3];
	int prevWeight = last & 0x000000ff;
	if (prevWeight > g_ImmortalWeight)	return;	//do not starve voxels that are once visible
	int newWeight = 0;
	if (prevWeight > starveWeight) {	//check for negative weight
		newWeight = prevWeight - starveWeight;
	}
	if (newWeight == 0) {
		//g_VoxelHash[4*idx+0] = 0; 
		//g_VoxelHash[4*idx+1] = 0;
		//g_VoxelHash[4*idx+2] = 0;
		last = 0;
	} else {
		last &= 0xffffff00;
		last |= newWeight;
	}
	g_VoxelHash[4*idx+3] = last;
}

[numthreads( groupthreads * groupthreads * 8, 1, 1)]
void starveCS(uint3 DTid : SV_DispatchThreadID, uint3 GTid : SV_GroupThreadID, uint GI: SV_GroupIndex)
{
	const uint idx = DTid.x;
	if (idx >= g_HashNumBuckets * g_HashBucketSize)	return;
	starveVoxel(idx, g_WeightStarve);
}




[numthreads( groupthreads * groupthreads * 8, 1, 1)]
void starveVisibleCS(uint3 DTid : SV_DispatchThreadID, uint3 GTid : SV_GroupThreadID, uint GI: SV_GroupIndex)
{
	const uint idx = DTid.x;
	if (idx >= g_HashNumBuckets * g_HashBucketSize)	return;

	float3 pos = getWorldVoxelPos(idx);
	float4 viewPos = mul(float4(pos.xyz, 1.0f), g_RigidTransform);
	uint2 screenPos = cameraToKinectScreen(viewPos.xyz);
	float z = cameraToKinectProjZ(viewPos.z);
	bool isOutFrustum = screenPos.x >= g_ImageWidth || screenPos.y >= g_ImageHeight || z < 0.0f || z > 1.0f;
	float prevZ = kinectProjToCameraZ(prevDepth[cameraToKinectScreen(viewPos.xyz)]);
	if (!isOutFrustum  && abs(prevZ - viewPos.z) < cullRange ) {
	//if (!isOutFrustum) {
		starveVoxel(idx, g_WeightStarve);
	}
}


[numthreads( groupthreads * groupthreads * 8, 1, 1)]
void starveRegulizeNoiseCS(uint3 DTid : SV_DispatchThreadID, uint3 GTid : SV_GroupThreadID, uint GI: SV_GroupIndex)
{
	const uint idx = DTid.x;
	if (idx >= g_HashNumBuckets * g_HashBucketSize)	return;

	float3 pos = getWorldVoxelPos(idx);
	float3 viewPos = mul(float4(pos.xyz, 1.0f), g_RigidTransform).xyz;

	uint2 screenPos = cameraToKinectScreen(viewPos.xyz);
	float3 prevPos = kinectProjToCamera(screenPos.x, screenPos.y, prevDepth[screenPos]);
	float3 prevNormal = prevNormals[screenPos];

	if (abs(prevPos.z - viewPos.z * prevNormal.z) < cullRange ) {
		int3 iViewPos = worldToVirtualVoxelPos(viewPos.xyz);
		int3 iPrevPos = worldToVirtualVoxelPos(viewPos.xyz);
		if (iViewPos.x != iPrevPos.x || iViewPos.y != iPrevPos.y || iViewPos.z != iPrevPos.z) {
			starveVoxel(idx, g_WeightStarve * 3);
		}
	}

	starveVoxel(idx, g_WeightStarve);
}