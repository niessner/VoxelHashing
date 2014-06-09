cbuffer consts : register(cb0)
{
	// Grid
	float3 g_gridPosition;
	float align0;

	int3 g_gridDimensions;
	int align1;

	float3 g_voxelExtends;
	int align2;
};

RWBuffer<int> g_voxelBuffer : register(u0);

#include "RayCastingUtil.h.hlsl"

[numthreads(groupthreads, groupthreads, groupthreads)]
void resetCS(int3 dTid : SV_DispatchThreadID)
{
	Voxel voxel;
	voxel.sdf = 0.0f;
	voxel.weight = 0;
	voxel.color = int3(0, 0, 0);

	setVoxel(g_voxelBuffer, linearizeIndex(dTid), voxel);
}
