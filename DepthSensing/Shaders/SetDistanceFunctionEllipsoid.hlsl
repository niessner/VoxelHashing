cbuffer consts : register(b0)
{
	// Grid
	float3 g_gridPosition;
	float align0;

	int3 g_gridDimensions;
	int align1;

	float3 g_voxelExtends;
	float align2;

	// Sphere
	float3 g_center;
	float align3;

	float g_a;
	float g_b;
	float g_c;
	float align4;
};

RWBuffer<int> g_voxelBuffer : register(u0);

#include "RayCastingUtil.h.hlsl"

float evalDistFuncEllipsoid(float3 center, float a, float b, float c, float3 pos)
{
	float3 p = pos-center;
	
	p.x /= a;
	p.y /= b;
	p.z /= c;
	
	float d = length(p);
	return d-1.0f;
}

[numthreads(groupthreads, groupthreads, groupthreads)]
void setDistanceFunctionEllipsoidCS(int3 dTid : SV_DispatchThreadID)
{
	float3 pos = computeSamplePositions(dTid);
	float sdf = evalDistFuncEllipsoid(g_center, g_a, g_b, g_c, pos);

	float truncation = 0.07f;
	if (sdf >= 0.0f)
	{
		sdf = min(truncation, sdf);
	}
	else
	{
		sdf = max(-truncation, sdf);
	}

	Voxel voxel;
	voxel.sdf = sdf;
	voxel.weight = 1;
	voxel.color = int3(0, 0, 0);

	setVoxel(g_voxelBuffer, linearizeIndex(dTid), voxel);
}
