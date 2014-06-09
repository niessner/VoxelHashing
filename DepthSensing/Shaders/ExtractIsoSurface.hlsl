cbuffer consts : register(cb0)
{
	// Grid
	float3 g_gridPosition;
	float align0;

	int3 g_gridDimensions;
	int align1;

	float3 g_voxelExtends;
	float align2;
};

struct Vertex
{
	float3 p;
	float3 c;
};

struct Triangle
{
	Vertex v0;
	Vertex v1;
	Vertex v2;
};

Buffer<int> g_voxelBuffer : register(t0);
AppendStructuredBuffer<Triangle> g_triangles : register(u0);

#include "RayCastingUtil.h.hlsl"
#include "Tables.h.hlsl"

Vertex VertexInterp(float isolevel, float3 p1, float3 p2, Voxel v1, Voxel v2)
{
	Vertex r1; r1.p = p1; r1.c = (float3)v1.color;
	Vertex r2; r2.p = p2; r2.c = (float3)v2.color;

	if(abs(isolevel-v1.sdf) < 0.00001f)		return r1;
	if(abs(isolevel-v2.sdf) < 0.00001f)		return r2;
	if(abs(v1.sdf-v2.sdf) < 0.00001f)		return r1;

	float mu = (isolevel - v1.sdf) / (v2.sdf - v1.sdf);
  
	Vertex res;
	res.p.x = p1.x + mu * (p2.x - p1.x); // Positions
	res.p.y = p1.y + mu * (p2.y - p1.y);
	res.p.z = p1.z + mu * (p2.z - p1.z);

	res.c.x = ((float3)v1.color).x + mu * (((float3)v2.color).x - ((float3)v1.color).x); // Color
	res.c.y = ((float3)v1.color).y + mu * (((float3)v2.color).y - ((float3)v1.color).y);
	res.c.z = ((float3)v1.color).z + mu * (((float3)v2.color).z - ((float3)v1.color).z);

	return res;
}

// DirectX implementation of
//
// Polygonising a scalar field
// Also known as: "3D Contouring", "Marching Cubes", "Surface Reconstruction" 
// Written by Paul Bourke
// May 1994 
// http://paulbourke.net/geometry/polygonise/

[numthreads(groupthreads, groupthreads, groupthreads)]
void extractIsoSurfaceCS(int3 dTid : SV_DispatchThreadID)
{
	const float isolevel = 0.0f;

	Voxel v000 = getVoxel(g_voxelBuffer, linearizeIndex(dTid+int3(0, 0, 0))); float3 p000 = computeSamplePositions(dTid+int3(0, 0, 0));
	Voxel v100 = getVoxel(g_voxelBuffer, linearizeIndex(dTid+int3(1, 0, 0))); float3 p100 = computeSamplePositions(dTid+int3(1, 0, 0));
	Voxel v010 = getVoxel(g_voxelBuffer, linearizeIndex(dTid+int3(0, 1, 0))); float3 p010 = computeSamplePositions(dTid+int3(0, 1, 0));
	Voxel v001 = getVoxel(g_voxelBuffer, linearizeIndex(dTid+int3(0, 0, 1))); float3 p001 = computeSamplePositions(dTid+int3(0, 0, 1));
	Voxel v110 = getVoxel(g_voxelBuffer, linearizeIndex(dTid+int3(1, 1, 0))); float3 p110 = computeSamplePositions(dTid+int3(1, 1, 0));
	Voxel v011 = getVoxel(g_voxelBuffer, linearizeIndex(dTid+int3(0, 1, 1))); float3 p011 = computeSamplePositions(dTid+int3(0, 1, 1));
	Voxel v101 = getVoxel(g_voxelBuffer, linearizeIndex(dTid+int3(1, 0, 1))); float3 p101 = computeSamplePositions(dTid+int3(1, 0, 1));
	Voxel v111 = getVoxel(g_voxelBuffer, linearizeIndex(dTid+int3(1, 1, 1))); float3 p111 = computeSamplePositions(dTid+int3(1, 1, 1));

	if(v000.weight == 0 || v100.weight == 0 || v010.weight == 0 || v001.weight == 0 || v110.weight == 0 || v011.weight == 0 || v101.weight == 0 || v111.weight == 0) return;
	
	uint cubeindex = 0;
	if(v010.sdf < isolevel) cubeindex += 1;
	if(v110.sdf < isolevel) cubeindex += 2;
	if(v100.sdf < isolevel) cubeindex += 4;
	if(v000.sdf < isolevel) cubeindex += 8;
	if(v011.sdf < isolevel) cubeindex += 16;
	if(v111.sdf < isolevel) cubeindex += 32;
	if(v101.sdf < isolevel) cubeindex += 64;
	if(v001.sdf < isolevel) cubeindex += 128;

	if(edgeTable[cubeindex] == 0 || edgeTable[cubeindex] == 255) return;  // added by me edgeTable[cubeindex] == 255 !!!

	Vertex vertlist[12];
	if(edgeTable[cubeindex] & 1)	vertlist[0]  = VertexInterp(isolevel, p010, p110, v010, v110);
	if(edgeTable[cubeindex] & 2)	vertlist[1]  = VertexInterp(isolevel, p110, p100, v110, v100);
	if(edgeTable[cubeindex] & 4)	vertlist[2]  = VertexInterp(isolevel, p100, p000, v100, v000);
	if(edgeTable[cubeindex] & 8)	vertlist[3]  = VertexInterp(isolevel, p000, p010, v000, v010);
	if(edgeTable[cubeindex] & 16)	vertlist[4]  = VertexInterp(isolevel, p011, p111, v011, v111);
	if(edgeTable[cubeindex] & 32)	vertlist[5]  = VertexInterp(isolevel, p111, p101, v111, v101);
	if(edgeTable[cubeindex] & 64)	vertlist[6]  = VertexInterp(isolevel, p101, p001, v101, v001);
	if(edgeTable[cubeindex] & 128)	vertlist[7]  = VertexInterp(isolevel, p001, p011, v001, v011);
	if(edgeTable[cubeindex] & 256)	vertlist[8]	 = VertexInterp(isolevel, p010, p011, v010, v011);
	if(edgeTable[cubeindex] & 512)	vertlist[9]  = VertexInterp(isolevel, p110, p111, v110, v111);
	if(edgeTable[cubeindex] & 1024) vertlist[10] = VertexInterp(isolevel, p100, p101, v100, v101);
	if(edgeTable[cubeindex] & 2048) vertlist[11] = VertexInterp(isolevel, p000, p001, v000, v001);

	[allow_uav_condition]
	for(int i=0; triTable[cubeindex][i] != -1; i+=3)
	{
		Triangle t;
		t.v0 = vertlist[triTable[cubeindex][i+0]];
		t.v1 = vertlist[triTable[cubeindex][i+1]];
		t.v2 = vertlist[triTable[cubeindex][i+2]];

		g_triangles.Append(t);
	}
}
