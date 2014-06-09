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

cbuffer consts : register(b2)
{
	uint boxEnabled;
	float3 minCorner;

	uint align5;
	float3 maxCorner;		
};

Buffer<int> g_Hash : register(t0);
Buffer<float> g_SDFBlocksSDF : register(t1);
Buffer<int> g_SDFBlocksRGBW : register(t2);

AppendStructuredBuffer<Triangle> g_triangles : register(u0);

#include "GlobalAppStateShaderBuffer.h.hlsl"
#include "SDFShaderBuffer.h.hlsl"
#include "KinectCameraUtil.h.hlsl"
#include "VoxelUtilHashSDF.h.hlsl"
#include "RayCastingUtilHashSDF.h.hlsl"
#include "Tables.h.hlsl"

Vertex VertexInterp(float isolevel, float3 p1, float3 p2, float d1, float d2, float3 c1, float3 c2)
{
	Vertex r1; r1.p = p1; r1.c = (float3)c1;
	Vertex r2; r2.p = p2; r2.c = (float3)c2;

	if(abs(isolevel-d1) < 0.00001f)		return r1;
	if(abs(isolevel-d2) < 0.00001f)		return r2;
	if(abs(d1-d2) < 0.00001f)			return r1;

	float mu = (isolevel - d1) / (d2 - d1);
  
	Vertex res;
	res.p.x = p1.x + mu * (p2.x - p1.x); // Positions
	res.p.y = p1.y + mu * (p2.y - p1.y);
	res.p.z = p1.z + mu * (p2.z - p1.z);

	res.c.x = c1.x + mu * (c2.x - c1.x); // Color
	res.c.y = c1.y + mu * (c2.y - c1.y);
	res.c.z = c1.z + mu * (c2.z - c1.z);

	return res;
}

// DirectX implementation of
//
// Polygonising a scalar field
// Also known as: "3D Contouring", "Marching Cubes", "Surface Reconstruction" 
// Written by Paul Bourke
// May 1994 
// http://paulbourke.net/geometry/polygonise/

#define NUM_GROUPS_X 1024 // has to be in sync with the other application code and the CPU !!!

bool isInBoxAA(float3 minCorner, float3 maxCorner, float3 pos)
{
	if(pos.x < minCorner.x || pos.x > maxCorner.x) return false;
	if(pos.y < minCorner.y || pos.y > maxCorner.y) return false;
	if(pos.z < minCorner.z || pos.z > maxCorner.z) return false;

	return true;
}

[numthreads(SDF_BLOCK_SIZE, SDF_BLOCK_SIZE, SDF_BLOCK_SIZE)]
void extractIsoSurfaceHashSDFCS(int3 dTid : SV_DispatchThreadID, uint3 GTid : SV_GroupThreadID, uint GI: SV_GroupIndex, uint3 GID : SV_GroupID)
{
	uint groupID = GID.x + GID.y * NUM_GROUPS_X;
	if(groupID < g_HashNumBuckets*g_HashBucketSize)
	{
		HashEntry entry = getHashEntry(g_Hash, groupID);
		
		if(entry.ptr != FREE_ENTRY)
		{
			int3 pi_base = SDFBlockToVirtualVoxelPos(entry.pos);
			int3 pi = pi_base + GTid;
			float3 worldPos = virtualVoxelPosToWorld(pi);
		
			if(boxEnabled == 1)
			{
				if(!isInBoxAA(minCorner, maxCorner, worldPos)) return;
			}
			
			const float isolevel = 0.0f;

			const float P = g_VirtualVoxelSize/2.0f;
			const float M = -P;

			float3 p000 = worldPos+float3(M, M, M); float dist000; float3 color000; bool valid000 = trilinearInterpolationSimpleFastFast(p000, dist000, color000);
			float3 p100 = worldPos+float3(P, M, M); float dist100; float3 color100; bool valid100 = trilinearInterpolationSimpleFastFast(p100, dist100, color100);
			float3 p010 = worldPos+float3(M, P, M); float dist010; float3 color010; bool valid010 = trilinearInterpolationSimpleFastFast(p010, dist010, color010);
			float3 p001 = worldPos+float3(M, M, P); float dist001; float3 color001; bool valid001 = trilinearInterpolationSimpleFastFast(p001, dist001, color001);
			float3 p110 = worldPos+float3(P, P, M); float dist110; float3 color110; bool valid110 = trilinearInterpolationSimpleFastFast(p110, dist110, color110);
			float3 p011 = worldPos+float3(M, P, P); float dist011; float3 color011; bool valid011 = trilinearInterpolationSimpleFastFast(p011, dist011, color011);
			float3 p101 = worldPos+float3(P, M, P); float dist101; float3 color101; bool valid101 = trilinearInterpolationSimpleFastFast(p101, dist101, color101);
			float3 p111 = worldPos+float3(P, P, P); float dist111; float3 color111; bool valid111 = trilinearInterpolationSimpleFastFast(p111, dist111, color111);

			if(!valid000 || !valid100 || !valid010 || !valid001 || !valid110 || !valid011 || !valid101 || !valid111) return;
	
			uint cubeindex = 0;
			if(dist010 < isolevel) cubeindex += 1;
			if(dist110 < isolevel) cubeindex += 2;
			if(dist100 < isolevel) cubeindex += 4;
			if(dist000 < isolevel) cubeindex += 8;
			if(dist011 < isolevel) cubeindex += 16;
			if(dist111 < isolevel) cubeindex += 32;
			if(dist101 < isolevel) cubeindex += 64;
			if(dist001 < isolevel) cubeindex += 128;

			const float thres = g_thresMarchingCubes;
			float distArray[] = {dist000, dist100, dist010, dist001, dist110, dist011, dist101, dist111};
			for(uint k = 0; k<8; k++)
			{
				for(uint l = 0; l<8; l++)
				{
					if(distArray[k]*distArray[l] < 0.0f)
					{
						if(abs(distArray[k]) + abs(distArray[l]) > thres) return;
					}
					else
					{
						if(abs(distArray[k]-distArray[l]) > thres) return;
					}
				}
			}

			if(abs(dist000) > g_thresMarchingCubes2) return;
			if(abs(dist100) > g_thresMarchingCubes2) return;
			if(abs(dist010) > g_thresMarchingCubes2) return;
			if(abs(dist001) > g_thresMarchingCubes2) return;
			if(abs(dist110) > g_thresMarchingCubes2) return;
			if(abs(dist011) > g_thresMarchingCubes2) return;
			if(abs(dist101) > g_thresMarchingCubes2) return;
			if(abs(dist111) > g_thresMarchingCubes2) return;

			if(edgeTable[cubeindex] == 0 || edgeTable[cubeindex] == 255) return; // added by me edgeTable[cubeindex] == 255 !!!

			/*Vertex vertlist[12];
			if(edgeTable[cubeindex] & 1)	vertlist[0]  = VertexInterp(isolevel, p010, p110, dist010, dist110, color010, color110);
			if(edgeTable[cubeindex] & 2)	vertlist[1]  = VertexInterp(isolevel, p110, p100, dist110, dist100, color110, color100);
			if(edgeTable[cubeindex] & 4)	vertlist[2]  = VertexInterp(isolevel, p100, p000, dist100, dist000, color100, color000);
			if(edgeTable[cubeindex] & 8)	vertlist[3]  = VertexInterp(isolevel, p000, p010, dist000, dist010, color000, color010);
			if(edgeTable[cubeindex] & 16)	vertlist[4]  = VertexInterp(isolevel, p011, p111, dist011, dist111, color011, color111);
			if(edgeTable[cubeindex] & 32)	vertlist[5]  = VertexInterp(isolevel, p111, p101, dist111, dist101, color111, color101);
			if(edgeTable[cubeindex] & 64)	vertlist[6]  = VertexInterp(isolevel, p101, p001, dist101, dist001, color101, color001);
			if(edgeTable[cubeindex] & 128)	vertlist[7]  = VertexInterp(isolevel, p001, p011, dist001, dist011, color001, color011);
			if(edgeTable[cubeindex] & 256)	vertlist[8]	 = VertexInterp(isolevel, p010, p011, dist010, dist011, color010, color011);
			if(edgeTable[cubeindex] & 512)	vertlist[9]  = VertexInterp(isolevel, p110, p111, dist110, dist111, color110, color111);
			if(edgeTable[cubeindex] & 1024) vertlist[10] = VertexInterp(isolevel, p100, p101, dist100, dist101, color100, color101);
			if(edgeTable[cubeindex] & 2048) vertlist[11] = VertexInterp(isolevel, p000, p001, dist000, dist001, color000, color001);*/

			Voxel v = getVoxel(g_SDFBlocksSDF, g_SDFBlocksRGBW, entry.ptr+linearizeVoxelPos(GTid));

			Vertex vertlist[12];
			if(edgeTable[cubeindex] & 1)	vertlist[0]  = VertexInterp(isolevel, p010, p110, dist010, dist110, v.color, v.color);
			if(edgeTable[cubeindex] & 2)	vertlist[1]  = VertexInterp(isolevel, p110, p100, dist110, dist100, v.color, v.color);
			if(edgeTable[cubeindex] & 4)	vertlist[2]  = VertexInterp(isolevel, p100, p000, dist100, dist000, v.color, v.color);
			if(edgeTable[cubeindex] & 8)	vertlist[3]  = VertexInterp(isolevel, p000, p010, dist000, dist010, v.color, v.color);
			if(edgeTable[cubeindex] & 16)	vertlist[4]  = VertexInterp(isolevel, p011, p111, dist011, dist111, v.color, v.color);
			if(edgeTable[cubeindex] & 32)	vertlist[5]  = VertexInterp(isolevel, p111, p101, dist111, dist101, v.color, v.color);
			if(edgeTable[cubeindex] & 64)	vertlist[6]  = VertexInterp(isolevel, p101, p001, dist101, dist001, v.color, v.color);
			if(edgeTable[cubeindex] & 128)	vertlist[7]  = VertexInterp(isolevel, p001, p011, dist001, dist011, v.color, v.color);
			if(edgeTable[cubeindex] & 256)	vertlist[8]	 = VertexInterp(isolevel, p010, p011, dist010, dist011, v.color, v.color);
			if(edgeTable[cubeindex] & 512)	vertlist[9]  = VertexInterp(isolevel, p110, p111, dist110, dist111, v.color, v.color);
			if(edgeTable[cubeindex] & 1024) vertlist[10] = VertexInterp(isolevel, p100, p101, dist100, dist101, v.color, v.color);
			if(edgeTable[cubeindex] & 2048) vertlist[11] = VertexInterp(isolevel, p000, p001, dist000, dist001, v.color, v.color);

			[allow_uav_condition]
			for(int i=0; triTable[cubeindex][i] != -1; i+=3)
			{
				Triangle t;
				t.v0 = vertlist[triTable[cubeindex][i+0]];
				t.v1 = vertlist[triTable[cubeindex][i+1]];
				t.v2 = vertlist[triTable[cubeindex][i+2]];

				t.v0.c /= 255.0f;
				t.v1.c /= 255.0f;
				t.v2.c /= 255.0f;

				g_triangles.Append(t);
			}
		}
	}
}
