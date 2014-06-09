Buffer<int>		g_VoxelHash : register(t0);		

Buffer<int>		g_FragmentCountBufferSRV : register(t1);
Buffer<int>		g_FragmentPrefixSumBufferSRV : register(t2);
Buffer<float>	g_FragmentSortedDepthBufferSRV : register(t3);

Buffer<float>	g_SDFBlocksSDFSRV : register(t5);
Buffer<int>		g_SDFBlocksRGBWSRV : register(t7);
Buffer<int>		g_DecisionArraySRV : register(t4);


RWBuffer<int>	g_FragmentCountBufferUAV : register(u0);
RWBuffer<int>	g_FragmentPrefixSumBufferUAV : register(u1);
RWBuffer<float>	g_FragmentSortedDepthBufferUAV : register(u2);

RWBuffer<int>	g_DecisionArrayUAV : register(u6);

#define NUM_GROUPS_X 1024 // to be in-sync with the define on the Cpu

cbuffer cbConstant : register(b1)
{
	float4x4	g_ViewMat;
	float4x4	g_ViewMatInverse;
	uint		g_RenderTargetWidth;
	uint		g_RenderTargetHeight;
	uint		g_dummyRayInteveral337;
	uint		g_dummyRayInteveral338;
};

#include "GlobalAppStateShaderBuffer.h.hlsl"
#include "SDFShaderBuffer.h.hlsl"
#include "KinectCameraUtil.h.hlsl"
#include "VoxelUtilHashSDF.h.hlsl"

////////////////////////////////////////////////////////////////////
// General setup
////////////////////////////////////////////////////////////////////

static const float4 offsets[4] = 
{
    float4( 0.5, -0.5, 0.0, 0.0),
    float4(-0.5, -0.5, 0.0, 0.0),
	
	float4( 0.5,  0.5, 0.0, 0.0),
    float4(-0.5,  0.5, 0.0, 0.0)
};

struct GS_INPUT
{
};

struct PS_INPUT
{
    float4 position	: SV_POSITION;
};

GS_INPUT VS()
{
    GS_INPUT output = (GS_INPUT)0;
 
    return output;
}

[maxvertexcount(4)]
void GS(point GS_INPUT points[1], uint primID : SV_PrimitiveID, inout TriangleStream<PS_INPUT> triStream)
{
	PS_INPUT output;

	HashEntry entry = getHashEntry(g_VoxelHash, primID);
	float3 worldCurrentVoxel = SDFBlockToWorld(entry.pos);

	float3 MINV = worldCurrentVoxel.xyz-g_VirtualVoxelSize/2.0;
	float3 maxv = MINV+SDF_BLOCK_SIZE*g_VirtualVoxelSize;

	float3 proj000 = cameraToKinectProj(mul(float4(MINV.x, MINV.y, MINV.z, 1.0f), g_ViewMat).xyz);
	float3 proj100 = cameraToKinectProj(mul(float4(maxv.x, MINV.y, MINV.z, 1.0f), g_ViewMat).xyz);
	float3 proj010 = cameraToKinectProj(mul(float4(MINV.x, maxv.y, MINV.z, 1.0f), g_ViewMat).xyz);
	float3 proj001 = cameraToKinectProj(mul(float4(MINV.x, MINV.y, maxv.z, 1.0f), g_ViewMat).xyz);
	float3 proj110 = cameraToKinectProj(mul(float4(maxv.x, maxv.y, MINV.z, 1.0f), g_ViewMat).xyz);
	float3 proj011 = cameraToKinectProj(mul(float4(MINV.x, maxv.y, maxv.z, 1.0f), g_ViewMat).xyz);
	float3 proj101 = cameraToKinectProj(mul(float4(maxv.x, MINV.y, maxv.z, 1.0f), g_ViewMat).xyz);
	float3 proj111 = cameraToKinectProj(mul(float4(maxv.x, maxv.y, maxv.z, 1.0f), g_ViewMat).xyz);

	// Tree Reduction Min
	float3 min00 = min(proj000, proj100);
	float3 min01 = min(proj010, proj001);
	float3 min10 = min(proj110, proj011);
	float3 min11 = min(proj101, proj111);

	float3 min0 = min(min00, min01);
	float3 min1 = min(min10, min11);

	float3 minFinal = min(min0, min1);

	// Tree Reduction Max 
	float3 max00 = max(proj000, proj100);
	float3 max01 = max(proj010, proj001);
	float3 max10 = max(proj110, proj011);
	float3 max11 = max(proj101, proj111);

	float3 max0 = max(max00, max01);
	float3 max1 = max(max10, max11);

	float3 maxFinal = max(max0, max1);
	
	float depth = minFinal.z;

	if(g_DecisionArraySRV[primID] == 1)
	{
		output.position = float4(maxFinal.x, minFinal.y, depth, 1.0f);
		triStream.Append(output);

		output.position = float4(minFinal.x, minFinal.y, depth, 1.0f);
		triStream.Append(output);

		output.position = float4(maxFinal.x, maxFinal.y, depth, 1.0f);
		triStream.Append(output);
		
		output.position = float4(minFinal.x, maxFinal.y, depth, 1.0f);
		triStream.Append(output);
	}
}

////////////////////////////////////////////////////////////////////
// For first pass // Count fragments per pixel
////////////////////////////////////////////////////////////////////

[numthreads(groupthreads*groupthreads, 1, 1)]
void clearCS(int3 dTid : SV_DispatchThreadID)
{
	if(dTid.x >= (int)(g_RenderTargetWidth*g_RenderTargetHeight)) return;
	
	g_FragmentCountBufferUAV[dTid.x] = 0;
}

void PS_Count(PS_INPUT input)
{
	int2 index = (int2)input.position.xy;
	int index1D = index.y*g_RenderTargetWidth+index.x;

	int original;
	InterlockedAdd(g_FragmentCountBufferUAV[index1D], 1, original);
}

////////////////////////////////////////////////////////////////////
// For third pass // Write fragments
////////////////////////////////////////////////////////////////////

void PS_Write(PS_INPUT input) 
{
	int2 index = (int2)input.position.xy;
	int index1D = index.y*g_RenderTargetWidth+index.x;

	int internalIndex;
	InterlockedAdd(g_FragmentCountBufferUAV[index1D], -1, internalIndex);
	int endIndex = g_FragmentPrefixSumBufferSRV[index1D];

	int offset = endIndex-internalIndex;

	g_FragmentSortedDepthBufferUAV[offset] = input.position.z;
}

////////////////////////////////////////////////////////////////////
// For forth pass // Sort fragments
////////////////////////////////////////////////////////////////////

static const uint maxLocalBufferSize = 30;  // adapt also in raycasting shader

[numthreads(groupthreads*groupthreads, 1, 1)]
void sortFragmentsCS(int3 dTid : SV_DispatchThreadID, int3 GTid : SV_GroupThreadID)
{
	if(dTid.x >= (int)(g_RenderTargetWidth*g_RenderTargetHeight)) return;

	// Allocate local memory
	float depthValuesLocalCopy[maxLocalBufferSize];

	int startIndex = 0;
	if(dTid.x != 0) startIndex = g_FragmentPrefixSumBufferSRV[dTid.x-1];
	int endIndex = g_FragmentPrefixSumBufferSRV[dTid.x];
	int length = min(endIndex-startIndex, maxLocalBufferSize);

	// Copy to local memory
	for(int k = 0; k < length; k++)
	{
		depthValuesLocalCopy[k] = g_FragmentSortedDepthBufferUAV[startIndex+k];
	}

	// Sort local copy
	for(int i = 0; i < length-1; i++)
	{
		float d = depthValuesLocalCopy[i];

		int swap_id = i;
		for(int j = i; j < length; j++)
		{
			float dj = depthValuesLocalCopy[j];
			if(dj < d)
			{
				d = dj;
				swap_id = j;
			}
		}

		if(swap_id != i)
		{			
			depthValuesLocalCopy[swap_id] = depthValuesLocalCopy[i];
			depthValuesLocalCopy[i] = d;
		}
	}

	// Copy to output buffer
	for(int l = 0; l < length; l++)
	{
		g_FragmentSortedDepthBufferUAV[startIndex+l] = depthValuesLocalCopy[l];
	}
}

////////////////////////////////////////////////////////////////////
// Pre-pass // Which blocks should be splatted
////////////////////////////////////////////////////////////////////

groupshared float shared_MinSDF[SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE / 2];

[numthreads( SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE / 2, 1, 1)]
void splatIdentifyCS(uint3 DTid : SV_DispatchThreadID, uint3 GTid : SV_GroupThreadID, uint GI: SV_GroupIndex, uint3 GID : SV_GroupID)
{
	const uint groupID = GID.x + GID.y * NUM_GROUPS_X;
	const uint hashIdx = groupID;
	if(hashIdx < g_NumOccupiedSDFBlocks)
	{ 
		HashEntry entry = getHashEntry(g_VoxelHash, hashIdx);
		uint idx0 = entry.ptr + 2*GTid.x+0;
		uint idx1 = entry.ptr + 2*GTid.x+1;
		Voxel v0 = getVoxel(g_SDFBlocksSDFSRV, g_SDFBlocksRGBWSRV, idx0);
		Voxel v1 = getVoxel(g_SDFBlocksSDFSRV, g_SDFBlocksRGBWSRV, idx1);
		if (v0.weight == 0)	v0.sdf = PINF;
		if (v1.weight == 0) v1.sdf = PINF;
		shared_MinSDF[GTid.x] = min(abs(v0.sdf), abs(v1.sdf));	//init shared memory
		
		uint numGroupThreads = SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE / 2;

		[loop]
		for(uint stride = 2; stride <= numGroupThreads; stride <<= 1)
		{
			GroupMemoryBarrierWithGroupSync();
			if((GTid.x  & (stride-1)) == (stride-1))
			{
				shared_MinSDF[GTid.x] = min(shared_MinSDF[GTid.x-stride/2], shared_MinSDF[GTid.x]);
			}
		}

		if(GTid.x == numGroupThreads - 1)
		{
            float minSDF = shared_MinSDF[GTid.x];
            
			g_DecisionArrayUAV[hashIdx] = 0;
			//float t = getTruncation(g_maxIntegrationDistance);
			
			if(minSDF < rayIncrement+g_VirtualVoxelSize)
			{
				g_DecisionArrayUAV[hashIdx] = 1;
			}
		}
	}
}
