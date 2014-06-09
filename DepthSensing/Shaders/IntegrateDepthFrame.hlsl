cbuffer consts : register(cb0)
{
	// Grid
	float3 g_gridPosition;
	float align0;

	int3 g_gridDimensions;
	int align1;

	float3 g_voxelExtends;
	float align2;

	int g_ImageWidth;
	int g_ImageHeight;
	int align4;
	int align5;

	float4x4 g_viewMat;
};

Texture2D<float> g_depth : register(t0);
Texture2D<float4> g_color : register(t1);
RWBuffer<int> g_voxelBuffer : register(u0);

#include "KinectCameraUtil.h.hlsl"
#include "RayCastingUtil.h.hlsl"

// Merges two voxels (v0 is the input voxel, v1 the currently stored voxel)
Voxel combineVoxel(in Voxel v0, in Voxel v1)
{
	Voxel v;
	v.color = (10*v0.weight * v0.color + v1.weight * v1.color)/(10*v0.weight + v1.weight);	//give the currently observed color more weight
	v.sdf = (v0.sdf * v0.weight + v1.sdf * v1.weight) / (v0.weight + v1.weight);
	v.weight = min(255, v0.weight + v1.weight);
	return v;
}

[numthreads(groupthreads, groupthreads, groupthreads)]
void integrateDepthFrameCS(int3 dTid : SV_DispatchThreadID)
{	
	float3 posWorld = computeSamplePositions(dTid);
	float3 posWorldTransformed = mul(float4(posWorld, 1.0f), g_viewMat).xyz;

	uint2 screenPos = uint2(cameraToKinectScreenInt(posWorldTransformed));
	if(screenPos.x < (uint)g_ImageWidth && screenPos.y < (uint)g_ImageHeight)
	{
		float depth = g_depth[screenPos];
		if (depth != MINF) //valid depth value
		{										
			float sdf = depth - posWorldTransformed.z;
			float truncation = 2.0f*0.05f;//+0.02f*posWorldTransformed.z; //based on distance
			if (sdf > -truncation) //check if in truncation range
			{
				if (sdf >= 0.0f)
				{
					sdf = min(truncation, sdf);
				}
				else
				{
					sdf = max(-truncation, sdf);
				}

				Voxel curr;	//construct current voxel
				curr.sdf = sdf;
				curr.weight = 1;
				curr.color = (int3)(g_color[screenPos].xyz*255.0f);

				Voxel prev = getVoxel(g_voxelBuffer, linearizeIndex(dTid));
				setVoxel(g_voxelBuffer, linearizeIndex(dTid), combineVoxel(curr, prev));
			}
		}
	}
}
