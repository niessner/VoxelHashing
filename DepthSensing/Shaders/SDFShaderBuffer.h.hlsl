#ifndef _SDF_SHADER_BUFFER_H_
#define _SDF_SHADER_BUFFER_H_

cbuffer cbCS : register( b0 )
{
	float4x4	g_RigidTransform;
	float4x4	g_RigidTransformInverse;
	uint		g_HashNumBuckets;
	uint		g_HashBucketSize;
	uint		g_ImageWidth;
	uint		g_ImageHeight;
	float		g_VirtualVoxelSize;
	float		g_VirtualVoxelResolutionScalar;
	uint		g_NumSDFBlocks;
	uint		g_NumOccupiedSDFBlocks;
};

cbuffer cbCS : register( b1 )
{
	float4x4	g_Other_RigidTransform;
	float4x4	g_Other_RigidTransformInverse;
	uint		g_Other_HashNumBuckets;
	uint		g_Other_HashBucketSize;
	uint		g_Other_ImageWidth;
	uint		g_Other_ImageHeight;
	float		g_Other_VirtualVoxelSize;
	float		g_Other_VirtualVoxelResolutionScalar;
	uint		g_Other_NumSDFBlocks;
	uint		g_Other_NumOccupiedSDFBlocks;
};


#endif
