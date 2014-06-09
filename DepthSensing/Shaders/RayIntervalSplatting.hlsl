Buffer<int>		g_VoxelHash : register(t0);
   
cbuffer cbConstant : register(b1)
{
	float4x4	g_ViewMat;
	float4x4	g_ViewMatInverse;
	uint		g_RenderTargetWidth;
	uint		g_RenderTargetHeight;
	uint		g_splatMinimum;
	uint		g_dummyRayInteveral337;
};

#include "GlobalAppStateShaderBuffer.h.hlsl"
#include "SDFShaderBuffer.h.hlsl"
#include "KinectCameraUtil.h.hlsl"
#include "VoxelUtilHashSDF.h.hlsl"

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


float3 cameraToKinectProjStereo(float3 pos)
{
	float2 proj = cameraToKinectScreenFloat(pos);

    float3 pImage = float3(proj.x, proj.y, pos.z);

	pImage.x = (2.0f*pImage.x - (g_RenderTargetWidth-1.0f))/(g_RenderTargetWidth-1.0f);
	pImage.y = ((g_RenderTargetHeight-1.0f) - 2.0f*pImage.y)/(g_RenderTargetHeight-1.0f);
	pImage.z = cameraToKinectProjZ(pImage.z);

	return pImage;
}

[maxvertexcount(4)]
void GS(point GS_INPUT points[1], uint primID : SV_PrimitiveID, inout TriangleStream<PS_INPUT> triStream)
{
	PS_INPUT output;

	HashEntry entry = getHashEntry(g_VoxelHash, primID);
	float3 worldCurrentVoxel = SDFBlockToWorld(entry.pos);

	float3 MINV = worldCurrentVoxel.xyz-g_VirtualVoxelSize/2.0;
	float3 maxv = MINV+SDF_BLOCK_SIZE*g_VirtualVoxelSize;

	float3 proj000 = cameraToKinectProjStereo(mul(float4(MINV.x, MINV.y, MINV.z, 1.0f), g_ViewMat).xyz);
	float3 proj100 = cameraToKinectProjStereo(mul(float4(maxv.x, MINV.y, MINV.z, 1.0f), g_ViewMat).xyz);
	float3 proj010 = cameraToKinectProjStereo(mul(float4(MINV.x, maxv.y, MINV.z, 1.0f), g_ViewMat).xyz);
	float3 proj001 = cameraToKinectProjStereo(mul(float4(MINV.x, MINV.y, maxv.z, 1.0f), g_ViewMat).xyz);
	float3 proj110 = cameraToKinectProjStereo(mul(float4(maxv.x, maxv.y, MINV.z, 1.0f), g_ViewMat).xyz);
	float3 proj011 = cameraToKinectProjStereo(mul(float4(MINV.x, maxv.y, maxv.z, 1.0f), g_ViewMat).xyz);
	float3 proj101 = cameraToKinectProjStereo(mul(float4(maxv.x, MINV.y, maxv.z, 1.0f), g_ViewMat).xyz);
	float3 proj111 = cameraToKinectProjStereo(mul(float4(maxv.x, maxv.y, maxv.z, 1.0f), g_ViewMat).xyz);

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
	
	float depth = maxFinal.z;
	if(g_splatMinimum == 1)
	{
		depth = minFinal.z;
	}

	output.position = float4(maxFinal.x, minFinal.y, depth, 1.0f);
	triStream.Append(output);

	output.position = float4(minFinal.x, minFinal.y, depth, 1.0f);
	triStream.Append(output);

	output.position = float4(maxFinal.x, maxFinal.y, depth, 1.0f);
	triStream.Append(output);
		
	output.position = float4(minFinal.x, maxFinal.y, depth, 1.0f);
	triStream.Append(output);


	
	/*float3 centerWorld = 0.5f*(MIN+max);
	float3 center = mul(float4(centerWorld, 1.0f), g_ViewMat).xyz;
	float3 dir = normalize(center).xyz;

	const float diameter = sqrt(3*SDF_BLOCK_SIZE*SDF_BLOCK_SIZE*g_VirtualVoxelSize*g_VirtualVoxelSize);
	const float radius = diameter/2.0f;

	const float4 scale = float4(diameter, diameter, 0.0, 0.0);

	float4 viewPos;
	if(g_splatMinimum == 1)
	{
		viewPos = float4(center-radius*dir, 1.0f);
	}
	else
	{
		viewPos = float4(center+radius*dir, 1.0f);
	}

	[unroll]
	for(uint c = 0; c < 4; ++c) {
		float4 corner = viewPos+offsets[c]*scale;
		output.position = float4(cameraToKinectProj(corner.xyz), 1.0f);
		triStream.Append(output);
	}*/
}

void PS(PS_INPUT input) 
{
}
