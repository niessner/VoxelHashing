cbuffer consts : register(cb0)
{
	int g_ImageWidth;
	int g_ImageHeight;
	uint dummy0;
	uint dummy1;
};

Texture2D<float> input : register(t0);
RWTexture2D<float4> output : register(u0);

// should be set by application
#ifndef groupthreads
#define groupthreads 1
#endif

//#ifndef NUI_CAMERA_DEPTH_IMAGE_TO_SKELETON_MULTIPLIER_320x240
//#define NUI_CAMERA_DEPTH_IMAGE_TO_SKELETON_MULTIPLIER_320x240 (3.501e-3f)
//#endif

#define MINF asfloat(0xff800000)

#include "GlobalAppStateShaderBuffer.h.hlsl"
#include "KinectCameraUtil.h.hlsl"

[numthreads(groupthreads, groupthreads, 1)]
void cameraSpaceProjectionCS(int3 dTid : SV_DispatchThreadID)
{
	output[dTid.xy] = float4(MINF, MINF, MINF, MINF);

	float depth = input[dTid.xy];

	if(depth != MINF)
	{
		output[dTid.xy] =  float4(kinectDepthToSkeleton(dTid.x, dTid.y, depth), 1.0f);
	}
} 

Texture2D<float4> input_cameraPos	: register(t1);
RWTexture2D<float> output_depth		: register(u1);

[numthreads(groupthreads, groupthreads, 1)]
void cameraSpaceToDepthMapCS(int3 dTid : SV_DispatchThreadID)
{
	//output_depth[dTid.xy] = MINF;
	float depth = input_cameraPos[dTid.xy].z;
	output_depth[dTid.xy] =  depth;
}
