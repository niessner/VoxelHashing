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

#define MINF asfloat(0xff800000)

#include "GlobalAppStateShaderBuffer.h.hlsl"
#include "KinectCameraUtil.h.hlsl"

[numthreads(groupthreads, groupthreads, 1)]
void stereoMaskCameraSpaceProjectionCS(int3 dTid : SV_DispatchThreadID)
{
	output[dTid.xy] = float4(MINF, MINF, MINF, MINF);

	float depth = input[dTid.xy];

	if(depth != MINF)
	{
		if(dTid.x >= 240 && dTid.x < 240+1440 && dTid.y >= 0 && dTid.y < 1080)
		{
			float4 camSpace = float4(kinectDepthToSkeleton(dTid.x, dTid.y, depth), 1.0f);

			float4 worldSpace = mul(camSpace, g_camToWorldStereo);
			float4 camOther = mul(worldSpace, g_worldToCamStereoOther);
			//camOther/=camOther.w;

			float4 p = float4(camOther.x, camOther.y, 0.0f, camOther.z);
			float4 proj = mul(p, g_intrinsicsStereoOther);
			float2 screen = float2(proj.x/proj.w, proj.y/proj.w);
			int2 screenInt = int2(screen + float2(0.5f, 0.5f));

			if(screenInt.x >= 240 && screenInt.x < 240+1440 && screenInt.y >= 0 && screenInt.y < 1080)
			{
				output[dTid.xy] = camSpace;	
			}
		}
	}
}
