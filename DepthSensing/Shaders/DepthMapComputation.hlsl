cbuffer consts : register(cb0)
{
	int g_ImageWidth;
	int g_ImageHeight;
	uint dummy0;
	uint dummy1;
};

Texture2D<uint> input : register(t0);
RWTexture2D<float> output : register(u0);

#include "GlobalAppStateShaderBuffer.h.hlsl"
#include "KinectCameraUtil.h.hlsl"

// should be set by application
#ifndef groupthreads
#define groupthreads 1
#endif

#define MINF asfloat(0xff800000)

[numthreads(groupthreads, groupthreads, 1)]
void depthMapComputationCS(int3 dTid : SV_DispatchThreadID)
{
	output[dTid.xy] = MINF;

	//const uint minDepth = ((uint)(g_SensorDepthWorldMin*1000.0f)) << 3;
	//const uint maxDepth = ((uint)(g_SensorDepthWorldMax*1000.0f)) << 3;

	const uint minDepth = ((uint)(g_SensorDepthWorldMin*1000.0f));
	const uint maxDepth = ((uint)(g_SensorDepthWorldMax*1000.0f));

	uint pixel = input[dTid.xy];

	if(pixel >= minDepth && pixel <= maxDepth)
	{
		//pixel = pixel >> 3;
		output[dTid.xy] = ((float)pixel)*0.001f;
	}
}
