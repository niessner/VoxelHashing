cbuffer consts : register(cb0)
{
	int g_ImageWidth;
	int g_ImageHeight;
	uint dummy0;
	uint dummy1;
};

Texture2D<float> input : register(t0);
RWTexture2D<float4> output : register(u0);

#include "GlobalAppStateShaderBuffer.h.hlsl"
#include "KinectCameraUtil.h.hlsl"

// should be set by application
#ifndef groupthreads
#define groupthreads 1
#endif

#define MINF asfloat(0xff800000)

float3 HSVtoRGB(float3 hsv)
{
	float H = hsv[0];
	float S = hsv[1];
	float V = hsv[2];

	float hd = H/60.0f;
	uint h = (uint)hd;
	float f = hd-h;

	float p = V*(1.0f-S);
	float q = V*(1.0f-S*f);
	float t = V*(1.0f-S*(1.0f-f));

	if(h == 0 || h == 6)
	{
		return float3(V, t, p);
	}
	else if(h == 1)
	{
		return float3(q, V, p);
	}
	else if(h == 2)
	{
		return float3(p, V, t);
	}
	else if(h == 3)
	{
		return float3(p, q, V);
	}
	else if(h == 4)
	{
		return float3(t, p, V);
	}
	else
	{
		return float3(V, p, q);
	}
}


[numthreads(groupthreads, groupthreads, 1)]
void HSVdepthMapComputationCS(int3 dTid : SV_DispatchThreadID)
{
	const float depthMin = g_SensorDepthWorldMin;
	const float depthMax = g_SensorDepthWorldMax;

	output[dTid.xy] = MINF;

	float pixel = input[dTid.xy];
	float depthZeroOne = (pixel - depthMin)/(depthMax - depthMin);

	if(pixel.x != MINF)
	{
		float3 colorRGB = HSVtoRGB(float3(240*clamp((1.0-depthZeroOne), 0.0f, 1.0f), 1.0, 0.5));
		output[dTid.xy] = float4(colorRGB, 1.0f);
	}
	
	// See Kinect SDK, Kinect Explorer Sample, input in mm
	//float intensityTable = 1.0f - min(255, log( ((pixel - 1000.0f*g_SensorDepthWorldMin) / 500) + 1) * 74) / 255.0f;
	//output[dTid.xy] = float4(intensityTable, intensityTable, intensityTable, 1.0f);
}
