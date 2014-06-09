cbuffer consts : register(cb0)
{
	int imageWidth;
	int imageHeigth;

	uint dummy0;
	uint dummy1;
};

Texture2D<float4> input : register(t0);
RWTexture2D<float4> output : register(u0);

// should be set by application
#ifndef groupthreads
#define groupthreads 1
#endif

[numthreads(groupthreads, groupthreads, 1)]
void splatWeightDivideCS(uint3 DTid : SV_DispatchThreadID)
{
	output[DTid.xy] = float4(input[DTid.xy].rgb/input[DTid.xy].a, 0.0f);
}
