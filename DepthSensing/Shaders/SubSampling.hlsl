cbuffer consts : register(cb0)
{
	int imageWidth;
	int imageHeigth;

	uint dummy0;
	uint dummy1;
};

Texture2D<float> input : register(t0);
RWTexture2D<float> output : register(u0);

// should be set by application
#ifndef groupthreads
#define groupthreads 1
#endif

#define MINF asfloat(0xff800000)

[numthreads(groupthreads, groupthreads, 1)]
void subSampleCS(uint3 DTid : SV_DispatchThreadID)
{
	output[DTid.xy] = input[2*DTid.xy];
}
