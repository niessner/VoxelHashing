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

#define MINF asfloat(0xff800000)

[numthreads(groupthreads, groupthreads, 1)]
void downsamplingCS(uint3 DTid : SV_DispatchThreadID)
{
	int valid = 0;
	if (input[2*DTid.xy].x != MINF)	valid++;
	if (input[2*DTid.xy+int2(1, 0)].x != MINF) valid++;
	if (input[2*DTid.xy+int2(0, 1)].x != MINF) valid++;
	if (input[2*DTid.xy+int2(1, 1)].x != MINF) valid++;
	output[DTid.xy] = (1.0f/(float)valid) * (input[2*DTid.xy]+input[2*DTid.xy+int2(1, 0)]+input[2*DTid.xy+int2(0, 1)]+input[2*DTid.xy+int2(1, 1)]); // Thats wrong !!!!!!! fix it !!!!!!!!
}

