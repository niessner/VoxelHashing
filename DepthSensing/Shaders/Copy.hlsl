cbuffer consts : register(cb0)
{
	int imageWidth;
	int imageHeigth;
	
	float2 dummy;
};

Texture2D<float4> input : register(t0);
RWTexture2D<float4> output : register(u0);

[numthreads(groupthreads, groupthreads, 1)]
void copyCS(int3 dTid : SV_DispatchThreadID)
{		
	output[dTid.xy] = input[dTid.xy];
}
