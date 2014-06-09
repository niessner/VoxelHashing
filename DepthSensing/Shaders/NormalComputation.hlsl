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
void normalComputationCS(int3 dTid : SV_DispatchThreadID)
{
	output[dTid.xy] = float4(MINF, MINF, MINF, MINF);

	if(dTid.x > 0 && dTid.x < imageWidth-1 && dTid.y > 0 && dTid.y < imageHeigth-1)
	{
		float4 CC = input[dTid.xy];
		float4 PC = input[dTid.xy+int2(1, 0)];
		float4 CP = input[dTid.xy+int2(0, 1)];
		float4 MC = input[dTid.xy+int2(-1, 0)];
		float4 CM = input[dTid.xy+int2(0, -1)];

		if (CC.x != MINF && PC.x != MINF && CP.x != MINF && MC.x != MINF && CM.x != MINF)
		{
			float3 n = cross(PC.xyz-MC.xyz, CP.xyz-CM.xyz);
			float l = length(n);

			if (l > 0.0f)
			{
				output[dTid.xy] = float4((n/l).xyz, 1.0f);
			}
		}
	}
}
