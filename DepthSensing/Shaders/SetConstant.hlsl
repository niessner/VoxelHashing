cbuffer consts : register(cb0)
{
	int3 g_gridDimensions;	
	float g_value;
};

RWTexture3D<float> output : register(u0);

[numthreads(groupthreads, groupthreads, groupthreads)]
void setConstantCS(int3 dTid : SV_DispatchThreadID)
{		
	output[dTid] = g_value;
}
