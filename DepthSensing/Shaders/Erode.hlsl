cbuffer consts : register(cb0)
{
	int g_imageWidth;
	int g_imageHeigth;
	float g_distThres;
	int g_stencilSize;
};

Texture2D<float> input : register(t0);
Texture2D<float4> inputColor : register(t1);
RWTexture2D<float> output : register(u0);

#define MINF asfloat(0xff800000)

[numthreads(groupthreads, groupthreads, 1)]
void erodeCS(int3 dTid : SV_DispatchThreadID)
{
	float intCenter = input[dTid.xy];
	float4 colorCenter = inputColor[dTid.xy];

	output[dTid.xy] = intCenter;

	if(intCenter != MINF)
	{
		for (int m = dTid.x-g_stencilSize; m <= dTid.x+g_stencilSize; m++)
		{
			for (int n = dTid.y-g_stencilSize; n <= dTid.y+g_stencilSize; n++)
			{
				if(m >= 0 && n >= 0 && m < g_imageWidth && n < g_imageHeigth)
				{
					uint2 pos = uint2(m, n);
					float intKerPos = input[pos];
					float4 colorKerPos = inputColor[pos];
					
					float dx = abs(intKerPos - intCenter);
					//float d = sqrt((m-dTid.x)*(m-dTid.x)+(n-dTid.y)*(n-dTid.y));
					
					float w = dx; //(1.0f/(1.0f+d))*(dx);


					if((intKerPos == MINF || w > g_distThres) && (length(colorKerPos.xyz-colorCenter.xyz) < 0.1f))
					{
						output[dTid.xy] = MINF;
					}
				}
			}
		}
	}
}
