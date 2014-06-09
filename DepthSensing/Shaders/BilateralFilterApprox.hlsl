cbuffer consts : register(cb0)
{
	int g_imageWidth;
	int g_imageHeigth;

	int g_kernelRadius;
	float g_thres;
};

Texture2D<float> input : register(t0);
RWTexture2D<float> output : register(u0);

// should be set by application
#ifndef groupthreads
#define groupthreads 1
#endif

#define MINF asfloat(0xff800000)

[numthreads(groupthreads, groupthreads, 1)]
void bilateralFilterApproxCS(int3 dTid : SV_DispatchThreadID)
{	
	float sum = 0.0f;
	float sumWeight = 0.0f;
	
	output[dTid.xy] = MINF;
	float intCenter = input[dTid.xy];
	
	if(intCenter != MINF)
	{
		for(int m = dTid.x-g_kernelRadius; m <= dTid.x+g_kernelRadius; m++)
		{
			for(int n = dTid.y-g_kernelRadius; n <= dTid.y+g_kernelRadius; n++)
			{
				if(m >= 0 && n >= 0 && m < g_imageWidth && n < g_imageHeigth)
				{
					float intKerPos = input[uint2(m, n)];
					float dx = abs(intKerPos - intCenter);

					if(intKerPos != MINF && dx < g_thres)
					{
						float d = sqrt((m-dTid.x)*(m-dTid.x)+(n-dTid.y)*(n-dTid.y));
						float weight = (1.0f/(1.0f+d))*(g_thres-dx);

						sumWeight += weight;
						sum += weight*intKerPos;
					}
				}
			}
		}

		if(sumWeight > 0.0f)
		{
			output[dTid.xy] = sum / sumWeight;
		}
	}
}
