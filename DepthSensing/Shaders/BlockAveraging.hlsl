cbuffer consts : register(cb0)
{
	int g_imageWidth;
	int g_imageHeigth;
	
	float g_sigmaD;
	float g_sigmaR;
};

Texture2D<float> input : register(t0);
RWTexture2D<float> output : register(u0);

// should be set by application
#ifndef groupthreads
#define groupthreads 1
#endif

#define MINF asfloat(0xff800000)

float  gaussD(float sigma, int x, int y)
{
	return exp(-((x*x+y*y)/(2.0f*sigma*sigma)));
}

[numthreads(groupthreads, groupthreads, 1)]
void averageCS(int3 dTid : SV_DispatchThreadID)
{
	int kernelRadius = (int)ceil(2.0*g_sigmaD);
	int kernelSize = 2*kernelRadius+1;
	float thres = 3.0f*g_sigmaR;
	
	float sum = 0.0f;
	float sumWeight = 0.0f;
	
	output[dTid.xy] = MINF;

	float intCenter = input[dTid.xy];
	if(intCenter != MINF)
	{
		for(int m = dTid.x-kernelRadius; m <= dTid.x+kernelRadius; m++)
		{
			for(int n = dTid.y-kernelRadius; n <= dTid.y+kernelRadius; n++)
			{
				if(m >= 0 && n >= 0 && m < g_imageWidth && n < g_imageHeigth)
				{
					uint2 pos = uint2(m, n);
					float intKerPos = input[pos];

					if(intKerPos != MINF && abs(intCenter-intKerPos) < thres)
					{
						float weight = gaussD(g_sigmaD, m-dTid.x, n-dTid.y);
								
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
