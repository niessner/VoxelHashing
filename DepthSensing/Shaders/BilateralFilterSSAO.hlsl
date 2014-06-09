cbuffer consts : register(cb0)
{
	int imageWidth;
	int imageHeigth;

	float sigmaD;
	float sigmaR;
};

Texture2D<float> inputDepth : register(t0);
Texture2D<float> inputSSAOMap : register(t1);

RWTexture2D<float> output : register(u0);

// should be set by application
#ifndef groupthreads
#define groupthreads 1
#endif

#define MINF asfloat(0xff800000)

float gaussR(float sigma, float dist)
{
	return exp(-(dist*dist)/(2.0*sigma*sigma));
}

float  gaussD(float sigma, int x, int y)
{
	return exp(-((x*x+y*y)/(2.0f*sigma*sigma)));
}

[numthreads(groupthreads, groupthreads, 1)]
void bilateralFilterSSAOCS(int3 dTid : SV_DispatchThreadID)
{
	int kernelRadius = (int)ceil(2.0*sigmaD);
	
	float sum = 0.0f;
	float sumWeight = 0.0f;
	
	output[dTid.xy] = 0.0f;

	float intCenterDepth = inputDepth[dTid.xy];
	
	for(int m = dTid.x-kernelRadius; m <= dTid.x+kernelRadius; m++)
	{
		for(int n = dTid.y-kernelRadius; n <= dTid.y+kernelRadius; n++)
		{
			if(m >= 0 && n >= 0 && m < imageWidth && n < imageHeigth)
			{
				uint2 pos = uint2(m, n);
				float intKerPosDepth = inputDepth[pos];
				float intKerPosSSAO = inputSSAOMap[pos];

				float weight = gaussD(sigmaD, m-dTid.x, n-dTid.y)*gaussR(sigmaR, intKerPosDepth-intCenterDepth);
								
				sumWeight += weight;
				sum += weight*intKerPosSSAO;
			}
		}
	}

	if(sumWeight > 0.0f)
	{
		output[dTid.xy] = sum / sumWeight;
	}
}
