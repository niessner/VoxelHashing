cbuffer consts : register(cb0)
{
	int imageWidth;
	int imageHeigth;

	float sigmaD;
	float sigmaR;
};

Texture2D<float> input : register(t0);
Texture2D<float4> input4F : register(t1);

RWTexture2D<float> output : register(u0);
RWTexture2D<float4> output4F : register(u1);

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

/////////////////////////////////////////
// Bilateral Completion w/ and w/o Color
/////////////////////////////////////////

static const float minFilterWeight = 10.0f;	// dunno this is just a wild guess TODO

[numthreads(groupthreads, groupthreads, 1)]
void bilateralCompletionCS(int3 dTid : SV_DispatchThreadID)
{
	int kernelRadius = (int)ceil(2.0*sigmaD);
	int kernelSize = 2*kernelRadius+1;
	
	float sum = 0.0f;
	float sumWeight = 0.0f;
	
	//output[dTid.xy] = MINF;

	float intCenter = input[dTid.xy];
	output[dTid.xy] = intCenter;
#ifdef WITH_COLOR
	output4F[dTid.xy] =  input4F[dTid.xy];
#endif

	if(intCenter == MINF)
	{
		//first find the average depth
		for (int m = dTid.x-kernelRadius; m <= dTid.x+kernelRadius; m++) {
			for (int n = dTid.y-kernelRadius; n <= dTid.y+kernelRadius; n++) {
				if(m >= 0 && n >= 0 && m < imageWidth && n < imageHeigth) {
					uint2 pos = uint2(m, n);
					float intKerPos = input[pos];

					if (intKerPos != MINF) {
						float weight = gaussD(sigmaD, m-dTid.x, n-dTid.y);
						sumWeight += weight;
						sum += weight*intKerPos;
					}
				}
			}
		}

		if(sumWeight == 0.0f) { return; }

		float avgDepth = sum / sumWeight;
		sum = sumWeight = 0.0f;
#ifdef WITH_COLOR
		float3 sumColor = 0.0f;
#endif
		for(int m = dTid.x-kernelRadius; m <= dTid.x+kernelRadius; m++)
		{
			for(int n = dTid.y-kernelRadius; n <= dTid.y+kernelRadius; n++)
			{
				if(m >= 0 && n >= 0 && m < imageWidth && n < imageHeigth)
				{
					uint2 pos = uint2(m, n);
					float intKerPos = input[pos];

					if(intKerPos != MINF)
					{
						float weight = gaussD(sigmaD, m-dTid.x, n-dTid.y)*gaussR(sigmaR, intKerPos-avgDepth);
								
						sumWeight += weight;
						sum += weight*intKerPos;
#ifdef WITH_COLOR
						float3 color = input4F[pos].xyz;
						sumColor += weight*color.xyz;
#endif
					}
				}
			}
		}

		if(sumWeight > minFilterWeight) // sumWeight != MIN
		{
			output[dTid.xy] = sum / sumWeight;
#ifdef WITH_COLOR
			output4F[dTid.xy] = float4(sumColor / sumWeight, 1.0f);
#endif
		}
	}
}

/////////////////////////////
// Standard Bilateral Filter
/////////////////////////////
[numthreads(groupthreads, groupthreads, 1)]
void bilateralFilterCS(int3 dTid : SV_DispatchThreadID)
{
	int kernelRadius = (int)ceil(2.0*sigmaD);
	
	float sum = 0.0f;
	float sumWeight = 0.0f;
	
	output[dTid.xy] = MINF;

	//if (dTid.x < kernelRadius || dTid.x >= imageWidth - kernelRadius || dTid.y < kernelRadius || dTid.y >= imageHeigth - kernelRadius)
	//	return;

	float intCenter = input[dTid.xy];
	if(intCenter != MINF)
	{
		for(int m = dTid.x-kernelRadius; m <= dTid.x+kernelRadius; m++)
		{
			for(int n = dTid.y-kernelRadius; n <= dTid.y+kernelRadius; n++)
			{
				if(m >= 0 && n >= 0 && m < imageWidth && n < imageHeigth)
				{
					uint2 pos = uint2(m, n);
					float intKerPos = input[pos];

					if(intKerPos != MINF)
					{
						float weight = gaussD(sigmaD, m-dTid.x, n-dTid.y)*gaussR(sigmaR, intKerPos-intCenter);
								
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

[numthreads(groupthreads, groupthreads, 1)]
void bilateralFilter4FCS(int3 dTid : SV_DispatchThreadID)
{
	int kernelRadius = (int)ceil(2.0*sigmaD);
	int kernelSize = 2*kernelRadius+1;
	
	float3 sum = float3(0.0f, 0.0f, 0.0f);
	float sumWeight = 0.0f;
	
	output4F[dTid.xy] = float4(MINF, MINF, MINF, MINF);

	float4 intCenter = input4F[dTid.xy];
	if(intCenter.x != MINF)
	{
		for(int m = dTid.x-kernelRadius; m <= dTid.x+kernelRadius; m++)
		{
			for(int n = dTid.y-kernelRadius; n <= dTid.y+kernelRadius; n++)
			{
				if(m >= 0 && n >= 0 && m < imageWidth && n < imageHeigth)
				{
					uint2 pos = uint2(m, n);
					float4 intKerPos = input4F[pos];

					if(intKerPos.x != MINF)
					{
						float d = distance(intKerPos.xyz, intCenter.xyz);
						float weight = gaussD(sigmaD, m-dTid.x, n-dTid.y)*gaussR(sigmaR, d);
								
						sumWeight += weight;
						sum += weight*intKerPos.xyz;
					}
				}
			}
		}

		if(sumWeight > 0.0f)
		{
			output4F[dTid.xy] = float4(sum / sumWeight, 1.0f);
		}
	}
}
