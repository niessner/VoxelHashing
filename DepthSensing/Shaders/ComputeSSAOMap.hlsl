cbuffer consts : register(cb0)
{
	int g_ImageWidth;
	int g_ImageHeight;
	uint dummy0;
	uint dummy1;

	float4 g_rotationVectors[16];
};

Texture2D<float> input : register(t0);
RWTexture2D<float> output : register(u0);

// should be set by application
#ifndef groupthreads
#define groupthreads 1
#endif

//#include "KinectCameraUtil.h.hlsl"

#define MINF asfloat(0xff800000)

///////////////////////////////////////////////////////////
// Implementation of
// SHADER X7 Article 6.1 Screen-Space Ambient Occlusion
// VLADIMIR KAJALIN
///////////////////////////////////////////////////////////

[numthreads(groupthreads, groupthreads, 1)]
void computeSSAOMapCS(int3 dTid : SV_DispatchThreadID)
{
	output[dTid.xy] = 0.0f;
	
	float fSceneDepthP = input[dTid.xy];
	
	if(fSceneDepthP.x != MINF)
	{
		int2 rotationTC = dTid.xy%4.0f;
		float3 vRotation = g_rotationVectors[4*rotationTC.x+rotationTC.y].xyz;
		
		float3x3 rotMat;
		float h = 1.0f/(1.0f+vRotation.z);
		rotMat._m00 = h*vRotation.y*vRotation.y+vRotation.z;
		rotMat._m01 = -h*vRotation.y*vRotation.x;
		rotMat._m02 = -vRotation.x;
		rotMat._m10 = -h*vRotation.y*vRotation.x;
		rotMat._m11 = h*vRotation.x*vRotation.x+vRotation.z;
		rotMat._m12 = -vRotation.y; rotMat._m20 = vRotation.x;
		rotMat._m21 = vRotation.y; rotMat._m22 = vRotation.z;
		
		const int nSampleNum = 16;
		float offsetScale = 0.001f;
		const float offsetScaleStep = 1+2.4f/nSampleNum;
		
		float Accessibility = 0.0f;
		
		[unroll]
		for(int i = 0; i<(nSampleNum/8); i++)
		{
			[unroll]
			for(int x=-1; x<=1; x+=2)
			{
				[unroll]
				for(int y=-1; y<=1; y+=2)
				{
					[unroll]
					for(int z=-1; z<=1; z+=2)
					{
						float3 vOffset = normalize(float3(x, y, z))*(offsetScale*=offsetScaleStep);
						
						float3 vRotatedOffset = mul(vOffset, rotMat);
						
						float3 vSamplePos = float3(dTid.xy/float2(g_ImageWidth, g_ImageHeight), fSceneDepthP);
						vSamplePos += float3(vRotatedOffset.xy, vRotatedOffset.z*fSceneDepthP*2.0f);
						
						float fSceneDepthS = input[(int2)(vSamplePos.xy*int2(g_ImageWidth, g_ImageHeight))];
						if(fSceneDepthS != MINF)
						{
							float fRangeIsValid = saturate(((fSceneDepthP - fSceneDepthS)/fSceneDepthS));
							Accessibility += lerp(fSceneDepthS>vSamplePos.z, 0.5f, fRangeIsValid);
						}
					}
				}
			}
		}
					
		Accessibility = Accessibility/nSampleNum;
		output[dTid.xy] = saturate(Accessibility*Accessibility+Accessibility);
	}
}
