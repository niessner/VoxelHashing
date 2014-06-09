cbuffer consts : register(b0)
{
	int g_ImageWidth;
	int g_ImageHeight;
	float distThres;
	float normalThres;
	float4x4 transform; // dx style transformation matrix !!!
	float g_levelFactor;
	float dummy00;
	float dummy01;
	float dummy02;
};

Texture2D<float4> target : register(t0);
Texture2D<float4> targetNormals : register(t1);
Texture2D<float4> targetColors : register(t2);
Texture2D<float4> input : register(t3);
Texture2D<float4> inputNormals : register(t4);
Texture2D<float4> inputColors : register(t5);

RWTexture2D<float4> output : register(u0);
RWTexture2D<float4> outputNormals : register(u1);

// should be set by application
#ifndef groupthreads
#define groupthreads 1
#endif

//#ifndef NUI_CAMERA_SKELETON_TO_DEPTH_IMAGE_MULTIPLIER_320x240
//#define NUI_CAMERA_SKELETON_TO_DEPTH_IMAGE_MULTIPLIER_320x240 (285.63f)
//#endif

//#ifndef FLT_EPSILON
//#define FLT_EPSILON 1.192092896e-07f
//#endif

#define MINF asfloat(0xff800000)
#define MAXF asfloat(0x7F7FFFFF)

#include "GlobalAppStateShaderBuffer.h.hlsl"
#include "KinectCameraUtil.h.hlsl"
#include "Util.h.hlsl"

void getBestCorrespondence1x1(uint2 screenPos, float4 pInput, float4 nInput, float4 cInput, out float4 pTarget, out float4 nTarget)
{
	pTarget = target[int2(screenPos.x, screenPos.y)];
	nTarget = targetNormals[int2(screenPos.x, screenPos.y)];
}

void getBestCorrespondence5x5(int2 screenPos, float4 pInput, float4 nInput, float4 cInput, out float4 pTarget, out float4 nTarget)
{
	float dMin = MAXF;
	pTarget = float4(MINF, MINF, MINF, MINF);
	nTarget = float4(MINF, MINF, MINF, MINF);

	for(int i = -2; i<=2; i++)
	{
		for(int j = -2; j<=2; j++)
		{
			int2 screenPosNew = screenPos + int2(i, j);

			if(screenPosNew.x >= 0 && screenPosNew.y >= 0 && screenPosNew.x < g_ImageWidth && screenPosNew.y < g_ImageHeight)
			{
				float4 p = target[int2(screenPosNew.x, screenPosNew.y)];
				float4 n = targetNormals[int2(screenPosNew.x, screenPosNew.y)];
				
				if(isValid(p) && isValid(n))
				{
					float d = length(pInput-p);

					if(d < dMin)
					{
						pTarget = p;
						nTarget = n;

						dMin = d;
					}
				}
			}
		}
	}
}

void getBestCorrespondence5x5Color(int2 screenPos, float4 pInput, float4 nInput, float4 cInput, out float4 pTarget, out float4 nTarget)
{
	float dMin = MAXF;
	pTarget = float4(MINF, MINF, MINF, MINF);
	nTarget = float4(MINF, MINF, MINF, MINF);

	for(int i = -2; i<=2; i++)
	{
		for(int j = -2; j<=2; j++)
		{
			int2 screenPosNew = screenPos + int2(i, j);

			if(screenPosNew.x >= 0 && screenPosNew.y >= 0 && screenPosNew.x < g_ImageWidth && screenPosNew.y < g_ImageHeight)
			{
				float4 p = target[int2(screenPosNew.x, screenPosNew.y)];
				float4 n = targetNormals[int2(screenPosNew.x, screenPosNew.y)];
				float4 c = targetColors[int2(screenPosNew.x, screenPosNew.y)];

				if(isValid(p) && isValid(n) && isValidCol(c)) // isValid(c)
				{
					float d = length(pInput.xyz-p.xyz);
					float cd = length(cInput.xyz-c.xyz);

					float dist = 4.0*d+cd;
					if(dist < dMin)
					{
						pTarget = p;
						nTarget = n;

						dMin = dist;
					}
				}
			}
		}
	}
}

[numthreads(groupthreads, groupthreads, 1)]
void projectiveCorrespondencesCS(int3 dTid : SV_DispatchThreadID)
{
	output[dTid.xy] = float4(MINF, MINF, MINF, MINF);
	
	float4 pInput = input[dTid.xy];
	float4 nInput = inputNormals[dTid.xy];
	float4 cInput = inputColors[dTid.xy];

	if(isValid(pInput) && isValid(nInput)) // && isValidCol(cInput)
	{
		pInput.w = 1.0f; // assert it is a point
		float4 pTransInput = mul(pInput, transform);

		nInput.w = 0.0f;  // assert it is a vector
		float4 nTransInput = mul(nInput, transform); // transformation is a rotation M^(-1)^T = M, translation is ignored because it is a vector

		//if(pTransInput.z > FLT_EPSILON) // really necessary
		{
			int2 screenPos = cameraToKinectScreenInt(pTransInput.xyz)/g_levelFactor;

			if(screenPos.x >= 0 && screenPos.y >= 0 && screenPos.x < g_ImageWidth && screenPos.y < g_ImageHeight)
			{
				float4 pTarget, nTarget;
				getBestCorrespondence1x1(screenPos, pTransInput, nTransInput, cInput, pTarget, nTarget);
				//getBestCorrespondence5x5(screenPos, pTransInput, nTransInput, cInput, pTarget, nTarget);
				//getBestCorrespondence5x5Color(screenPos, pTransInput, nTransInput, cInput, pTarget, nTarget);

				if(isValid(pTarget) && isValid(nTarget))
				{
					float d = length(pTransInput.xyz-pTarget.xyz);
					float dNormal = dot(nTransInput.xyz, nTarget.xyz);
						
					if(d <= distThres && dNormal >= normalThres)
					{
						output[dTid.xy] = pTarget;
						
						nTarget.w = max(0.0, 0.5f*((1.0f-d/distThres)+(1.0f-cameraToKinectProjZ(pTransInput.z)))); // for weighted ICP;
						outputNormals[dTid.xy] = nTarget;
					}
				}
			}
		}
	}
}
