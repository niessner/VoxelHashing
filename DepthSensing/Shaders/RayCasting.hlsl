cbuffer consts : register(cb0)
{
	uint g_ImageWidth;
	uint g_ImageHeight;
	uint align0;
	uint align1;

	float4x4 g_lastRigidTransform;
	float4x4 g_viewMat;

	float3 g_gridPosition;
	float align2;

	int3 g_gridDimensions;
	int align3;

	float3 g_voxelExtends;
	float align4;
};

#include "GlobalAppStateShaderBuffer.h.hlsl"
#include "KinectCameraUtil.h.hlsl"

Buffer<int> g_voxelBuffer : register(t0);
RWTexture2D<float> g_output : register(u0);
RWTexture2D<float4> g_outputNormals : register(u1);
RWTexture2D<float4> g_outputColors : register(u2);

#define PINF asfloat(0x7f800000)
#define MINF asfloat(0xff800000)

#include "RayCastingUtil.h.hlsl"

struct Sample
{
	float sdf;
	float alpha;
	uint weight;
};

float3 getColor(float3 pos)
{
	int3 posVoxel = posWorldToVoxel(pos);
	Voxel v000 = getVoxel(g_voxelBuffer, linearizeIndex(posVoxel));
	return (float3)v000.color/255.0f;
}

float distanceForPointTriLinearUnSafe(float3 pos)
{
	int3 posVoxel = posWorldToVoxel(pos);

	Voxel v000 = getVoxel(g_voxelBuffer, linearizeIndex(posVoxel+int3(0, 0, 0)));
	Voxel v100 = getVoxel(g_voxelBuffer, linearizeIndex(posVoxel+int3(1, 0, 0)));
	Voxel v010 = getVoxel(g_voxelBuffer, linearizeIndex(posVoxel+int3(0, 1, 0)));
	Voxel v001 = getVoxel(g_voxelBuffer, linearizeIndex(posVoxel+int3(0, 0, 1)));
	Voxel v110 = getVoxel(g_voxelBuffer, linearizeIndex(posVoxel+int3(1, 1, 0)));
	Voxel v011 = getVoxel(g_voxelBuffer, linearizeIndex(posVoxel+int3(0, 1, 1)));
	Voxel v101 = getVoxel(g_voxelBuffer, linearizeIndex(posVoxel+int3(1, 0, 1)));
	Voxel v111 = getVoxel(g_voxelBuffer, linearizeIndex(posVoxel+int3(1, 1, 1)));

	float3 posVoxelFloat = posWorldToVoxelFloat(pos);
	float3 weight = frac(posVoxelFloat);

	float tmp0 = lerp(lerp(v000.sdf, v100.sdf, weight.x), lerp(v010.sdf, v110.sdf, weight.x), weight.y);
	float tmp1 = lerp(lerp(v001.sdf, v101.sdf, weight.x), lerp(v011.sdf, v111.sdf, weight.x), weight.y);

	return lerp(tmp0, tmp1, weight.z);
}

bool distanceForPointTriLinear(float3 pos, float thresh, out float dist)
{
	int3 posVoxel = posWorldToVoxel(pos);

	if(!isValidId(posVoxel)) return false;

	Voxel v000 = getVoxel(g_voxelBuffer, linearizeIndex(posVoxel+int3(0, 0, 0)));
	Voxel v100 = getVoxel(g_voxelBuffer, linearizeIndex(posVoxel+int3(1, 0, 0)));
	Voxel v010 = getVoxel(g_voxelBuffer, linearizeIndex(posVoxel+int3(0, 1, 0)));
	Voxel v001 = getVoxel(g_voxelBuffer, linearizeIndex(posVoxel+int3(0, 0, 1)));
	Voxel v110 = getVoxel(g_voxelBuffer, linearizeIndex(posVoxel+int3(1, 1, 0)));
	Voxel v011 = getVoxel(g_voxelBuffer, linearizeIndex(posVoxel+int3(0, 1, 1)));
	Voxel v101 = getVoxel(g_voxelBuffer, linearizeIndex(posVoxel+int3(1, 0, 1)));
	Voxel v111 = getVoxel(g_voxelBuffer, linearizeIndex(posVoxel+int3(1, 1, 1)));

	if(v000.weight == 0 ||	v100.weight == 0 || v010.weight == 0 ||	v001.weight == 0 ||	v110.weight == 0 ||	v011.weight == 0 ||	v101.weight == 0 || v111.weight == 0) return false;

	//if(abs(v000.sdf - v100.sdf) > thresh ||	abs(v010.sdf - v110.sdf) > thresh || abs(v001.sdf - v101.sdf) > thresh || abs(v011.sdf - v111.sdf) > thresh) return false;

	float3 posVoxelFloat = posWorldToVoxelFloat(pos);
	float3 weight = frac(posVoxelFloat);

	float v00 = lerp(v000.sdf, v100.sdf, weight.x);
	float v01 = lerp(v010.sdf, v110.sdf, weight.x);
	float v10 = lerp(v001.sdf, v101.sdf, weight.x);
	float v11 = lerp(v011.sdf, v111.sdf, weight.x);

	//if (abs(v00 - v01) > thresh || abs(v10 - v11) > thresh) return false;

	float v0 = lerp(v00, v01, weight.y);
	float v1 = lerp(v10, v11, weight.y);
	
	//if (abs(v0 - v1) > thresh) return false;

	dist = lerp(v0, v1, weight.z);

	return true;
}

float3 gradientForPoint(float3 pos)
{
	float3 offset = g_voxelExtends;
	int3 posVoxel = posWorldToVoxel(pos);

	float dist000 = distanceForPointTriLinearUnSafe(pos+float3(0.0f, 0.0f, 0.0f));
	float dist100 = distanceForPointTriLinearUnSafe(pos+float3(offset.x, 0.0f, 0.0f));
	float dist010 = distanceForPointTriLinearUnSafe(pos+float3(0.0f, offset.y, 0.0f));
	float dist001 = distanceForPointTriLinearUnSafe(pos+float3(0.0f, 0.0f, offset.z));

	float3 grad = float3((dist000-dist100)/offset.x, (dist000-dist010)/offset.y, (dist000-dist001)/offset.z);

	float l = length(grad);
	if(l == 0.0f)
	{
		return float3(0.0f, 0.0f, 0.0f);
	}

	return -grad/l;
}

void traverseCoarseGridSimple(float3 worldCamPos, float3 worldDir, float3 camDir, int3 dTid)
{
	const float thresSampleDist = 0.03f;
	const float thresDist = 0.01f;

	// Last Sample
	Sample lastSample; lastSample.sdf = 0.0f; lastSample.alpha = 0.0f; lastSample.weight = 0;
	const float rayIncrement = 0.01f;
	const float depthToRayLength = 1.0f/camDir.z; // scale factor to convert from depth to ray length

	float rayCurrent = depthToRayLength*g_SensorDepthWorldMin;
	float rayEnd = depthToRayLength*g_SensorDepthWorldMax;

	[allow_uav_condition]
	while(rayCurrent < rayEnd)
	{
		float3 currentPosWorld = worldCamPos+rayCurrent*worldDir;

		float dist;
		[branch]
		if(distanceForPointTriLinear(currentPosWorld, 0.01f, dist))
		{
			if(lastSample.weight > 0 && lastSample.sdf > 0.0f && dist < 0.0f) // current sample is always valid here
			{
				const float alpha = findIntersectionLinear(lastSample.alpha, rayCurrent, lastSample.sdf, dist);
				float3 currentIso = worldCamPos+alpha*worldDir;

				if(abs(lastSample.sdf - dist) < thresSampleDist && distanceForPointTriLinear(currentIso, thresDist, dist))
				{
					if(abs(dist) < thresDist)
					{
						g_output[dTid.xy] = alpha/depthToRayLength;
						g_outputNormals[dTid.xy] = mul(float4(gradientForPoint(currentIso), 1.0f), g_viewMat); // world space normal atm !!!
						g_outputColors[dTid.xy] = float4(getColor(currentIso), 1.0f);
						
						return;
					}
				}
			}

			lastSample.weight = 1;
		}
		else
		{
			lastSample.weight = 0;
		}

		// Save last sample
		lastSample.sdf = dist;
		lastSample.alpha = rayCurrent;

		rayCurrent+=rayIncrement;
	}
}


[numthreads(groupthreads, groupthreads, 1)]
void renderCS(int3 dTid : SV_DispatchThreadID)
{
	g_output[dTid.xy] = MINF;
	
	float3 camDir = normalize(kinectProjToCamera(dTid.x, dTid.y, 1.0f));

	float3 worldCamPos = mul(float4(0.0f, 0.0f, 0.0f, 1.0f), g_lastRigidTransform).xyz;
	float3 worldDir = normalize(mul(float4(camDir, 0.0f), g_lastRigidTransform).xyz);
	
	traverseCoarseGridSimple(worldCamPos, worldDir, camDir, dTid);
}
