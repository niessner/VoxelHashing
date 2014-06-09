
Buffer<int>	g_Hash					: register( t0 );
Buffer<float>	 g_SDFBlocksSDF		: register( t1 );
Texture2D<float> g_RayIntervalMin   : register( t2 );
Texture2D<float> g_RayIntervalMax   : register( t3 );
Buffer<int>		 g_SDFBlocksRGBW	: register( t4 );
 
Buffer<int>		g_FragmentPrefixSumBufferSRV	: register( t5 );
Buffer<float>	g_FragmentSortedDepthBufferSRV	: register( t6 );
 
#include "GlobalAppStateShaderBuffer.h.hlsl"
#include "SDFShaderBuffer.h.hlsl"
#include "KinectCameraUtil.h.hlsl"
#include "VoxelUtilHashSDF.h.hlsl"
#include "RayCastingUtilHashSDF.h.hlsl"
      
RWTexture2D<float> g_output : register(u0);
RWTexture2D<float4> g_outputColor : register(u1);
RWTexture2D<float4> g_outputNormals : register(u2);
  
cbuffer cbConstant : register(b1)
{ 
	float4x4	g_ViewMat;
	float4x4	g_ViewMatInverse;
	uint		g_RenderTargetWidth;
	uint		g_RenderTargetHeight;
	uint		g_splatMinimum;
	uint		g_dummyRayInteveral337;
};

#include "RayCastingHashSDFTraversalSimple.h.hlsl"
 
// rayCurrent and rayEnd need to be be the intersections between ray and the coarse grid box
/*bool traverseFineGridSimple(HashEntry entry, float3 worldCamPos, float3 worldDir, float3 camDir, in out float rayCurrent, float rayEnd, float rayIncrement, in out Sample lastSample, int3 dTid)
{
	const float depthToRayLength = 1.0f/camDir.z; // scale factor to convert from depth to ray length
	 
	[allow_uav_condition]
	while(rayCurrent < rayEnd)
	{		
		float dist;	float3 color;
		if(trilinearInterpolationSimple(worldCamPos+rayCurrent*worldDir, dist, color))
		{
			if(lastSample.weight > 0 && lastSample.sdf > 0.0f && dist < 0.0f) // current sample is always valid here
			{
				float alpha = 0.0f; //findIntersectionLinear(lastSample.alpha, rayCurrent, lastSample.sdf, dist);
				findIntersectionBisection(worldCamPos, worldDir, lastSample.sdf, lastSample.alpha, dist, rayCurrent, alpha);

				g_output[dTid.xy] = alpha/depthToRayLength; // Convert ray length to depth
				g_outputColor[dTid.xy] = float4(color/255.0f, 1.0f);
		
				return true;
			}

			lastSample.sdf = dist;
			lastSample.alpha = rayCurrent;
			lastSample.weight = 1;
			lastSample.color = color;
		}
		else
		{
			lastSample.weight = 0;
		}
		
		rayCurrent+=rayIncrement;
	}

	rayCurrent-=rayIncrement; // Go to last valid sample position in the sdf block
	return false;
}

void traverseCoarseGridSimple(float3 worldCamPos, float3 worldDir, float3 camDir, int3 dTid)
{
	// Last Sample
	Sample lastSample; lastSample.sdf = 0.0f; lastSample.alpha = 0.0f; lastSample.weight = 0; lastSample.color = int3(0, 0, 0);
	int3 lastBlockID = int3(INTMAX, INTMAX, INTMAX);

	const float rayIncrement = g_Truncation/4.0f;
	const float depthToRayLength = 1.0f/camDir.z; // scale factor to convert from depth to ray length

	int2 screenPos = cameraToKinectScreenInt(camDir);
	float rayCurrent = depthToRayLength*max(g_SensorDepthWorldMin, kinectProjZToCamera(g_RayIntervalMin[screenPos])); // Convert depth to raylength
	float rayEnd = depthToRayLength*min(g_SensorDepthWorldMax, kinectProjZToCamera(g_RayIntervalMax[screenPos])); // Convert depth to raylength
	//rayCurrent = depthToRayLength*g_SensorDepthWorldMin;
	//rayEnd = depthToRayLength*g_SensorDepthWorldMax;

	[allow_uav_condition]
	while(rayCurrent < rayEnd)
	{
		float3 currentPosWorld = worldCamPos+rayCurrent*worldDir;
		int3 blockID = worldToSDFBlock(currentPosWorld);
		if(blockID.x != lastBlockID.x || blockID.y != lastBlockID.y || blockID.z != lastBlockID.z)
		{
			HashEntry entry = getHashEntryForSDFBlockPos(g_Hash, blockID);

			if(entry.ptr != FREE_ENTRY)
			{
				float3 minCorner = SDFBlockToWorld(entry.pos)-g_VirtualVoxelSize/2.0;
				float3 maxCorner = minCorner+SDF_BLOCK_SIZE*g_VirtualVoxelSize;

				float alphaMax = computeAlphaOut(worldCamPos, worldDir, minCorner, maxCorner);
				if(traverseFineGridSimple(entry, worldCamPos, worldDir, camDir, rayCurrent, alphaMax, rayIncrement, lastSample, dTid)) return; // Changes rayCurrent and lastSample
			}

			lastBlockID = blockID;
		}

		rayCurrent+=rayIncrement;
	}
}*/

[numthreads(groupthreads, groupthreads, 1)]
void renderCS(int3 dTid : SV_DispatchThreadID)
{
	g_output[dTid.xy] = MINF;
	//g_outputColor[dTid.xy] = float4(0.0f, 0.0f, 0.0f, 1.0f);
	
	if(dTid.x >= g_RenderTargetWidth || dTid.y >= g_RenderTargetHeight) return;

	float3 camDir = normalize(kinectProjToCamera(dTid.x, dTid.y, 1.0f));

	float3 worldCamPos = mul(float4(0.0f, 0.0f, 0.0f, 1.0f), g_ViewMatInverse).xyz;
	float3 worldDir = normalize(mul(float4(camDir, 0.0f), g_ViewMatInverse).xyz);

	if(g_enableMultiLayerSplatting)
	{
		//traverseCoarseGridSimpleSampleAllMultiLayer(worldCamPos, worldDir, camDir, dTid);
	}
	else
	{
		traverseCoarseGridSimpleSampleAll(worldCamPos, worldDir, camDir, dTid);
	} 
}
   