#ifndef _GLOBAL_APP_STATE_BUFFER_H_
#define _GLOBAL_APP_STATE_BUFFER_H_

cbuffer cbGlobalAppState : register( b8 )
{
	uint	g_WeightSample;			//weight per sample (per integration step)
	uint	g_WeightMax;			//maximum weight per voxel
	float	g_Truncation;	
	float	g_maxIntegrationDistance;

	float3 m_voxelExtends;
	float	g_TruncScale;			//how to adapt the truncation: per distance in meter
	
	int3 m_gridDimensions;
	int nBitsInT;

	int3 m_minGridPos;
	float rayIncrement;

	float g_thresSampleDist;
	float g_thresDist;
	float g_thresMarchingCubes;
	float materialShininess;
	
	float4 materialDiffuse;
	float4 materialSpecular;
	float4 lightAmbient;
	float4 lightDiffuse;
	float4 lightSpecular;
	float4 g_LightDir;

	uint	g_MaxLoopIterCount;
	float	g_thresMarchingCubes2;
	uint	g_useGradients;
	uint	g_enableMultiLayerSplatting;

	//float4x4 g_intrinsics;
	//float4x4 g_intrinsicsInv;
	float4x4 g_intrinsicsCoeff;		//coeffs

	float4x4 g_intrinsicsStereo;
	float4x4 g_intrinsicsInvStereo;

	float4x4 g_intrinsicsStereoOther;
	float4x4 g_intrinsicsInvStereoOther;

	float4x4 g_worldToCamStereo;
	float4x4 g_camToWorldStereo;

	float4x4 g_worldToCamStereoOther;
	float4x4 g_camToWorldStereoOther;

	uint g_stereoEnabled;
	float g_SensorDepthWorldMin;
	float g_SensorDepthWorldMax;
	uint g_dummy02Glob;
};

#endif
