#ifndef _KINECTCAMERA_UTIL_H_
#define _KINECTCAMERA_UTIL_H_

//#ifndef NUI_CAMERA_SKELETON_TO_DEPTH_IMAGE_MULTIPLIER_320x240
//#define NUI_CAMERA_SKELETON_TO_DEPTH_IMAGE_MULTIPLIER_320x240 (285.63f)
//#endif

//#ifndef NUI_CAMERA_DEPTH_IMAGE_TO_SKELETON_MULTIPLIER_320x240
//#define NUI_CAMERA_DEPTH_IMAGE_TO_SKELETON_MULTIPLIER_320x240 (3.501e-3f)
//#endif

//#define DEPTH_WORLD_MIN 0.4f
//#define DEPTH_WORLD_MAX 12.0f
//set by buffer
#define DEPTH_WORLD_MIN g_SensorDepthWorldMin
#define DEPTH_WORLD_MAX g_SensorDepthWorldMax


///////////////////////////////////////////////////////////////
// Camera to Screen
///////////////////////////////////////////////////////////////

float2 cameraToKinectScreenFloat(float3 pos)
{
	//float4 p = float4(pos.x, pos.y, 0.0f, pos.z);
	//float4x4 projMat = g_intrinsics;
	//if(g_stereoEnabled) projMat = g_intrinsicsStereo;

	//float4 proj = mul(p, projMat);
	//return float2(proj.x/proj.w, proj.y/proj.w);

	//undistort
	float fx = g_intrinsicsCoeff[0][0];
	float fy = g_intrinsicsCoeff[1][0];
	float mx = g_intrinsicsCoeff[2][0];
	float my = g_intrinsicsCoeff[3][0];
	float k1 = g_intrinsicsCoeff[0][1];
	float k2 = g_intrinsicsCoeff[1][1];
	float k3 = g_intrinsicsCoeff[2][1];
	float p1 = g_intrinsicsCoeff[3][1];
	float p2 = g_intrinsicsCoeff[0][2];

	//float2 p = float2(pos.x/pos.z, pos.y/pos.z);
	//float r2 = p.x*p.x + p.y*p.y;

	//float2 pos2;
	//pos2.x = p.x * (1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2) + 2.0f*p1*p.x*p.y + p2*(r2*2.0f*p.x*p.x);
	//pos2.y = p.y * (1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2) + 2.0f*p2*p.x*p.y + p1*(r2*2.0f*p.y*p.y);
	//
	//pos2.x = pos2.x*fx + mx;
	//pos2.y = my - pos2.y*fy;

	//return pos2;
	return float2(pos.x*fx/pos.z + mx, my - pos.y*fy/pos.z);
}

int2 cameraToKinectScreenInt(float3 pos)
{
	float2 pImage = cameraToKinectScreenFloat(pos);
	return int2(pImage + float2(0.5f, 0.5f));
}

uint2 cameraToKinectScreen(float3 pos)
{
    int2 p = cameraToKinectScreenInt(pos);

	return uint2(p.xy);
}

float cameraToKinectProjZ(float z)
{
	return (z - DEPTH_WORLD_MIN)/(DEPTH_WORLD_MAX - DEPTH_WORLD_MIN);
}

float kinectProjZToCamera(float z)
{
	return DEPTH_WORLD_MIN+z*(DEPTH_WORLD_MAX - DEPTH_WORLD_MIN);
}

float3 cameraToKinectProj(float3 pos)
{
	float2 proj = cameraToKinectScreenFloat(pos);

    float3 pImage = float3(proj.x, proj.y, pos.z);

	pImage.x = (2.0f*pImage.x - (g_ImageWidth-1.0f))/(g_ImageWidth-1.0f);
	pImage.y = ((g_ImageHeight-1.0f) - 2.0f*pImage.y)/(g_ImageHeight-1.0f);
	pImage.z = cameraToKinectProjZ(pImage.z);

	return pImage;
}

///////////////////////////////////////////////////////////////
// Screen to Camera (depth in meters)
///////////////////////////////////////////////////////////////

float3 kinectDepthToSkeleton(uint ux, uint uy, float depth)
{
	//float4x4 projInv = g_intrinsicsInv;
	//if(g_stereoEnabled) projInv = g_intrinsicsInvStereo;

	//float4 camera = mul(float4((float)ux*depth, (float)uy*depth, 0.0f, depth), projInv);
	//
	//return float3(camera.x, camera.y, camera.w);
	
	//undistort
	float fx = g_intrinsicsCoeff[0][0];
	float fy = g_intrinsicsCoeff[1][0];
	float mx = g_intrinsicsCoeff[2][0];
	float my = g_intrinsicsCoeff[3][0];
	float k1 = g_intrinsicsCoeff[0][1];
	float k2 = g_intrinsicsCoeff[1][1];
	float k3 = g_intrinsicsCoeff[2][1];
	float p1 = g_intrinsicsCoeff[3][1];
	float p2 = g_intrinsicsCoeff[0][2];

	const float x = ((float)ux-mx) / fx;
	const float y = (my-(float)uy) / fy;

	//float r2 = x*x + y*y;
	//float2 pos2;
	//pos2.x = x * (1 + k1 * r2 + k2 * r2*r2 + k3 * r2*r2*r2) + 2.0f * p1 * x * y + p2 * (r2 * 2.0f * x*x);
	//pos2.y = y * (1 + k1 * r2 + k2 * r2*r2 + k3 * r2*r2*r2) + 2.0f * p2 * x * y + p1 * (r2 * 2.0f * y*y); 

	//return float3(pos2.x*depth, pos2.y*depth, depth);
	return float3(depth*x, depth*y, depth);
}

///////////////////////////////////////////////////////////////
// RenderScreen to Camera -- ATTENTION ASSUMES [1,0]-Z range!!!!
///////////////////////////////////////////////////////////////

float kinectProjToCameraZ(float z)
{
	return z * (DEPTH_WORLD_MAX - DEPTH_WORLD_MIN) + DEPTH_WORLD_MIN;
}

// z has to be in [0, 1]
float3 kinectProjToCamera(uint ux, uint uy, float z)
{
	float fSkeletonZ = kinectProjToCameraZ(z);
	return kinectDepthToSkeleton(ux, uy, fSkeletonZ);
}

#endif
