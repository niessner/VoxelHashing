#pragma once

/***************************************************************************/
/* Global App state: reads and stores all app parameters (except tracking) */
/***************************************************************************/

// Only works when Kinect 2.0 SDK is installed
//#define KINECT_ONE

// Only works when OpenNI 2 SDK is installed
//#define OPEN_NI

#include "DX11Utils.h"
#include "DXUT.h"

#define SDF_BLOCK_SIZE 8
#define RENDERMODE_INTEGRATE 0 
#define RENDERMODE_VIEW 1

#include "DepthSensor.h"

#include "stdafx.h"

#include <string>

#define X_GLOBAL_APP_STATE_FIELDS \
	X(unsigned int, s_sensorIdx) \
	X(bool, s_timingsDetailledEnabled) \
	X(bool, s_timingsStepsEnabled) \
	X(bool, s_timingsTotalEnabled) \
	X(unsigned int, s_windowWidth) \
	X(unsigned int, s_windowHeight) \
	X(unsigned int, s_outputWindowWidth) \
	X(unsigned int, s_outputWindowHeight) \
	X(unsigned int, s_MaxCollisionLinkedListSize) \
	X(unsigned int, s_RenderMode) \
	X(bool, s_bRenderModeChanged) \
	X(unsigned int, s_DisplayTexture) \
	X(bool, s_bFilterKinectInputData) \
	X(bool, s_bEnableGlobalLocalStreaming) \
	X(bool, s_bEnableGarbageCollection) \
	X(unsigned int, s_hashNumBucketsLocal) \
	X(unsigned int, s_hashNumBucketsGlobal) \
	X(unsigned int, s_hashNumSDFBlocks) \
	X(unsigned int, s_hashStreamOutParts) \
	X(unsigned int, s_initialChunkListSize) \
	X(unsigned int, s_hashBucketSizeLocal) \
	X(unsigned int, s_hashBucketSizeGlobal) \
	X(float, s_virtualVoxelSize) \
	X(float, s_thresMarchingCubes) \
	X(float, s_maxIntegrationDistance) \
	X(float, s_SensorDepthWorldMin) \
	X(float, s_SensorDepthWorldMax) \
	X(bool, s_applicationDisabled) \
	X(float, s_StreamingRadius) \
	X(vec4f, s_StreamingPos) \
	X(vec3f, s_voxelExtends) \
	X(vec3i, s_gridDimensions) \
	X(vec3i, s_minGridPos) \
	X(int, s_nBitsInT) \
	X(unsigned int,	s_WeightSample) \
	X(unsigned int,	s_WeightMax) \
	X(float, s_Truncation) \
	X(float, s_TruncScale) \
	X(float, s_rayIncrement) \
	X(float, s_thresSampleDist) \
	X(float, s_thresDist) \
	X(float, s_materialShininess) \
	X(vec4f, s_materialDiffuse) \
	X(vec4f, s_materialSpecular) \
	X(vec4f, s_lightAmbient) \
	X(vec4f, s_lightDiffuse) \
	X(vec4f, s_lightSpecular) \
	X(vec4f, s_lightDir) \
	X(unsigned int, s_MaxLoopIterCount) \
	X(float, s_thresMarchingCubes2) \
	X(bool, s_useGradients) \
	X(std::string, s_BinaryDumpReaderSourceFile) \
	X(bool, s_usePreComputedCameraTrajectory) \
	X(std::string, s_PreComputedCameraTrajectoryPath) \
	X(std::string, s_DataDumpPath) \
	X(bool, s_DataDumpDepthData) \
	X(bool, s_DataDumpColorData) \
	X(bool, s_DataDumpRigidTransform) \
	X(std::string, s_ImageReaderSensorSourcePath) \
	X(unsigned int, s_ImageReaderSensorNumFrames) \
	X(unsigned int, s_HANDLE_COLLISIONS) \
	X(bool, s_enableMultiLayerSplatting) \
	X(bool, s_stereoEnabled) \
	X(unsigned int, s_windowWidthStereo) \
	X(unsigned int, s_windowHeightStereo) \
	X(bool, s_currentlyInStereoMode) \
	X(bool, s_bRegistrationEnabled) \
	X(bool, s_bDisableIntegration) \
	X(bool, s_bApplicationEnabled) \
	X(bool, s_RecordData) \
	X(std::string, s_RecordDataFile) \
	X(unsigned int, s_DumpVoxelGridFrames) \
	X(std::string, s_DumpVoxelGridFile) \
	X(bool, s_DumpAllRendering)

#ifndef VAR_NAME
#define VAR_NAME(x) #x
#endif

class GlobalAppState
{	
	public:

#define X(type, name) type name;
		X_GLOBAL_APP_STATE_FIELDS
#undef X

		//! sets the parameter file and reads
		void readMembers(const ParameterFile& parameterFile) {
			s_ParameterFile = parameterFile;
			readMembers();
		}

		//! reads all the members from the given parameter file (could be called for reloading)
		void readMembers() {
#define X(type, name) \
	if (!s_ParameterFile.readParameter(std::string(#name), name)) {MLIB_WARNING(std::string(#name).append(" ").append("uninitialized"));	name = type();}
			X_GLOBAL_APP_STATE_FIELDS
#undef X
			m_bIsInitialized = true;
		}

		void print() {
#define X(type, name) \
	std::cout << #name " = " << name << std::endl;
			X_GLOBAL_APP_STATE_FIELDS
#undef X
		}

		static GlobalAppState& getInstance() {
			static GlobalAppState s;
			return s;
		}	

		//! constructor
		GlobalAppState();

		//! destructor
		~GlobalAppState();


		//! matrices are set by the corresponding sensor
		float s_intrinsics[9];	
		//D3DXMATRIX s_intrinsics;
		//D3DXMATRIX s_intrinsicsInv;
		D3DXMATRIX s_intrinsicsStereo;
		D3DXMATRIX s_intrinsicsInvStereo;

		D3DXMATRIX s_intrinsicsStereoOther;
		D3DXMATRIX s_intrinsicsInvStereoOther;

		D3DXMATRIX s_worldToCamStereo;
		D3DXMATRIX s_camToWorldStereo;

		D3DXMATRIX s_worldToCamStereoOther;
		D3DXMATRIX s_camToWorldStereoOther;

		Timer		s_Timer;



		HRESULT OnD3D11CreateDevice(ID3D11Device* pd3dDevice);

		void OnD3D11DestroyDevice();

		void WaitForGPU();

		void StereoCameraFrustum(D3DXMATRIX& proj, D3DXMATRIX& projInv, D3DXMATRIX& worldToCam, D3DXMATRIX& camToWorld, float Convergence, float EyeSeparation, float AspectRatio, float FOV, float n, float f, float width, float height, bool isLeftEye, mat4f& lastRigidTransform);
		
		void MapConstantBuffer(ID3D11DeviceContext* context);

		ID3D11Buffer* MapAndGetConstantBuffer(ID3D11DeviceContext* context);		

		//! Returns the  current depth sensor (it's stored as a static variable)
		static DepthSensor* getDepthSensor();
private:
	bool m_bIsInitialized;
	ParameterFile s_ParameterFile;
	ID3D11Buffer* m_constantBuffer;
	ID3D11Query* s_pQuery;

	struct CB_GLOBAL_APP_STATE
	{
		unsigned int g_WeightSample;
		unsigned int g_WeightMax;
		float g_Truncation;	
		float g_maxIntegrationDistance;

		float3 m_voxelExtends;
		float	g_TruncScale;

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
		float4 lightDir;

		unsigned int g_MaxLoopIterCount;
		float g_thresMarchingCubes2;
		unsigned int g_useGradients;
		unsigned int g_enableMultiLayerSplatting;

		//D3DXMATRIX g_intrinsics;
		//D3DXMATRIX g_intrinsicsInv;
		D3DXMATRIX g_intrinsics;

		D3DXMATRIX g_intrinsicsStereo;
		D3DXMATRIX g_intrinsicsInvStereo;

		D3DXMATRIX g_intrinsicsStereoOther;
		D3DXMATRIX g_intrinsicsInvStereoOther;

		D3DXMATRIX g_worldToCamStereo;
		D3DXMATRIX g_camToWorldStereo;

		D3DXMATRIX g_worldToCamStereoOther;
		D3DXMATRIX g_camToWorldStereoOther;

		unsigned int g_stereoEnabled;
		float g_SensorDepthWorldMin;
		float g_SensorDepthWorldMax;
		unsigned int dummy02glob;
	};
};
