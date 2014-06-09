#include "stdafx.h"

#include "GlobalAppState.h"

#include "ImageReaderSensor.h"
#include "ChristophSensor.h"
#include "BinaryDumpReader.h"
#include "KinectSensor.h"
#include "PrimeSenseSensor.h"
#ifdef KINECT_ONE
#include "KinectOneSensor.h"
#endif

GlobalAppState::GlobalAppState()
{
	m_constantBuffer = NULL;
	s_pQuery = NULL;
	m_bIsInitialized = false;
}

GlobalAppState::~GlobalAppState()
{

}


HRESULT GlobalAppState::OnD3D11CreateDevice( ID3D11Device* pd3dDevice )
{
	HRESULT hr = S_OK;

	// Create constant buffer		
	D3D11_BUFFER_DESC bDesc;
	ZeroMemory(&bDesc, sizeof(D3D11_BUFFER_DESC));
	bDesc.BindFlags	= D3D11_BIND_CONSTANT_BUFFER;
	bDesc.Usage	= D3D11_USAGE_DYNAMIC;
	bDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	bDesc.MiscFlags	= 0;
	bDesc.ByteWidth	= sizeof(CB_GLOBAL_APP_STATE);
	V_RETURN(pd3dDevice->CreateBuffer(&bDesc, NULL, &m_constantBuffer));

	/////////////////////////////////////////////////////
	// Query
	/////////////////////////////////////////////////////

	D3D11_QUERY_DESC queryDesc;
	queryDesc.Query = D3D11_QUERY_EVENT;
	queryDesc.MiscFlags = 0;

	hr = pd3dDevice->CreateQuery(&queryDesc, &s_pQuery);
	if(FAILED(hr)) return hr;

	memcpy(s_intrinsics, getDepthSensor()->getIntrinsics().coeff, sizeof(float)*9);
	//s_intrinsics = *(D3DXMATRIX*)&getDepthSensor()->getIntrinsics();
	//s_intrinsicsInv = *(D3DXMATRIX*)&getDepthSensor()->getIntrinsicsInv();

	return  hr;
}

void GlobalAppState::OnD3D11DestroyDevice()
{
	SAFE_RELEASE(s_pQuery);
	SAFE_RELEASE(m_constantBuffer);
}

void GlobalAppState::WaitForGPU()
{
	DXUTGetD3D11DeviceContext()->Flush();
	DXUTGetD3D11DeviceContext()->End(s_pQuery);
	DXUTGetD3D11DeviceContext()->Flush();

	while (S_OK != DXUTGetD3D11DeviceContext()->GetData(s_pQuery, NULL, 0, 0));
}

void GlobalAppState::StereoCameraFrustum( D3DXMATRIX& proj, D3DXMATRIX& projInv, D3DXMATRIX& worldToCam, D3DXMATRIX& camToWorld, float Convergence, float EyeSeparation, float AspectRatio, float FOV, float n, float f, float width, float height, bool isLeftEye, mat4f& lastRigidTransform )
{
	float PI = 3.14159265359f;
	FOV = FOV * PI / 180.0f;

	float t, b, l, r;

	t  = n * tan(FOV/2);
	b  = -t;

	float a = AspectRatio * tan(FOV/2) * Convergence;

	float B = a - EyeSeparation/2;
	float C = a + EyeSeparation/2;

	if(isLeftEye)
	{
		l   = -B * n/Convergence;
		r   =  C * n/Convergence;
	}
	else
	{
		l   = -C * n/Convergence;
		r   =  B * n/Convergence;
	}

	float fovX = (width/2.0f)*(2.0f*n)/(r-l);
	float fovY = -(height/2.0f)*(2.0f*n)/(t-b);
	float centerX = (width/2.0f)*(r+l)/(r-l)+(width/2.0f);
	float centerY = (height/2.0f)*(t+b)/(t-b)+(height/2.0f);
	//float centerY = (height/2.0f);

	proj = D3DXMATRIX(fovX, 0.0f, 0.0f, centerX,
		0.0f, fovY, 0.0f, centerY,
		0.0f, 0.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f);


	projInv = D3DXMATRIX(1.0f/fovX, 0.0f, 0.0f, -centerX*1.0f/fovX,
		0.0f, 1.0f/fovY, 0.0f, -centerY*1.0f/fovY,
		0.0f, 0.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f);

	mat4f worldToLastKinectSpace = lastRigidTransform.getInverse();

	memcpy(&worldToCam, &worldToLastKinectSpace, sizeof(mat4f));
	memcpy(&camToWorld, &lastRigidTransform, sizeof(mat4f));
}

void GlobalAppState::MapConstantBuffer( ID3D11DeviceContext* context )
{
	HRESULT hr = S_OK;
	D3D11_MAPPED_SUBRESOURCE mappedResource;
	V(context->Map(m_constantBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource));
	CB_GLOBAL_APP_STATE* cbuffer = (CB_GLOBAL_APP_STATE*)mappedResource.pData;

	cbuffer->g_WeightSample = s_WeightSample;
	cbuffer->g_WeightMax = s_WeightMax;
	cbuffer->g_Truncation = s_Truncation;
	cbuffer->g_maxIntegrationDistance = s_maxIntegrationDistance;

	memcpy(&cbuffer->m_voxelExtends, &s_voxelExtends.array[0], sizeof(vec3f));

	cbuffer->g_TruncScale = s_TruncScale;
	memcpy(&cbuffer->m_gridDimensions, &s_gridDimensions.array[0], sizeof(vec3i));

	memcpy(&cbuffer->m_minGridPos, &s_minGridPos.array[0], sizeof(vec3i));
	cbuffer->nBitsInT = s_nBitsInT;

	cbuffer->rayIncrement = s_rayIncrement;
	cbuffer->g_thresSampleDist = s_thresSampleDist;
	cbuffer->g_thresDist = s_thresDist;
	cbuffer->g_thresMarchingCubes = s_thresMarchingCubes;

	cbuffer->materialShininess = s_materialShininess;

	memcpy(&cbuffer->materialDiffuse,  &s_materialDiffuse.array[0], sizeof(vec4f));
	memcpy(&cbuffer->materialSpecular, &s_materialSpecular.array[0], sizeof(vec4f));
	memcpy(&cbuffer->lightAmbient, &s_lightAmbient.array[0], sizeof(vec4f));
	memcpy(&cbuffer->lightDiffuse, &s_lightDiffuse.array[0], sizeof(vec4f));
	memcpy(&cbuffer->lightSpecular, &s_lightSpecular.array[0], sizeof(vec4f));


	memcpy(&cbuffer->lightDir, &s_lightDir.array[0], sizeof(vec4f));

	cbuffer->g_MaxLoopIterCount = s_MaxLoopIterCount;

	cbuffer->g_thresMarchingCubes2 = s_thresMarchingCubes2;
	cbuffer->g_useGradients = s_useGradients ? 1 : 0;

	cbuffer->g_enableMultiLayerSplatting = s_enableMultiLayerSplatting ? 1 : 0;

	memcpy(&cbuffer->g_intrinsics, &s_intrinsics, sizeof(float)*9);
	//memcpy(&cbuffer->g_intrinsicsInv, &s_intrinsicsInv, sizeof(D3DXMATRIX));

	memcpy(&cbuffer->g_intrinsicsStereo, &s_intrinsicsStereo, sizeof(D3DXMATRIX));
	memcpy(&cbuffer->g_intrinsicsInvStereo, &s_intrinsicsInvStereo, sizeof(D3DXMATRIX));

	memcpy(&cbuffer->g_intrinsicsStereoOther, &s_intrinsicsStereoOther, sizeof(D3DXMATRIX));
	memcpy(&cbuffer->g_intrinsicsInvStereoOther, &s_intrinsicsInvStereoOther, sizeof(D3DXMATRIX));

	memcpy(&cbuffer->g_worldToCamStereo, &s_worldToCamStereo, sizeof(D3DXMATRIX));
	memcpy(&cbuffer->g_camToWorldStereo, &s_camToWorldStereo, sizeof(D3DXMATRIX));

	memcpy(&cbuffer->g_worldToCamStereoOther, &s_worldToCamStereoOther, sizeof(D3DXMATRIX));
	memcpy(&cbuffer->g_camToWorldStereoOther, &s_camToWorldStereoOther, sizeof(D3DXMATRIX));

	cbuffer->g_stereoEnabled = s_currentlyInStereoMode ? 1 : 0;
	cbuffer->g_SensorDepthWorldMin = s_SensorDepthWorldMin;
	cbuffer->g_SensorDepthWorldMax = s_SensorDepthWorldMax;

	context->Unmap(m_constantBuffer, 0);
}

ID3D11Buffer* GlobalAppState::MapAndGetConstantBuffer( ID3D11DeviceContext* context )
{
	MapConstantBuffer(context);
	return m_constantBuffer;
}




DepthSensor* GlobalAppState::getDepthSensor()
{
	if(GlobalAppState::getInstance().s_sensorIdx == 0)
	{
		static KinectSensor s_kinect;
		return &s_kinect;
	}
	else if(GlobalAppState::getInstance().s_sensorIdx == 1)
	{
#ifdef OPEN_NI
		static PrimeSenseSensor s_primeSense;
		return &s_primeSense;
#else
		throw MLIB_EXCEPTION("Requires OpenNI 2 SDK and enable OPEN_NI macro");
#endif
	}
	

	else if (GlobalAppState::getInstance().s_sensorIdx == 4) 
	{
		static ImageReaderSensor s_imageReader;
		s_imageReader.setBaseFilePath(GlobalAppState::getInstance().s_ImageReaderSensorSourcePath);
		s_imageReader.setNumFrames(GlobalAppState::getInstance().s_ImageReaderSensorNumFrames);
		return &s_imageReader;
	}
	else if (GlobalAppState::getInstance().s_sensorIdx == 5) {
#ifdef KINECT_ONE
		static KinectOneSensor s_kinectOne;
		return &s_kinectOne;
#else
		throw MLIB_EXCEPTION("Requires Kinect 2.0 SDK and enable KINECT_ONE macro");
#endif
	}
	else if (GlobalAppState::getInstance().s_sensorIdx == 6) {
		static ChristophSensor s_christophSesnor;
		return &s_christophSesnor;
	} 
	else if (GlobalAppState::getInstance().s_sensorIdx == 7) {
		static BinaryDumpReader s_binaryDumpReader;
		return &s_binaryDumpReader;
	}
	else {
		throw MLIB_EXCEPTION("Invalid sensor idx");
		return NULL;
	}
}

