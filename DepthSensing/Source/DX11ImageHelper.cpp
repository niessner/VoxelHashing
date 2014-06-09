#include "stdafx.h"

#include "DX11ImageHelper.h"

#include "GlobalAppState.h"

#include <vector>
#include <iostream>

/////////////////////////////////////////////////////
// Bilateral Completion
/////////////////////////////////////////////////////

unsigned int DX11ImageHelper::m_blockSizeBilateralCompletion = 16;

ID3D11Buffer* DX11ImageHelper::m_constantBufferBilateralCompletion = NULL;
ID3D11ComputeShader* DX11ImageHelper::m_pComputeShaderBilateralCompletion = NULL;
ID3D11ComputeShader* DX11ImageHelper::m_pComputeShaderBilateralCompletionWithColor = NULL;

/////////////////////////////////////////////////////
// Bilateral Filter
/////////////////////////////////////////////////////

unsigned int DX11ImageHelper::m_blockSizeBilateralFilter = 16;

ID3D11Buffer* DX11ImageHelper::m_constantBufferBilateralFilter = NULL;
ID3D11ComputeShader* DX11ImageHelper::m_pComputeShaderBilateralFilter = NULL;

/////////////////////////////////////////////////////
// Bilateral Filter SSAO
/////////////////////////////////////////////////////

unsigned int DX11ImageHelper::m_blockSizeBilateralFilterForSSAO = 16;

ID3D11Buffer* DX11ImageHelper::m_constantBufferBilateralFilterForSSAO = NULL;
ID3D11ComputeShader* DX11ImageHelper::m_pComputeShaderBilateralFilterForSSAO = NULL;

/////////////////////////////////////////////////////
// Bilateral Approximation
/////////////////////////////////////////////////////

unsigned int DX11ImageHelper::m_blockSizeBilateralFilterApprox = 16;

ID3D11Buffer* DX11ImageHelper::m_constantBufferBilateralFilterApprox = NULL;
ID3D11ComputeShader* DX11ImageHelper::m_pComputeShaderBilateralFilterApprox = NULL;

/////////////////////////////////////////////////////
// Bilateral Filter 4F
/////////////////////////////////////////////////////

unsigned int DX11ImageHelper::m_blockSizeBilateralFilter4F = 16;

ID3D11Buffer* DX11ImageHelper::m_constantBufferBilateralFilter4F = NULL;
ID3D11ComputeShader* DX11ImageHelper::m_pComputeShaderBilateralFilter4F = NULL;

/////////////////////////////////////////////////////
// Normal Computation
/////////////////////////////////////////////////////

unsigned int DX11ImageHelper::m_blockSizeNormalComputation = 16;

ID3D11Buffer* DX11ImageHelper::m_constantBufferNormalComputation = NULL;
ID3D11ComputeShader* DX11ImageHelper::m_pComputeShaderNormalComputation = NULL;

/////////////////////////////////////////////////////
// Camera Space To Depth Map
/////////////////////////////////////////////////////

unsigned int DX11ImageHelper::m_blockSizeCameraSpace = 16;

ID3D11Buffer* DX11ImageHelper::m_constantBufferCameraSpace = NULL;
ID3D11ComputeShader* DX11ImageHelper::m_pComputeShaderCameraSpace = NULL;

/////////////////////////////////////////////////////
// Compute Depth Map
/////////////////////////////////////////////////////

unsigned int DX11ImageHelper::m_blockSizeDepthMap = 16;

ID3D11Buffer* DX11ImageHelper::m_constantBufferDepthMap = NULL;
ID3D11ComputeShader* DX11ImageHelper::m_pComputeShaderDepthMap = NULL;

/////////////////////////////////////////////////////
// Compute SSAO Map
/////////////////////////////////////////////////////

unsigned int DX11ImageHelper::m_blockSizeSSAOMap = 16;

ID3D11Buffer* DX11ImageHelper::m_constantBufferSSAOMap = NULL;
ID3D11ComputeShader* DX11ImageHelper::m_pComputeShaderSSAOMap = NULL;

std::vector<vec4f> DX11ImageHelper::m_randomRotations;

/////////////////////////////////////////////////////
// Compute HSV Depth Map
/////////////////////////////////////////////////////

unsigned int DX11ImageHelper::m_blockSizeHSVDepth = 16;

ID3D11Buffer* DX11ImageHelper::m_constantBufferHSVDepth = NULL;
ID3D11ComputeShader* DX11ImageHelper::m_pComputeShaderHSVDepth = NULL;

/////////////////////////////////////////////////////
// Camera Space Projection
/////////////////////////////////////////////////////

unsigned int DX11ImageHelper::m_blockSizeCameraSpaceProjection = 16;

ID3D11Buffer* DX11ImageHelper::m_constantBufferCameraSpaceProjection = NULL;
ID3D11ComputeShader* DX11ImageHelper::m_pComputeShaderCameraSpaceProjection = NULL;

/////////////////////////////////////////////////////
// Stereo Mask Camera Space Projection
/////////////////////////////////////////////////////

unsigned int DX11ImageHelper::m_blockSizeStereoCameraSpaceProjection = 16;

ID3D11Buffer* DX11ImageHelper::m_constantBufferStereoCameraSpaceProjection = NULL;
ID3D11ComputeShader* DX11ImageHelper::m_pComputeShaderStereoCameraSpaceProjection = NULL;

/////////////////////////////////////////////////////
// Projective Correspondences
/////////////////////////////////////////////////////

unsigned int DX11ImageHelper::m_blockSizeProjectiveCorrespondences = 16;

ID3D11Buffer* DX11ImageHelper::m_constantBufferProjectiveCorrespondences = NULL;
ID3D11ComputeShader* DX11ImageHelper::m_pComputeShaderProjectiveCorrespondences = NULL;

/////////////////////////////////////////////////////
// Depth aware block averaging
/////////////////////////////////////////////////////

unsigned int DX11ImageHelper::m_blockSizeBlockAverage = 16;
ID3D11Buffer* DX11ImageHelper::m_constantBufferBlockAverage = NULL;
ID3D11ComputeShader* DX11ImageHelper::m_pComputeShaderBlockAverage = NULL;

/////////////////////////////////////////////////////
// Subsample
/////////////////////////////////////////////////////

unsigned int DX11ImageHelper::m_blockSizeSubSamp = 16;
ID3D11Buffer* DX11ImageHelper::m_constantBufferSubSamp = NULL;
ID3D11ComputeShader* DX11ImageHelper::m_pComputeShaderSubSamp = NULL;

/////////////////////////////////////////////////////
// Downsampling
/////////////////////////////////////////////////////

unsigned int DX11ImageHelper::m_blockSizeDownsampling = 16;
ID3D11Buffer* DX11ImageHelper::m_constantBufferDownsampling = NULL;
ID3D11ComputeShader* DX11ImageHelper::m_pComputeShaderDownsampling = NULL;

/////////////////////////////////////////////////////
// Copy
/////////////////////////////////////////////////////

unsigned int DX11ImageHelper::m_blockSizeCopy = 16;
ID3D11Buffer* DX11ImageHelper::m_constantBufferCopy = NULL;
ID3D11ComputeShader* DX11ImageHelper::m_pComputeShaderCopy = NULL;

/////////////////////////////////////////////////////
// Erode
/////////////////////////////////////////////////////

unsigned int DX11ImageHelper::m_blockSizeErode = 16;
ID3D11Buffer* DX11ImageHelper::m_constantBufferErode = NULL;
ID3D11ComputeShader* DX11ImageHelper::m_pComputeShaderErode = NULL;

/////////////////////////////////////////////////////
// Timer
/////////////////////////////////////////////////////

Timer DX11ImageHelper::m_timer;

HRESULT DX11ImageHelper::OnD3D11CreateDevice( ID3D11Device* pd3dDevice )
{
	HRESULT hr = S_OK;

	/////////////////////////////////////////////////////
	// Bilateral Completion
	/////////////////////////////////////////////////////

	char BLOCK_SIZE_BilateralCompletion[5];
	sprintf_s(BLOCK_SIZE_BilateralCompletion, "%d", m_blockSizeBilateralCompletion);

	D3D_SHADER_MACRO shaderDefinesBilateralCompletion[] = { { "groupthreads", BLOCK_SIZE_BilateralCompletion }, { 0 } };

	ID3DBlob* pBlob = NULL;
	hr = CompileShaderFromFile(L"Shaders\\BilateralFilter.hlsl", "bilateralCompletionCS", "cs_5_0", &pBlob, shaderDefinesBilateralCompletion);
	if(FAILED(hr)) return hr;

	hr = pd3dDevice->CreateComputeShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &m_pComputeShaderBilateralCompletion);
	if(FAILED(hr)) return hr;

	D3D_SHADER_MACRO shaderDefinesBilateralCompletionWithColor[] = { { "groupthreads", BLOCK_SIZE_BilateralCompletion }, { "WITH_COLOR", "1"}, { 0 } };

	hr = CompileShaderFromFile(L"Shaders\\BilateralFilter.hlsl", "bilateralCompletionCS", "cs_5_0", &pBlob, shaderDefinesBilateralCompletionWithColor);
	if(FAILED(hr)) return hr;

	hr = pd3dDevice->CreateComputeShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &m_pComputeShaderBilateralCompletionWithColor);
	if(FAILED(hr)) return hr;

	SAFE_RELEASE(pBlob);

	D3D11_BUFFER_DESC bDesc;
	bDesc.BindFlags	= D3D11_BIND_CONSTANT_BUFFER;
	bDesc.Usage	= D3D11_USAGE_DYNAMIC;
	bDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	bDesc.MiscFlags	= 0;

	bDesc.ByteWidth	= sizeof(CBufferBilateralCompletion) ;
	hr = pd3dDevice->CreateBuffer(&bDesc, NULL, &m_constantBufferBilateralCompletion);
	if(FAILED(hr)) return hr;

	/////////////////////////////////////////////////////
	// Bilateral Filter
	/////////////////////////////////////////////////////

	char BLOCK_SIZE_BF[5];
	sprintf_s(BLOCK_SIZE_BF, "%d", m_blockSizeBilateralFilter);

	D3D_SHADER_MACRO shaderDefinesBF[] = { { "groupthreads", BLOCK_SIZE_BF }, { 0 } };

	hr = CompileShaderFromFile(L"Shaders\\BilateralFilter.hlsl", "bilateralFilterCS", "cs_5_0", &pBlob, shaderDefinesBF);
	if(FAILED(hr)) return hr;

	hr = pd3dDevice->CreateComputeShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &m_pComputeShaderBilateralFilter);
	if(FAILED(hr)) return hr;

	SAFE_RELEASE(pBlob);

	bDesc.BindFlags	= D3D11_BIND_CONSTANT_BUFFER;
	bDesc.Usage	= D3D11_USAGE_DYNAMIC;
	bDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	bDesc.MiscFlags	= 0;

	bDesc.ByteWidth	= sizeof(CBufferBilateralFilter) ;
	hr = pd3dDevice->CreateBuffer(&bDesc, NULL, &m_constantBufferBilateralFilter);
	if(FAILED(hr)) return hr;

	/////////////////////////////////////////////////////
	// Bilateral Filter SSAO
	/////////////////////////////////////////////////////

	char BLOCK_SIZE_BFSSAO[5];
	sprintf_s(BLOCK_SIZE_BFSSAO, "%d", m_blockSizeBilateralFilterForSSAO);

	D3D_SHADER_MACRO shaderDefinesBFSSAO[] = { { "groupthreads", BLOCK_SIZE_BFSSAO }, { 0 } };

	hr = CompileShaderFromFile(L"Shaders\\BilateralFilterSSAO.hlsl", "bilateralFilterSSAOCS", "cs_5_0", &pBlob, shaderDefinesBFSSAO);
	if(FAILED(hr)) return hr;

	hr = pd3dDevice->CreateComputeShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &m_pComputeShaderBilateralFilterForSSAO);
	if(FAILED(hr)) return hr;

	SAFE_RELEASE(pBlob);

	bDesc.BindFlags	= D3D11_BIND_CONSTANT_BUFFER;
	bDesc.Usage	= D3D11_USAGE_DYNAMIC;
	bDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	bDesc.MiscFlags	= 0;

	bDesc.ByteWidth	= sizeof(CBufferBilateralFilterSSAO) ;
	hr = pd3dDevice->CreateBuffer(&bDesc, NULL, &m_constantBufferBilateralFilterForSSAO);
	if(FAILED(hr)) return hr;

	/////////////////////////////////////////////////////
	// Bilateral Filter Approximation
	/////////////////////////////////////////////////////

	char BLOCK_SIZE_BFApprox[5];
	sprintf_s(BLOCK_SIZE_BFApprox, "%d", m_blockSizeBilateralFilterApprox);

	D3D_SHADER_MACRO shaderDefinesBFApprox[] = { { "groupthreads", BLOCK_SIZE_BFApprox }, { 0 } };

	hr = CompileShaderFromFile(L"Shaders\\BilateralFilterApprox.hlsl", "bilateralFilterApproxCS", "cs_5_0", &pBlob, shaderDefinesBFApprox);
	if(FAILED(hr)) return hr;

	hr = pd3dDevice->CreateComputeShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &m_pComputeShaderBilateralFilterApprox);
	if(FAILED(hr)) return hr;

	SAFE_RELEASE(pBlob);

	bDesc.BindFlags	= D3D11_BIND_CONSTANT_BUFFER;
	bDesc.Usage	= D3D11_USAGE_DYNAMIC;
	bDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	bDesc.MiscFlags	= 0;

	bDesc.ByteWidth	= sizeof(CBufferBilateralFilterApprox) ;
	hr = pd3dDevice->CreateBuffer(&bDesc, NULL, &m_constantBufferBilateralFilterApprox);
	if(FAILED(hr)) return hr;

	/////////////////////////////////////////////////////
	// Bilateral Filter 4F
	/////////////////////////////////////////////////////

	char BLOCK_SIZE_BF4F[5];
	sprintf_s(BLOCK_SIZE_BF4F, "%d", m_blockSizeBilateralFilter4F);

	D3D_SHADER_MACRO shaderDefinesBF4F[] = { { "groupthreads", BLOCK_SIZE_BF4F }, { 0 } };

	hr = CompileShaderFromFile(L"Shaders\\BilateralFilter.hlsl", "bilateralFilter4FCS", "cs_5_0", &pBlob, shaderDefinesBF4F);
	if(FAILED(hr)) return hr;

	hr = pd3dDevice->CreateComputeShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &m_pComputeShaderBilateralFilter4F);
	if(FAILED(hr)) return hr;

	SAFE_RELEASE(pBlob);

	bDesc.BindFlags	= D3D11_BIND_CONSTANT_BUFFER;
	bDesc.Usage	= D3D11_USAGE_DYNAMIC;
	bDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	bDesc.MiscFlags	= 0;

	bDesc.ByteWidth	= sizeof(CBufferBilateralFilter4F) ;
	hr = pd3dDevice->CreateBuffer(&bDesc, NULL, &m_constantBufferBilateralFilter4F);
	if(FAILED(hr)) return hr;

	/////////////////////////////////////////////////////
	// Normal Computation
	/////////////////////////////////////////////////////

	char BLOCK_SIZE_NC[5];
	sprintf_s(BLOCK_SIZE_NC, "%d", m_blockSizeNormalComputation);

	D3D_SHADER_MACRO shaderDefinesNC[] = { { "groupthreads", BLOCK_SIZE_NC }, { 0 } };

	hr = CompileShaderFromFile(L"Shaders\\NormalComputation.hlsl", "normalComputationCS", "cs_5_0", &pBlob, shaderDefinesNC);
	if(FAILED(hr)) return hr;

	hr = pd3dDevice->CreateComputeShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &m_pComputeShaderNormalComputation);
	if(FAILED(hr)) return hr;

	SAFE_RELEASE(pBlob);

	bDesc.ByteWidth	= sizeof(CBufferNormalComputation) ;
	hr = pd3dDevice->CreateBuffer(&bDesc, NULL, &m_constantBufferNormalComputation);
	if(FAILED(hr)) return hr;

	/////////////////////////////////////////////////////
	// Camera Space To Depth Map
	/////////////////////////////////////////////////////

	char BLOCK_SIZE_CS[5];
	sprintf_s(BLOCK_SIZE_CS, "%d", m_blockSizeCameraSpace);

	D3D_SHADER_MACRO shaderDefinesCS[] = { { "groupthreads", BLOCK_SIZE_CS }, { 0 } };

	hr = CompileShaderFromFile(L"Shaders\\CameraSpaceProjection.hlsl", "cameraSpaceToDepthMapCS", "cs_5_0", &pBlob, shaderDefinesCS);
	if(FAILED(hr)) return hr;

	hr = pd3dDevice->CreateComputeShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &m_pComputeShaderCameraSpace);
	if(FAILED(hr)) return hr;

	SAFE_RELEASE(pBlob);

	bDesc.ByteWidth	= sizeof(CBufferCameraSpace) ;
	hr = pd3dDevice->CreateBuffer(&bDesc, NULL, &m_constantBufferCameraSpace);
	if(FAILED(hr)) return hr;

	/////////////////////////////////////////////////////
	// Compute Depth Maps
	/////////////////////////////////////////////////////

	char BLOCK_SIZE_DM[5];
	sprintf_s(BLOCK_SIZE_DM, "%d", m_blockSizeDepthMap);

	D3D_SHADER_MACRO shaderDefinesDM[] = { { "groupthreads", BLOCK_SIZE_DM }, { 0 } };

	hr = CompileShaderFromFile(L"Shaders\\DepthMapComputation.hlsl", "depthMapComputationCS", "cs_5_0", &pBlob, shaderDefinesDM);
	if(FAILED(hr)) return hr;

	hr = pd3dDevice->CreateComputeShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &m_pComputeShaderDepthMap);
	if(FAILED(hr)) return hr;

	SAFE_RELEASE(pBlob);

	bDesc.ByteWidth	= sizeof(CBufferDepthMap);
	hr = pd3dDevice->CreateBuffer(&bDesc, NULL, &m_constantBufferDepthMap);
	if(FAILED(hr)) return hr;


	/////////////////////////////////////////////////////
	// Compute SSAO Map
	/////////////////////////////////////////////////////

	char BLOCK_SIZE_SSAOMap[5];
	sprintf_s(BLOCK_SIZE_SSAOMap, "%d", m_blockSizeSSAOMap);

	D3D_SHADER_MACRO shaderDefinesSSAOMap[] = { { "groupthreads", BLOCK_SIZE_SSAOMap }, { 0 } };

	hr = CompileShaderFromFile(L"Shaders\\ComputeSSAOMap.hlsl", "computeSSAOMapCS", "cs_5_0", &pBlob, shaderDefinesSSAOMap);
	if(FAILED(hr)) return hr;

	hr = pd3dDevice->CreateComputeShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &m_pComputeShaderSSAOMap);
	if(FAILED(hr)) return hr;

	SAFE_RELEASE(pBlob);

	bDesc.ByteWidth	= sizeof(CBufferSSAOMap);
	hr = pd3dDevice->CreateBuffer(&bDesc, NULL, &m_constantBufferSSAOMap);
	if(FAILED(hr)) return hr;

	for(unsigned int i = 0; i<16; i++)
	{
		float r0 = (float)std::rand()/(float)RAND_MAX; r0-=0.5f; r0*=2.0f;
		float r1 = (float)std::rand()/(float)RAND_MAX; r1-=0.5f; r1*=2.0f;
		float r2 = (float)std::rand()/(float)RAND_MAX; r2-=0.5f; r2*=2.0f;

		m_randomRotations.push_back(vec4f(r0, r1, r2, 1.0));
	}

	/////////////////////////////////////////////////////
	// Compute HSV Depth Maps
	/////////////////////////////////////////////////////

	char BLOCK_SIZE_HSVDepth[5];
	sprintf_s(BLOCK_SIZE_HSVDepth, "%d", m_blockSizeHSVDepth);

	D3D_SHADER_MACRO shaderDefinesHSVDepth[] = { { "groupthreads", BLOCK_SIZE_HSVDepth }, { 0 } };

	hr = CompileShaderFromFile(L"Shaders\\HSVDepthMapComputation.hlsl", "HSVdepthMapComputationCS", "cs_5_0", &pBlob, shaderDefinesHSVDepth);
	if(FAILED(hr)) return hr;

	hr = pd3dDevice->CreateComputeShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &m_pComputeShaderHSVDepth);
	if(FAILED(hr)) return hr;

	SAFE_RELEASE(pBlob);

	bDesc.ByteWidth	= sizeof(CBufferHSVDepth);
	hr = pd3dDevice->CreateBuffer(&bDesc, NULL, &m_constantBufferHSVDepth);
	if(FAILED(hr)) return hr;

	/////////////////////////////////////////////////////
	// Camera Space Projection
	/////////////////////////////////////////////////////

	char BLOCK_SIZE_CP[5];
	sprintf_s(BLOCK_SIZE_CP, "%d", m_blockSizeCameraSpaceProjection);

	D3D_SHADER_MACRO shaderDefinesCP[] = { { "groupthreads", BLOCK_SIZE_CP }, { 0 } };

	hr = CompileShaderFromFile(L"Shaders\\CameraSpaceProjection.hlsl", "cameraSpaceProjectionCS", "cs_5_0", &pBlob, shaderDefinesCP);
	if(FAILED(hr)) return hr;

	hr = pd3dDevice->CreateComputeShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &m_pComputeShaderCameraSpaceProjection);
	if(FAILED(hr)) return hr;

	SAFE_RELEASE(pBlob);

	bDesc.ByteWidth	= sizeof(CBufferCameraSpaceProjection) ;
	hr = pd3dDevice->CreateBuffer(&bDesc, NULL, &m_constantBufferCameraSpaceProjection);
	if(FAILED(hr)) return hr;

	/////////////////////////////////////////////////////
	// Stereo Mask Camera Space Projection
	/////////////////////////////////////////////////////

	char BLOCK_SIZE_SM[5];
	sprintf_s(BLOCK_SIZE_SM, "%d", m_blockSizeStereoCameraSpaceProjection);

	D3D_SHADER_MACRO shaderDefinesSM[] = { { "groupthreads", BLOCK_SIZE_SM }, { 0 } };

	hr = CompileShaderFromFile(L"Shaders\\StereoMaskCameraSpaceProjection.hlsl", "stereoMaskCameraSpaceProjectionCS", "cs_5_0", &pBlob, shaderDefinesSM);
	if(FAILED(hr)) return hr;

	hr = pd3dDevice->CreateComputeShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &m_pComputeShaderStereoCameraSpaceProjection);
	if(FAILED(hr)) return hr;

	SAFE_RELEASE(pBlob);

	bDesc.ByteWidth	= sizeof(CBufferStereoCameraSpaceProjection) ;
	hr = pd3dDevice->CreateBuffer(&bDesc, NULL, &m_constantBufferStereoCameraSpaceProjection);
	if(FAILED(hr)) return hr;

	/////////////////////////////////////////////////////
	// Projective Correspondences
	/////////////////////////////////////////////////////

	char BLOCK_SIZE_PC[5];
	sprintf_s(BLOCK_SIZE_PC, "%d", m_blockSizeProjectiveCorrespondences);

	D3D_SHADER_MACRO shaderDefinesPC[] = { { "groupthreads", BLOCK_SIZE_PC }, { 0 } };

	hr = CompileShaderFromFile(L"Shaders\\ProjectiveCorrespondences.hlsl", "projectiveCorrespondencesCS", "cs_5_0", &pBlob, shaderDefinesPC);
	V_RETURN(hr);

	hr = pd3dDevice->CreateComputeShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &m_pComputeShaderProjectiveCorrespondences);
	V_RETURN(hr);

	SAFE_RELEASE(pBlob);

	bDesc.ByteWidth	= sizeof(CBufferProjectiveCorrespondences) ;
	hr = pd3dDevice->CreateBuffer(&bDesc, NULL, &m_constantBufferProjectiveCorrespondences);
	V_RETURN(hr);

	/////////////////////////////////////////////////////
	// Downsampling
	/////////////////////////////////////////////////////

	char BLOCK_SIZE_DS[5];
	sprintf_s(BLOCK_SIZE_DS, "%d", m_blockSizeDownsampling);

	D3D_SHADER_MACRO shaderDefinesDS[] = { { "groupthreads", BLOCK_SIZE_DS }, { 0 } };

	hr = CompileShaderFromFile(L"Shaders\\Downsampling.hlsl", "downsamplingCS", "cs_5_0", &pBlob, shaderDefinesDS);
	V_RETURN(hr);

	hr = pd3dDevice->CreateComputeShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &m_pComputeShaderDownsampling);
	V_RETURN(hr);

	SAFE_RELEASE(pBlob);

	bDesc.ByteWidth	= sizeof(CBufferDS) ;
	hr = pd3dDevice->CreateBuffer(&bDesc, NULL, &m_constantBufferDownsampling);
	V_RETURN(hr);

	/////////////////////////////////////////////////////
	// Subsample
	/////////////////////////////////////////////////////

	char BLOCK_SIZE_SubSamp[5];
	sprintf_s(BLOCK_SIZE_SubSamp, "%d", m_blockSizeSubSamp);

	D3D_SHADER_MACRO shaderDefinesSubSamp[] = { { "groupthreads", BLOCK_SIZE_SubSamp }, { 0 } };

	hr = CompileShaderFromFile(L"Shaders\\SubSampling.hlsl", "subSampleCS", "cs_5_0", &pBlob, shaderDefinesSubSamp);
	V_RETURN(hr);

	hr = pd3dDevice->CreateComputeShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &m_pComputeShaderSubSamp);
	V_RETURN(hr);

	SAFE_RELEASE(pBlob);

	bDesc.ByteWidth	= sizeof(CBufferSubSamp) ;
	hr = pd3dDevice->CreateBuffer(&bDesc, NULL, &m_constantBufferSubSamp);
	V_RETURN(hr);

	/////////////////////////////////////////////////////
	// Depth aware block averaging
	/////////////////////////////////////////////////////

	char BLOCK_SIZE_BlockAverage[5];
	sprintf_s(BLOCK_SIZE_BlockAverage, "%d", m_blockSizeBlockAverage);

	D3D_SHADER_MACRO shaderDefinesBlockAverage[] = { { "groupthreads", BLOCK_SIZE_BlockAverage }, { 0 } };

	hr = CompileShaderFromFile(L"Shaders\\BlockAveraging.hlsl", "averageCS", "cs_5_0", &pBlob, shaderDefinesBlockAverage);
	V_RETURN(hr);

	hr = pd3dDevice->CreateComputeShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &m_pComputeShaderBlockAverage);
	V_RETURN(hr);

	SAFE_RELEASE(pBlob);

	bDesc.ByteWidth	= sizeof(CBufferBlockAverage) ;
	hr = pd3dDevice->CreateBuffer(&bDesc, NULL, &m_constantBufferBlockAverage);
	V_RETURN(hr);

	/////////////////////////////////////////////////////
	// Copy
	/////////////////////////////////////////////////////

	char BLOCK_SIZE_Copy[5];
	sprintf_s(BLOCK_SIZE_Copy, "%d", m_blockSizeCopy);

	D3D_SHADER_MACRO shaderDefinesCopy[] = { { "groupthreads", BLOCK_SIZE_Copy }, { 0 } };

	hr = CompileShaderFromFile(L"Shaders\\Copy.hlsl", "copyCS", "cs_5_0", &pBlob, shaderDefinesCopy);
	V_RETURN(hr);

	hr = pd3dDevice->CreateComputeShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &m_pComputeShaderCopy);
	V_RETURN(hr);

	SAFE_RELEASE(pBlob);

	bDesc.ByteWidth	= sizeof(CBufferCopy) ;
	hr = pd3dDevice->CreateBuffer(&bDesc, NULL, &m_constantBufferCopy);
	V_RETURN(hr);

	/////////////////////////////////////////////////////
	// Erode
	/////////////////////////////////////////////////////

	char BLOCK_SIZE_Erode[5];
	sprintf_s(BLOCK_SIZE_Erode, "%d", m_blockSizeErode);

	D3D_SHADER_MACRO shaderDefinesErode[] = { { "groupthreads", BLOCK_SIZE_Erode }, { 0 } };

	hr = CompileShaderFromFile(L"Shaders\\Erode.hlsl", "erodeCS", "cs_5_0", &pBlob, shaderDefinesErode);
	V_RETURN(hr);

	hr = pd3dDevice->CreateComputeShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &m_pComputeShaderErode);
	V_RETURN(hr);

	SAFE_RELEASE(pBlob);

	bDesc.ByteWidth	= sizeof(CBufferErode) ;
	hr = pd3dDevice->CreateBuffer(&bDesc, NULL, &m_constantBufferErode);
	V_RETURN(hr);

	return  hr;
}

HRESULT DX11ImageHelper::applyBilateralFilter( ID3D11DeviceContext* context, ID3D11ShaderResourceView* inputSRV, ID3D11UnorderedAccessView* outputUAV, unsigned int imageWidth, unsigned int imageHeight, float sigmaD /*= 5.0f*/, float sigmaR /*= 0.1f*/ )
{
	HRESULT hr = S_OK;

	// Initialize Constant Buffer
	D3D11_MAPPED_SUBRESOURCE mappedResource;
	hr = context->Map(m_constantBufferBilateralFilter, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
	V_RETURN(hr);

	CBufferBilateralFilter *cbuffer = (CBufferBilateralFilter*)mappedResource.pData;
	cbuffer->imageWidth = (int)imageWidth;
	cbuffer->imageHeigth = (int)imageHeight;
	cbuffer->sigmaD = sigmaD;
	cbuffer->sigmaR = sigmaR;

	context->Unmap(m_constantBufferBilateralFilter, 0);

	// Setup Pipeline
	context->CSSetShaderResources(0, 1, &inputSRV);
	context->CSSetUnorderedAccessViews( 0, 1, &outputUAV, 0);
	context->CSSetConstantBuffers(0, 1, &m_constantBufferBilateralFilter);
	context->CSSetShader(m_pComputeShaderBilateralFilter, 0, 0);

	// Start Compute Shader
	unsigned int dimX = (unsigned int)ceil(((float)imageWidth)/m_blockSizeBilateralFilter);
	unsigned int dimY = (unsigned int)ceil(((float)imageHeight)/m_blockSizeBilateralFilter);
	context->Dispatch(dimX, dimY, 1);
	assert(dimX <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);
	assert(dimY <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);

	// De-Initialize Pipeline
	ID3D11ShaderResourceView* nullSAV[1] = { NULL };
	ID3D11UnorderedAccessView* nullUAV[1] = { NULL };
	ID3D11Buffer* nullCB[1] = { NULL };

	context->CSSetShaderResources(0, 1, nullSAV);
	context->CSSetUnorderedAccessViews(0, 1, nullUAV, 0);
	context->CSSetConstantBuffers(0, 1, nullCB);
	context->CSSetShader(0, 0, 0);

	return hr;
}

HRESULT DX11ImageHelper::applyBilateralFilterForSSAO( ID3D11DeviceContext* context, ID3D11ShaderResourceView* inputDepthSRV, ID3D11ShaderResourceView* inputSRV, ID3D11UnorderedAccessView* outputUAV, unsigned int imageWidth, unsigned int imageHeight, float sigmaD /*= 5.0f*/, float sigmaR /*= 0.1f*/ )
{
	HRESULT hr = S_OK;

	// Initialize Constant Buffer
	D3D11_MAPPED_SUBRESOURCE mappedResource;
	hr = context->Map(m_constantBufferBilateralFilterForSSAO, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
	V_RETURN(hr);

	CBufferBilateralFilterSSAO *cbuffer = (CBufferBilateralFilterSSAO*)mappedResource.pData;
	cbuffer->imageWidth = (int)imageWidth;
	cbuffer->imageHeigth = (int)imageHeight;
	cbuffer->sigmaD = sigmaD;
	cbuffer->sigmaR = sigmaR;

	context->Unmap(m_constantBufferBilateralFilterForSSAO, 0);

	// Setup Pipeline
	context->CSSetShaderResources(0, 1, &inputDepthSRV);
	context->CSSetShaderResources(1, 1, &inputSRV);
	context->CSSetUnorderedAccessViews( 0, 1, &outputUAV, 0);
	context->CSSetConstantBuffers(0, 1, &m_constantBufferBilateralFilterForSSAO);
	context->CSSetShader(m_pComputeShaderBilateralFilterForSSAO, 0, 0);

	// Start Compute Shader
	unsigned int dimX = (unsigned int)ceil(((float)imageWidth)/m_blockSizeBilateralFilterForSSAO);
	unsigned int dimY = (unsigned int)ceil(((float)imageHeight)/m_blockSizeBilateralFilterForSSAO);
	context->Dispatch(dimX, dimY, 1);
	assert(dimX <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);
	assert(dimY <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);

	// De-Initialize Pipeline
	ID3D11ShaderResourceView* nullSAV[] = { NULL, NULL };
	ID3D11UnorderedAccessView* nullUAV[] = { NULL };
	ID3D11Buffer* nullCB[1] = { NULL };

	context->CSSetShaderResources(0, 2, nullSAV);
	context->CSSetUnorderedAccessViews(0, 1, nullUAV, 0);
	context->CSSetConstantBuffers(0, 1, nullCB);
	context->CSSetShader(0, 0, 0);

	return hr;
}

HRESULT DX11ImageHelper::applyBilateralFilterApprox( ID3D11DeviceContext* context, ID3D11ShaderResourceView* inputSRV, ID3D11UnorderedAccessView* outputUAV, unsigned int imageWidth, unsigned int imageHeight, unsigned int kernelRadius, float distThres )
{
	HRESULT hr = S_OK;

	// Initialize Constant Buffer
	D3D11_MAPPED_SUBRESOURCE mappedResource;
	hr = context->Map(m_constantBufferBilateralFilterApprox, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
	V_RETURN(hr);

	CBufferBilateralFilterApprox *cbuffer = (CBufferBilateralFilterApprox*)mappedResource.pData;
	cbuffer->imageWidth = (int)imageWidth;
	cbuffer->imageHeigth = (int)imageHeight;
	cbuffer->kernelRadius = kernelRadius;
	cbuffer->thres = distThres;

	context->Unmap(m_constantBufferBilateralFilterApprox, 0);

	// Setup Pipeline
	context->CSSetShaderResources(0, 1, &inputSRV);
	context->CSSetUnorderedAccessViews( 0, 1, &outputUAV, 0);
	context->CSSetConstantBuffers(0, 1, &m_constantBufferBilateralFilterApprox);
	context->CSSetShader(m_pComputeShaderBilateralFilterApprox, 0, 0);

	// Start Compute Shader
	unsigned int dimX = (unsigned int)ceil(((float)imageWidth)/m_blockSizeBilateralFilterApprox);
	unsigned int dimY = (unsigned int)ceil(((float)imageHeight)/m_blockSizeBilateralFilterApprox);
	context->Dispatch(dimX, dimY, 1);
	assert(dimX <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);
	assert(dimY <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);

	// De-Initialize Pipeline
	ID3D11ShaderResourceView* nullSAV[1] = { NULL };
	ID3D11UnorderedAccessView* nullUAV[1] = { NULL };
	ID3D11Buffer* nullCB[1] = { NULL };

	context->CSSetShaderResources(0, 1, nullSAV);
	context->CSSetUnorderedAccessViews(0, 1, nullUAV, 0);
	context->CSSetConstantBuffers(0, 1, nullCB);
	context->CSSetShader(0, 0, 0);

	return hr;
}

HRESULT DX11ImageHelper::applyBilateralCompletion( ID3D11DeviceContext* context, ID3D11ShaderResourceView* inputSRV, ID3D11UnorderedAccessView* outputUAV, unsigned int imageWidth, unsigned int imageHeight, float sigmaD /*= 5.0f*/, float sigmaR /*= 0.1f*/ )
{
	HRESULT hr = S_OK;

	// Initialize Constant Buffer
	D3D11_MAPPED_SUBRESOURCE mappedResource;
	hr = context->Map(m_constantBufferBilateralCompletion, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
	V_RETURN(hr);

	CBufferBilateralCompletion *cbuffer = (CBufferBilateralCompletion*)mappedResource.pData;
	cbuffer->imageWidth = (int)imageWidth;
	cbuffer->imageHeigth = (int)imageHeight;
	cbuffer->sigmaD = sigmaD;
	cbuffer->sigmaR = sigmaR;

	context->Unmap(m_constantBufferBilateralCompletion, 0);

	// Setup Pipeline
	context->CSSetShaderResources(0, 1, &inputSRV);
	context->CSSetUnorderedAccessViews( 0, 1, &outputUAV, 0);
	context->CSSetConstantBuffers(0, 1, &m_constantBufferBilateralCompletion);
	context->CSSetShader(m_pComputeShaderBilateralCompletion, 0, 0);

	// Start Compute Shader
	unsigned int dimX = (unsigned int)ceil(((float)imageWidth)/m_blockSizeBilateralCompletion);
	unsigned int dimY = (unsigned int)ceil(((float)imageHeight)/m_blockSizeBilateralCompletion);
	context->Dispatch(dimX, dimY, 1);
	assert(dimX <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);
	assert(dimY <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);

	// De-Initialize Pipeline
	ID3D11ShaderResourceView* nullSAV[1] = { NULL };
	ID3D11UnorderedAccessView* nullUAV[1] = { NULL };
	ID3D11Buffer* nullCB[1] = { NULL };

	context->CSSetShaderResources(0, 1, nullSAV);
	context->CSSetUnorderedAccessViews(0, 1, nullUAV, 0);
	context->CSSetConstantBuffers(0, 1, nullCB);
	context->CSSetShader(0, 0, 0);

	return hr;
}

HRESULT DX11ImageHelper::applyBilateralCompletionWithColor( ID3D11DeviceContext* context, ID3D11ShaderResourceView* inputDepthSRV, ID3D11UnorderedAccessView* outputDepthUAV, ID3D11ShaderResourceView* inputColorSRV, ID3D11UnorderedAccessView* outputColorUAV, unsigned int imageWidth, unsigned int imageHeight, float sigmaD /*= 5.0f*/, float sigmaR /*= 0.1f*/ )
{
	HRESULT hr = S_OK;

	// Initialize Constant Buffer
	D3D11_MAPPED_SUBRESOURCE mappedResource;
	hr = context->Map(m_constantBufferBilateralCompletion, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
	V_RETURN(hr);

	CBufferBilateralCompletion *cbuffer = (CBufferBilateralCompletion*)mappedResource.pData;
	cbuffer->imageWidth = (int)imageWidth;
	cbuffer->imageHeigth = (int)imageHeight;
	cbuffer->sigmaD = sigmaD;
	cbuffer->sigmaR = sigmaR;

	context->Unmap(m_constantBufferBilateralCompletion, 0);

	// Setup Pipeline
	context->CSSetShaderResources(0, 1, &inputDepthSRV);
	context->CSSetShaderResources(1, 1, &inputColorSRV);
	context->CSSetUnorderedAccessViews( 0, 1, &outputDepthUAV, 0);
	context->CSSetUnorderedAccessViews( 1, 1, &outputColorUAV, 0);
	context->CSSetConstantBuffers(0, 1, &m_constantBufferBilateralCompletion);
	context->CSSetShader(m_pComputeShaderBilateralCompletionWithColor, 0, 0);

	// Start Compute Shader
	unsigned int dimX = (unsigned int)ceil(((float)imageWidth)/m_blockSizeBilateralCompletion);
	unsigned int dimY = (unsigned int)ceil(((float)imageHeight)/m_blockSizeBilateralCompletion);
	context->Dispatch(dimX, dimY, 1);
	assert(dimX <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);
	assert(dimY <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);

	// De-Initialize Pipeline
	ID3D11ShaderResourceView* nullSRV[] = { NULL, NULL };
	ID3D11UnorderedAccessView* nullUAV[] = { NULL, NULL };
	ID3D11Buffer* nullCB[] = { NULL };

	context->CSSetShaderResources(0, 2, nullSRV);
	context->CSSetUnorderedAccessViews(0, 2, nullUAV, 0);
	context->CSSetConstantBuffers(0, 1, nullCB);
	context->CSSetShader(0, 0, 0);

	return hr;
}

HRESULT DX11ImageHelper::applyBilateralFilter4F( ID3D11DeviceContext* context, ID3D11ShaderResourceView* inputSRV, ID3D11UnorderedAccessView* outputUAV, unsigned int imageWidth, unsigned int imageHeight, float sigmaD /*= 5.0f*/, float sigmaR /*= 0.1f*/ )
{
	HRESULT hr = S_OK;

	// Initialize Constant Buffer
	D3D11_MAPPED_SUBRESOURCE mappedResource;
	hr = context->Map(m_constantBufferBilateralFilter4F, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
	V_RETURN(hr);

	CBufferBilateralFilter4F *cbuffer = (CBufferBilateralFilter4F*)mappedResource.pData;
	cbuffer->imageWidth = (int)imageWidth;
	cbuffer->imageHeigth = (int)imageHeight;
	cbuffer->sigmaD = sigmaD;
	cbuffer->sigmaR = sigmaR;

	context->Unmap(m_constantBufferBilateralFilter4F, 0);

	// Setup Pipeline
	context->CSSetShaderResources(1, 1, &inputSRV);
	context->CSSetUnorderedAccessViews(1, 1, &outputUAV, 0);
	context->CSSetConstantBuffers(0, 1, &m_constantBufferBilateralFilter4F);
	context->CSSetShader(m_pComputeShaderBilateralFilter4F, 0, 0);

	// Start Compute Shader
	unsigned int dimX = (unsigned int)ceil(((float)imageWidth)/m_blockSizeBilateralFilter4F);
	unsigned int dimY = (unsigned int)ceil(((float)imageHeight)/m_blockSizeBilateralFilter4F);
	context->Dispatch(dimX, dimY, 1);
	assert(dimX <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);
	assert(dimY <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);

	// De-Initialize Pipeline
	ID3D11ShaderResourceView* nullSAV[1] = { NULL };
	ID3D11UnorderedAccessView* nullUAV[1] = { NULL };
	ID3D11Buffer* nullCB[1] = { NULL };

	context->CSSetShaderResources(1, 1, nullSAV);
	context->CSSetUnorderedAccessViews(1, 1, nullUAV, 0);
	context->CSSetConstantBuffers(0, 1, nullCB);
	context->CSSetShader(0, 0, 0);

	return hr;
}

HRESULT DX11ImageHelper::applyNormalComputation( ID3D11DeviceContext* context, ID3D11ShaderResourceView* inputSRV, ID3D11UnorderedAccessView* outputUAV, unsigned int imageWidth, unsigned int imageHeight )
{
	HRESULT hr = S_OK;

	// Initialize Constant Buffer
	D3D11_MAPPED_SUBRESOURCE mappedResource;
	hr = context->Map(m_constantBufferNormalComputation, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
	if(FAILED(hr)) return hr;

	CBufferNormalComputation *cbuffer = (CBufferNormalComputation*)mappedResource.pData;
	cbuffer->imageWidth = (int)imageWidth;
	cbuffer->imageHeigth = (int)imageHeight;

	context->Unmap(m_constantBufferNormalComputation, 0);

	// Setup Pipeline
	context->CSSetShaderResources(0, 1, &inputSRV);
	context->CSSetUnorderedAccessViews( 0, 1, &outputUAV, 0);
	context->CSSetConstantBuffers(0, 1, &m_constantBufferNormalComputation);
	context->CSSetShader(m_pComputeShaderNormalComputation, 0, 0);

	// Start Compute Shader
	unsigned int dimX = (unsigned int)ceil(((float)imageWidth)/m_blockSizeNormalComputation);
	unsigned int dimY = (unsigned int)ceil(((float)imageHeight)/m_blockSizeNormalComputation);
	context->Dispatch(dimX, dimY, 1);
	assert(dimX <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);
	assert(dimY <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);

	// De-Initialize Pipeline
	ID3D11ShaderResourceView* nullSAV[1] = { NULL };
	ID3D11UnorderedAccessView* nullUAV[1] = { NULL };
	ID3D11Buffer* nullCB[1] = { NULL };

	context->CSSetShaderResources(0, 1, nullSAV);
	context->CSSetUnorderedAccessViews(0, 1, nullUAV, 0);
	context->CSSetConstantBuffers(0, 1, nullCB);
	context->CSSetShader(0, 0, 0);

	return hr;
}

HRESULT DX11ImageHelper::applyCameraSpace( ID3D11DeviceContext* context, ID3D11ShaderResourceView* inputSRV, ID3D11UnorderedAccessView* outputUAV, unsigned int imageWidth, unsigned int imageHeight )
{
	HRESULT hr = S_OK;

	// Initialize Constant Buffer
	D3D11_MAPPED_SUBRESOURCE mappedResource;
	hr = context->Map(m_constantBufferCameraSpace, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
	if(FAILED(hr)) return hr;

	CBufferCameraSpace *cbuffer = (CBufferCameraSpace*)mappedResource.pData;
	cbuffer->imageWidth = (int)imageWidth;
	cbuffer->imageHeigth = (int)imageHeight;

	context->Unmap(m_constantBufferCameraSpace, 0);

	// Setup Pipeline
	context->CSSetShaderResources(1, 1, &inputSRV);
	context->CSSetUnorderedAccessViews(1, 1, &outputUAV, 0);
	context->CSSetConstantBuffers(0, 1, &m_constantBufferCameraSpace);
	context->CSSetShader(m_pComputeShaderCameraSpace, 0, 0);

	// Start Compute Shader
	unsigned int dimX = (unsigned int)ceil(((float)imageWidth)/m_blockSizeCameraSpace);
	unsigned int dimY = (unsigned int)ceil(((float)imageHeight)/m_blockSizeCameraSpace);
	context->Dispatch(dimX, dimY, 1);
	assert(dimX <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);
	assert(dimY <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);

	// De-Initialize Pipeline
	ID3D11ShaderResourceView* nullSAV[1] = { NULL };
	ID3D11UnorderedAccessView* nullUAV[1] = { NULL };
	ID3D11Buffer* nullCB[1] = { NULL };

	context->CSSetShaderResources(1, 1, nullSAV);
	context->CSSetUnorderedAccessViews(1, 1, nullUAV, 0);
	context->CSSetConstantBuffers(0, 1, nullCB);
	context->CSSetShader(0, 0, 0);

	return hr;
}

HRESULT DX11ImageHelper::applyCameraSpaceProjection( ID3D11DeviceContext* context, ID3D11ShaderResourceView* inputSRV, ID3D11UnorderedAccessView* outputUAV, unsigned int imageWidth, unsigned int imageHeight )
{
	HRESULT hr = S_OK;

	// Initialize Constant Buffer
	D3D11_MAPPED_SUBRESOURCE mappedResource;
	hr = context->Map(m_constantBufferCameraSpaceProjection, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
	if(FAILED(hr)) return hr;

	CBufferCameraSpaceProjection *cbuffer = (CBufferCameraSpaceProjection*)mappedResource.pData;
	cbuffer->imageWidth = (int)imageWidth;
	cbuffer->imageHeigth = (int)imageHeight;

	context->Unmap(m_constantBufferCameraSpaceProjection, 0);

	// Setup Pipeline
	context->CSSetShaderResources(0, 1, &inputSRV);
	context->CSSetUnorderedAccessViews( 0, 1, &outputUAV, 0);
	context->CSSetConstantBuffers(0, 1, &m_constantBufferCameraSpaceProjection);
	ID3D11Buffer* CBGlobalAppState = GlobalAppState::getInstance().MapAndGetConstantBuffer(context);
	context->CSSetConstantBuffers(8, 1, &CBGlobalAppState);
	context->CSSetShader(m_pComputeShaderCameraSpaceProjection, 0, 0);

	// Start Compute Shader
	unsigned int dimX = (unsigned int)ceil(((float)imageWidth)/m_blockSizeCameraSpaceProjection);
	unsigned int dimY = (unsigned int)ceil(((float)imageHeight)/m_blockSizeCameraSpaceProjection);
	context->Dispatch(dimX, dimY, 1);
	assert(dimX <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);
	assert(dimY <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);

	// De-Initialize Pipeline
	ID3D11ShaderResourceView* nullSAV[1] = { NULL };
	ID3D11UnorderedAccessView* nullUAV[1] = { NULL };
	ID3D11Buffer* nullCB[1] = { NULL };

	context->CSSetShaderResources(0, 1, nullSAV);
	context->CSSetUnorderedAccessViews(0, 1, nullUAV, 0);
	context->CSSetConstantBuffers(0, 1, nullCB);
	context->CSSetConstantBuffers(8, 1, nullCB);
	context->CSSetShader(0, 0, 0);

	return hr;
}

HRESULT DX11ImageHelper::StereoCameraSpaceProjection( ID3D11DeviceContext* context, ID3D11ShaderResourceView* inputSRV, ID3D11UnorderedAccessView* outputUAV, unsigned int imageWidth, unsigned int imageHeight )
{
	HRESULT hr = S_OK;

	// Initialize Constant Buffer
	D3D11_MAPPED_SUBRESOURCE mappedResource;
	hr = context->Map(m_constantBufferStereoCameraSpaceProjection, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
	if(FAILED(hr)) return hr;

	CBufferStereoCameraSpaceProjection *cbuffer = (CBufferStereoCameraSpaceProjection*)mappedResource.pData;
	cbuffer->imageWidth = (int)imageWidth;
	cbuffer->imageHeigth = (int)imageHeight;

	context->Unmap(m_constantBufferStereoCameraSpaceProjection, 0);

	// Setup Pipeline
	context->CSSetShaderResources(0, 1, &inputSRV);
	context->CSSetUnorderedAccessViews( 0, 1, &outputUAV, 0);
	context->CSSetConstantBuffers(0, 1, &m_constantBufferStereoCameraSpaceProjection);
	ID3D11Buffer* CBGlobalAppState = GlobalAppState::getInstance().MapAndGetConstantBuffer(context);
	context->CSSetConstantBuffers(8, 1, &CBGlobalAppState);
	context->CSSetShader(m_pComputeShaderStereoCameraSpaceProjection, 0, 0);

	// Start Compute Shader
	unsigned int dimX = (unsigned int)ceil(((float)imageWidth)/m_blockSizeStereoCameraSpaceProjection);
	unsigned int dimY = (unsigned int)ceil(((float)imageHeight)/m_blockSizeStereoCameraSpaceProjection);
	context->Dispatch(dimX, dimY, 1);
	assert(dimX <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);
	assert(dimY <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);

	// De-Initialize Pipeline
	ID3D11ShaderResourceView* nullSAV[1] = { NULL };
	ID3D11UnorderedAccessView* nullUAV[1] = { NULL };
	ID3D11Buffer* nullCB[1] = { NULL };

	context->CSSetShaderResources(0, 1, nullSAV);
	context->CSSetUnorderedAccessViews(0, 1, nullUAV, 0);
	context->CSSetConstantBuffers(0, 1, nullCB);
	context->CSSetConstantBuffers(8, 1, nullCB);
	context->CSSetShader(0, 0, 0);

	return hr;
}

HRESULT DX11ImageHelper::applyDepthMap( ID3D11DeviceContext* context, ID3D11ShaderResourceView* inputSRV, ID3D11UnorderedAccessView* outputUAV, unsigned int imageWidth, unsigned int imageHeight )
{
	HRESULT hr = S_OK;

	// Initialize Constant Buffer
	D3D11_MAPPED_SUBRESOURCE mappedResource;
	hr = context->Map(m_constantBufferDepthMap, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
	if(FAILED(hr)) return hr;

	CBufferDepthMap *cbuffer = (CBufferDepthMap*)mappedResource.pData;
	cbuffer->imageWidth = (int)imageWidth;
	cbuffer->imageHeigth = (int)imageHeight;

	context->Unmap(m_constantBufferDepthMap, 0);

	// Setup Pipeline
	context->CSSetShaderResources(0, 1, &inputSRV);
	context->CSSetUnorderedAccessViews( 0, 1, &outputUAV, 0);
	context->CSSetConstantBuffers(0, 1, &m_constantBufferDepthMap);
	ID3D11Buffer* CBGlobalAppState = GlobalAppState::getInstance().MapAndGetConstantBuffer(context);
	context->CSSetConstantBuffers(8, 1, &CBGlobalAppState);
	context->CSSetShader(m_pComputeShaderDepthMap, 0, 0);

	// Start Compute Shader
	unsigned int dimX = (unsigned int)ceil(((float)imageWidth)/m_blockSizeDepthMap);
	unsigned int dimY = (unsigned int)ceil(((float)imageHeight)/m_blockSizeDepthMap);
	context->Dispatch(dimX, dimY, 1);
	assert(dimX <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);
	assert(dimY <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);

	// De-Initialize Pipeline
	ID3D11ShaderResourceView* nullSAV[1] = { NULL };
	ID3D11UnorderedAccessView* nullUAV[1] = { NULL };
	ID3D11Buffer* nullCB[1] = { NULL };

	context->CSSetShaderResources(0, 1, nullSAV);
	context->CSSetUnorderedAccessViews(0, 1, nullUAV, 0);
	context->CSSetConstantBuffers(0, 1, nullCB);
	context->CSSetConstantBuffers(8, 1, nullCB);
	context->CSSetShader(0, 0, 0);

	return hr;
}

HRESULT DX11ImageHelper::applySSAOMap( ID3D11DeviceContext* context, ID3D11ShaderResourceView* inputSRV, ID3D11UnorderedAccessView* outputUAV, unsigned int imageWidth, unsigned int imageHeight )
{
	HRESULT hr = S_OK;

	// Initialize Constant Buffer
	D3D11_MAPPED_SUBRESOURCE mappedResource;
	hr = context->Map(m_constantBufferSSAOMap, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
	if(FAILED(hr)) return hr;

	CBufferSSAOMap *cbuffer = (CBufferSSAOMap*)mappedResource.pData;
	cbuffer->imageWidth = (int)imageWidth;
	cbuffer->imageHeigth = (int)imageHeight;

	memcpy(&cbuffer->rotationVectors[0], &m_randomRotations[0], 16*sizeof(float4));

	context->Unmap(m_constantBufferSSAOMap, 0);

	// Setup Pipeline
	context->CSSetShaderResources(0, 1, &inputSRV);
	context->CSSetUnorderedAccessViews( 0, 1, &outputUAV, 0);
	context->CSSetConstantBuffers(0, 1, &m_constantBufferSSAOMap);
	context->CSSetShader(m_pComputeShaderSSAOMap, 0, 0);

	// Start Compute Shader
	unsigned int dimX = (unsigned int)ceil(((float)imageWidth)/m_blockSizeSSAOMap);
	unsigned int dimY = (unsigned int)ceil(((float)imageHeight)/m_blockSizeSSAOMap);
	context->Dispatch(dimX, dimY, 1);
	assert(dimX <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);
	assert(dimY <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);

	// De-Initialize Pipeline
	ID3D11ShaderResourceView* nullSAV[1] = { NULL };
	ID3D11UnorderedAccessView* nullUAV[1] = { NULL };
	ID3D11Buffer* nullCB[1] = { NULL };

	context->CSSetShaderResources(0, 1, nullSAV);
	context->CSSetUnorderedAccessViews(0, 1, nullUAV, 0);
	context->CSSetConstantBuffers(0, 1, nullCB);
	context->CSSetShader(0, 0, 0);

	return hr;
}

HRESULT DX11ImageHelper::applyHSVDepth( ID3D11DeviceContext* context, ID3D11ShaderResourceView* inputSRV, ID3D11UnorderedAccessView* outputUAV, unsigned int imageWidth, unsigned int imageHeight )
{
	HRESULT hr = S_OK;

	// Initialize Constant Buffer
	D3D11_MAPPED_SUBRESOURCE mappedResource;
	hr = context->Map(m_constantBufferHSVDepth, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
	if(FAILED(hr)) return hr;

	CBufferHSVDepth *cbuffer = (CBufferHSVDepth*)mappedResource.pData;
	cbuffer->imageWidth = (int)imageWidth;
	cbuffer->imageHeigth = (int)imageHeight;

	context->Unmap(m_constantBufferHSVDepth, 0);

	// Setup Pipeline
	context->CSSetShaderResources(0, 1, &inputSRV);
	context->CSSetUnorderedAccessViews( 0, 1, &outputUAV, 0);
	context->CSSetConstantBuffers(0, 1, &m_constantBufferHSVDepth);
	ID3D11Buffer* CBGlobalAppState = GlobalAppState::getInstance().MapAndGetConstantBuffer(context);
	context->CSSetConstantBuffers(8, 1, &CBGlobalAppState);
	context->CSSetShader(m_pComputeShaderHSVDepth, 0, 0);

	// Start Compute Shader
	unsigned int dimX = (unsigned int)ceil(((float)imageWidth)/m_blockSizeHSVDepth);
	unsigned int dimY = (unsigned int)ceil(((float)imageHeight)/m_blockSizeHSVDepth);
	context->Dispatch(dimX, dimY, 1);
	assert(dimX <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);
	assert(dimY <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);

	// De-Initialize Pipeline
	ID3D11ShaderResourceView* nullSAV[1] = { NULL };
	ID3D11UnorderedAccessView* nullUAV[1] = { NULL };
	ID3D11Buffer* nullCB[1] = { NULL };

	context->CSSetShaderResources(0, 1, nullSAV);
	context->CSSetUnorderedAccessViews(0, 1, nullUAV, 0);
	context->CSSetConstantBuffers(0, 1, nullCB);
	context->CSSetConstantBuffers(8, 1, nullCB);
	context->CSSetShader(0, 0, 0);

	return hr;
}

HRESULT DX11ImageHelper::applyProjectiveCorrespondences( ID3D11DeviceContext* context, ID3D11ShaderResourceView* inputSRV, ID3D11ShaderResourceView* inputNormalsSRV, ID3D11ShaderResourceView* inputColorsSRV, ID3D11ShaderResourceView* targetSRV, ID3D11ShaderResourceView* targetNormalsSRV, ID3D11ShaderResourceView* targetColorsSRV, ID3D11UnorderedAccessView* outputUAV, ID3D11UnorderedAccessView* outputNormalUAV, const Eigen::Matrix4f& deltaTransform, unsigned int imageWidth, unsigned int imageHeight, float distThres, float normalThres, float levelFactor )
{
	HRESULT hr = S_OK;

	D3D11_MAPPED_SUBRESOURCE mappedResource;
	hr = context->Map(m_constantBufferProjectiveCorrespondences, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
	if(FAILED(hr)) return hr;

	Eigen::Matrix4f deltaTransformT = deltaTransform.transpose();

	CBufferProjectiveCorrespondences *cbuffer = (CBufferProjectiveCorrespondences*)mappedResource.pData;
	cbuffer->imageWidth = (int)imageWidth;
	cbuffer->imageHeigth = (int)imageHeight;
	cbuffer->distThres = distThres;
	cbuffer->normalThres = normalThres;
	memcpy(cbuffer->transform, deltaTransformT.data(), 16*sizeof(float));
	cbuffer->levelFactor = levelFactor;

	context->Unmap(m_constantBufferProjectiveCorrespondences, 0);

	// Setup Pipeline
	ID3D11ShaderResourceView* srvs[] = {targetSRV, targetNormalsSRV, targetColorsSRV, inputSRV, inputNormalsSRV, inputColorsSRV};
	context->CSSetShaderResources(0, 6, srvs);
	context->CSSetUnorderedAccessViews(0, 1, &outputUAV, 0);
	context->CSSetUnorderedAccessViews(1, 1, &outputNormalUAV, 0);
	context->CSSetConstantBuffers(0, 1, &m_constantBufferProjectiveCorrespondences);
	ID3D11Buffer* CBGlobalAppState = GlobalAppState::getInstance().MapAndGetConstantBuffer(context);
	context->CSSetConstantBuffers(8, 1, &CBGlobalAppState);
	context->CSSetShader(m_pComputeShaderProjectiveCorrespondences, 0, 0);

	// Start Compute Shader
	unsigned int dimX = (unsigned int)ceil(((float)imageWidth)/m_blockSizeProjectiveCorrespondences);
	unsigned int dimY = (unsigned int)ceil(((float)imageHeight)/m_blockSizeProjectiveCorrespondences);
	context->Dispatch(dimX, dimY, 1);
	assert(dimX <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);
	assert(dimY <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);

	// De-Initialize Pipeline
	ID3D11ShaderResourceView* nullSRV[] = { NULL, NULL, NULL, NULL, NULL, NULL };
	ID3D11UnorderedAccessView* nullUAV[] = { NULL, NULL };
	ID3D11Buffer* nullCB[1] = { NULL };

	context->CSSetShaderResources(0, 6, nullSRV);
	context->CSSetUnorderedAccessViews(0, 2, nullUAV, 0);
	context->CSSetConstantBuffers(0, 1, nullCB);
	context->CSSetConstantBuffers(8, 1, nullCB);
	context->CSSetShader(0, 0, 0);

	return hr;
}

HRESULT DX11ImageHelper::applySubSamp( ID3D11DeviceContext* context, ID3D11ShaderResourceView* inputSRV, ID3D11UnorderedAccessView* outputUAV, unsigned int targetImageWidth, unsigned int targetImageHeight )
{
	HRESULT hr = S_OK;

	// Initialize Constant Buffer
	D3D11_MAPPED_SUBRESOURCE mappedResource;
	hr = context->Map(m_constantBufferSubSamp, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
	V_RETURN(hr);

	CBufferSubSamp *cbuffer = (CBufferSubSamp*)mappedResource.pData;
	cbuffer->imageWidth = (int)targetImageWidth;
	cbuffer->imageHeigth = (int)targetImageHeight;

	context->Unmap(m_constantBufferSubSamp, 0);

	// Setup Pipeline
	context->CSSetShaderResources(0, 1, &inputSRV);
	context->CSSetUnorderedAccessViews( 0, 1, &outputUAV, 0);
	context->CSSetConstantBuffers(0, 1, &m_constantBufferSubSamp);
	context->CSSetShader(m_pComputeShaderSubSamp, 0, 0);

	// Start Compute Shader
	unsigned int dimX = (unsigned int)ceil(((float)targetImageWidth)/m_blockSizeSubSamp);
	unsigned int dimY = (unsigned int)ceil(((float)targetImageHeight)/m_blockSizeSubSamp);
	context->Dispatch(dimX, dimY, 1);
	assert(dimX <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);
	assert(dimY <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);

	// De-Initialize Pipeline
	ID3D11ShaderResourceView* nullSAV[1] = { NULL };
	ID3D11UnorderedAccessView* nullUAV[1] = { NULL };
	ID3D11Buffer* nullCB[1] = { NULL };

	context->CSSetShaderResources(0, 1, nullSAV);
	context->CSSetUnorderedAccessViews(0, 1, nullUAV, 0);
	context->CSSetConstantBuffers(0, 1, nullCB);
	context->CSSetShader(0, 0, 0);

	return hr;
}

HRESULT DX11ImageHelper::applyDownsampling( ID3D11DeviceContext* context, ID3D11ShaderResourceView* inputSRV, ID3D11UnorderedAccessView* outputUAV, unsigned int targetImageWidth, unsigned int targetImageHeight )
{
	HRESULT hr = S_OK;

	// Initialize Constant Buffer
	D3D11_MAPPED_SUBRESOURCE mappedResource;
	hr = context->Map(m_constantBufferDownsampling, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
	V_RETURN(hr);

	CBufferDS *cbuffer = (CBufferDS*)mappedResource.pData;
	cbuffer->imageWidth = (int)targetImageWidth;
	cbuffer->imageHeigth = (int)targetImageHeight;

	context->Unmap(m_constantBufferDownsampling, 0);

	// Setup Pipeline
	context->CSSetShaderResources(0, 1, &inputSRV);
	context->CSSetUnorderedAccessViews( 0, 1, &outputUAV, 0);
	context->CSSetConstantBuffers(0, 1, &m_constantBufferDownsampling);
	context->CSSetShader(m_pComputeShaderDownsampling, 0, 0);

	// Start Compute Shader
	unsigned int dimX = (unsigned int)ceil(((float)targetImageWidth)/m_blockSizeDownsampling);
	unsigned int dimY = (unsigned int)ceil(((float)targetImageHeight)/m_blockSizeDownsampling);
	context->Dispatch(dimX, dimY, 1);
	assert(dimX <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);
	assert(dimY <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);

	// De-Initialize Pipeline
	ID3D11ShaderResourceView* nullSAV[1] = { NULL };
	ID3D11UnorderedAccessView* nullUAV[1] = { NULL };
	ID3D11Buffer* nullCB[1] = { NULL };

	context->CSSetShaderResources(0, 1, nullSAV);
	context->CSSetUnorderedAccessViews(0, 1, nullUAV, 0);
	context->CSSetConstantBuffers(0, 1, nullCB);
	context->CSSetShader(0, 0, 0);

	return hr;
}

HRESULT DX11ImageHelper::applyBlockAveraging( ID3D11DeviceContext* context, ID3D11ShaderResourceView* inputSRV, ID3D11UnorderedAccessView* outputUAV, unsigned int imageWidth, unsigned int imageHeight, float sigmaD, float sigmaR )
{
	HRESULT hr = S_OK;

	// Initialize Constant Buffer
	D3D11_MAPPED_SUBRESOURCE mappedResource;
	V_RETURN(context->Map(m_constantBufferBlockAverage, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource));

	CBufferBlockAverage *cbuffer = (CBufferBlockAverage*)mappedResource.pData;
	cbuffer->imageWidth = (int)imageWidth;
	cbuffer->imageHeigth = (int)imageHeight;
	cbuffer->sigmaD = sigmaD;
	cbuffer->sigmaR = sigmaR;

	context->Unmap(m_constantBufferBlockAverage, 0);

	// Setup Pipeline
	context->CSSetShaderResources(0, 1, &inputSRV);
	context->CSSetUnorderedAccessViews(0, 1, &outputUAV, 0);
	context->CSSetConstantBuffers(0, 1, &m_constantBufferBlockAverage);
	context->CSSetShader(m_pComputeShaderBlockAverage, 0, 0);

	// Start Compute Shader
	unsigned int dimX = (unsigned int)ceil(((float)imageWidth)/m_blockSizeBlockAverage);
	unsigned int dimY = (unsigned int)ceil(((float)imageHeight)/m_blockSizeBlockAverage);
	context->Dispatch(dimX, dimY, 1);
	assert(dimX <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);
	assert(dimY <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);

	// De-Initialize Pipeline
	ID3D11ShaderResourceView* nullSAV[1] = { NULL };
	ID3D11UnorderedAccessView* nullUAV[1] = { NULL };
	ID3D11Buffer* nullCB[1] = { NULL };

	context->CSSetShaderResources(0, 1, nullSAV);
	context->CSSetUnorderedAccessViews(0, 1, nullUAV, 0);
	context->CSSetConstantBuffers(0, 1, nullCB);
	context->CSSetShader(0, 0, 0);

	return hr;
}

HRESULT DX11ImageHelper::applyCopy( ID3D11DeviceContext* context, ID3D11ShaderResourceView* inputSRV, ID3D11UnorderedAccessView* outputUAV, unsigned int imageWidth, unsigned int imageHeight )
{
	HRESULT hr = S_OK;

	// Initialize Constant Buffer
	D3D11_MAPPED_SUBRESOURCE mappedResource;
	hr = context->Map(m_constantBufferCopy, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
	V_RETURN(hr);

	CBufferCopy *cbuffer = (CBufferCopy*)mappedResource.pData;
	cbuffer->imageWidth = (int)imageWidth;
	cbuffer->imageHeigth = (int)imageHeight;

	context->Unmap(m_constantBufferCopy, 0);

	// Setup Pipeline
	context->CSSetShaderResources(0, 1, &inputSRV);
	context->CSSetUnorderedAccessViews( 0, 1, &outputUAV, 0);
	context->CSSetConstantBuffers(0, 1, &m_constantBufferCopy);
	context->CSSetShader(m_pComputeShaderCopy, 0, 0);

	// Start Compute Shader
	unsigned int dimX = (unsigned int)ceil(((float)imageWidth)/m_blockSizeCopy);
	unsigned int dimY = (unsigned int)ceil(((float)imageHeight)/m_blockSizeCopy);
	context->Dispatch(dimX, dimY, 1);
	assert(dimX <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);
	assert(dimY <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);

	// De-Initialize Pipeline
	ID3D11ShaderResourceView* nullSAV[1] = { NULL };
	ID3D11UnorderedAccessView* nullUAV[1] = { NULL };
	ID3D11Buffer* nullCB[1] = { NULL };

	context->CSSetShaderResources(0, 1, nullSAV);
	context->CSSetUnorderedAccessViews(0, 1, nullUAV, 0);
	context->CSSetConstantBuffers(0, 1, nullCB);
	context->CSSetShader(0, 0, 0);

	return hr;
}

HRESULT DX11ImageHelper::applyErode( ID3D11DeviceContext* context, ID3D11ShaderResourceView* inputSRV, ID3D11ShaderResourceView* inputColorSRV, ID3D11UnorderedAccessView* outputUAV, float distThres, int stencilSize, unsigned int imageWidth, unsigned int imageHeight )
{
	HRESULT hr = S_OK;

	// Initialize Constant Buffer
	D3D11_MAPPED_SUBRESOURCE mappedResource;
	V_RETURN(context->Map(m_constantBufferErode, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource));
	CBufferErode *cbuffer = (CBufferErode*)mappedResource.pData;
	cbuffer->imageWidth = (int)imageWidth;
	cbuffer->imageHeigth = (int)imageHeight;
	cbuffer->distThres = distThres;
	cbuffer->stencilSize = stencilSize;
	context->Unmap(m_constantBufferErode, 0);

	// Setup Pipeline
	context->CSSetShaderResources(0, 1, &inputSRV);
	context->CSSetShaderResources(1, 1, &inputColorSRV);
	context->CSSetUnorderedAccessViews( 0, 1, &outputUAV, 0);
	context->CSSetConstantBuffers(0, 1, &m_constantBufferErode);
	context->CSSetShader(m_pComputeShaderErode, 0, 0);

	// Start Compute Shader
	unsigned int dimX = (unsigned int)ceil(((float)imageWidth)/m_blockSizeErode);
	unsigned int dimY = (unsigned int)ceil(((float)imageHeight)/m_blockSizeErode);
	context->Dispatch(dimX, dimY, 1);
	assert(dimX <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);
	assert(dimY <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);

	// De-Initialize Pipeline
	ID3D11ShaderResourceView* nullSAV[] = { NULL, NULL};
	ID3D11UnorderedAccessView* nullUAV[] = { NULL };
	ID3D11Buffer* nullCB[] = { NULL };

	context->CSSetShaderResources(0, 2, nullSAV);
	context->CSSetUnorderedAccessViews(0, 1, nullUAV, 0);
	context->CSSetConstantBuffers(0, 1, nullCB);
	context->CSSetShader(0, 0, 0);

	return hr;
}

void DX11ImageHelper::OnD3D11DestroyDevice()
{
	/////////////////////////////////////////////////////
	// Bilateral Completion
	/////////////////////////////////////////////////////

	SAFE_RELEASE(m_constantBufferBilateralCompletion);
	SAFE_RELEASE(m_pComputeShaderBilateralCompletion);
	SAFE_RELEASE(m_pComputeShaderBilateralCompletionWithColor);

	/////////////////////////////////////////////////////
	// Bilateral Filter
	/////////////////////////////////////////////////////

	SAFE_RELEASE(m_constantBufferBilateralFilter);
	SAFE_RELEASE(m_pComputeShaderBilateralFilter);

	/////////////////////////////////////////////////////
	// Bilateral Filter
	/////////////////////////////////////////////////////

	SAFE_RELEASE(m_constantBufferBilateralFilterForSSAO);
	SAFE_RELEASE(m_pComputeShaderBilateralFilterForSSAO);

	/////////////////////////////////////////////////////
	// Bilateral Filter Approximation
	/////////////////////////////////////////////////////

	SAFE_RELEASE(m_constantBufferBilateralFilterApprox);
	SAFE_RELEASE(m_pComputeShaderBilateralFilterApprox);

	/////////////////////////////////////////////////////
	// Bilateral Filter 4F
	/////////////////////////////////////////////////////

	SAFE_RELEASE(m_constantBufferBilateralFilter4F);
	SAFE_RELEASE(m_pComputeShaderBilateralFilter4F);

	/////////////////////////////////////////////////////
	// Normal Computation
	/////////////////////////////////////////////////////

	SAFE_RELEASE(m_constantBufferNormalComputation);
	SAFE_RELEASE(m_pComputeShaderNormalComputation);

	/////////////////////////////////////////////////////
	// Camera Space To Dept Map
	/////////////////////////////////////////////////////

	SAFE_RELEASE(m_constantBufferCameraSpace);
	SAFE_RELEASE(m_pComputeShaderCameraSpace);

	/////////////////////////////////////////////////////
	// Compute Depth Maps
	/////////////////////////////////////////////////////

	SAFE_RELEASE(m_constantBufferDepthMap);
	SAFE_RELEASE(m_pComputeShaderDepthMap);

	/////////////////////////////////////////////////////
	// Compute SSAO Map
	/////////////////////////////////////////////////////

	SAFE_RELEASE(m_constantBufferSSAOMap);
	SAFE_RELEASE(m_pComputeShaderSSAOMap);

	/////////////////////////////////////////////////////
	// Compute HSV Depth Maps
	/////////////////////////////////////////////////////

	SAFE_RELEASE(m_constantBufferHSVDepth);
	SAFE_RELEASE(m_pComputeShaderHSVDepth);

	/////////////////////////////////////////////////////
	// Camera Space Projection
	/////////////////////////////////////////////////////

	SAFE_RELEASE(m_constantBufferCameraSpaceProjection);
	SAFE_RELEASE(m_pComputeShaderCameraSpaceProjection);

	/////////////////////////////////////////////////////
	// Stereo Mask Camera Space Projection
	/////////////////////////////////////////////////////

	SAFE_RELEASE(m_constantBufferStereoCameraSpaceProjection);
	SAFE_RELEASE(m_pComputeShaderStereoCameraSpaceProjection);

	/////////////////////////////////////////////////////
	// Projective Correspondences
	/////////////////////////////////////////////////////

	SAFE_RELEASE(m_constantBufferProjectiveCorrespondences);
	SAFE_RELEASE(m_pComputeShaderProjectiveCorrespondences);

	/////////////////////////////////////////////////////
	// Subsample
	/////////////////////////////////////////////////////

	SAFE_RELEASE(m_constantBufferSubSamp);
	SAFE_RELEASE(m_pComputeShaderSubSamp);

	/////////////////////////////////////////////////////
	// Downsampling
	/////////////////////////////////////////////////////

	SAFE_RELEASE(m_constantBufferDownsampling);
	SAFE_RELEASE(m_pComputeShaderDownsampling);

	/////////////////////////////////////////////////////
	// Depth aware block averaging
	/////////////////////////////////////////////////////

	SAFE_RELEASE(m_constantBufferBlockAverage);
	SAFE_RELEASE(m_pComputeShaderBlockAverage);

	/////////////////////////////////////////////////////
	// Copy
	/////////////////////////////////////////////////////

	SAFE_RELEASE(m_constantBufferCopy);
	SAFE_RELEASE(m_pComputeShaderCopy);

	/////////////////////////////////////////////////////
	// Erode
	/////////////////////////////////////////////////////

	SAFE_RELEASE(m_constantBufferErode);
	SAFE_RELEASE(m_pComputeShaderErode);
}
