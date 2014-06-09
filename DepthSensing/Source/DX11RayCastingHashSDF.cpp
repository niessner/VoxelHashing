#include "stdafx.h"

#include "DX11RayCastingHashSDF.h"

#include "GlobalAppState.h"

unsigned int DX11RayCastingHashSDF::m_blockSize = 16;

ID3D11ComputeShader* DX11RayCastingHashSDF::m_pComputeShader = NULL;
ID3D11Buffer* DX11RayCastingHashSDF::m_constantBuffer = NULL;

ID3D11Texture2D* DX11RayCastingHashSDF::m_pOutputImage2D = NULL;
ID3D11ShaderResourceView* DX11RayCastingHashSDF::m_pOutputImage2DSRV = NULL;
ID3D11UnorderedAccessView* DX11RayCastingHashSDF::m_pOutputImage2DUAV = NULL;

ID3D11Texture2D* DX11RayCastingHashSDF::m_pSSAOMap = NULL;
ID3D11ShaderResourceView* DX11RayCastingHashSDF::m_pSSAOMapSRV = NULL;
ID3D11UnorderedAccessView* DX11RayCastingHashSDF::m_pSSAOMapUAV = NULL;

ID3D11Texture2D* DX11RayCastingHashSDF::m_pSSAOMapFiltered = NULL;
ID3D11ShaderResourceView* DX11RayCastingHashSDF::m_pSSAOMapFilteredSRV = NULL;
ID3D11UnorderedAccessView* DX11RayCastingHashSDF::m_pSSAOMapFilteredUAV = NULL;

// Output
ID3D11Texture2D* DX11RayCastingHashSDF::s_pColors = NULL;
ID3D11ShaderResourceView* DX11RayCastingHashSDF::s_pColorsSRV = NULL;
ID3D11UnorderedAccessView* DX11RayCastingHashSDF::s_pColorsUAV = NULL;

ID3D11Texture2D* DX11RayCastingHashSDF::s_pPositions = NULL;
ID3D11ShaderResourceView* DX11RayCastingHashSDF::s_pPositionsSRV = NULL;
ID3D11UnorderedAccessView* DX11RayCastingHashSDF::s_pPositionsUAV = NULL;

ID3D11Texture2D* DX11RayCastingHashSDF::s_pNormals = NULL;
ID3D11ShaderResourceView* DX11RayCastingHashSDF::s_pNormalsSRV = NULL;
ID3D11UnorderedAccessView* DX11RayCastingHashSDF::s_pNormalsUAV = NULL;

// Ray Interval	
ID3D11Buffer* DX11RayCastingHashSDF::s_ConstantBufferSplatting = NULL;

ID3D11VertexShader*	DX11RayCastingHashSDF::s_pVertexShaderSplatting = NULL;
ID3D11GeometryShader* DX11RayCastingHashSDF::s_pGeometryShaderSplatting = NULL;
ID3D11PixelShader* DX11RayCastingHashSDF::s_pPixelShaderSplatting = NULL;

ID3D11Texture2D* DX11RayCastingHashSDF::s_pDepthStencilSplattingMin = NULL;
ID3D11DepthStencilView*	DX11RayCastingHashSDF::s_pDepthStencilSplattingMinDSV = NULL;
ID3D11ShaderResourceView* DX11RayCastingHashSDF::s_pDepthStencilSplattingMinSRV = NULL;

ID3D11Texture2D* DX11RayCastingHashSDF::s_pDepthStencilSplattingMax = NULL;
ID3D11DepthStencilView*	DX11RayCastingHashSDF::s_pDepthStencilSplattingMaxDSV = NULL;
ID3D11ShaderResourceView* DX11RayCastingHashSDF::s_pDepthStencilSplattingMaxSRV = NULL;

ID3D11DepthStencilState* DX11RayCastingHashSDF::s_pDepthStencilStateSplattingMin = NULL;
ID3D11DepthStencilState* DX11RayCastingHashSDF::s_pDepthStencilStateSplattingMax = NULL;

ID3D11BlendState* DX11RayCastingHashSDF::s_pBlendStateDefault = NULL;
ID3D11BlendState* DX11RayCastingHashSDF::s_pBlendStateColorWriteDisabled = NULL;

ID3D11RasterizerState* DX11RayCastingHashSDF::s_pRastState = NULL;

// Stereo
ID3D11Texture2D* DX11RayCastingHashSDF::m_pOutputImage2DStereo = NULL;
ID3D11Texture2D* DX11RayCastingHashSDF::m_pSSAOMapStereo = NULL;
ID3D11Texture2D* DX11RayCastingHashSDF::m_pSSAOMapFilteredStereo = NULL;

ID3D11ShaderResourceView* DX11RayCastingHashSDF::m_pOutputImage2DStereoSRV = NULL;
ID3D11ShaderResourceView* DX11RayCastingHashSDF::m_pSSAOMapStereoSRV = NULL;
ID3D11ShaderResourceView* DX11RayCastingHashSDF::m_pSSAOMapFilteredStereoSRV = NULL;

ID3D11UnorderedAccessView* DX11RayCastingHashSDF::m_pOutputImage2DStereoUAV = NULL;
ID3D11UnorderedAccessView* DX11RayCastingHashSDF::m_pSSAOMapStereoUAV = NULL;
ID3D11UnorderedAccessView* DX11RayCastingHashSDF::m_pSSAOMapFilteredStereoUAV = NULL;

ID3D11Texture2D* DX11RayCastingHashSDF::s_pColorsStereo = NULL;
ID3D11ShaderResourceView* DX11RayCastingHashSDF::s_pColorsStereoSRV = NULL;
ID3D11UnorderedAccessView* DX11RayCastingHashSDF::s_pColorsStereoUAV = NULL;

ID3D11Texture2D* DX11RayCastingHashSDF::s_pPositionsStereo = NULL;
ID3D11ShaderResourceView* DX11RayCastingHashSDF::s_pPositionsStereoSRV = NULL;
ID3D11UnorderedAccessView* DX11RayCastingHashSDF::s_pPositionsStereoUAV = NULL;

ID3D11Texture2D* DX11RayCastingHashSDF::s_pNormalsStereo = NULL;
ID3D11ShaderResourceView* DX11RayCastingHashSDF::s_pNormalsStereoSRV = NULL;
ID3D11UnorderedAccessView* DX11RayCastingHashSDF::s_pNormalsStereoUAV = NULL;

ID3D11Texture2D* DX11RayCastingHashSDF::s_pDepthStencilSplattingMinStereo = NULL;
ID3D11DepthStencilView*	DX11RayCastingHashSDF::s_pDepthStencilSplattingMinDSVStereo = NULL;
ID3D11ShaderResourceView* DX11RayCastingHashSDF::s_pDepthStencilSplattingMinSRVStereo = NULL;

ID3D11Texture2D* DX11RayCastingHashSDF::s_pDepthStencilSplattingMaxStereo = NULL;
ID3D11DepthStencilView* DX11RayCastingHashSDF::s_pDepthStencilSplattingMaxDSVStereo = NULL;
ID3D11ShaderResourceView* DX11RayCastingHashSDF::s_pDepthStencilSplattingMaxSRVStereo = NULL;

Timer DX11RayCastingHashSDF::m_timer;

HRESULT DX11RayCastingHashSDF::OnD3D11CreateDevice( ID3D11Device* pd3dDevice )
{
	HRESULT hr = S_OK;

	V_RETURN(initialize(pd3dDevice));

	return  hr;
}

void DX11RayCastingHashSDF::OnD3D11DestroyDevice()
{
	destroy();
}

HRESULT DX11RayCastingHashSDF::Render( ID3D11DeviceContext* context, ID3D11ShaderResourceView* hash, ID3D11ShaderResourceView* hashCompact, ID3D11ShaderResourceView* sdfBlocksSDF, ID3D11ShaderResourceView* sdfBlocksRGBW, unsigned int hashNumValidBuckets, unsigned int renderTargetWidth, unsigned int renderTargetHeight, const mat4f* lastRigidTransform, ID3D11Buffer* CBsceneRepSDF )
{
	return RenderToTexture( context, hash, hashCompact, sdfBlocksSDF, sdfBlocksRGBW, hashNumValidBuckets, renderTargetWidth, renderTargetHeight, lastRigidTransform, CBsceneRepSDF,
		s_pDepthStencilSplattingMinSRV, s_pDepthStencilSplattingMaxSRV,
		s_pDepthStencilSplattingMinDSV, s_pDepthStencilSplattingMaxDSV,
		m_pOutputImage2DSRV, m_pOutputImage2DUAV,
		s_pPositionsSRV, s_pPositionsUAV,
		s_pColorsUAV,
		s_pNormalsSRV, s_pNormalsUAV,
		m_pSSAOMapSRV, m_pSSAOMapUAV, m_pSSAOMapFilteredUAV);
}


HRESULT DX11RayCastingHashSDF::RenderStereo( ID3D11DeviceContext* context, ID3D11ShaderResourceView* hash, ID3D11ShaderResourceView* hashCompact, ID3D11ShaderResourceView* sdfBlocksSDF, ID3D11ShaderResourceView* sdfBlocksRGBW, unsigned int hashNumValidBuckets, unsigned int renderTargetWidth, unsigned int renderTargetHeight, const mat4f* lastRigidTransform, ID3D11Buffer* CBsceneRepSDF )
{
	HRESULT hr = S_OK;

	if(GlobalAppState::getInstance().s_stereoEnabled)
	{
		GlobalAppState::getInstance().s_currentlyInStereoMode = true;

		D3D11_VIEWPORT vp;
		vp.Width = (float)renderTargetWidth;
		vp.Height = (float)renderTargetHeight;
		vp.MinDepth = 0;
		vp.MaxDepth = 1;
		vp.TopLeftX = 0;
		vp.TopLeftY = 0;

		context->RSSetViewports( 1, &vp );


		static unsigned int whichShot = 0;

		bool useColor = false;
		float eyeSep = 0.08f;
		float C = -2.0f;
		float aspect = 16.0f/9.0f;
		float fovY = 45.0f;

		{
			mat4f t; t.setTranslation(-eyeSep/2.0f, 0.0, 0.0);
			mat4f lastRigidTransformNew = (*lastRigidTransform)*t;
			GlobalAppState::getInstance().StereoCameraFrustum(GlobalAppState::getInstance().s_intrinsicsStereo, GlobalAppState::getInstance().s_intrinsicsInvStereo, GlobalAppState::getInstance().s_worldToCamStereo, GlobalAppState::getInstance().s_camToWorldStereo, C, eyeSep, aspect, fovY, 0.2f, 8.0f, (float)GlobalAppState::getInstance().s_windowWidthStereo, (float)GlobalAppState::getInstance().s_windowHeightStereo, true, lastRigidTransformNew);

			mat4f tOther; tOther.setTranslation(eyeSep/2.0f, 0.0, 0.0);
			mat4f lastRigidTransformNewOther = (*lastRigidTransform)*tOther;
			GlobalAppState::getInstance().StereoCameraFrustum(GlobalAppState::getInstance().s_intrinsicsStereoOther, GlobalAppState::getInstance().s_intrinsicsInvStereoOther, GlobalAppState::getInstance().s_worldToCamStereoOther, GlobalAppState::getInstance().s_camToWorldStereoOther, C, eyeSep, aspect, fovY, 0.2f, 8.0f, (float)GlobalAppState::getInstance().s_windowWidthStereo, (float)GlobalAppState::getInstance().s_windowHeightStereo, false, lastRigidTransformNewOther);

			V_RETURN(RenderToTexture(context, hash, hashCompact, sdfBlocksSDF, sdfBlocksRGBW, hashNumValidBuckets, renderTargetWidth, renderTargetHeight, &lastRigidTransformNew, CBsceneRepSDF,
				s_pDepthStencilSplattingMinSRVStereo, s_pDepthStencilSplattingMaxSRVStereo,
				s_pDepthStencilSplattingMinDSVStereo, s_pDepthStencilSplattingMaxDSVStereo,
				m_pOutputImage2DStereoSRV, m_pOutputImage2DStereoUAV,
				s_pPositionsStereoSRV, s_pPositionsStereoUAV,
				s_pColorsStereoUAV,
				s_pNormalsStereoSRV, s_pNormalsStereoUAV,
				m_pSSAOMapStereoSRV, m_pSSAOMapStereoUAV, m_pSSAOMapFilteredStereoUAV));

			DX11PhongLighting::renderStereo(context, s_pPositionsStereoSRV, s_pNormalsStereoSRV, s_pColorsStereoSRV, m_pSSAOMapStereoSRV, useColor, false);

			std::stringstream toNumber; toNumber << whichShot;
			unsigned int numZeros = (unsigned int)(5 - toNumber.str().length());					

			wchar_t sz[200];
			unsigned int j = swprintf_s(sz, 200, L"movie\\stereo_left.");
			for (unsigned int i = 0; i < numZeros; i++) j += swprintf_s(sz + j, 200 - j, L"0");
			swprintf_s(sz + j, 200 - j, L"%d.bmp", whichShot);

			V_RETURN(D3DX11SaveTextureToFile(context, DX11PhongLighting::getStereoImage(), D3DX11_IFF_BMP, sz));
		}

		{
			mat4f t; t.setTranslation(eyeSep/2.0f, 0.0, 0.0);
			mat4f lastRigidTransformNew = (*lastRigidTransform)*t;
			GlobalAppState::getInstance().StereoCameraFrustum(GlobalAppState::getInstance().s_intrinsicsStereo, GlobalAppState::getInstance().s_intrinsicsInvStereo, GlobalAppState::getInstance().s_worldToCamStereo, GlobalAppState::getInstance().s_camToWorldStereo, C, eyeSep, aspect, fovY, 0.2f, 8.0f, (float)GlobalAppState::getInstance().s_windowWidthStereo, (float)GlobalAppState::getInstance().s_windowHeightStereo, false, lastRigidTransformNew);

			mat4f tOther; tOther.setTranslation(-eyeSep/2.0f, 0.0, 0.0);
			mat4f lastRigidTransformNewOther = (*lastRigidTransform)*tOther;
			GlobalAppState::getInstance().StereoCameraFrustum(GlobalAppState::getInstance().s_intrinsicsStereoOther, GlobalAppState::getInstance().s_intrinsicsInvStereoOther, GlobalAppState::getInstance().s_worldToCamStereoOther, GlobalAppState::getInstance().s_camToWorldStereoOther, C, eyeSep, aspect, fovY, 0.2f, 8.0f, (float)GlobalAppState::getInstance().s_windowWidthStereo, (float)GlobalAppState::getInstance().s_windowHeightStereo, true, lastRigidTransformNewOther);

			V_RETURN(RenderToTexture(context, hash, hashCompact, sdfBlocksSDF, sdfBlocksRGBW, hashNumValidBuckets, renderTargetWidth, renderTargetHeight, &lastRigidTransformNew, CBsceneRepSDF,
				s_pDepthStencilSplattingMinSRVStereo, s_pDepthStencilSplattingMaxSRVStereo,
				s_pDepthStencilSplattingMinDSVStereo, s_pDepthStencilSplattingMaxDSVStereo,
				m_pOutputImage2DStereoSRV, m_pOutputImage2DStereoUAV,
				s_pPositionsStereoSRV, s_pPositionsStereoUAV,
				s_pColorsStereoUAV,
				s_pNormalsStereoSRV, s_pNormalsStereoUAV,
				m_pSSAOMapStereoSRV, m_pSSAOMapStereoUAV, m_pSSAOMapFilteredStereoUAV));

			DX11PhongLighting::renderStereo(context, s_pPositionsStereoSRV, s_pNormalsStereoSRV, s_pColorsStereoSRV, m_pSSAOMapStereoSRV, useColor, false);

			std::stringstream toNumber; toNumber << whichShot;
			unsigned int numZeros = (unsigned int)(5 - toNumber.str().length());					

			wchar_t sz[200];
			unsigned int j = swprintf_s(sz, 200, L"movie\\stereo_right.");
			for (unsigned int i = 0; i < numZeros; i++) j += swprintf_s(sz + j, 200 - j, L"0");
			swprintf_s(sz + j, 200 - j, L"%d.bmp", whichShot);

			V_RETURN(D3DX11SaveTextureToFile(context, DX11PhongLighting::getStereoImage(), D3DX11_IFF_BMP, sz));
		}

		whichShot++;

		vp.Width = (float)GlobalAppState::getInstance().s_windowWidth;
		vp.Height = (float)GlobalAppState::getInstance().s_windowHeight;
		vp.MinDepth = 0;
		vp.MaxDepth = 1;
		vp.TopLeftX = 0;
		vp.TopLeftY = 0;

		context->RSSetViewports( 1, &vp );

		GlobalAppState::getInstance().s_currentlyInStereoMode = false;
	}

	return hr;
}


HRESULT DX11RayCastingHashSDF::RenderToTexture( ID3D11DeviceContext* context, ID3D11ShaderResourceView* hash, ID3D11ShaderResourceView* hashCompact, ID3D11ShaderResourceView* sdfBlocksSDF, ID3D11ShaderResourceView* sdfBlocksRGBW, unsigned int hashNumValidBuckets, unsigned int renderTargetWidth, unsigned int renderTargetHeight, const mat4f* lastRigidTransform, ID3D11Buffer* CBsceneRepSDF, ID3D11ShaderResourceView* pDepthStencilSplattingMinSRV, ID3D11ShaderResourceView* pDepthStencilSplattingMaxSRV, ID3D11DepthStencilView* pDepthStencilSplattingMinDSV, ID3D11DepthStencilView* pDepthStencilSplattingMaxDSV, ID3D11ShaderResourceView* pOutputImage2DSRV, ID3D11UnorderedAccessView* pOutputImage2DUAV, ID3D11ShaderResourceView* pPositionsSRV, ID3D11UnorderedAccessView* pPositionsUAV, ID3D11UnorderedAccessView* pColorsUAV, ID3D11ShaderResourceView* pNormalsSRV, ID3D11UnorderedAccessView* pNormalsUAV, ID3D11ShaderResourceView* pSSAOMapSRV, ID3D11UnorderedAccessView* pSSAOMapUAV, ID3D11UnorderedAccessView* pSSAOMapFilteredUAV )
{
	HRESULT hr = S_OK;

	// Splat Ray Interval Images
	if(GlobalAppState::getInstance().s_enableMultiLayerSplatting)
	{
		// TODO !!! adapt to stereo setup !!!
		//V_RETURN(DX11RayMarchingStepsSplatting::rayMarchingStepsSplatting(context, hashCompact, sdfBlocksSDF, sdfBlocksRGBW, hashNumValidBuckets, lastRigidTransform, renderTargetWidth, renderTargetHeight, CBsceneRepSDF));
	}
	else
	{
		V_RETURN(rayIntervalSplattingRenderToTexture(context, hashCompact, hashNumValidBuckets, lastRigidTransform, renderTargetWidth, renderTargetHeight, CBsceneRepSDF, pDepthStencilSplattingMinDSV, pDepthStencilSplattingMaxDSV));

		//V_RETURN(rayIntervalSplatting(context, hashCompact, hashNumValidBuckets, lastRigidTransform, renderTargetWidth, renderTargetHeight, CBsceneRepSDF));
		//DX11QuadDrawer::RenderQuad(context, s_pDepthStencilSplattingMinSRV);
	}

	// Initialize constant buffer
	D3D11_MAPPED_SUBRESOURCE mappedResource;
	V_RETURN(context->Map(s_ConstantBufferSplatting, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource));
	CBuffer *cbuffer = (CBuffer*)mappedResource.pData;
	cbuffer->m_RenderTargetWidth = renderTargetWidth;
	cbuffer->m_RenderTargetHeight = renderTargetHeight;
	mat4f worldToLastKinectSpace = lastRigidTransform->getInverse();
	memcpy(&cbuffer->m_ViewMat, &worldToLastKinectSpace, sizeof(mat4f));
	memcpy(&cbuffer->m_ViewMatInverse, lastRigidTransform, sizeof(mat4f));
	cbuffer->m_SplatMinimum = 1;				
	context->Unmap(s_ConstantBufferSplatting, 0);

	// Setup pipeline
	context->CSSetShaderResources(0, 1, &hash);
	context->CSSetShaderResources(1, 1, &sdfBlocksSDF);
	context->CSSetShaderResources(4, 1, &sdfBlocksRGBW);
	context->CSSetShaderResources(2, 1, &pDepthStencilSplattingMinSRV);
	context->CSSetShaderResources(3, 1, &pDepthStencilSplattingMaxSRV);
	ID3D11ShaderResourceView* srvPrefix = DX11RayMarchingStepsSplatting::getFragmentPrefixSumBufferSRV();
	ID3D11ShaderResourceView* srvSortedDepth = DX11RayMarchingStepsSplatting::getFragmentSortedDepthBufferSRV();
	context->CSSetShaderResources(5, 1, &srvPrefix);
	context->CSSetShaderResources(6, 1, &srvSortedDepth);
	context->CSSetUnorderedAccessViews(0, 1, &pOutputImage2DUAV, 0);
	context->CSSetUnorderedAccessViews(1, 1, &pColorsUAV, 0);
	context->CSSetUnorderedAccessViews(2, 1, &pNormalsUAV, 0);

	//context->CSSetConstantBuffers(0, 1, &m_constantBuffer);
	context->CSSetConstantBuffers(0, 1, &CBsceneRepSDF);
	context->CSSetConstantBuffers(1, 1, &s_ConstantBufferSplatting);
	ID3D11Buffer* CBGlobalAppState = GlobalAppState::getInstance().MapAndGetConstantBuffer(context);
	context->CSSetConstantBuffers(8, 1, &CBGlobalAppState);
	context->CSSetShader(m_pComputeShader, 0, 0);

	// Run compute shader
	unsigned int dimX = (unsigned int)ceil(((float)renderTargetWidth)/m_blockSize);
	unsigned int dimY = (unsigned int)ceil(((float)renderTargetHeight)/m_blockSize);

	// Start query for timing
	if(GlobalAppState::getInstance().s_timingsDetailledEnabled)
	{
		GlobalAppState::getInstance().WaitForGPU();
		m_timer.start();
	}

	context->Dispatch(dimX, dimY, 1);
	assert(dimX <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);
	assert(dimY <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);

	// Wait for query
	if(GlobalAppState::getInstance().s_timingsDetailledEnabled)
	{
		GlobalAppState::getInstance().WaitForGPU();
		TimingLog::totalTimeRayCast+=m_timer.getElapsedTimeMS();
		TimingLog::countRayCast++;
	}

	// Cleanup
	ID3D11ShaderResourceView* nullSRV[] = {NULL, NULL, NULL, NULL, NULL};
	ID3D11UnorderedAccessView* nullUAV[] = {NULL};
	ID3D11Buffer* nullB[] = {NULL, NULL};

	context->CSSetShaderResources(0, 5, nullSRV);
	context->CSSetShaderResources(5, 1, nullSRV);
	context->CSSetShaderResources(6, 1, nullSRV);
	context->CSSetUnorderedAccessViews(0, 1, nullUAV, 0);
	context->CSSetUnorderedAccessViews(1, 1, nullUAV, 0);
	context->CSSetUnorderedAccessViews(2, 1, nullUAV, 0);
	context->CSSetConstantBuffers(0, 2, nullB);
	context->CSSetConstantBuffers(8, 1, nullB);
	context->CSSetShader(0, 0, 0);

	// Output
	if(GlobalAppState::getInstance().s_currentlyInStereoMode)
	{
		DX11ImageHelper::StereoCameraSpaceProjection(context, pOutputImage2DSRV, pPositionsUAV, renderTargetWidth, renderTargetHeight);
	}
	else
	{
		DX11ImageHelper::applyCameraSpaceProjection(context, pOutputImage2DSRV, pPositionsUAV, renderTargetWidth, renderTargetHeight);
	}

	if(!GlobalAppState::getInstance().s_useGradients)
	{
		DX11ImageHelper::applyNormalComputation(context, pPositionsSRV, pNormalsUAV, renderTargetWidth, renderTargetHeight);
	}

	// Compute SSAO Maps
	DX11ImageHelper::applySSAOMap(context, pOutputImage2DSRV, pSSAOMapUAV, renderTargetWidth, renderTargetHeight);
	DX11ImageHelper::applyBilateralFilterForSSAO(context, pOutputImage2DSRV, pSSAOMapSRV, pSSAOMapFilteredUAV, renderTargetWidth, renderTargetHeight, 2.0f, 0.1f);

	return hr;
}

HRESULT DX11RayCastingHashSDF::initialize( ID3D11Device* pd3dDevice )
{
	HRESULT hr = S_OK;

	char SDFBLOCKSIZE[5];
	sprintf_s(SDFBLOCKSIZE, "%d", SDF_BLOCK_SIZE);

	char HANDLECOLLISIONS[5];
	sprintf_s(HANDLECOLLISIONS, "%d", GlobalAppState::getInstance().s_HANDLE_COLLISIONS);

	char BLOCK_SIZE[5];
	sprintf_s(BLOCK_SIZE, "%d", m_blockSize);

	D3D_SHADER_MACRO shaderDefines[] = {{"groupthreads", BLOCK_SIZE}, { "SDF_BLOCK_SIZE", SDFBLOCKSIZE }, {"HANDLE_COLLISIONS", HANDLECOLLISIONS }, {0}};
	D3D_SHADER_MACRO shaderDefinesWithout[] = {{"groupthreads", BLOCK_SIZE}, { "SDF_BLOCK_SIZE", SDFBLOCKSIZE }, {0}};


	D3D_SHADER_MACRO* validDefines = shaderDefines;
	if(GlobalAppState::getInstance().s_HANDLE_COLLISIONS == 0)
	{
		validDefines = shaderDefinesWithout;
	}

	ID3DBlob* pBlob = NULL;

	V_RETURN(CompileShaderFromFile(L"Shaders\\RayCastingHashSDF.hlsl", "renderCS", "cs_5_0", &pBlob, validDefines)); //, D3DCOMPILE_SKIP_VALIDATION | D3DCOMPILE_SKIP_OPTIMIZATION | D3DCOMPILE_PREFER_FLOW_CONTROL | D3DCOMPILE_OPTIMIZATION_LEVEL0));
	V_RETURN(pd3dDevice->CreateComputeShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &m_pComputeShader))
		SAFE_RELEASE(pBlob);


	// Create Output Image
	D3D11_TEXTURE2D_DESC depthTexDesc = {0};
	depthTexDesc.Width = GlobalAppState::getInstance().s_windowWidth;
	depthTexDesc.Height = GlobalAppState::getInstance().s_windowHeight;
	depthTexDesc.MipLevels = 1;
	depthTexDesc.ArraySize = 1;
	depthTexDesc.Format = DXGI_FORMAT_R32_FLOAT;
	depthTexDesc.SampleDesc.Count = 1;
	depthTexDesc.SampleDesc.Quality = 0;
	depthTexDesc.Usage = D3D11_USAGE_DEFAULT;
	depthTexDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
	depthTexDesc.CPUAccessFlags = 0;
	depthTexDesc.MiscFlags = 0;

	V_RETURN(pd3dDevice->CreateTexture2D(&depthTexDesc, NULL, &m_pOutputImage2D));
	V_RETURN(pd3dDevice->CreateTexture2D(&depthTexDesc, NULL, &m_pSSAOMap));
	V_RETURN(pd3dDevice->CreateTexture2D(&depthTexDesc, NULL, &m_pSSAOMapFiltered));

	//create shader resource views
	V_RETURN(pd3dDevice->CreateShaderResourceView(m_pOutputImage2D, NULL, &m_pOutputImage2DSRV));
	V_RETURN(pd3dDevice->CreateShaderResourceView(m_pSSAOMap, NULL, &m_pSSAOMapSRV));
	V_RETURN(pd3dDevice->CreateShaderResourceView(m_pSSAOMapFiltered, NULL, &m_pSSAOMapFilteredSRV));

	//create unordered access views
	V_RETURN(pd3dDevice->CreateUnorderedAccessView(m_pOutputImage2D, NULL, &m_pOutputImage2DUAV));
	V_RETURN(pd3dDevice->CreateUnorderedAccessView(m_pSSAOMap, NULL, &m_pSSAOMapUAV));
	V_RETURN(pd3dDevice->CreateUnorderedAccessView(m_pSSAOMapFiltered, NULL, &m_pSSAOMapFilteredUAV));


	D3D11_BUFFER_DESC bDesc;
	bDesc.BindFlags	= D3D11_BIND_CONSTANT_BUFFER;
	bDesc.Usage	= D3D11_USAGE_DYNAMIC;
	bDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	bDesc.MiscFlags	= 0;

	bDesc.ByteWidth	= sizeof(CBuffer);
	V_RETURN(pd3dDevice->CreateBuffer(&bDesc, NULL, &m_constantBuffer));

	//Output
	D3D11_TEXTURE2D_DESC descTex;
	ZeroMemory(&descTex, sizeof(D3D11_TEXTURE2D_DESC));
	descTex.Usage = D3D11_USAGE_DEFAULT;
	descTex.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
	descTex.CPUAccessFlags = 0;
	descTex.MiscFlags = 0;
	descTex.SampleDesc.Count = 1;
	descTex.SampleDesc.Quality = 0;
	descTex.ArraySize = 1;
	descTex.MipLevels = 1;
	descTex.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
	descTex.Width = GlobalAppState::getInstance().s_windowWidth;
	descTex.Height = GlobalAppState::getInstance().s_windowHeight;

	// Color
	V_RETURN(pd3dDevice->CreateTexture2D(&descTex, NULL, &s_pColors));
	V_RETURN(pd3dDevice->CreateShaderResourceView(s_pColors, NULL, &s_pColorsSRV));
	V_RETURN(pd3dDevice->CreateUnorderedAccessView(s_pColors, NULL, &s_pColorsUAV));

	// Position
	V_RETURN(pd3dDevice->CreateTexture2D(&descTex, NULL, &s_pPositions));
	V_RETURN(pd3dDevice->CreateShaderResourceView(s_pPositions, NULL, &s_pPositionsSRV));
	V_RETURN(pd3dDevice->CreateUnorderedAccessView(s_pPositions, NULL, &s_pPositionsUAV));

	// Normals
	V_RETURN(pd3dDevice->CreateTexture2D(&descTex, NULL, &s_pNormals));
	V_RETURN(pd3dDevice->CreateShaderResourceView(s_pNormals, NULL, &s_pNormalsSRV));
	V_RETURN(pd3dDevice->CreateUnorderedAccessView(s_pNormals, NULL, &s_pNormalsUAV));


	// Ray Interval
	V_RETURN(CompileShaderFromFile(L"Shaders\\RayIntervalSplatting.hlsl", "VS", "vs_5_0", &pBlob, validDefines));
	V_RETURN(pd3dDevice->CreateVertexShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &s_pVertexShaderSplatting));
	SAFE_RELEASE(pBlob);

	V_RETURN(CompileShaderFromFile(L"Shaders\\RayIntervalSplatting.hlsl", "GS", "gs_5_0", &pBlob, validDefines));
	V_RETURN(pd3dDevice->CreateGeometryShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &s_pGeometryShaderSplatting));
	SAFE_RELEASE(pBlob);

	V_RETURN(CompileShaderFromFile(L"Shaders\\RayIntervalSplatting.hlsl", "PS", "ps_5_0", &pBlob, validDefines));
	V_RETURN(pd3dDevice->CreatePixelShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &s_pPixelShaderSplatting));
	SAFE_RELEASE(pBlob);

	// Create Constant Buffers
	D3D11_BUFFER_DESC cbDesc;
	ZeroMemory(&cbDesc, sizeof(D3D11_BUFFER_DESC));
	cbDesc.Usage = D3D11_USAGE_DYNAMIC;
	cbDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	cbDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	cbDesc.MiscFlags = 0;
	cbDesc.ByteWidth = sizeof(CBuffer);

	V_RETURN(pd3dDevice->CreateBuffer(&cbDesc, NULL, &s_ConstantBufferSplatting));

	//for first pass
	ZeroMemory(&descTex, sizeof(D3D11_TEXTURE2D_DESC));
	descTex.Usage = D3D11_USAGE_DEFAULT;
	descTex.BindFlags = D3D11_BIND_DEPTH_STENCIL | D3D11_BIND_SHADER_RESOURCE;
	descTex.CPUAccessFlags = 0;
	descTex.MiscFlags = 0;
	descTex.SampleDesc.Count = 1;
	descTex.SampleDesc.Quality = 0;
	descTex.ArraySize = 1;
	descTex.MipLevels = 1;
	descTex.Format = DXGI_FORMAT_R32_TYPELESS;
	descTex.Width = GlobalAppState::getInstance().s_windowWidth;
	descTex.Height = GlobalAppState::getInstance().s_windowHeight;
	V_RETURN(pd3dDevice->CreateTexture2D(&descTex, NULL, &s_pDepthStencilSplattingMin));
	V_RETURN(pd3dDevice->CreateTexture2D(&descTex, NULL, &s_pDepthStencilSplattingMax));

	D3D11_DEPTH_STENCIL_VIEW_DESC descDSV;
	ZeroMemory( &descDSV, sizeof(D3D11_DEPTH_STENCIL_VIEW_DESC) );
	descDSV.Format = DXGI_FORMAT_D32_FLOAT;
	descDSV.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
	descDSV.Texture2D.MipSlice = 0;
	V_RETURN(pd3dDevice->CreateDepthStencilView(s_pDepthStencilSplattingMin, &descDSV, &s_pDepthStencilSplattingMinDSV));
	V_RETURN(pd3dDevice->CreateDepthStencilView(s_pDepthStencilSplattingMax, &descDSV, &s_pDepthStencilSplattingMaxDSV));

	D3D11_SHADER_RESOURCE_VIEW_DESC descSRV;
	ZeroMemory(&descSRV, sizeof(D3D11_SHADER_RESOURCE_VIEW_DESC));
	descSRV.Format = DXGI_FORMAT_R32_FLOAT;
	descSRV.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
	descSRV.Texture2D.MipLevels = 1;
	descSRV.Texture2D.MostDetailedMip = 0;
	V_RETURN(pd3dDevice->CreateShaderResourceView(s_pDepthStencilSplattingMin, &descSRV, &s_pDepthStencilSplattingMinSRV));
	V_RETURN(pd3dDevice->CreateShaderResourceView(s_pDepthStencilSplattingMax, &descSRV, &s_pDepthStencilSplattingMaxSRV));

	// Blend State
	D3D11_BLEND_DESC blendDesc;
	blendDesc.AlphaToCoverageEnable = false;
	blendDesc.IndependentBlendEnable = false;
	blendDesc.RenderTarget[0].BlendEnable = false;
	blendDesc.RenderTarget[0].SrcBlend = D3D11_BLEND_ONE;
	blendDesc.RenderTarget[0].DestBlend = D3D11_BLEND_ZERO;
	blendDesc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
	blendDesc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
	blendDesc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ZERO;
	blendDesc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
	blendDesc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
	pd3dDevice->CreateBlendState(&blendDesc, &s_pBlendStateDefault);

	blendDesc.RenderTarget[0].RenderTargetWriteMask = 0; // Disable Color Write
	pd3dDevice->CreateBlendState(&blendDesc, &s_pBlendStateColorWriteDisabled);

	// Depth Stencil
	D3D11_DEPTH_STENCIL_DESC stenDesc;
	ZeroMemory(&stenDesc, sizeof(D3D11_DEPTH_STENCIL_DESC));
	stenDesc.DepthEnable = true;
	stenDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
	stenDesc.DepthFunc =  D3D11_COMPARISON_LESS;
	stenDesc.StencilEnable = false;
	V_RETURN(pd3dDevice->CreateDepthStencilState(&stenDesc, &s_pDepthStencilStateSplattingMin))

		stenDesc.DepthFunc =  D3D11_COMPARISON_GREATER;
	V_RETURN(pd3dDevice->CreateDepthStencilState(&stenDesc, &s_pDepthStencilStateSplattingMax))

		// Rasterizer Stage
		D3D11_RASTERIZER_DESC rastDesc;
	ZeroMemory(&rastDesc, sizeof(D3D11_RASTERIZER_DESC));
	rastDesc.FillMode = D3D11_FILL_SOLID;
	rastDesc.CullMode = D3D11_CULL_NONE;
	rastDesc.FrontCounterClockwise = false;

	V_RETURN(pd3dDevice->CreateRasterizerState(&rastDesc, &s_pRastState))

		// Stereo render targets
		if(GlobalAppState::getInstance().s_stereoEnabled)
		{
			// Create Output Image
			D3D11_TEXTURE2D_DESC depthTexDesc = {0};
			depthTexDesc.Width = GlobalAppState::getInstance().s_windowWidthStereo;
			depthTexDesc.Height = GlobalAppState::getInstance().s_windowHeightStereo;
			depthTexDesc.MipLevels = 1;
			depthTexDesc.ArraySize = 1;
			depthTexDesc.Format = DXGI_FORMAT_R32_FLOAT;
			depthTexDesc.SampleDesc.Count = 1;
			depthTexDesc.SampleDesc.Quality = 0;
			depthTexDesc.Usage = D3D11_USAGE_DEFAULT;
			depthTexDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
			depthTexDesc.CPUAccessFlags = 0;
			depthTexDesc.MiscFlags = 0;

			V_RETURN(pd3dDevice->CreateTexture2D(&depthTexDesc, NULL, &m_pOutputImage2DStereo));
			V_RETURN(pd3dDevice->CreateTexture2D(&depthTexDesc, NULL, &m_pSSAOMapStereo));
			V_RETURN(pd3dDevice->CreateTexture2D(&depthTexDesc, NULL, &m_pSSAOMapFilteredStereo));

			//create shader resource views
			V_RETURN(pd3dDevice->CreateShaderResourceView(m_pOutputImage2DStereo, NULL, &m_pOutputImage2DStereoSRV));
			V_RETURN(pd3dDevice->CreateShaderResourceView(m_pSSAOMapStereo, NULL, &m_pSSAOMapStereoSRV));
			V_RETURN(pd3dDevice->CreateShaderResourceView(m_pSSAOMapFilteredStereo, NULL, &m_pSSAOMapFilteredStereoSRV));

			//create unordered access views
			V_RETURN(pd3dDevice->CreateUnorderedAccessView(m_pOutputImage2DStereo, NULL, &m_pOutputImage2DStereoUAV));
			V_RETURN(pd3dDevice->CreateUnorderedAccessView(m_pSSAOMapStereo, NULL, &m_pSSAOMapStereoUAV));
			V_RETURN(pd3dDevice->CreateUnorderedAccessView(m_pSSAOMapFilteredStereo, NULL, &m_pSSAOMapFilteredStereoUAV));

			//Output
			D3D11_TEXTURE2D_DESC descTex;
			ZeroMemory(&descTex, sizeof(D3D11_TEXTURE2D_DESC));
			descTex.Usage = D3D11_USAGE_DEFAULT;
			descTex.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
			descTex.CPUAccessFlags = 0;
			descTex.MiscFlags = 0;
			descTex.SampleDesc.Count = 1;
			descTex.SampleDesc.Quality = 0;
			descTex.ArraySize = 1;
			descTex.MipLevels = 1;
			descTex.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
			descTex.Width = GlobalAppState::getInstance().s_windowWidthStereo;
			descTex.Height = GlobalAppState::getInstance().s_windowHeightStereo;

			// Color
			V_RETURN(pd3dDevice->CreateTexture2D(&descTex, NULL, &s_pColorsStereo));
			V_RETURN(pd3dDevice->CreateShaderResourceView(s_pColorsStereo, NULL, &s_pColorsStereoSRV));
			V_RETURN(pd3dDevice->CreateUnorderedAccessView(s_pColorsStereo, NULL, &s_pColorsStereoUAV));

			// Position
			V_RETURN(pd3dDevice->CreateTexture2D(&descTex, NULL, &s_pPositionsStereo));
			V_RETURN(pd3dDevice->CreateShaderResourceView(s_pPositionsStereo, NULL, &s_pPositionsStereoSRV));
			V_RETURN(pd3dDevice->CreateUnorderedAccessView(s_pPositionsStereo, NULL, &s_pPositionsStereoUAV));

			// Normals
			V_RETURN(pd3dDevice->CreateTexture2D(&descTex, NULL, &s_pNormalsStereo));
			V_RETURN(pd3dDevice->CreateShaderResourceView(s_pNormalsStereo, NULL, &s_pNormalsStereoSRV));
			V_RETURN(pd3dDevice->CreateUnorderedAccessView(s_pNormalsStereo, NULL, &s_pNormalsStereoUAV));

			//for first pass
			ZeroMemory(&descTex, sizeof(D3D11_TEXTURE2D_DESC));
			descTex.Usage = D3D11_USAGE_DEFAULT;
			descTex.BindFlags = D3D11_BIND_DEPTH_STENCIL | D3D11_BIND_SHADER_RESOURCE;
			descTex.CPUAccessFlags = 0;
			descTex.MiscFlags = 0;
			descTex.SampleDesc.Count = 1;
			descTex.SampleDesc.Quality = 0;
			descTex.ArraySize = 1;
			descTex.MipLevels = 1;
			descTex.Format = DXGI_FORMAT_R32_TYPELESS;
			descTex.Width = GlobalAppState::getInstance().s_windowWidthStereo;
			descTex.Height = GlobalAppState::getInstance().s_windowHeightStereo;
			V_RETURN(pd3dDevice->CreateTexture2D(&descTex, NULL, &s_pDepthStencilSplattingMinStereo));
			V_RETURN(pd3dDevice->CreateTexture2D(&descTex, NULL, &s_pDepthStencilSplattingMaxStereo));

			D3D11_DEPTH_STENCIL_VIEW_DESC descDSV;
			ZeroMemory( &descDSV, sizeof(D3D11_DEPTH_STENCIL_VIEW_DESC) );
			descDSV.Format = DXGI_FORMAT_D32_FLOAT;
			descDSV.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
			descDSV.Texture2D.MipSlice = 0;
			V_RETURN(pd3dDevice->CreateDepthStencilView(s_pDepthStencilSplattingMinStereo, &descDSV, &s_pDepthStencilSplattingMinDSVStereo));
			V_RETURN(pd3dDevice->CreateDepthStencilView(s_pDepthStencilSplattingMaxStereo, &descDSV, &s_pDepthStencilSplattingMaxDSVStereo));

			D3D11_SHADER_RESOURCE_VIEW_DESC descSRV;
			ZeroMemory(&descSRV, sizeof(D3D11_SHADER_RESOURCE_VIEW_DESC));
			descSRV.Format = DXGI_FORMAT_R32_FLOAT;
			descSRV.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
			descSRV.Texture2D.MipLevels = 1;
			descSRV.Texture2D.MostDetailedMip = 0;
			V_RETURN(pd3dDevice->CreateShaderResourceView(s_pDepthStencilSplattingMinStereo, &descSRV, &s_pDepthStencilSplattingMinSRVStereo));
			V_RETURN(pd3dDevice->CreateShaderResourceView(s_pDepthStencilSplattingMaxStereo, &descSRV, &s_pDepthStencilSplattingMaxSRVStereo));
		}

		return  hr;
}

HRESULT DX11RayCastingHashSDF::rayIntervalSplatting( ID3D11DeviceContext* context, ID3D11ShaderResourceView* hash, unsigned int hashNumValidBuckets, const mat4f* lastRigidTransform, unsigned int renderTargetWidth, unsigned int renderTargetHeight, ID3D11Buffer* CBsceneRepSDF )
{
	return  rayIntervalSplattingRenderToTexture(	context, hash, hashNumValidBuckets, lastRigidTransform, renderTargetWidth, renderTargetHeight, CBsceneRepSDF,
		s_pDepthStencilSplattingMinDSV, s_pDepthStencilSplattingMaxDSV);
}

HRESULT DX11RayCastingHashSDF::rayIntervalSplattingRenderToTexture( ID3D11DeviceContext* context, ID3D11ShaderResourceView* hash, unsigned int hashNumValidBuckets, const mat4f* lastRigidTransform, unsigned int renderTargetWidth, unsigned int renderTargetHeight, ID3D11Buffer* CBsceneRepSDF, ID3D11DepthStencilView* pDepthStencilSplattingMinDSV, ID3D11DepthStencilView* pDepthStencilSplattingMaxDSV )
{
	HRESULT hr = S_OK;

	// Initialize constant buffer
	D3D11_MAPPED_SUBRESOURCE mappedResource;
	V_RETURN(context->Map(s_ConstantBufferSplatting, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource));
	CBuffer *cbuffer = (CBuffer*)mappedResource.pData;
	cbuffer->m_RenderTargetWidth = renderTargetWidth;
	cbuffer->m_RenderTargetHeight = renderTargetHeight;
	mat4f worldToLastKinectSpace = lastRigidTransform->getInverse();
	memcpy(&cbuffer->m_ViewMat, &worldToLastKinectSpace, sizeof(mat4f));
	memcpy(&cbuffer->m_ViewMatInverse, lastRigidTransform, sizeof(mat4f));
	cbuffer->m_SplatMinimum = 1;				
	context->Unmap(s_ConstantBufferSplatting, 0);

	// Setup Pipeline
	context->OMSetDepthStencilState(s_pDepthStencilStateSplattingMin, 0);
	context->OMSetBlendState(s_pBlendStateColorWriteDisabled, NULL, 0xffffffff);
	context->RSSetState(s_pRastState);

	context->ClearDepthStencilView(pDepthStencilSplattingMinDSV, D3D11_CLEAR_DEPTH, 1.0f, 0);
	context->OMSetRenderTargets(0, 0, pDepthStencilSplattingMinDSV);

	unsigned int stride = 0;
	unsigned int offset = 0;
	context->IASetVertexBuffers(0, 0, NULL, &stride, &offset);		
	context->IASetInputLayout(NULL);
	context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_POINTLIST);

	context->VSSetShader(s_pVertexShaderSplatting, 0, 0);

	context->GSSetConstantBuffers(0, 1, &CBsceneRepSDF);
	context->GSSetConstantBuffers(1, 1, &s_ConstantBufferSplatting);
	ID3D11Buffer* CBGlobalAppState = GlobalAppState::getInstance().MapAndGetConstantBuffer(context);
	context->GSSetConstantBuffers(8, 1, &CBGlobalAppState);
	context->GSSetShaderResources(0, 1, &hash);

	context->GSSetShader(s_pGeometryShaderSplatting, 0, 0);
	context->PSSetShader(s_pPixelShaderSplatting, 0, 0);

	// Start query for timing
	if(GlobalAppState::getInstance().s_timingsDetailledEnabled)
	{
		GlobalAppState::getInstance().WaitForGPU();
		m_timer.start();
	}

	/////////////////////////////////////////////////////////////////////////////
	// Splat minimum
	/////////////////////////////////////////////////////////////////////////////

	context->Draw(hashNumValidBuckets, 0);

	/////////////////////////////////////////////////////////////////////////////
	// Splat maximum
	/////////////////////////////////////////////////////////////////////////////

	// Initialize constant buffer

	V_RETURN(context->Map(s_ConstantBufferSplatting, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource));
	cbuffer = (CBuffer*)mappedResource.pData;
	cbuffer->m_RenderTargetWidth = renderTargetWidth;
	cbuffer->m_RenderTargetHeight = renderTargetHeight;
	worldToLastKinectSpace = lastRigidTransform->getInverse();
	memcpy(&cbuffer->m_ViewMat, &worldToLastKinectSpace, sizeof(mat4f));
	memcpy(&cbuffer->m_ViewMatInverse, lastRigidTransform, sizeof(mat4f));
	cbuffer->m_SplatMinimum = 0;				
	context->Unmap(s_ConstantBufferSplatting, 0);

	context->OMSetDepthStencilState(s_pDepthStencilStateSplattingMax, 0);

	context->ClearDepthStencilView(pDepthStencilSplattingMaxDSV, D3D11_CLEAR_DEPTH, 0.0f, 0);
	context->OMSetRenderTargets(0, 0, pDepthStencilSplattingMaxDSV);

	context->GSSetConstantBuffers(0, 1, &CBsceneRepSDF);
	context->GSSetConstantBuffers(1, 1, &s_ConstantBufferSplatting);
	context->GSSetConstantBuffers(8, 1, &CBGlobalAppState);

	context->Draw(hashNumValidBuckets, 0);

	// Wait for query
	if(GlobalAppState::getInstance().s_timingsDetailledEnabled)
	{
		GlobalAppState::getInstance().WaitForGPU();
		TimingLog::totalTimeRayIntervalSplatting+=m_timer.getElapsedTimeMS();
		TimingLog::countRayIntervalSplatting++;
	}

	// Reset Pipeline
	ID3D11RenderTargetView* rtv = DXUTGetD3D11RenderTargetView();
	ID3D11DepthStencilView* dsv = DXUTGetD3D11DepthStencilView();
	context->OMSetRenderTargets(1, &rtv, dsv);

	context->RSSetState(0);
	context->OMSetDepthStencilState(s_pDepthStencilStateSplattingMin, 0); // Min is also default state
	context->OMSetBlendState(s_pBlendStateDefault, NULL, 0xffffffff);

	ID3D11ShaderResourceView* nullSRV[] = { NULL };
	ID3D11UnorderedAccessView* nullUAV[] = { NULL };
	ID3D11Buffer* nullCB[] = { NULL, NULL };

	context->GSSetConstantBuffers(0, 2, nullCB);
	context->GSSetShaderResources(0, 1, nullSRV);
	context->GSSetConstantBuffers(8, 1, nullCB);
	context->PSSetConstantBuffers(0, 2, nullCB);
	context->PSSetConstantBuffers(8, 1, nullCB);

	context->VSSetShader(0, 0, 0);
	context->GSSetShader(0, 0, 0);
	context->PSSetShader(0, 0, 0);

	return hr;
}

void DX11RayCastingHashSDF::destroy()
{
	SAFE_RELEASE(m_pComputeShader);
	SAFE_RELEASE(m_constantBuffer);

	SAFE_RELEASE(m_pOutputImage2D);
	SAFE_RELEASE(m_pOutputImage2DSRV);
	SAFE_RELEASE(m_pOutputImage2DUAV);

	SAFE_RELEASE(m_pSSAOMap);
	SAFE_RELEASE(m_pSSAOMapSRV);
	SAFE_RELEASE(m_pSSAOMapUAV);

	SAFE_RELEASE(m_pSSAOMapFiltered);
	SAFE_RELEASE(m_pSSAOMapFilteredSRV);
	SAFE_RELEASE(m_pSSAOMapFilteredUAV);

	// Output
	SAFE_RELEASE(s_pColors);
	SAFE_RELEASE(s_pColorsSRV);
	SAFE_RELEASE(s_pColorsUAV);

	SAFE_RELEASE(s_pPositions);
	SAFE_RELEASE(s_pPositionsSRV);
	SAFE_RELEASE(s_pPositionsUAV);

	SAFE_RELEASE(s_pNormals);
	SAFE_RELEASE(s_pNormalsSRV);
	SAFE_RELEASE(s_pNormalsUAV);

	// Ray Interval	
	SAFE_RELEASE(s_ConstantBufferSplatting);

	SAFE_RELEASE(s_pVertexShaderSplatting);
	SAFE_RELEASE(s_pGeometryShaderSplatting);
	SAFE_RELEASE(s_pPixelShaderSplatting);

	SAFE_RELEASE(s_pDepthStencilSplattingMin);
	SAFE_RELEASE(s_pDepthStencilSplattingMinDSV);
	SAFE_RELEASE(s_pDepthStencilSplattingMinSRV);

	SAFE_RELEASE(s_pDepthStencilSplattingMax);
	SAFE_RELEASE(s_pDepthStencilSplattingMaxDSV);
	SAFE_RELEASE(s_pDepthStencilSplattingMaxSRV);

	SAFE_RELEASE(s_pDepthStencilStateSplattingMin);
	SAFE_RELEASE(s_pDepthStencilStateSplattingMax);

	SAFE_RELEASE(s_pBlendStateDefault);
	SAFE_RELEASE(s_pBlendStateColorWriteDisabled);

	SAFE_RELEASE(s_pRastState);

	// Stereo
	SAFE_RELEASE(m_pOutputImage2DStereo);
	SAFE_RELEASE(m_pSSAOMapStereo);
	SAFE_RELEASE(m_pSSAOMapFilteredStereo);

	SAFE_RELEASE(m_pOutputImage2DStereoSRV);
	SAFE_RELEASE(m_pSSAOMapStereoSRV);
	SAFE_RELEASE(m_pSSAOMapFilteredStereoSRV);

	SAFE_RELEASE(m_pOutputImage2DStereoUAV);
	SAFE_RELEASE(m_pSSAOMapStereoUAV);
	SAFE_RELEASE(m_pSSAOMapFilteredStereoUAV);

	SAFE_RELEASE(s_pColorsStereo);
	SAFE_RELEASE(s_pColorsStereoSRV);
	SAFE_RELEASE(s_pColorsStereoUAV);

	SAFE_RELEASE(s_pPositionsStereo);
	SAFE_RELEASE(s_pPositionsStereoSRV);
	SAFE_RELEASE(s_pPositionsStereoUAV);

	SAFE_RELEASE(s_pNormalsStereo);
	SAFE_RELEASE(s_pNormalsStereoSRV);
	SAFE_RELEASE(s_pNormalsStereoUAV);

	SAFE_RELEASE(s_pDepthStencilSplattingMinStereo);
	SAFE_RELEASE(s_pDepthStencilSplattingMinDSVStereo);
	SAFE_RELEASE(s_pDepthStencilSplattingMinSRVStereo);

	SAFE_RELEASE(s_pDepthStencilSplattingMaxStereo);
	SAFE_RELEASE(s_pDepthStencilSplattingMaxDSVStereo);
	SAFE_RELEASE(s_pDepthStencilSplattingMaxSRVStereo);
}
