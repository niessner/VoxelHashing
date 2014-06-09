#include "stdafx.h"

#include "DX11DepthSensor.h"

#include "GlobalAppState.h"

DX11Sensor::DX11Sensor()
{
	m_pDepthTextureUS2D = NULL;
	m_pDepthTextureUSSRV = NULL;


	m_pDepthTextureF2D = NULL;
	m_pDepthTextureFSRV = NULL;
	m_pDepthTextureFUAV = NULL;

	m_pDepthTextureFEroded2D = NULL;
	m_pDepthTextureFErodedSRV = NULL;
	m_pDepthTextureFErodedUAV = NULL;

	m_pDepthTextureFFiltered2D = NULL;
	m_pDepthTextureFFilteredSRV = NULL;
	m_pDepthTextureFFilteredUAV = NULL;

	m_pDepthTextureFloat42D = NULL;
	m_pDepthTextureFloat4SRV = NULL;
	m_pDepthTextureFloat4UAV = NULL;

	m_pDepthTextureFloat4NoSmoothing2D = NULL;
	m_pDepthTextureFloat4NoSmoothingSRV = NULL;
	m_pDepthTextureFloat4NoSmoothingUAV = NULL;

	m_pHSVDepthTextureFloat42D = NULL;
	m_pHSVDepthTextureFloat4SRV = NULL;
	m_pHSVDepthTextureFloat4UAV = NULL;

	m_pNormalTextureFloat42D = NULL;
	m_pNormalTextureFloat4SRV = NULL;
	m_pNormalTextureFloat4UAV = NULL;

	m_pColorTexture2D = NULL;
	m_pColorTextureSRV = NULL;

	m_FrameNumberDepth = 0;
}

DX11Sensor::~DX11Sensor()
{
	OnD3D11DestroyDevice();
}

void DX11Sensor::OnD3D11DestroyDevice()
{
	SAFE_RELEASE(m_pDepthTextureUS2D);
	SAFE_RELEASE(m_pDepthTextureUSSRV);

	//! depth texture float
	SAFE_RELEASE(m_pDepthTextureF2D);
	SAFE_RELEASE(m_pDepthTextureFSRV);
	SAFE_RELEASE(m_pDepthTextureFUAV);

	SAFE_RELEASE(m_pDepthTextureFEroded2D);
	SAFE_RELEASE(m_pDepthTextureFErodedSRV);
	SAFE_RELEASE(m_pDepthTextureFErodedUAV);

	SAFE_RELEASE(m_pDepthTextureFFiltered2D);
	SAFE_RELEASE(m_pDepthTextureFFilteredSRV);
	SAFE_RELEASE(m_pDepthTextureFFilteredUAV);

	SAFE_RELEASE(m_pHSVDepthTextureFloat42D);
	SAFE_RELEASE(m_pHSVDepthTextureFloat4SRV);
	SAFE_RELEASE(m_pHSVDepthTextureFloat4UAV);

	SAFE_RELEASE(m_pDepthTextureFloat42D);
	SAFE_RELEASE(m_pDepthTextureFloat4SRV);
	SAFE_RELEASE(m_pDepthTextureFloat4UAV);

	SAFE_RELEASE(m_pDepthTextureFloat4NoSmoothing2D);
	SAFE_RELEASE(m_pDepthTextureFloat4NoSmoothingSRV);
	SAFE_RELEASE(m_pDepthTextureFloat4NoSmoothingUAV);

	SAFE_RELEASE(m_pNormalTextureFloat42D);
	SAFE_RELEASE(m_pNormalTextureFloat4SRV);
	SAFE_RELEASE(m_pNormalTextureFloat4UAV);

	SAFE_RELEASE(m_pColorTexture2D);
	SAFE_RELEASE(m_pColorTextureSRV);
}

HRESULT DX11Sensor::OnD3D11CreateDevice( ID3D11Device* device, DepthSensor* depthSensor )
{
	HRESULT hr = S_OK;

	m_depthSensor = depthSensor;
	
	// Create depth texture
	D3D11_TEXTURE2D_DESC depthTexDesc = {0};
	depthTexDesc.Width = GlobalAppState::getInstance().s_windowWidth;
	depthTexDesc.Height = GlobalAppState::getInstance().s_windowHeight;
	depthTexDesc.MipLevels = 1;
	depthTexDesc.ArraySize = 1;
	depthTexDesc.Format = DXGI_FORMAT_R16_UINT;
	depthTexDesc.SampleDesc.Count = 1;
	depthTexDesc.SampleDesc.Quality = 0;
	depthTexDesc.Usage = D3D11_USAGE_DYNAMIC;
	depthTexDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
	depthTexDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	depthTexDesc.MiscFlags = 0;

	hr = device->CreateTexture2D(&depthTexDesc, NULL, &m_pDepthTextureUS2D);
	if ( FAILED(hr) ) { return hr; }

	depthTexDesc.Usage = D3D11_USAGE_DEFAULT;
	depthTexDesc.CPUAccessFlags = 0;
	depthTexDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
	depthTexDesc.Format = DXGI_FORMAT_R32_FLOAT;

	hr = device->CreateTexture2D(&depthTexDesc, NULL, &m_pDepthTextureF2D);
	if ( FAILED(hr) ) { return hr; }

	hr = device->CreateTexture2D(&depthTexDesc, NULL, &m_pDepthTextureFEroded2D);
	if ( FAILED(hr) ) { return hr; }

	hr = device->CreateTexture2D(&depthTexDesc, NULL, &m_pDepthTextureFFiltered2D);
	if ( FAILED(hr) ) { return hr; }

	depthTexDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;

	hr = device->CreateTexture2D(&depthTexDesc, NULL, &m_pHSVDepthTextureFloat42D);
	if ( FAILED(hr) ) { return hr; }

	hr = device->CreateTexture2D(&depthTexDesc, NULL, &m_pDepthTextureFloat42D);
	if ( FAILED(hr) ) { return hr; }

	hr = device->CreateTexture2D(&depthTexDesc, NULL, &m_pDepthTextureFloat4NoSmoothing2D);
	if ( FAILED(hr) ) { return hr; }

	hr = device->CreateTexture2D(&depthTexDesc, NULL, &m_pNormalTextureFloat42D);
	if ( FAILED(hr) ) { return hr; }

	//create shader resource views
	hr = device->CreateShaderResourceView(m_pDepthTextureUS2D, NULL, &m_pDepthTextureUSSRV);
	if ( FAILED(hr) ) { return hr; }

	hr = device->CreateShaderResourceView(m_pDepthTextureF2D, NULL, &m_pDepthTextureFSRV);
	if ( FAILED(hr) ) { return hr; }

	hr = device->CreateShaderResourceView(m_pDepthTextureFEroded2D, NULL, &m_pDepthTextureFErodedSRV);
	if ( FAILED(hr) ) { return hr; }

	hr = device->CreateShaderResourceView(m_pDepthTextureFFiltered2D, NULL, &m_pDepthTextureFFilteredSRV);
	if ( FAILED(hr) ) { return hr; }

	hr = device->CreateShaderResourceView(m_pHSVDepthTextureFloat42D, NULL, &m_pHSVDepthTextureFloat4SRV);
	if ( FAILED(hr) ) { return hr; }

	hr = device->CreateShaderResourceView(m_pDepthTextureFloat42D, NULL, &m_pDepthTextureFloat4SRV);
	if ( FAILED(hr) ) { return hr; }

	hr = device->CreateShaderResourceView(m_pDepthTextureFloat4NoSmoothing2D, NULL, &m_pDepthTextureFloat4NoSmoothingSRV);
	if ( FAILED(hr) ) { return hr; }

	hr = device->CreateShaderResourceView(m_pNormalTextureFloat42D, NULL, &m_pNormalTextureFloat4SRV);
	if ( FAILED(hr) ) { return hr; }

	//create unordered access views
	hr = device->CreateUnorderedAccessView(m_pDepthTextureF2D, NULL, &m_pDepthTextureFUAV);
	if ( FAILED(hr) ) { return hr; }

	hr = device->CreateUnorderedAccessView(m_pDepthTextureFEroded2D, NULL, &m_pDepthTextureFErodedUAV);
	if ( FAILED(hr) ) { return hr; }

	hr = device->CreateUnorderedAccessView(m_pDepthTextureFFiltered2D, NULL, &m_pDepthTextureFFilteredUAV);
	if ( FAILED(hr) ) { return hr; }

	hr = device->CreateUnorderedAccessView(m_pHSVDepthTextureFloat42D, NULL, &m_pHSVDepthTextureFloat4UAV);
	if ( FAILED(hr) ) { return hr; }

	V_RETURN(device->CreateUnorderedAccessView(m_pDepthTextureFloat42D, NULL, &m_pDepthTextureFloat4UAV));
	V_RETURN(device->CreateUnorderedAccessView(m_pDepthTextureFloat4NoSmoothing2D, NULL, &m_pDepthTextureFloat4NoSmoothingUAV));
	V_RETURN(device->CreateUnorderedAccessView(m_pNormalTextureFloat42D, NULL, &m_pNormalTextureFloat4UAV));

	// Create color texture
	D3D11_TEXTURE2D_DESC colorTexDesc = {0};
	colorTexDesc.Width = GlobalAppState::getInstance().s_windowWidth;
	colorTexDesc.Height = GlobalAppState::getInstance().s_windowHeight;
	colorTexDesc.MipLevels = 1;
	colorTexDesc.ArraySize = 1;
	colorTexDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
	colorTexDesc.SampleDesc.Count = 1;
	colorTexDesc.SampleDesc.Quality = 0;
	colorTexDesc.Usage = D3D11_USAGE_DYNAMIC;
	colorTexDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
	colorTexDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	colorTexDesc.MiscFlags = 0;

	V_RETURN(device->CreateTexture2D(&colorTexDesc, NULL, &m_pColorTexture2D));
	V_RETURN(device->CreateShaderResourceView(m_pColorTexture2D, NULL, &m_pColorTextureSRV));

	return hr;
}

HRESULT DX11Sensor::processDepth( ID3D11DeviceContext* context )	{

	HRESULT hr = S_OK;

	//get data from Kinect to GPU
	if (m_depthSensor->processDepth() == S_FALSE)	return S_FALSE;
	m_FrameNumberDepth++;

	// copy to our d3d 11 depth texture
	D3D11_MAPPED_SUBRESOURCE msT;
	V_RETURN(context->Map(m_pDepthTextureUS2D, NULL, D3D11_MAP_WRITE_DISCARD, NULL, &msT));

	if (GlobalAppState::getInstance().s_windowWidth != m_depthSensor->getDepthWidth() || 
		GlobalAppState::getInstance().s_windowHeight != m_depthSensor->getDepthHeight()) {

		float convWidth = (float)m_depthSensor->getDepthWidth() / (float)GlobalAppState::getInstance().s_windowWidth;
		float convHeight = (float)m_depthSensor->getDepthHeight() / (float)GlobalAppState::getInstance().s_windowHeight;

		for (unsigned int i = 0; i < GlobalAppState::getInstance().s_windowHeight; i++) {
			for (unsigned int j = 0; j < GlobalAppState::getInstance().s_windowWidth; j++) {
				const unsigned int tarIdx = i*msT.RowPitch/2 + j;	//2 bytes per entry
				const unsigned int srcIdx = (unsigned int)((float)i*convHeight+0.5f)*m_depthSensor->getDepthWidth() + (unsigned int)((float)j*convWidth+0.5f);
				((USHORT*)msT.pData)[tarIdx] = m_depthSensor->getDepthD16()[srcIdx];
			}
		}
	} else {
		//TODO actually there should be a pitch as well... (for odd width sizes this doesn't work...)
		memcpy(msT.pData, m_depthSensor->getDepthD16(), GlobalAppState::getInstance().s_windowWidth * GlobalAppState::getInstance().s_windowHeight * sizeof(USHORT));
	}
	context->Unmap(m_pDepthTextureUS2D, NULL);

	DX11ImageHelper::applyDepthMap(context, m_pDepthTextureUSSRV, m_pDepthTextureFUAV, GlobalAppState::getInstance().s_windowWidth, GlobalAppState::getInstance().s_windowHeight);

	// Start query for timing
	if(GlobalAppState::getInstance().s_timingsDetailledEnabled)
	{
		GlobalAppState::getInstance().WaitForGPU();
		m_timer.start();
	}

	DX11ImageHelper::applyErode(context, m_pDepthTextureFSRV, m_pColorTextureSRV, m_pDepthTextureFErodedUAV, 0.1f, 0, GlobalAppState::getInstance().s_windowWidth, GlobalAppState::getInstance().s_windowHeight);

	// Wait for query
	if(GlobalAppState::getInstance().s_timingsDetailledEnabled)
	{
		GlobalAppState::getInstance().WaitForGPU();
		TimingLog::totalTimeErode += m_timer.getElapsedTimeMS(); TimingLog::countErode++;
	}		

	DX11ImageHelper::applyCameraSpaceProjection(context, m_pDepthTextureFErodedSRV, m_pDepthTextureFloat4NoSmoothingUAV, GlobalAppState::getInstance().s_windowWidth, GlobalAppState::getInstance().s_windowHeight);

	if (m_bFilterDepthValues)
	{
		{
			// Start query for timing
			if(GlobalAppState::getInstance().s_timingsDetailledEnabled)
			{
				GlobalAppState::getInstance().WaitForGPU();
				m_timer.start();
			}


			DX11ImageHelper::applyBilateralFilter(context, m_pDepthTextureFErodedSRV, m_pDepthTextureFFilteredUAV, GlobalAppState::getInstance().s_windowWidth, GlobalAppState::getInstance().s_windowHeight, m_fBilateralFilterSigmaD, m_fBilateralFilterSigmaR);

			//DX11ImageHelper::applyBFApprox(context, m_pDepthTextureFErodedSRV, m_pDepthTextureFFilteredUAV, getDepthImageWidth(), getDepthImageHeight(), 10, 0.1f);
			//DX11ImageHelper::applyBlockAveraging(context, m_pDepthTextureFErodedSRV, m_pDepthTextureFFilteredUAV, getDepthImageWidth(), getDepthImageHeight(), m_fBilateralFilterSigmaD, m_fBilateralFilterSigmaR);
			//DX11ImageHelper::applySubSamp(context, m_pDepthTextureFErodedSRV, m_pDepthTextureFFilteredUAV, getDepthImageWidth()/2, getDepthImageHeight()/2);

			// Wait for query
			if(GlobalAppState::getInstance().s_timingsDetailledEnabled)
			{
				GlobalAppState::getInstance().WaitForGPU();
				TimingLog::totalTimeBilateralFiltering += m_timer.getElapsedTimeMS(); TimingLog::countBilateralFiltering++;
			}
		}

		DX11ImageHelper::applyHSVDepth(context, m_pDepthTextureFFilteredSRV, m_pHSVDepthTextureFloat4UAV, GlobalAppState::getInstance().s_windowWidth, GlobalAppState::getInstance().s_windowHeight);

		DX11ImageHelper::applyCameraSpaceProjection(context, m_pDepthTextureFFilteredSRV, m_pDepthTextureFloat4UAV, GlobalAppState::getInstance().s_windowWidth, GlobalAppState::getInstance().s_windowHeight);
	}
	else
	{
		DX11ImageHelper::applyHSVDepth(context, m_pDepthTextureFErodedSRV, m_pHSVDepthTextureFloat4UAV, GlobalAppState::getInstance().s_windowWidth, GlobalAppState::getInstance().s_windowHeight);

		DX11ImageHelper::applyCameraSpaceProjection(context, m_pDepthTextureFErodedSRV, m_pDepthTextureFloat4UAV, GlobalAppState::getInstance().s_windowWidth, GlobalAppState::getInstance().s_windowHeight);
	}

	DX11ImageHelper::applyNormalComputation(context, m_pDepthTextureFloat4SRV, m_pNormalTextureFloat4UAV, GlobalAppState::getInstance().s_windowWidth, GlobalAppState::getInstance().s_windowHeight);

	return hr;
}

HRESULT DX11Sensor::processColor( ID3D11DeviceContext* context )
{
	if (m_depthSensor->processColor() == S_FALSE)	return S_FALSE;

	HRESULT hr = S_OK; // No remapping of color -> therefore color is off at the moment!!

	//// copy to our d3d 11 depth texture
	D3D11_MAPPED_SUBRESOURCE msT;

	// copy to our d3d 11 color texture
	V_RETURN(context->Map(m_pColorTexture2D, NULL, D3D11_MAP_WRITE_DISCARD, NULL, &msT));

	if (GlobalAppState::getInstance().s_windowWidth != m_depthSensor->getColorWidth() || 
		GlobalAppState::getInstance().s_windowHeight != m_depthSensor->getColorHeight()) {

			float convWidth = (float)m_depthSensor->getColorWidth() / (float)GlobalAppState::getInstance().s_windowWidth;
			float convHeight = (float)m_depthSensor->getColorHeight() / (float)GlobalAppState::getInstance().s_windowHeight;

			for (unsigned int i = 0; i < GlobalAppState::getInstance().s_windowHeight; i++) {
				for (unsigned int j = 0; j < GlobalAppState::getInstance().s_windowWidth; j++) {
					const unsigned int tarIdx = i*msT.RowPitch/4 + j;	//4 bytes per entry
					const unsigned int srcIdx = (unsigned int)((float)i*convHeight+0.5f)*m_depthSensor->getColorWidth() + (unsigned int)((float)j*convWidth+0.5f);
					((unsigned int*)msT.pData)[tarIdx] = ((const unsigned int*)m_depthSensor->getColorRGBX())[srcIdx];
				}
			}
	} else {
		//TODO actually there should be a pitch as well... (for odd width sizes this doesn't work...)
		memcpy(msT.pData, m_depthSensor->getColorRGBX(), sizeof(unsigned int)*GlobalAppState::getInstance().s_windowWidth*GlobalAppState::getInstance().s_windowHeight);
	}

	context->Unmap(m_pColorTexture2D, NULL);

	return hr;
}
