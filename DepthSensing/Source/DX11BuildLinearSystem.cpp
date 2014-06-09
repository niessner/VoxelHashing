#include "stdafx.h"

#include "DX11BuildLinearSystem.h"

#include "GlobalAppState.h"
#include "GlobalCameraTrackingState.h"

#include <iostream>

/////////////////////////////////////////////////////
// Build Linear System
/////////////////////////////////////////////////////

unsigned int DX11BuildLinearSystem::m_blockSizeBL = 64;

ID3D11Buffer* DX11BuildLinearSystem::m_constantBufferBL = NULL;

ID3D11ComputeShader** DX11BuildLinearSystem::m_pComputeShaderBL = NULL;

ID3D11Buffer* DX11BuildLinearSystem::m_pOutputFloat = NULL;
ID3D11UnorderedAccessView* DX11BuildLinearSystem::m_pOutputFloatUAV = NULL;
ID3D11Buffer* DX11BuildLinearSystem::m_pOutputFloatCPU = NULL;

/////////////////////////////////////////////////////
// Timer
/////////////////////////////////////////////////////

Timer DX11BuildLinearSystem::m_timer;

HRESULT DX11BuildLinearSystem::OnD3D11CreateDevice( ID3D11Device* pd3dDevice )
{
	HRESULT hr = S_OK;

	/////////////////////////////////////////////////////
	// Build Linear System
	/////////////////////////////////////////////////////

	m_pComputeShaderBL = new ID3D11ComputeShader*[GlobalCameraTrackingState::getInstance().s_maxLevels];

	char BLOCK_SIZE_BL[5];
	sprintf_s(BLOCK_SIZE_BL, "%d", m_blockSizeBL);

	for(int level = GlobalCameraTrackingState::getInstance().s_maxLevels-1; level >= 0; level--)
	{
		char LOCALWINDOWSIZE_BL[5];
		sprintf_s(LOCALWINDOWSIZE_BL, "%d", GlobalCameraTrackingState::getInstance().s_localWindowSize[level]);

		// Build Linearized System
		char ARRAYSIZE_BL[5];
		sprintf_s(ARRAYSIZE_BL, "%d", 30);
		D3D_SHADER_MACRO shaderDefinesBL[] = { { "groupthreads", BLOCK_SIZE_BL }, { "LOCALWINDOWSIZE", LOCALWINDOWSIZE_BL }, {"ARRAYSIZE", ARRAYSIZE_BL }, { 0 } };

		ID3DBlob* pBlob = NULL;
		hr = CompileShaderFromFile(L"Shaders\\BuildLinearSystem.hlsl", "scanScanElementsCS", "cs_5_0", &pBlob, shaderDefinesBL);
		if(FAILED(hr)) return hr;

		hr = pd3dDevice->CreateComputeShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &m_pComputeShaderBL[level]);
		if(FAILED(hr)) return hr;

		SAFE_RELEASE(pBlob);
	}

	D3D11_BUFFER_DESC bDesc;
	bDesc.BindFlags	= D3D11_BIND_CONSTANT_BUFFER;
	bDesc.Usage	= D3D11_USAGE_DYNAMIC;
	bDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	bDesc.MiscFlags	= 0;
	bDesc.ByteWidth	= sizeof(CBufferBL);

	hr = pd3dDevice->CreateBuffer(&bDesc, NULL, &m_constantBufferBL);
	if(FAILED(hr)) return hr;

	unsigned int dimX = (unsigned int)ceil(((float)GlobalAppState::getInstance().s_windowWidth*GlobalAppState::getInstance().s_windowHeight)/(GlobalCameraTrackingState::getInstance().s_localWindowSize[0]*m_blockSizeBL));

	// Create Output Buffer
	D3D11_BUFFER_DESC Desc;
	Desc.Usage = D3D11_USAGE_DEFAULT;
	Desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
	Desc.CPUAccessFlags = 0;
	Desc.MiscFlags = 0;
	Desc.ByteWidth = 30*sizeof(float)*dimX; // Buffer is reused for all levels -> maximal dimX*dimY==400 elements on each level
	V_RETURN(pd3dDevice->CreateBuffer(&Desc, NULL, &m_pOutputFloat));

	// Create Output Buffer CPU
	Desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
	Desc.BindFlags = 0;
	Desc.Usage = D3D11_USAGE_STAGING;
	V_RETURN(pd3dDevice->CreateBuffer(&Desc, NULL, &m_pOutputFloatCPU));

	// Create Output UAV
	D3D11_UNORDERED_ACCESS_VIEW_DESC DescUAV;
	ZeroMemory( &DescUAV, sizeof(DescUAV));
	DescUAV.Format = DXGI_FORMAT_R32_FLOAT;
	DescUAV.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
	DescUAV.Buffer.FirstElement = 0;
	DescUAV.Buffer.NumElements = 30*dimX;
	V_RETURN( pd3dDevice->CreateUnorderedAccessView(m_pOutputFloat, &DescUAV, &m_pOutputFloatUAV));

	return  hr;
}

HRESULT DX11BuildLinearSystem::applyBL( ID3D11DeviceContext* context, ID3D11ShaderResourceView* inputSRV, ID3D11ShaderResourceView* correspondenceSRV, ID3D11ShaderResourceView* correspondenceNormalsSRV, D3DXVECTOR3& mean, float meanStDev, Eigen::Matrix4f& deltaTransform, unsigned int imageWidth, unsigned int imageHeight, unsigned int level, Matrix6x7f& res, LinearSystemConfidence& conf )
{
	HRESULT hr = S_OK;

	unsigned int dimX = (unsigned int)ceil(((float)imageWidth*imageHeight)/(GlobalCameraTrackingState::getInstance().s_localWindowSize[level]*m_blockSizeBL));

	Eigen::Matrix4f deltaTransformT = deltaTransform.transpose();

	// Initialize Constant Buffer
	D3D11_MAPPED_SUBRESOURCE mappedResource;
	V_RETURN(context->Map(m_constantBufferBL, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource))

	CBufferBL *cbuffer = (CBufferBL*)mappedResource.pData;
	cbuffer->imageWidth = (int)imageWidth;
	cbuffer->imageHeigth = (int)imageHeight;
	memcpy(cbuffer->deltaTransform, deltaTransformT.data(), 16*sizeof(float));
	cbuffer->imageHeigth = (int)imageHeight;
	memcpy(cbuffer->mean, (float*)mean, 3*sizeof(float));
	cbuffer->meanStDevInv = 1.0f/meanStDev;			
	context->Unmap(m_constantBufferBL, 0);

	// Setup Pipeline
	context->CSSetShaderResources(0, 1, &inputSRV);
	context->CSSetShaderResources(1, 1, &correspondenceSRV);
	context->CSSetShaderResources(2, 1, &correspondenceNormalsSRV);
	context->CSSetUnorderedAccessViews(0, 1, &m_pOutputFloatUAV, 0);
	context->CSSetConstantBuffers(0, 1, &m_constantBufferBL);
	context->CSSetShader(m_pComputeShaderBL[level], 0, 0);

	// Start Compute Shader
	context->Dispatch(dimX, 1, 1);
	assert(dimX <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);

	// De-Initialize Pipeline
	ID3D11ShaderResourceView* nullSAV[1] = { NULL };
	ID3D11UnorderedAccessView* nullUAV[1] = { NULL };
	ID3D11Buffer* nullCB[1] = { NULL };

	context->CSSetShaderResources(0, 1, nullSAV);
	context->CSSetShaderResources(1, 1, nullSAV);
	context->CSSetShaderResources(2, 1, nullSAV);
	context->CSSetUnorderedAccessViews(0, 1, nullUAV, 0);
	context->CSSetConstantBuffers(0, 1, nullCB);
	context->CSSetShader(0, 0, 0);

	// Copy to CPU
	context->CopyResource(m_pOutputFloatCPU, m_pOutputFloat);
	V_RETURN(context->Map(m_pOutputFloatCPU, 0, D3D11_MAP_READ, 0, &mappedResource));
	res = reductionSystemCPU((float*)mappedResource.pData, dimX, conf);

	context->Unmap(m_pOutputFloatCPU, 0);

	return hr;
}

Matrix6x7f DX11BuildLinearSystem::reductionSystemCPU( const float* data, unsigned int nElems, LinearSystemConfidence& conf )
{
	Matrix6x7f res; res.setZero();

	conf.reset();
	float numCorrF = 0.0f;

	for(unsigned int k = 0; k<nElems; k++)
	{
		unsigned int linRowStart = 0;

		for(unsigned int i = 0; i<6; i++)
		{
			for(unsigned int j = i; j<6; j++)
			{
				res(i, j) += data[30*k+linRowStart+j-i];
			}

			linRowStart += 6-i;

			res(i, 6) += data[30*k+21+i];
		}

		conf.sumRegError += data[30*k+27];
		conf.sumRegWeight += data[30*k+28];
		numCorrF += data[30*k+29];
	}

	// Fill lower triangle
	for(unsigned int i = 0; i<6; i++)
	{
		for(unsigned int j = i; j<6; j++)
		{
			res(j, i) = res(i, j);
		}
	}

	conf.numCorr = (unsigned int)numCorrF;

	return res;
}

void DX11BuildLinearSystem::OnD3D11DestroyDevice()
{
	/////////////////////////////////////////////////////
	// Build Linear System
	/////////////////////////////////////////////////////

	SAFE_RELEASE(m_constantBufferBL);

	for(int level = GlobalCameraTrackingState::getInstance().s_maxLevels-1; level >= 0; level--)
	{
		SAFE_RELEASE(m_pComputeShaderBL[level]);
	}

	SAFE_RELEASE(m_pOutputFloat);
	SAFE_RELEASE(m_pOutputFloatUAV);
	SAFE_RELEASE(m_pOutputFloatCPU);

	SAFE_DELETE_ARRAY(m_pComputeShaderBL);
}
