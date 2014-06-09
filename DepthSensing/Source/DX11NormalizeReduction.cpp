#include "stdafx.h"

#include "DX11NormalizeReduction.h"

#include "GlobalCameraTrackingState.h"

#include <iostream>

ID3D11Buffer* DX11NormalizeReduction::m_constantBufferNorm = NULL;

/////////////////////////////////////////////////////
// Center of Gravity and Variance
/////////////////////////////////////////////////////

ID3D11ComputeShader** DX11NormalizeReduction::m_pComputeShaderNorm = NULL;
ID3D11ComputeShader** DX11NormalizeReduction::m_pComputeShader2Norm = NULL;

ID3D11Buffer** DX11NormalizeReduction::m_pAuxBufNorm = NULL;
ID3D11Buffer** DX11NormalizeReduction::m_pAuxBufNormCPU = NULL;

ID3D11ShaderResourceView** DX11NormalizeReduction::m_pAuxBufNormSRV = NULL;
ID3D11UnorderedAccessView** DX11NormalizeReduction::m_pAuxBufNormUAV = NULL;

/////////////////////////////////////////////////////
// Timer
/////////////////////////////////////////////////////

Timer DX11NormalizeReduction::m_timer;

HRESULT DX11NormalizeReduction::OnD3D11CreateDevice( ID3D11Device* pd3dDevice )
{
	HRESULT hr = S_OK;

	m_pComputeShaderNorm = new ID3D11ComputeShader*[GlobalCameraTrackingState::getInstance().s_maxLevels];
	m_pComputeShader2Norm = new ID3D11ComputeShader*[GlobalCameraTrackingState::getInstance().s_maxLevels];

	m_pAuxBufNorm = new ID3D11Buffer*[GlobalCameraTrackingState::getInstance().s_maxLevels];
	m_pAuxBufNormCPU = new ID3D11Buffer*[GlobalCameraTrackingState::getInstance().s_maxLevels];

	m_pAuxBufNormSRV = new ID3D11ShaderResourceView*[GlobalCameraTrackingState::getInstance().s_maxLevels];
	m_pAuxBufNormUAV = new ID3D11UnorderedAccessView*[GlobalCameraTrackingState::getInstance().s_maxLevels];

	/////////////////////////////////////////////////////
	// Normalize Reduction
	/////////////////////////////////////////////////////

	D3D11_BUFFER_DESC Desc;
	Desc.Usage = D3D11_USAGE_DYNAMIC;
	Desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	Desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	Desc.MiscFlags = 0;
	Desc.ByteWidth = sizeof(CBufferNorm);
	V_RETURN(pd3dDevice->CreateBuffer(&Desc, NULL, &m_constantBufferNorm));

	for(int level = GlobalCameraTrackingState::getInstance().s_maxLevels-1; level >= 0; level--)
	{		
		char BUCKET_SIZE_NORM[5];
		char NUM_BUCKETS_NORM[5];

		sprintf_s(BUCKET_SIZE_NORM, "%d", GlobalCameraTrackingState::getInstance().s_blockSizeNormalize[level]);
		sprintf_s(NUM_BUCKETS_NORM, "%d", GlobalCameraTrackingState::getInstance().s_numBucketsNormalize[level]);

		D3D_SHADER_MACRO shaderDefinesScanInBucketNorm[] = { { "groupthreads", BUCKET_SIZE_NORM }, { 0 } };
		D3D_SHADER_MACRO shaderDefinesScanBucketResultNorm[] = { { "groupthreads", NUM_BUCKETS_NORM }, {"BUCKET_SIZE", BUCKET_SIZE_NORM}, { 0 } };

		ID3DBlob* pBlobCS = NULL;
		V_RETURN(CompileShaderFromFile( L"Shaders\\NormalizeReduction.hlsl", "CSScanInBucket", "cs_5_0", &pBlobCS, shaderDefinesScanInBucketNorm));
		V_RETURN(pd3dDevice->CreateComputeShader(pBlobCS->GetBufferPointer(), pBlobCS->GetBufferSize(), NULL, &m_pComputeShaderNorm[level]));
		SAFE_RELEASE(pBlobCS);

		V_RETURN(CompileShaderFromFile(L"Shaders\\NormalizeReduction.hlsl", "CSScanBucketResult", "cs_5_0", &pBlobCS, shaderDefinesScanBucketResultNorm));
		V_RETURN(pd3dDevice->CreateComputeShader(pBlobCS->GetBufferPointer(), pBlobCS->GetBufferSize(), NULL, &m_pComputeShader2Norm[level]));
		SAFE_RELEASE(pBlobCS);

		D3D11_BUFFER_DESC Desc;
		ZeroMemory(&Desc, sizeof(Desc));
		Desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
		Desc.StructureByteStride = 8*sizeof(float);
		Desc.ByteWidth = Desc.StructureByteStride*GlobalCameraTrackingState::getInstance().s_numBucketsNormalize[level];
		Desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
		Desc.Usage = D3D11_USAGE_DEFAULT;
		V_RETURN(pd3dDevice->CreateBuffer(&Desc, NULL, &m_pAuxBufNorm[level]));

		D3D11_SHADER_RESOURCE_VIEW_DESC DescRV;
		ZeroMemory( &DescRV, sizeof( DescRV ) );
		DescRV.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
		DescRV.Format = DXGI_FORMAT_UNKNOWN;
		DescRV.Buffer.FirstElement = 0;
		DescRV.Buffer.NumElements = GlobalCameraTrackingState::getInstance().s_numBucketsNormalize[level];
		V_RETURN(pd3dDevice->CreateShaderResourceView(m_pAuxBufNorm[level], &DescRV, &m_pAuxBufNormSRV[level]));

		D3D11_UNORDERED_ACCESS_VIEW_DESC DescUAV;
		ZeroMemory( &DescUAV, sizeof(DescUAV) );
		DescUAV.Format = DXGI_FORMAT_UNKNOWN;
		DescUAV.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
		DescUAV.Buffer.FirstElement = 0;
		DescUAV.Buffer.NumElements = GlobalCameraTrackingState::getInstance().s_numBucketsNormalize[level];
		V_RETURN( pd3dDevice->CreateUnorderedAccessView(m_pAuxBufNorm[level], &DescUAV, &m_pAuxBufNormUAV[level]));

		// Create Output Buffer CPU
		Desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
		Desc.BindFlags = 0;
		Desc.Usage = D3D11_USAGE_STAGING;
		V_RETURN(pd3dDevice->CreateBuffer(&Desc, NULL, &m_pAuxBufNormCPU[level]));
	}

	return hr;
}

HRESULT DX11NormalizeReduction::applyNorm( ID3D11DeviceContext* context, ID3D11ShaderResourceView* inputSRV, unsigned int level, unsigned int imageWidth, unsigned int imageHeight, D3DXVECTOR3& mean, float& meanStDev, float& nValidCorres )
{
	HRESULT hr = S_OK;

	mean = D3DXVECTOR3(0.0f, 0.0f, 0.0f);
	meanStDev = 1.0f;

	return hr;

	// Initialize Constant Buffer
	D3D11_MAPPED_SUBRESOURCE mappedResource;
	V_RETURN(context->Map(m_constantBufferNorm, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource));
	CBufferNorm *cbuffer = (CBufferNorm*)mappedResource.pData;
	cbuffer->imageWidth = imageWidth;
	cbuffer->numElements = (int)imageWidth*imageHeight;
	context->Unmap(m_constantBufferNorm, 0);

	// first pass, scan in each bucket
	context->CSSetConstantBuffers(0, 1, &m_constantBufferNorm);
	context->CSSetShader(m_pComputeShaderNorm[level], NULL, 0);
	context->CSSetShaderResources(0, 1, &inputSRV);
	context->CSSetUnorderedAccessViews(0, 1, &m_pAuxBufNormUAV[level], NULL);

	context->Dispatch(GlobalCameraTrackingState::getInstance().s_numBucketsNormalize[level], 1, 1);

	ID3D11UnorderedAccessView* ppUAViewNULL[1] = { NULL };
	ID3D11ShaderResourceView* ppSRVNULL[1] = { NULL };
	ID3D11Buffer* nullCB[1] = { NULL };

	context->CSSetUnorderedAccessViews(0, 1, ppUAViewNULL, NULL);
	context->CSSetShaderResources(0, 1, ppSRVNULL);
	context->CSSetConstantBuffers(0, 1, nullCB);
	context->CSSetShader(0, 0, 0);

	// Copy to CPU
	context->CopyResource(m_pAuxBufNormCPU[level], m_pAuxBufNorm[level]);
	V_RETURN(context->Map(m_pAuxBufNormCPU[level], 0, D3D11_MAP_READ, 0, &mappedResource))

		reductionCPU((float*)mappedResource.pData, GlobalCameraTrackingState::getInstance().s_numBucketsNormalize[level], mean, meanStDev, nValidCorres);

	context->Unmap(m_pAuxBufNormCPU[level], 0);

	return hr;
}

void DX11NormalizeReduction::reductionCPU( const float* data, unsigned int nElems, D3DXVECTOR3& mean, float& meanStDev, float& nValidCorres )
{
	float buffer[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
	for(unsigned int k = 0; k<nElems; k++)
	{
		for(unsigned int i = 0; i<8; i++)
		{
			buffer[i] += data[8*k+i];
		}
	}

	nValidCorres = buffer[6];

	if(nValidCorres >= 10.0f)
	{
		mean = D3DXVECTOR3(buffer[0]/nValidCorres, buffer[1]/nValidCorres, buffer[2]/nValidCorres);
		D3DXVECTOR3 meanSquared = D3DXVECTOR3(buffer[3]/nValidCorres, buffer[4]/nValidCorres, buffer[5]/nValidCorres);
		D3DXVECTOR3 var = D3DXVECTOR3(meanSquared.x-mean.x*mean.x, meanSquared.y-mean.y*mean.y, meanSquared.z-mean.z*mean.z);
		meanStDev = (sqrt(var.x)+sqrt(var.y)+sqrt(var.z))/3.0f;
	}
	else
	{
		mean = D3DXVECTOR3(0.0f, 0.0f, 0.0f);
		meanStDev = 1.0f;
	}
}

void DX11NormalizeReduction::OnD3D11DestroyDevice()
{
	SAFE_RELEASE(m_constantBufferNorm);

	for(int level = GlobalCameraTrackingState::getInstance().s_maxLevels-1; level >= 0; level--)
	{
		SAFE_RELEASE(m_pComputeShaderNorm[level]);
		SAFE_RELEASE(m_pComputeShader2Norm[level]);

		SAFE_RELEASE(m_pAuxBufNorm[level]);
		SAFE_RELEASE(m_pAuxBufNormCPU[level]);

		SAFE_RELEASE(m_pAuxBufNormSRV[level]);
		SAFE_RELEASE(m_pAuxBufNormUAV[level]);
	}

	SAFE_DELETE_ARRAY(m_pComputeShaderNorm);
	SAFE_DELETE_ARRAY(m_pComputeShader2Norm);
	SAFE_DELETE_ARRAY(m_pAuxBufNorm);
	SAFE_DELETE_ARRAY(m_pAuxBufNormCPU);
	SAFE_DELETE_ARRAY(m_pAuxBufNormSRV);
	SAFE_DELETE_ARRAY(m_pAuxBufNormUAV);
}
