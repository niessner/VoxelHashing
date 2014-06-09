#include "stdafx.h"

#include "DX11HistogramHashSDF.h"

#include <iostream>

#include "GlobalAppState.h"


unsigned int DX11HistogramHashSDF::s_blockSize = 256;

ID3D11ComputeShader* DX11HistogramHashSDF::m_pComputeShader = NULL;
ID3D11ComputeShader* DX11HistogramHashSDF::m_pComputeShaderReset = NULL;
	
ID3D11Buffer* DX11HistogramHashSDF::s_pHistogram = NULL;
ID3D11UnorderedAccessView* DX11HistogramHashSDF::s_pHistogramUAV = NULL;
ID3D11Buffer* DX11HistogramHashSDF::s_pHistogramCPU = NULL;

HRESULT DX11HistogramHashSDF::OnD3D11CreateDevice( ID3D11Device* pd3dDevice )
{
	HRESULT hr = S_OK;

	V_RETURN(initialize(pd3dDevice));

	return  hr;
}

void DX11HistogramHashSDF::OnD3D11DestroyDevice()
{
	destroy();
}

HRESULT DX11HistogramHashSDF::computeHistrogram( ID3D11DeviceContext* context, ID3D11ShaderResourceView* hash, ID3D11Buffer* CBsceneRepSDF, unsigned int hashNumBuckets, unsigned int hashBucketSize, std::string hashName )
{
	HRESULT hr = S_OK;

	// Reset Histogram
	context->CSSetUnorderedAccessViews(0, 1, &s_pHistogramUAV, NULL);
	context->CSSetShaderResources(0, 1, &hash);
	context->CSSetConstantBuffers(0, 1, &CBsceneRepSDF);
	ID3D11Buffer* CBGlobalAppState = GlobalAppState::getInstance().MapAndGetConstantBuffer(context);
	context->CSSetConstantBuffers(8, 1, &CBGlobalAppState);
	context->CSSetShader(m_pComputeShaderReset, 0, 0);

	unsigned int dimX = hashBucketSize+1+GlobalAppState::getInstance().s_MaxCollisionLinkedListSize;

	context->Dispatch(dimX, 1, 1);
	assert(dimX <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);

	// Cleanup
	ID3D11UnorderedAccessView* nullUAV[1] = {NULL};
	ID3D11ShaderResourceView* nullSRV[1] = {NULL};
	ID3D11Buffer* nullB[1] = {NULL};
	context->CSSetUnorderedAccessViews(0, 1, nullUAV, 0);
	context->CSSetShaderResources(0, 1, nullSRV);
	context->CSSetConstantBuffers(0, 1, nullB);
	context->CSSetConstantBuffers(8, 1, nullB);
	context->CSSetShader(0, 0, 0);

	// Setup Pipeline
	context->CSSetUnorderedAccessViews(0, 1, &s_pHistogramUAV, NULL);
	context->CSSetShaderResources(0, 1, &hash);
	context->CSSetConstantBuffers(0, 1, &CBsceneRepSDF);
	context->CSSetConstantBuffers(8, 1, &CBGlobalAppState);
	context->CSSetShader(m_pComputeShader, 0, 0);

	// Run compute shader
	dimX = NUM_GROUPS_X;
	unsigned int dimY = ((hashNumBuckets + s_blockSize - 1)/s_blockSize + NUM_GROUPS_X - 1) / NUM_GROUPS_X;
	assert(dimX <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);
	assert(dimY <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);

	context->Dispatch(dimX, dimY, 1);

	// Cleanup
	context->CSSetUnorderedAccessViews(0, 1, nullUAV, 0);
	context->CSSetShaderResources(0, 1, nullSRV);
	context->CSSetConstantBuffers(0, 1, nullB);
	context->CSSetConstantBuffers(8, 1, nullB);
	context->CSSetShader(0, 0, 0);

	// Copy to CPU
	context->CopyResource(s_pHistogramCPU, s_pHistogram);

	D3D11_MAPPED_SUBRESOURCE mappedResource;
	V_RETURN(context->Map(s_pHistogramCPU, 0, D3D11_MAP_READ, 0, &mappedResource));
	printHistogram((unsigned int*)mappedResource.pData, hashNumBuckets, hashBucketSize, hashName);
	context->Unmap(s_pHistogramCPU, 0);

	return hr;
}

HRESULT DX11HistogramHashSDF::initialize( ID3D11Device* pd3dDevice )
{
	HRESULT hr = S_OK;

	ID3DBlob* pBlob = NULL;

	char SDFBLOCKSIZE[5];
	sprintf_s(SDFBLOCKSIZE, "%d", SDF_BLOCK_SIZE);

	char HANDLECOLLISIONS[5];
	sprintf_s(HANDLECOLLISIONS, "%d", GlobalAppState::getInstance().s_HANDLE_COLLISIONS);

	char MAXLINKEDLISTSIZE[5];
	sprintf_s(MAXLINKEDLISTSIZE, "%d", GlobalAppState::getInstance().s_MaxCollisionLinkedListSize);

	char BLOCK_SIZE[5];
	sprintf_s(BLOCK_SIZE, "%d", s_blockSize);
	D3D_SHADER_MACRO shaderDefines[] = { { "groupthreads", BLOCK_SIZE }, { "SDF_BLOCK_SIZE", SDFBLOCKSIZE }, {"MAX_COLLISION_LINKED_LIST_SIZE", MAXLINKEDLISTSIZE}, {"HANDLE_COLLISIONS", HANDLECOLLISIONS }, { 0 } };
	D3D_SHADER_MACRO shaderDefinesWithout[] = { { "groupthreads", BLOCK_SIZE }, { "SDF_BLOCK_SIZE", SDFBLOCKSIZE }, {"MAX_COLLISION_LINKED_LIST_SIZE", MAXLINKEDLISTSIZE}, { 0 } };

	D3D_SHADER_MACRO* validDefines = shaderDefines;
	if(GlobalAppState::getInstance().s_HANDLE_COLLISIONS == 0)
	{
		validDefines = shaderDefinesWithout;
	}


	V_RETURN(CompileShaderFromFile(L"Shaders\\HistogramHashSDF.hlsl", "computeHistogramHashSDFCS", "cs_5_0", &pBlob, validDefines));
	V_RETURN(pd3dDevice->CreateComputeShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &m_pComputeShader))
		SAFE_RELEASE(pBlob);

	V_RETURN(CompileShaderFromFile(L"Shaders\\HistogramHashSDF.hlsl", "resetHistogramHashSDFCS", "cs_5_0", &pBlob, validDefines));
	V_RETURN(pd3dDevice->CreateComputeShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &m_pComputeShaderReset))
		SAFE_RELEASE(pBlob);

	// Output Buffer
	D3D11_BUFFER_DESC bDesc;
	ZeroMemory(&bDesc, sizeof(D3D11_BUFFER_DESC));
	bDesc.BindFlags	= D3D11_BIND_UNORDERED_ACCESS;
	bDesc.Usage	= D3D11_USAGE_DEFAULT;
	bDesc.CPUAccessFlags = 0;
	bDesc.MiscFlags	= 0;
	bDesc.ByteWidth	= sizeof(unsigned int)*(std::max(GlobalAppState::getInstance().s_hashBucketSizeLocal, GlobalAppState::getInstance().s_hashBucketSizeGlobal)+1 + GlobalAppState::getInstance().s_MaxCollisionLinkedListSize);
	V_RETURN(pd3dDevice->CreateBuffer(&bDesc, NULL, &s_pHistogram));

	D3D11_UNORDERED_ACCESS_VIEW_DESC DescUAV;
	ZeroMemory(&DescUAV, sizeof(DescUAV));
	DescUAV.Format = DXGI_FORMAT_R32_UINT;
	DescUAV.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
	DescUAV.Buffer.FirstElement = 0;
	DescUAV.Buffer.NumElements = std::max(GlobalAppState::getInstance().s_hashBucketSizeLocal, GlobalAppState::getInstance().s_hashBucketSizeGlobal)+1 + GlobalAppState::getInstance().s_MaxCollisionLinkedListSize;
	V_RETURN(pd3dDevice->CreateUnorderedAccessView(s_pHistogram, &DescUAV, &s_pHistogramUAV));

	// Create Output Buffer CPU
	bDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
	bDesc.BindFlags = 0;
	bDesc.Usage = D3D11_USAGE_STAGING;
	V_RETURN(pd3dDevice->CreateBuffer(&bDesc, NULL, &s_pHistogramCPU));

	return  hr;
}

void DX11HistogramHashSDF::printHistogram( const unsigned int* data, unsigned int hashNumBuckets, unsigned int hashBucketSize, std::string hashName )
{
	std::streamsize oldPrec = std::cout.precision(4);
	std::ios_base::fmtflags oldFlags = std::cout.setf( std::ios::fixed, std:: ios::floatfield );

	unsigned int nTotal = 0;
	unsigned int nElements = 0;
	for(unsigned int i = 0; i<hashBucketSize+1; i++)
	{
		nTotal += data[i];
		nElements += data[i]*i;
	}

	std::cout << nTotal << std::endl;
	std::cout << "Histogram for hash " << hashName << " with " << (unsigned int)nElements <<" of " << hashNumBuckets*hashBucketSize << " elements:" << std::endl;
	std::cout << "--------------------------------------------------------------" << std::endl;
	for(unsigned int i = 0; i<hashBucketSize+1; i++)
	{
		float percent = 100.0f*(float)data[i]/(float)nTotal;

		std::cout << i << ":\t" << (percent < 10.0f ? " " : "" ) << percent << "%\tabsolute: " << data[i] << std::endl;
	}
	std::cout << std::endl;
	for (unsigned int i = hashBucketSize+1; i < hashBucketSize+1+GlobalAppState::getInstance().s_MaxCollisionLinkedListSize-1; i++) {

		float percent = 100.0f*(float)data[i]/(float)hashNumBuckets;
		std::cout << "listLen " << (i - (hashBucketSize+1)) << ":\t" << (percent < 10.0f ? " " : "" ) << percent << "%\tabsolute: " << data[i] << std::endl;
	}
	std::cout << "--------------------------------------------------------------" << std::endl;

	std::cout.precision(oldPrec);
	std::cout.setf(oldFlags);
}

void DX11HistogramHashSDF::destroy()
{
	SAFE_RELEASE(m_pComputeShader);
	SAFE_RELEASE(m_pComputeShaderReset);

	SAFE_RELEASE(s_pHistogram);
	SAFE_RELEASE(s_pHistogramUAV);
	SAFE_RELEASE(s_pHistogramCPU);
}
