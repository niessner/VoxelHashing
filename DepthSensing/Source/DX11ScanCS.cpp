#include "stdafx.h"

#include "DXUT.h"
#include "DX11ScanCS.h"

#include <iostream>
#include <cassert>


DX11ScanCS::DX11ScanCS()    
{
	m_pcbCS = NULL;

	m_pBucketResults = NULL;
	m_pBucketResultsSRV = NULL;
	m_pBucketResultsUAV = NULL;

	m_pBucketBlockResults = NULL;
	m_pBucketBlockResultsSRV = NULL;
	m_pBucketBlockResultsUAV = NULL;

	m_PrefixSumLastElemStagging = NULL;
	m_PrefixSumLastElem = NULL;
	m_PrefixSumLastElemUAV = NULL;
	m_PrefixSumLastElemSRV = NULL;

	m_pScanBucketsCS = NULL;
	m_pScanBucketResultsCS = NULL;
	m_pScanApplyBucketBlockResultsCS = NULL;
	m_pScanApplyBucketResultsCS = NULL;


	m_bIsCreated = false;
	UINT mySize = 512;	//about 130 mio
	m_BucketSize = mySize;
	m_BucketBlockSize = mySize;

	m_NumBuckets = mySize*mySize;
	m_NumBucketBlocks = mySize;
}

HRESULT DX11ScanCS::OnD3D11CreateDevice( ID3D11Device* pd3dDevice )
{
    HRESULT hr = S_OK;
	if (m_bIsCreated)	return hr;

	char BUCKET_SIZE[10];
	char NUM_BUCKETS[10];
	sprintf_s(BUCKET_SIZE, "%d", m_BucketSize);
	sprintf_s(NUM_BUCKETS, "%d", m_NumBuckets);


	//D3D_SHADER_MACRO shaderDefinesScanInBucket[] = { { "groupthreads", BUCKET_SIZE }, { 0 } };
	//D3D_SHADER_MACRO shaderDefinesScanBucketResult[] = { { "groupthreads", NUM_BUCKETS }, {"BUCKET_SIZE", BUCKET_SIZE}, { 0 } };	

	ID3DBlob* pBlob = NULL;

	V_RETURN( CompileShaderFromFile( L"ScanCS.hlsl", "CSScanBucket", "cs_5_0", &pBlob, NULL ) ); 
	V_RETURN( pd3dDevice->CreateComputeShader( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &m_pScanBucketsCS ) );   

	V_RETURN( CompileShaderFromFile( L"ScanCS.hlsl", "CSScanBucketResults", "cs_5_0", &pBlob, NULL ) ); 
	V_RETURN( pd3dDevice->CreateComputeShader( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &m_pScanBucketResultsCS ) );   

	V_RETURN( CompileShaderFromFile( L"ScanCS.hlsl", "CSScanBucketBlockResults", "cs_5_0", &pBlob, NULL ) ); 
	V_RETURN( pd3dDevice->CreateComputeShader( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &m_pScanBucketBlockResultsCS ) );   
	
	V_RETURN( CompileShaderFromFile( L"ScanCS.hlsl", "CSScanApplyBucketBlockResults", "cs_5_0", &pBlob, NULL ) ); 
	V_RETURN( pd3dDevice->CreateComputeShader( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &m_pScanApplyBucketBlockResultsCS ) );  

	V_RETURN( CompileShaderFromFile( L"ScanCS.hlsl", "CSScanApplyBucketResults", "cs_5_0", &pBlob, NULL ) ); 
	V_RETURN( pd3dDevice->CreateComputeShader( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &m_pScanApplyBucketResultsCS ) );   

	SAFE_RELEASE( pBlob );

	




	D3D11_BUFFER_DESC Desc;
	D3D11_UNORDERED_ACCESS_VIEW_DESC DescUAV;
	D3D11_SHADER_RESOURCE_VIEW_DESC DescSRV;

	/////////
	// Aux Buffers for BucketResults
	/////////
	ZeroMemory( &Desc, sizeof(Desc) );
	Desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
	Desc.StructureByteStride = sizeof(int); 
	Desc.ByteWidth = Desc.StructureByteStride * m_NumBuckets;
	Desc.Usage = D3D11_USAGE_DEFAULT;
	//Desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
	V_RETURN( pd3dDevice->CreateBuffer(&Desc, NULL, &m_pBucketResults) );

	ZeroMemory( &DescSRV, sizeof( DescSRV ) );
	DescSRV.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
	//DescRV.Format = DXGI_FORMAT_UNKNOWN;
	DescSRV.Format = DXGI_FORMAT_R32_SINT;
	DescSRV.Buffer.FirstElement = 0;
	DescSRV.Buffer.NumElements = m_NumBuckets;
	V_RETURN( pd3dDevice->CreateShaderResourceView( m_pBucketResults, &DescSRV, &m_pBucketResultsSRV ) );

	ZeroMemory( &DescUAV, sizeof(DescUAV) );
	//DescUAV.Format = DXGI_FORMAT_UNKNOWN;
	DescUAV.Format = DXGI_FORMAT_R32_SINT;
	DescUAV.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
	DescUAV.Buffer.FirstElement = 0;
	DescUAV.Buffer.NumElements = m_NumBuckets;
	V_RETURN( pd3dDevice->CreateUnorderedAccessView( m_pBucketResults, &DescUAV, &m_pBucketResultsUAV ) );


	/////////
	// Aux Buffers for BucketBlockResults
	/////////
    ZeroMemory( &Desc, sizeof(Desc) );
    Desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
    Desc.StructureByteStride = sizeof(int); 
    Desc.ByteWidth = Desc.StructureByteStride * m_NumBucketBlocks;
    Desc.Usage = D3D11_USAGE_DEFAULT;
	//Desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    V_RETURN( pd3dDevice->CreateBuffer(&Desc, NULL, &m_pBucketBlockResults) );

    ZeroMemory( &DescSRV, sizeof( DescSRV ) );
    DescSRV.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
	//DescRV.Format = DXGI_FORMAT_UNKNOWN;
	DescSRV.Format = DXGI_FORMAT_R32_SINT;
    DescSRV.Buffer.FirstElement = 0;
    DescSRV.Buffer.NumElements = m_NumBucketBlocks;
    V_RETURN( pd3dDevice->CreateShaderResourceView( m_pBucketBlockResults, &DescSRV, &m_pBucketBlockResultsSRV ) );

    ZeroMemory( &DescUAV, sizeof(DescUAV) );
    //DescUAV.Format = DXGI_FORMAT_UNKNOWN;
    DescUAV.Format = DXGI_FORMAT_R32_SINT;
	DescUAV.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
    DescUAV.Buffer.FirstElement = 0;
    DescUAV.Buffer.NumElements = m_NumBucketBlocks;
    V_RETURN( pd3dDevice->CreateUnorderedAccessView( m_pBucketBlockResults, &DescUAV, &m_pBucketBlockResultsUAV ) );

	


	ZeroMemory( &Desc, sizeof(D3D11_BUFFER_DESC) );
	Desc.ByteWidth = sizeof(int);
	Desc.Usage = D3D11_USAGE_STAGING;
	Desc.BindFlags = 0;
	Desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
	Desc.StructureByteStride = sizeof(int);
	V_RETURN(pd3dDevice->CreateBuffer(&Desc, NULL, &m_PrefixSumLastElemStagging));


	//D3D11_BUFFER_DESC Desc;
	ZeroMemory( &Desc, sizeof(D3D11_BUFFER_DESC) );
	Desc.ByteWidth = sizeof(int);
	Desc.Usage = D3D11_USAGE_DEFAULT;
	Desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
	Desc.StructureByteStride = sizeof(int);
	V_RETURN(pd3dDevice->CreateBuffer(&Desc, NULL, &m_PrefixSumLastElem));

	//	D3D11_UNORDERED_ACCESS_VIEW_DESC DescUAV;
	ZeroMemory( &DescUAV, sizeof(D3D11_UNORDERED_ACCESS_VIEW_DESC) );
	DescUAV.Format = DXGI_FORMAT_R32_SINT;
	DescUAV.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
	DescUAV.Buffer.FirstElement = 0;
	DescUAV.Buffer.NumElements = 1;
	DescUAV.Buffer.Flags = 0;
	V_RETURN(pd3dDevice->CreateUnorderedAccessView(m_PrefixSumLastElem, &DescUAV, &m_PrefixSumLastElemUAV));

	//	D3D11_SHADER_RESOURCE_VIEW_DESC DescRV;
	ZeroMemory( &DescSRV, sizeof(D3D11_SHADER_RESOURCE_VIEW_DESC) );
	DescSRV.Format = DXGI_FORMAT_R32_SINT;
	DescSRV.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
	DescSRV.Buffer.FirstElement = 0;
	DescSRV.Buffer.NumElements = 1;
	V_RETURN( pd3dDevice->CreateShaderResourceView( m_PrefixSumLastElem, &DescSRV, &m_PrefixSumLastElemSRV ));


	Desc.Usage = D3D11_USAGE_DYNAMIC;
	Desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	Desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	Desc.MiscFlags = 0;    
	Desc.ByteWidth = sizeof( CBScanCS );
	V_RETURN( pd3dDevice->CreateBuffer( &Desc, NULL, &m_pcbCS ) );


	m_bIsCreated = true;
    return hr;
}

void DX11ScanCS::OnD3D11DestroyDevice()
{
	SAFE_RELEASE( m_pBucketResultsSRV );
	SAFE_RELEASE( m_pBucketResultsUAV );
	SAFE_RELEASE( m_pBucketResults );

    SAFE_RELEASE( m_pBucketBlockResultsSRV );
    SAFE_RELEASE( m_pBucketBlockResultsUAV );
    SAFE_RELEASE( m_pBucketBlockResults );

    SAFE_RELEASE( m_pcbCS );;

	SAFE_RELEASE(m_PrefixSumLastElemStagging);
	SAFE_RELEASE(m_PrefixSumLastElem);
	SAFE_RELEASE(m_PrefixSumLastElemUAV);
	SAFE_RELEASE(m_PrefixSumLastElemSRV);

	SAFE_RELEASE( m_pScanBucketsCS );
	SAFE_RELEASE( m_pScanBucketResultsCS );
	SAFE_RELEASE( m_pScanBucketBlockResultsCS );
	SAFE_RELEASE( m_pScanApplyBucketBlockResultsCS );
	SAFE_RELEASE( m_pScanApplyBucketResultsCS );

	m_bIsCreated = false;
}


UINT DX11ScanCS::ScanCS( ID3D11DeviceContext* pd3dImmediateContext,
	UINT numElements,
	ID3D11ShaderResourceView* input_SRV,
	ID3D11UnorderedAccessView* input_UAV,
	ID3D11ShaderResourceView* output_SRV,
	ID3D11UnorderedAccessView* output_UAV,
	ID3D11Buffer* output_Buf)
{
	assert(numElements <= GetMaxScanSize());

	ID3D11UnorderedAccessView* ppUAViewNULL[] = { NULL, NULL };
	ID3D11ShaderResourceView* ppSRVNULL[] = { NULL, NULL };

	/////////
	// PASS 1 Up-sweep
	/////////		
	pd3dImmediateContext->CSSetShader(m_pScanBucketsCS, NULL, 0);

	pd3dImmediateContext->CSSetShaderResources(0, 1, &input_SRV);
	pd3dImmediateContext->CSSetUnorderedAccessViews(0, 1, &output_UAV, NULL);

	UINT groupsPass1 = (numElements + m_BucketSize - 1) / m_BucketSize;
	UINT dimX = 128;
	UINT dimY = (groupsPass1 + dimX - 1) / dimX;
	assert(dimX <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);
	assert(dimY <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);
	pd3dImmediateContext->Dispatch(dimX,dimY,1);
	//pd3dImmediateContext->Dispatch(groupsPass1,1,1);

	pd3dImmediateContext->CSSetUnorderedAccessViews(0, 1, ppUAViewNULL,NULL);
	pd3dImmediateContext->CSSetShaderResources( 0, 1, ppSRVNULL );

	//ID3D11Buffer* tmpbuf;
	//output_UAV->GetResource((ID3D11Resource**)&tmpbuf);
	//INT* cpuMemory1 = (INT*)CreateAndCopyToDebugBuf(g_pd3dDevice, pd3dImmediateContext, tmpbuf, true);


	/////////
	// PASS 2 Up-sweep
	/////////		
	pd3dImmediateContext->CSSetShader(m_pScanBucketResultsCS, NULL, 0);

	pd3dImmediateContext->CSSetShaderResources(0, 1, &output_SRV);
	pd3dImmediateContext->CSSetUnorderedAccessViews(0, 1, &m_pBucketResultsUAV, NULL);

	UINT groupsPass2 = (groupsPass1 + m_BucketBlockSize - 1) / m_BucketBlockSize;
	pd3dImmediateContext->Dispatch(groupsPass2,1,1);
	assert(groupsPass2 <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);
			

	pd3dImmediateContext->CSSetUnorderedAccessViews(0, 1, ppUAViewNULL,NULL);
	pd3dImmediateContext->CSSetShaderResources( 0, 1, ppSRVNULL );

	//INT* cpuMemory2 = (INT*)CreateAndCopyToDebugBuf(g_pd3dDevice, pd3dImmediateContext, m_pBucketResults, true);

	//////////
	// PASS 3 Up-sweep
	//////////
	pd3dImmediateContext->CSSetShader(m_pScanBucketBlockResultsCS, NULL, 0);

	pd3dImmediateContext->CSSetShaderResources(0, 1, &m_pBucketResultsSRV);
	pd3dImmediateContext->CSSetUnorderedAccessViews(0, 1, &m_pBucketBlockResultsUAV, NULL);

	pd3dImmediateContext->Dispatch(1,1,1);

	pd3dImmediateContext->CSSetUnorderedAccessViews(0, 1, ppUAViewNULL,NULL);
	pd3dImmediateContext->CSSetShaderResources( 0, 1, ppSRVNULL );

	//INT* cpuMemory3 = (INT*)CreateAndCopyToDebugBuf(g_pd3dDevice, pd3dImmediateContext, m_pBucketBlockResults, true);

	/////////
	// PASS 4 Down-sweep
	/////////
	pd3dImmediateContext->CSSetShader(m_pScanApplyBucketBlockResultsCS, NULL, 0);

	pd3dImmediateContext->CSSetShaderResources(0, 1, &m_pBucketBlockResultsSRV);
	pd3dImmediateContext->CSSetUnorderedAccessViews(0, 1, &m_pBucketResultsUAV, NULL);

	pd3dImmediateContext->Dispatch(groupsPass2,1,1);
	assert(groupsPass2 <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);

	pd3dImmediateContext->CSSetUnorderedAccessViews(0, 1, ppUAViewNULL,NULL);
	pd3dImmediateContext->CSSetShaderResources( 0, 1, ppSRVNULL );

	//INT* cpuMemory4 = (INT*)CreateAndCopyToDebugBuf(g_pd3dDevice, pd3dImmediateContext, m_pBucketResults, true);

	/////////
	// PASS 5 Down-sweep
	/////////
	pd3dImmediateContext->CSSetShader(m_pScanApplyBucketResultsCS, NULL, 0);

	pd3dImmediateContext->CSSetShaderResources(0, 1, &m_pBucketResultsSRV);
	pd3dImmediateContext->CSSetUnorderedAccessViews(0, 1, &output_UAV, NULL);

	pd3dImmediateContext->Dispatch(dimX,dimY,1);
	assert(dimX <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);
	assert(dimY <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);

	//pd3dImmediateContext->Dispatch(groupsPass1,1,1);

	pd3dImmediateContext->CSSetUnorderedAccessViews(0, 1, ppUAViewNULL,NULL);
	pd3dImmediateContext->CSSetShaderResources( 0, 1, ppSRVNULL );

	////ID3D11Buffer* tmpbuf;
	////output_UAV->GetResource((ID3D11Resource**)&tmpbuf);
	//INT* cpuMemory5 = (INT*)CreateAndCopyToDebugBuf(g_pd3dDevice, pd3dImmediateContext, tmpbuf, true);

	//ID3D11Buffer* tmpbufinput;
	//input_UAV->GetResource((ID3D11Resource**)&tmpbufinput);
	//INT* cpuInput = (INT*)CreateAndCopyToDebugBuf(g_pd3dDevice, pd3dImmediateContext, tmpbufinput, true);

	//UINT size = 64;
	//INT* ref1 = new INT[numElements];
	//for (unsigned int i = 0; i < numElements; i = i + size) {
	//	INT prev = 0;
	//	for (unsigned int k = 0; k < size; k++) {
	//		ref1[i+k] = prev + cpuInput[i+k];
	//		prev = ref1[i+k];
	//	}
	//}
	//for (unsigned int i = 0; i < numElements; i++) {
	//	assert(ref1[i] == cpuMemory1[i]);
	//}
	//INT* ref2 = new INT[numElements/size];
	//for (unsigned int i = 0; i < numElements/size; i = i +size) {
	//	INT prev = 0;
	//	for (unsigned int k = 0; k < size; k++) {

	//		ref2[i+k] = prev + ref1[(i+k)*size+size-1];
	//		prev = ref2[i+k];
	//	}
	//}
	//for (unsigned int i = 0; i < numElements/size; i++) {
	//	assert(ref2[i] == cpuMemory2[i]);
	//}

	//INT* ref = new INT[numElements];
	//INT prev = 0;
	//for (int i = 0; i < numElements; i++) {
	//	ref[i] = prev + cpuInput[i];
	//	prev = ref[i];
	//}
	//for (int i = 0; i < numElements; i++) {
	//	assert(ref[i] == cpuMemory5[i]);
	//}




	int sum = 0;
	if (output_Buf) {	//if output buffer is set return value
		D3D11_BOX sourceRegion;
		sourceRegion.left = 4*(numElements-1);
		sourceRegion.right = 4*numElements;
		sourceRegion.top = sourceRegion.front = 0;
		sourceRegion.bottom = sourceRegion.back = 1;

		// Get last element of the prefix sum (sum of all elements)
		pd3dImmediateContext->CopySubresourceRegion( m_PrefixSumLastElemStagging, 0, 0, 0, 0, 
			output_Buf, 0, &sourceRegion);

		D3D11_MAPPED_SUBRESOURCE MappedResource;
		pd3dImmediateContext->Map(m_PrefixSumLastElemStagging, 0, D3D11_MAP_READ, 0, &MappedResource);   
		sum = ((int*)MappedResource.pData)[0];

		pd3dImmediateContext->Unmap( m_PrefixSumLastElemStagging, 0 );
	}
	return sum;
}



UINT DX11ScanCS::GetMaxScanSize()
{
	return m_NumBuckets * m_BucketSize;
}

