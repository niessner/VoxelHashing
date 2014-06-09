#include "stdafx.h"

#include "DX11RayMarchingStepsSplatting.h"

#include "GlobalAppState.h"


////////////////////////////////////////////////////////////////////
// General
////////////////////////////////////////////////////////////////////

unsigned int DX11RayMarchingStepsSplatting::m_blockSize = 8;

ID3D11VertexShader* DX11RayMarchingStepsSplatting::s_pVertexShaderSplatting = NULL;
ID3D11GeometryShader* DX11RayMarchingStepsSplatting::s_pGeometryShaderSplatting = NULL;

ID3D11Buffer* DX11RayMarchingStepsSplatting::s_ConstantBufferSplatting = NULL;

ID3D11DepthStencilState* DX11RayMarchingStepsSplatting::s_pDepthStencilStateSplatting = NULL;

ID3D11BlendState* DX11RayMarchingStepsSplatting::s_pBlendStateDefault = NULL;
ID3D11BlendState* DX11RayMarchingStepsSplatting::s_pBlendStateColorWriteDisabled = NULL;

ID3D11RasterizerState* DX11RayMarchingStepsSplatting::s_pRastState = NULL;

////////////////////////////////////////////////////////////////////
// Pre-pass // Which blocks should be splatted
////////////////////////////////////////////////////////////////////

ID3D11ComputeShader* DX11RayMarchingStepsSplatting::s_pComputeShaderDecisionArray = NULL;

ID3D11Buffer* DX11RayMarchingStepsSplatting::s_pDecisionArrayBuffer = NULL;
ID3D11ShaderResourceView* DX11RayMarchingStepsSplatting::s_pDecisionArrayBufferSRV = NULL;
ID3D11UnorderedAccessView* DX11RayMarchingStepsSplatting::s_pDecisionArrayBufferUAV = NULL;

////////////////////////////////////////////////////////////////////
// For first pass // Count fragments per pixel
////////////////////////////////////////////////////////////////////

ID3D11ComputeShader* DX11RayMarchingStepsSplatting::s_pComputeShaderClear = NULL;

ID3D11PixelShader* DX11RayMarchingStepsSplatting::s_pPixelShaderSplatting_Count = NULL;

ID3D11Buffer* DX11RayMarchingStepsSplatting::m_FragmentCountBuffer = NULL;
ID3D11ShaderResourceView* DX11RayMarchingStepsSplatting::m_FragmentCountBufferSRV = NULL;
ID3D11UnorderedAccessView* DX11RayMarchingStepsSplatting::m_FragmentCountBufferUAV = NULL;

////////////////////////////////////////////////////////////////////
// For second pass // Perform prefix sum
////////////////////////////////////////////////////////////////////

DX11ScanCS* DX11RayMarchingStepsSplatting::m_Scan = NULL;

ID3D11Buffer* DX11RayMarchingStepsSplatting::m_FragmentPrefixSumBuffer = NULL;
ID3D11ShaderResourceView* DX11RayMarchingStepsSplatting::m_FragmentPrefixSumBufferSRV = NULL;
ID3D11UnorderedAccessView* DX11RayMarchingStepsSplatting::m_FragmentPrefixSumBufferUAV = NULL;

////////////////////////////////////////////////////////////////////
// For third pass // Write fragments
////////////////////////////////////////////////////////////////////

unsigned int DX11RayMarchingStepsSplatting::m_maxNumberOfPossibleFragments = 20000000;

ID3D11Buffer* DX11RayMarchingStepsSplatting::m_FragmentSortedDepthBuffer = NULL;
ID3D11ShaderResourceView* DX11RayMarchingStepsSplatting::m_FragmentSortedDepthBufferSRV = NULL;
ID3D11UnorderedAccessView* DX11RayMarchingStepsSplatting::m_FragmentSortedDepthBufferUAV = NULL;

ID3D11PixelShader* DX11RayMarchingStepsSplatting::s_pPixelShaderSplatting_Write = NULL;

////////////////////////////////////////////////////////////////////
// For forth pass // Sort fragments
////////////////////////////////////////////////////////////////////

ID3D11ComputeShader* DX11RayMarchingStepsSplatting::m_pComputeShaderSort = NULL;

////////////////////////////////////////////////////////////////////
// Timer
////////////////////////////////////////////////////////////////////

Timer DX11RayMarchingStepsSplatting::m_timer;

HRESULT DX11RayMarchingStepsSplatting::OnD3D11CreateDevice( ID3D11Device* pd3dDevice )
{
	HRESULT hr = S_OK;

	V_RETURN(initialize(pd3dDevice));

	return  hr;
}

void DX11RayMarchingStepsSplatting::OnD3D11DestroyDevice()
{
	destroy();
}

HRESULT DX11RayMarchingStepsSplatting::generalSetup( ID3D11DeviceContext* context, ID3D11ShaderResourceView* hash, ID3D11ShaderResourceView* SDFBlocksSDFSRV, ID3D11ShaderResourceView* SDFBlocksRGBWSRV, unsigned int hashNumValidBuckets, const mat4f* lastRigidTransform, unsigned int renderTargetWidth, unsigned int renderTargetHeight, ID3D11Buffer* CBsceneRepSDF )
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
	context->Unmap(s_ConstantBufferSplatting, 0);

	// Clear Count Buffer
	context->CSSetConstantBuffers(1, 1, &s_ConstantBufferSplatting);
	context->CSSetUnorderedAccessViews(0, 1, &m_FragmentCountBufferUAV, 0);
	context->CSSetShader(s_pComputeShaderClear, 0, 0);

	unsigned int dimX = (unsigned int)ceil(((float)renderTargetWidth*renderTargetHeight)/(m_blockSize*m_blockSize));
	context->Dispatch(dimX, 1, 1);
	assert(dimX <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);

	ID3D11UnorderedAccessView* nullUAVCSClear[] = { NULL };
	ID3D11Buffer* nullCBCSClear[1] = { NULL };

	context->CSSetUnorderedAccessViews(0, 1, nullUAVCSClear, 0);
	context->CSSetConstantBuffers(1, 1, nullCBCSClear);
	context->CSSetShader(0, 0, 0);

	return hr;
}

void DX11RayMarchingStepsSplatting::prePass( ID3D11DeviceContext* context, ID3D11ShaderResourceView* hash, ID3D11ShaderResourceView* SDFBlocksSDFSRV, ID3D11ShaderResourceView* SDFBlocksRGBWSRV, unsigned int hashNumValidBuckets, ID3D11Buffer* CBsceneRepSDF )
{
	ID3D11ShaderResourceView* srvsNULL[] = { NULL, NULL, NULL };
	ID3D11UnorderedAccessView* uavsNULL[] = { NULL, NULL, NULL };
	ID3D11Buffer* cbsNULL[] = { NULL, NULL, NULL };

	context->CSSetShaderResources(0, 1, &hash);
	context->CSSetShaderResources(5, 1, &SDFBlocksSDFSRV);
	context->CSSetShaderResources(7, 1, &SDFBlocksRGBWSRV);

	context->CSSetUnorderedAccessViews(6, 1, &s_pDecisionArrayBufferUAV, NULL);

	context->CSSetConstantBuffers(0, 1, &CBsceneRepSDF);
	ID3D11Buffer* CBGlobalAppState = GlobalAppState::getInstance().MapAndGetConstantBuffer(context);
	context->CSSetConstantBuffers(8, 1, &CBGlobalAppState);

	context->CSSetShader(s_pComputeShaderDecisionArray, NULL, 0);

	unsigned int dimX = NUM_GROUPS_X;
	unsigned int dimY = (hashNumValidBuckets + NUM_GROUPS_X - 1) / NUM_GROUPS_X;
	assert(dimX <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);
	assert(dimY <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);

	context->Dispatch(dimX, dimY, 1);

	context->CSSetShaderResources(0, 1, srvsNULL);
	context->CSSetShaderResources(5, 1, srvsNULL);
	context->CSSetShaderResources(7, 1, srvsNULL);
	context->CSSetUnorderedAccessViews(6, 1, uavsNULL, NULL);
	context->CSSetConstantBuffers(0, 1, cbsNULL);
	context->CSSetConstantBuffers(8, 1, cbsNULL);
	context->CSSetShader(0, 0, 0);

	//debugDecisionArrayBuffer(context);
}

void DX11RayMarchingStepsSplatting::firstPass( ID3D11DeviceContext* context, ID3D11ShaderResourceView* hash, unsigned int hashNumValidBuckets,ID3D11Buffer* CBsceneRepSDF )
{
	ID3D11ShaderResourceView* srvs[] = { m_FragmentCountBufferSRV, m_FragmentPrefixSumBufferSRV, m_FragmentSortedDepthBufferSRV };
	ID3D11UnorderedAccessView* uavs[] = { m_FragmentCountBufferUAV, m_FragmentPrefixSumBufferUAV, m_FragmentSortedDepthBufferUAV };
	ID3D11ShaderResourceView* srvsNULL[] = { NULL, NULL, NULL };
	ID3D11UnorderedAccessView* uavsNULL[] = { NULL, NULL, NULL };
	ID3D11Buffer* cbsNULL[] = { NULL, NULL, NULL };

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
	context->GSSetShaderResources(4, 1, &s_pDecisionArrayBufferSRV);

	context->GSSetShader(s_pGeometryShaderSplatting, 0, 0);

	context->PSSetShader(s_pPixelShaderSplatting_Count, 0, 0);
	context->PSSetConstantBuffers(1, 1, &s_ConstantBufferSplatting);
	context->OMSetRenderTargetsAndUnorderedAccessViews(0, NULL, NULL, 0, 3, uavs, NULL);

	// Splat
	context->Draw(hashNumValidBuckets, 0);

	// Reset
	context->OMSetRenderTargetsAndUnorderedAccessViews(0, NULL, NULL, 0, 3, uavsNULL, NULL);

	ID3D11RenderTargetView* rtv = DXUTGetD3D11RenderTargetView();
	ID3D11DepthStencilView* dsv = DXUTGetD3D11DepthStencilView();
	context->OMSetRenderTargets(1, &rtv, dsv);

	ID3D11ShaderResourceView* nullSRV[] = { NULL };
	ID3D11UnorderedAccessView* nullUAV[] = { NULL };
	ID3D11Buffer* nullCB[] = { NULL, NULL };

	context->GSSetConstantBuffers(0, 2, nullCB);
	context->GSSetShaderResources(0, 1, nullSRV);
	context->GSSetShaderResources(4, 1, nullSRV);
	context->GSSetConstantBuffers(8, 1, nullCB);
	context->PSSetConstantBuffers(1, 1, nullCB);

	context->VSSetShader(0, 0, 0);
	context->GSSetShader(0, 0, 0);
	context->PSSetShader(0, 0, 0);

	//debugCountBuffer(context);
}

void DX11RayMarchingStepsSplatting::thirdPass( ID3D11DeviceContext* context, ID3D11ShaderResourceView* hash, unsigned int hashNumValidBuckets,ID3D11Buffer* CBsceneRepSDF )
{
	ID3D11ShaderResourceView* srvs[] = { m_FragmentCountBufferSRV, m_FragmentPrefixSumBufferSRV, m_FragmentSortedDepthBufferSRV };
	ID3D11UnorderedAccessView* uavs3[] = { m_FragmentCountBufferUAV, NULL, m_FragmentSortedDepthBufferUAV };
	ID3D11ShaderResourceView* srvsNULL[] = { NULL, NULL, NULL };
	ID3D11UnorderedAccessView* uavsNULL[] = { NULL, NULL, NULL };
	ID3D11Buffer* cbsNULL[] = { NULL, NULL, NULL };

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
	context->GSSetShaderResources(4, 1, &s_pDecisionArrayBufferSRV);

	context->GSSetShader(s_pGeometryShaderSplatting, 0, 0);

	context->PSSetShader(s_pPixelShaderSplatting_Write, 0, 0);
	context->PSSetConstantBuffers(1, 1, &s_ConstantBufferSplatting);
	context->PSSetShaderResources(2, 1, &m_FragmentPrefixSumBufferSRV);
	context->OMSetRenderTargetsAndUnorderedAccessViews(0, NULL, NULL, 0, 3, uavs3, NULL);

	// Splat
	context->Draw(hashNumValidBuckets, 0);

	// Reset
	context->OMSetRenderTargetsAndUnorderedAccessViews(0, NULL, NULL, 0, 3, uavsNULL, NULL);

	ID3D11RenderTargetView* rtv = DXUTGetD3D11RenderTargetView();
	ID3D11DepthStencilView* dsv = DXUTGetD3D11DepthStencilView();
	context->OMSetRenderTargets(1, &rtv, dsv);

	ID3D11ShaderResourceView* nullSRV[] = { NULL };
	ID3D11UnorderedAccessView* nullUAV[] = { NULL };
	ID3D11Buffer* nullCB[] = { NULL, NULL };

	context->GSSetConstantBuffers(0, 2, nullCB);
	context->GSSetShaderResources(0, 1, nullSRV);
	context->GSSetShaderResources(4, 1, nullSRV);
	context->GSSetConstantBuffers(8, 1, nullCB);
	context->PSSetConstantBuffers(1, 1, nullCB);
	context->PSSetShaderResources(2, 1, nullSRV);

	context->VSSetShader(0, 0, 0);
	context->GSSetShader(0, 0, 0);
	context->PSSetShader(0, 0, 0);

	//debugDepthBuffer(context);
}

void DX11RayMarchingStepsSplatting::fourthPass( ID3D11DeviceContext* context, ID3D11ShaderResourceView* hash, unsigned int renderTargetWidth, unsigned int renderTargetHeight, unsigned int hashNumValidBuckets,ID3D11Buffer* CBsceneRepSDF )
{
	ID3D11ShaderResourceView* srvs4[] = { m_FragmentCountBufferSRV, m_FragmentPrefixSumBufferSRV, NULL };
	ID3D11UnorderedAccessView* uavs4[] = { NULL, NULL, m_FragmentSortedDepthBufferUAV };
	ID3D11ShaderResourceView* srvsNULL[] = { NULL, NULL, NULL };
	ID3D11UnorderedAccessView* uavsNULL[] = { NULL, NULL, NULL };

	context->CSSetShaderResources(1, 3, srvs4);
	context->CSSetUnorderedAccessViews(0, 3, uavs4, NULL);

	// Setup Pipeline
	context->CSSetConstantBuffers(1, 1, &s_ConstantBufferSplatting);
	context->CSSetShader(m_pComputeShaderSort, 0, 0);

	// Start Compute Shader
	unsigned int dimX = (unsigned int)ceil(((float)renderTargetWidth*renderTargetHeight)/(m_blockSize*m_blockSize));
	assert(dimX <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);
	context->Dispatch(dimX, 1, 1);

	// De-Initialize Pipeline
	context->CSSetShaderResources(1, 3, srvsNULL);
	context->CSSetUnorderedAccessViews(0, 3, uavsNULL, NULL);

	ID3D11Buffer* nullCBCS[1] = { NULL };
	context->CSSetConstantBuffers(0, 1, nullCBCS);
	context->CSSetShader(0, 0, 0);

	//debugSortedDepthBuffer(context);
}

HRESULT DX11RayMarchingStepsSplatting::rayMarchingStepsSplatting( ID3D11DeviceContext* context, ID3D11ShaderResourceView* hash, ID3D11ShaderResourceView* SDFBlocksSDFSRV, ID3D11ShaderResourceView* SDFBlocksRGBWSRV, unsigned int hashNumValidBuckets, const mat4f* lastRigidTransform, unsigned int renderTargetWidth, unsigned int renderTargetHeight, ID3D11Buffer* CBsceneRepSDF )
{
	HRESULT hr = S_OK;

	// Start query for timing
	if(GlobalAppState::getInstance().s_timingsDetailledEnabled)
	{
		GlobalAppState::getInstance().WaitForGPU();
		m_timer.start();
	}

	// General setup
	V_RETURN(generalSetup(context, hash, SDFBlocksSDFSRV, SDFBlocksRGBWSRV, hashNumValidBuckets, lastRigidTransform, renderTargetWidth, renderTargetHeight, CBsceneRepSDF));

	// Pre-Pass: Decision Array
	prePass(context, hash, SDFBlocksSDFSRV, SDFBlocksRGBWSRV,  hashNumValidBuckets, CBsceneRepSDF);

	// Setup Pipeline
	context->OMSetDepthStencilState(s_pDepthStencilStateSplatting, 0);
	context->OMSetBlendState(s_pBlendStateColorWriteDisabled, NULL, 0xffffffff);
	context->RSSetState(s_pRastState);

	// First pass // Count fragments per pixel
	firstPass(context, hash, hashNumValidBuckets, CBsceneRepSDF);

	// Second pass // Perform prefix sum
	m_Scan->ScanCS(context, GlobalAppState::getInstance().s_windowWidth * GlobalAppState::getInstance().s_windowHeight, m_FragmentCountBufferSRV, m_FragmentCountBufferUAV, m_FragmentPrefixSumBufferSRV, m_FragmentPrefixSumBufferUAV, NULL);
	//debugPrefixSumBuffer(context);

	// Third pass // Write fragments
	thirdPass(context, hash, hashNumValidBuckets, CBsceneRepSDF);

	context->RSSetState(0);
	context->OMSetDepthStencilState(s_pDepthStencilStateSplatting, 0); //is also default state
	context->OMSetBlendState(s_pBlendStateDefault, NULL, 0xffffffff);

	// Forth pass // Sort fragments
	fourthPass(context, hash, renderTargetWidth, renderTargetHeight, hashNumValidBuckets, CBsceneRepSDF);

	// Wait for query
	if(GlobalAppState::getInstance().s_timingsDetailledEnabled)
	{
		GlobalAppState::getInstance().WaitForGPU();
		TimingLog::totalTimeRayMarchingStepsSplatting+=m_timer.getElapsedTimeMS();
		TimingLog::countRayMarchingStepsSplatting++;
	}

	return hr;
}

void DX11RayMarchingStepsSplatting::debugCountBuffer( ID3D11DeviceContext* context )
{
	int* countBuffer = (int*)CreateAndCopyToDebugBuf(DXUTGetD3D11Device(), context, m_FragmentCountBuffer, true);

	int minVal = countBuffer[0];
	int maxVal = countBuffer[0];
	float avgVal = 0;

	const unsigned int n = GlobalAppState::getInstance().s_windowWidth*GlobalAppState::getInstance().s_windowHeight;
	for(unsigned int i = 0; i<n; i++)
	{
		avgVal += countBuffer[i];
		minVal = std::min(minVal, countBuffer[i]);
		maxVal = std::max(maxVal, countBuffer[i]);
	}

	avgVal /= n;


	int b = 0;
	delete countBuffer;
}

void DX11RayMarchingStepsSplatting::debugPrefixSumBuffer( ID3D11DeviceContext* context )
{
	int* countBuffer = (int*)CreateAndCopyToDebugBuf(DXUTGetD3D11Device(), context, m_FragmentCountBuffer, true);
	int* countPrefix = (int*)CreateAndCopyToDebugBuf(DXUTGetD3D11Device(), context, m_FragmentPrefixSumBuffer, true);

	int minVal = std::numeric_limits<int>::max();
	int maxVal = std::numeric_limits<int>::min();
	float avgVal = 0.0f;

	const unsigned int n = GlobalAppState::getInstance().s_windowWidth*GlobalAppState::getInstance().s_windowHeight;
	for(unsigned int i = 0; i<n; i++)
	{
		int startIndex = 0;
		if(i > 0) startIndex = countPrefix[i-1];
		int endIndex = countPrefix[i]-1;
		int currentIndex = startIndex;

		int length = endIndex-startIndex+1;

		avgVal += length;
		minVal = std::min(minVal, length);
		maxVal = std::max(maxVal, length);

		unsigned int iter = 0;
		while(currentIndex <= endIndex)
		{
			++currentIndex;
			iter++;
		}

		if(iter > 2)
		{
			std::cout << iter << std::endl;
		}
	}

	avgVal /= n;


	int b = 0;
	delete countBuffer;
	delete countPrefix;
}

void DX11RayMarchingStepsSplatting::debugDepthBuffer( ID3D11DeviceContext* context )
{
	int* countBuffer = (int*)CreateAndCopyToDebugBuf(DXUTGetD3D11Device(), context, m_FragmentCountBuffer, true);
	int* countPrefix = (int*)CreateAndCopyToDebugBuf(DXUTGetD3D11Device(), context, m_FragmentPrefixSumBuffer, true);
	float* depthBuffer = (float*)CreateAndCopyToDebugBuf(DXUTGetD3D11Device(), context, m_FragmentSortedDepthBuffer, true);

	int b = 0;
	delete countBuffer;
	delete countPrefix;
	delete depthBuffer;
}

void DX11RayMarchingStepsSplatting::debugSortedDepthBuffer( ID3D11DeviceContext* context )
{
	int* countBuffer = (int*)CreateAndCopyToDebugBuf(DXUTGetD3D11Device(), context, m_FragmentCountBuffer, true);
	int* countPrefix = (int*)CreateAndCopyToDebugBuf(DXUTGetD3D11Device(), context, m_FragmentPrefixSumBuffer, true);
	float* depthBufferSorted = (float*)CreateAndCopyToDebugBuf(DXUTGetD3D11Device(), context, m_FragmentSortedDepthBuffer, true);

	int b = 0;
	delete countBuffer;
	delete countPrefix;
	delete depthBufferSorted;
}
