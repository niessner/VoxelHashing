#pragma once

/***************************************************************************************/
/* DX11 Splatting class for SDF blocks to gain good start depth values for ray casting */
/***************************************************************************************/

#include "stdafx.h"

#include <D3D11.h>
#include "DXUT.h"
#include "DX11Utils.h"
#include "TimingLog.h"
#include "DX11ImageHelper.h"
#include "DX11QuadDrawer.h"
#include "DX11ScanCS.h"
#include "GlobalAppState.h"

class DX11RayMarchingStepsSplatting
{
	#define NUM_GROUPS_X 1024 // to be in-sync with the define in the shader

	public:
		
		static HRESULT OnD3D11CreateDevice(ID3D11Device* pd3dDevice);

		static void OnD3D11DestroyDevice();


		////////////////////////////////////////////////////////////////////
		// Ray Marching Steps Splatting
		////////////////////////////////////////////////////////////////////

		static HRESULT rayMarchingStepsSplatting(ID3D11DeviceContext* context, ID3D11ShaderResourceView* hash, ID3D11ShaderResourceView* SDFBlocksSDFSRV, ID3D11ShaderResourceView* SDFBlocksRGBWSRV, unsigned int hashNumValidBuckets, const mat4f* lastRigidTransform, unsigned int renderTargetWidth, unsigned int renderTargetHeight, ID3D11Buffer* CBsceneRepSDF);
		
		static void debugCountBuffer(ID3D11DeviceContext* context);

		static void debugPrefixSumBuffer(ID3D11DeviceContext* context);

		static void debugDepthBuffer(ID3D11DeviceContext* context);

		static void debugSortedDepthBuffer(ID3D11DeviceContext* context);

		static void debugDecisionArrayBuffer(ID3D11DeviceContext* context)	{
			int* decisionArrayBuffer = (int*)CreateAndCopyToDebugBuf(DXUTGetD3D11Device(), context, s_pDecisionArrayBuffer, true);

			int b = 0;
			delete decisionArrayBuffer;
		}


		static ID3D11ShaderResourceView* getFragmentPrefixSumBufferSRV() {
			return m_FragmentPrefixSumBufferSRV;
		}

		static ID3D11ShaderResourceView* getFragmentSortedDepthBufferSRV()	{
			return m_FragmentSortedDepthBufferSRV;
		}

	private:
		////////////////////////////////////////////////////////////////////
		// General setup
		////////////////////////////////////////////////////////////////////

		static HRESULT generalSetup(ID3D11DeviceContext* context, ID3D11ShaderResourceView* hash, ID3D11ShaderResourceView* SDFBlocksSDFSRV, ID3D11ShaderResourceView* SDFBlocksRGBWSRV, unsigned int hashNumValidBuckets, const mat4f* lastRigidTransform, unsigned int renderTargetWidth, unsigned int renderTargetHeight, ID3D11Buffer* CBsceneRepSDF);

		////////////////////////////////////////////////////////////////////
		// Pre-Pass // Which blocks should be splatted
		////////////////////////////////////////////////////////////////////

		static void prePass(ID3D11DeviceContext* context, ID3D11ShaderResourceView* hash, ID3D11ShaderResourceView* SDFBlocksSDFSRV, ID3D11ShaderResourceView* SDFBlocksRGBWSRV, unsigned int hashNumValidBuckets, ID3D11Buffer* CBsceneRepSDF);

		////////////////////////////////////////////////////////////////////
		// First pass // Count fragments per pixel
		////////////////////////////////////////////////////////////////////

		static void firstPass(ID3D11DeviceContext* context, ID3D11ShaderResourceView* hash, unsigned int hashNumValidBuckets,ID3D11Buffer* CBsceneRepSDF);

		////////////////////////////////////////////////////////////////////
		// Third pass // Write fragments
		////////////////////////////////////////////////////////////////////

		static void thirdPass(ID3D11DeviceContext* context, ID3D11ShaderResourceView* hash, unsigned int hashNumValidBuckets,ID3D11Buffer* CBsceneRepSDF);

		////////////////////////////////////////////////////////////////////
		// Fourth pass // Sort fragments
		////////////////////////////////////////////////////////////////////

		static void fourthPass(ID3D11DeviceContext* context, ID3D11ShaderResourceView* hash, unsigned int renderTargetWidth, unsigned int renderTargetHeight, unsigned int hashNumValidBuckets,ID3D11Buffer* CBsceneRepSDF);



		static HRESULT initialize(ID3D11Device* pd3dDevice)
		{
			HRESULT hr = S_OK;

			////////////////////////////////////////////////////////////////////
			// General setup
			////////////////////////////////////////////////////////////////////

			ID3DBlob* pBlob = NULL;
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

			V_RETURN(CompileShaderFromFile(L"Shaders\\RayMarchingStepsSplatting.hlsl", "VS", "vs_5_0", &pBlob, validDefines));
			V_RETURN(pd3dDevice->CreateVertexShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &s_pVertexShaderSplatting));
			SAFE_RELEASE(pBlob);
	
			V_RETURN(CompileShaderFromFile(L"Shaders\\RayMarchingStepsSplatting.hlsl", "GS", "gs_5_0", &pBlob, validDefines));
			V_RETURN(pd3dDevice->CreateGeometryShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &s_pGeometryShaderSplatting));
			SAFE_RELEASE(pBlob);

			D3D11_BUFFER_DESC cbDesc;
			ZeroMemory(&cbDesc, sizeof(D3D11_BUFFER_DESC));
			cbDesc.Usage = D3D11_USAGE_DYNAMIC;
			cbDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
			cbDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
			cbDesc.MiscFlags = 0;
			cbDesc.ByteWidth = sizeof(CBuffer);
			V_RETURN(pd3dDevice->CreateBuffer(&cbDesc, NULL, &s_ConstantBufferSplatting));

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
			stenDesc.DepthEnable = false;
			stenDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;
			stenDesc.StencilEnable = false;
			V_RETURN(pd3dDevice->CreateDepthStencilState(&stenDesc, &s_pDepthStencilStateSplatting))

			// Rasterizer Stage
			D3D11_RASTERIZER_DESC rastDesc;
			ZeroMemory(&rastDesc, sizeof(D3D11_RASTERIZER_DESC));
			rastDesc.FillMode = D3D11_FILL_SOLID;
			rastDesc.CullMode = D3D11_CULL_NONE;
			rastDesc.FrontCounterClockwise = false;
			V_RETURN(pd3dDevice->CreateRasterizerState(&rastDesc, &s_pRastState))

			////////////////////////////////////////////////////////////////////
			// Pre-pass // Which blocks should be splatted
			////////////////////////////////////////////////////////////////////

			V_RETURN(CompileShaderFromFile(L"Shaders\\RayMarchingStepsSplatting.hlsl", "splatIdentifyCS", "cs_5_0", &pBlob, validDefines));
			V_RETURN(pd3dDevice->CreateComputeShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &s_pComputeShaderDecisionArray));
			SAFE_RELEASE(pBlob);

			D3D11_BUFFER_DESC bDesc;
			ZeroMemory(&bDesc, sizeof(D3D11_BUFFER_DESC));
			bDesc.BindFlags	= D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
			bDesc.Usage	= D3D11_USAGE_DEFAULT;
			bDesc.CPUAccessFlags = 0;
			bDesc.MiscFlags	= 0;
			bDesc.ByteWidth	= sizeof(int) * GlobalAppState::getInstance().s_hashBucketSizeLocal*GlobalAppState::getInstance().s_hashNumBucketsLocal;
			bDesc.StructureByteStride = sizeof(int);

			D3D11_SHADER_RESOURCE_VIEW_DESC SRVDesc;
			ZeroMemory( &SRVDesc, sizeof(D3D11_SHADER_RESOURCE_VIEW_DESC) );
			SRVDesc.Format = DXGI_FORMAT_R32_SINT;
			SRVDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
			SRVDesc.Buffer.FirstElement = 0;
			SRVDesc.Buffer.NumElements = GlobalAppState::getInstance().s_hashBucketSizeLocal*GlobalAppState::getInstance().s_hashNumBucketsLocal;

			D3D11_UNORDERED_ACCESS_VIEW_DESC UAVDesc;
			ZeroMemory( &UAVDesc, sizeof(D3D11_UNORDERED_ACCESS_VIEW_DESC) );
			UAVDesc.Format = DXGI_FORMAT_R32_SINT;;
			UAVDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
			UAVDesc.Buffer.FirstElement = 0;
			UAVDesc.Buffer.NumElements = GlobalAppState::getInstance().s_hashBucketSizeLocal*GlobalAppState::getInstance().s_hashNumBucketsLocal;

			V_RETURN(pd3dDevice->CreateBuffer(&bDesc, NULL, &s_pDecisionArrayBuffer));
			V_RETURN(pd3dDevice->CreateShaderResourceView(s_pDecisionArrayBuffer, &SRVDesc, &s_pDecisionArrayBufferSRV));
			V_RETURN(pd3dDevice->CreateUnorderedAccessView(s_pDecisionArrayBuffer, &UAVDesc, &s_pDecisionArrayBufferUAV));

			////////////////////////////////////////////////////////////////////
			// For first pass // Count fragments per pixel
			////////////////////////////////////////////////////////////////////

			V_RETURN(CompileShaderFromFile(L"Shaders\\RayMarchingStepsSplatting.hlsl", "clearCS", "cs_5_0", &pBlob, validDefines));
			V_RETURN(pd3dDevice->CreateComputeShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &s_pComputeShaderClear));
			SAFE_RELEASE(pBlob);

			V_RETURN(CompileShaderFromFile(L"Shaders\\RayMarchingStepsSplatting.hlsl", "PS_Count", "ps_5_0", &pBlob, validDefines));
			V_RETURN(pd3dDevice->CreatePixelShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &s_pPixelShaderSplatting_Count));
			SAFE_RELEASE(pBlob);

			ZeroMemory(&bDesc, sizeof(D3D11_BUFFER_DESC));
			bDesc.BindFlags	= D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
			bDesc.Usage	= D3D11_USAGE_DEFAULT;
			bDesc.CPUAccessFlags = 0;
			bDesc.MiscFlags	= 0;
			bDesc.ByteWidth	= sizeof(int) * GlobalAppState::getInstance().s_windowWidth * GlobalAppState::getInstance().s_windowHeight; // Has to match the size of the render target
			bDesc.StructureByteStride = sizeof(int);

			ZeroMemory( &SRVDesc, sizeof(D3D11_SHADER_RESOURCE_VIEW_DESC) );
			SRVDesc.Format = DXGI_FORMAT_R32_SINT;
			SRVDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
			SRVDesc.Buffer.FirstElement = 0;
			SRVDesc.Buffer.NumElements = GlobalAppState::getInstance().s_windowWidth * GlobalAppState::getInstance().s_windowHeight;

			ZeroMemory( &UAVDesc, sizeof(D3D11_UNORDERED_ACCESS_VIEW_DESC) );
			UAVDesc.Format = DXGI_FORMAT_R32_SINT;;
			UAVDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
			UAVDesc.Buffer.FirstElement = 0;
			UAVDesc.Buffer.NumElements =  GlobalAppState::getInstance().s_windowWidth * GlobalAppState::getInstance().s_windowHeight;

			V_RETURN(pd3dDevice->CreateBuffer(&bDesc, NULL, &m_FragmentCountBuffer));
			V_RETURN(pd3dDevice->CreateShaderResourceView(m_FragmentCountBuffer, &SRVDesc, &m_FragmentCountBufferSRV));
			V_RETURN(pd3dDevice->CreateUnorderedAccessView(m_FragmentCountBuffer, &UAVDesc, &m_FragmentCountBufferUAV));

			////////////////////////////////////////////////////////////////////
			// For second pass // Perform prefix sum
			////////////////////////////////////////////////////////////////////

			m_Scan = new DX11ScanCS();
			m_Scan->OnD3D11CreateDevice(pd3dDevice);

			V_RETURN(pd3dDevice->CreateBuffer(&bDesc, NULL, &m_FragmentPrefixSumBuffer));
			V_RETURN(pd3dDevice->CreateShaderResourceView(m_FragmentPrefixSumBuffer, &SRVDesc, &m_FragmentPrefixSumBufferSRV));
			V_RETURN(pd3dDevice->CreateUnorderedAccessView(m_FragmentPrefixSumBuffer, &UAVDesc, &m_FragmentPrefixSumBufferUAV));

			////////////////////////////////////////////////////////////////////
			// For third pass // Write fragments
			////////////////////////////////////////////////////////////////////

			ZeroMemory(&bDesc, sizeof(D3D11_BUFFER_DESC));
			bDesc.BindFlags	= D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
			bDesc.Usage	= D3D11_USAGE_DEFAULT;
			bDesc.CPUAccessFlags = 0;
			bDesc.MiscFlags	= 0;
			bDesc.ByteWidth	= sizeof(float) * m_maxNumberOfPossibleFragments;
			bDesc.StructureByteStride = sizeof(float);

			ZeroMemory( &SRVDesc, sizeof(D3D11_SHADER_RESOURCE_VIEW_DESC) );
			SRVDesc.Format = DXGI_FORMAT_R32_FLOAT;
			SRVDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
			SRVDesc.Buffer.FirstElement = 0;
			SRVDesc.Buffer.NumElements = m_maxNumberOfPossibleFragments;

			ZeroMemory( &UAVDesc, sizeof(D3D11_UNORDERED_ACCESS_VIEW_DESC) );
			UAVDesc.Format = DXGI_FORMAT_R32_FLOAT;
			UAVDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
			UAVDesc.Buffer.FirstElement = 0;
			UAVDesc.Buffer.NumElements =  m_maxNumberOfPossibleFragments;
			V_RETURN(pd3dDevice->CreateBuffer(&bDesc, NULL, &m_FragmentSortedDepthBuffer));
			V_RETURN(pd3dDevice->CreateShaderResourceView(m_FragmentSortedDepthBuffer, &SRVDesc, &m_FragmentSortedDepthBufferSRV));
			V_RETURN(pd3dDevice->CreateUnorderedAccessView(m_FragmentSortedDepthBuffer, &UAVDesc, &m_FragmentSortedDepthBufferUAV));

			V_RETURN(CompileShaderFromFile(L"Shaders\\RayMarchingStepsSplatting.hlsl", "PS_Write", "ps_5_0", &pBlob, validDefines));
			V_RETURN(pd3dDevice->CreatePixelShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &s_pPixelShaderSplatting_Write));
			SAFE_RELEASE(pBlob);

			////////////////////////////////////////////////////////////////////
			// For forth pass // Sort fragments
			////////////////////////////////////////////////////////////////////

			V_RETURN(CompileShaderFromFile(L"Shaders\\RayMarchingStepsSplatting.hlsl", "sortFragmentsCS", "cs_5_0", &pBlob, validDefines));
			V_RETURN(pd3dDevice->CreateComputeShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &m_pComputeShaderSort));
			SAFE_RELEASE(pBlob);

			return  hr;
		}

		static void destroy()
		{
			////////////////////////////////////////////////////////////////////
			// General
			////////////////////////////////////////////////////////////////////

			SAFE_RELEASE(s_pVertexShaderSplatting);
			SAFE_RELEASE(s_pGeometryShaderSplatting);

			SAFE_RELEASE(s_ConstantBufferSplatting);

			SAFE_RELEASE(s_pDepthStencilStateSplatting);

			SAFE_RELEASE(s_pBlendStateDefault);
			SAFE_RELEASE(s_pBlendStateColorWriteDisabled);


			SAFE_RELEASE(s_pRastState);

			////////////////////////////////////////////////////////////////////
			// Pre-pass // Which blocks should be splatted
			////////////////////////////////////////////////////////////////////

			SAFE_RELEASE(s_pComputeShaderDecisionArray);

			SAFE_RELEASE(s_pDecisionArrayBuffer);
			SAFE_RELEASE(s_pDecisionArrayBufferSRV);
			SAFE_RELEASE(s_pDecisionArrayBufferUAV);

			////////////////////////////////////////////////////////////////////
			// For first pass // Count fragments per pixel
			////////////////////////////////////////////////////////////////////
			
			SAFE_RELEASE(s_pComputeShaderClear);

			SAFE_RELEASE(s_pPixelShaderSplatting_Count);

			SAFE_RELEASE(m_FragmentCountBuffer);
			SAFE_RELEASE(m_FragmentCountBufferSRV);
			SAFE_RELEASE(m_FragmentCountBufferUAV);

			////////////////////////////////////////////////////////////////////
			// For second pass // Perform prefix sum
			////////////////////////////////////////////////////////////////////

			m_Scan->OnD3D11DestroyDevice();
			SAFE_DELETE(m_Scan);

			SAFE_RELEASE(m_FragmentPrefixSumBuffer);
			SAFE_RELEASE(m_FragmentPrefixSumBufferSRV);
			SAFE_RELEASE(m_FragmentPrefixSumBufferUAV);

			////////////////////////////////////////////////////////////////////
			// For third pass // Write fragments
			////////////////////////////////////////////////////////////////////

			SAFE_RELEASE(m_FragmentSortedDepthBuffer);
			SAFE_RELEASE(m_FragmentSortedDepthBufferSRV);
			SAFE_RELEASE(m_FragmentSortedDepthBufferUAV);

			SAFE_RELEASE(s_pPixelShaderSplatting_Write);

			////////////////////////////////////////////////////////////////////
			// For forth pass // Sort fragments
			////////////////////////////////////////////////////////////////////

			SAFE_RELEASE(m_pComputeShaderSort);
		}


		////////////////////////////////////////////////////////////////////
		// General
		////////////////////////////////////////////////////////////////////

		struct CBuffer
		{
			float m_ViewMat[16];
			float m_ViewMatInverse[16];
			unsigned int m_RenderTargetWidth;
			unsigned int m_RenderTargetHeight;
			unsigned int align0;
			unsigned int align1;
		};

		static unsigned int m_blockSize;

		static ID3D11VertexShader* s_pVertexShaderSplatting;
		static ID3D11GeometryShader* s_pGeometryShaderSplatting;

		static ID3D11Buffer* s_ConstantBufferSplatting;

		static ID3D11DepthStencilState* s_pDepthStencilStateSplatting;

		static ID3D11BlendState* s_pBlendStateDefault;
		static ID3D11BlendState* s_pBlendStateColorWriteDisabled;


		static ID3D11RasterizerState* s_pRastState;

		////////////////////////////////////////////////////////////////////
		// Pre-pass // Which blocks should be splatted
		////////////////////////////////////////////////////////////////////

		static ID3D11ComputeShader* s_pComputeShaderDecisionArray;

		static ID3D11Buffer* s_pDecisionArrayBuffer;
		static ID3D11ShaderResourceView* s_pDecisionArrayBufferSRV;
		static ID3D11UnorderedAccessView* s_pDecisionArrayBufferUAV;

		////////////////////////////////////////////////////////////////////
		// For first pass // Count fragments per pixel
		////////////////////////////////////////////////////////////////////

		static ID3D11ComputeShader* s_pComputeShaderClear;

		static ID3D11PixelShader* s_pPixelShaderSplatting_Count;

		static ID3D11Buffer* m_FragmentCountBuffer;
		static ID3D11ShaderResourceView* m_FragmentCountBufferSRV;
		static ID3D11UnorderedAccessView* m_FragmentCountBufferUAV;

		////////////////////////////////////////////////////////////////////
		// For second pass // Perform prefix sum
		////////////////////////////////////////////////////////////////////

		static DX11ScanCS* m_Scan;

		static ID3D11Buffer*  m_FragmentPrefixSumBuffer;
		static ID3D11ShaderResourceView* m_FragmentPrefixSumBufferSRV;
		static ID3D11UnorderedAccessView* m_FragmentPrefixSumBufferUAV;

		////////////////////////////////////////////////////////////////////
		// For third pass // Write fragments
		////////////////////////////////////////////////////////////////////

		static unsigned int m_maxNumberOfPossibleFragments;
		static ID3D11Buffer* m_FragmentSortedDepthBuffer;
		static ID3D11ShaderResourceView* m_FragmentSortedDepthBufferSRV;
		static ID3D11UnorderedAccessView* m_FragmentSortedDepthBufferUAV;

		static ID3D11PixelShader* s_pPixelShaderSplatting_Write;

		////////////////////////////////////////////////////////////////////
		// For forth pass // Sort fragments
		////////////////////////////////////////////////////////////////////

		static ID3D11ComputeShader* m_pComputeShaderSort;

		////////////////////////////////////////////////////////////////////
		// Timer
		////////////////////////////////////////////////////////////////////

		static Timer m_timer;
};
