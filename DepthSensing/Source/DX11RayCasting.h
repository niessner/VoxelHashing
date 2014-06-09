#pragma once

#include <D3D11.h>
#include "DXUT.h"
#include "DX11Utils.h"
#include "DX11VoxelGrid.h"
#include "matrix4x4.h"
#include "GlobalAppState.h"
#include "TimingLog.h"
//#include "KinectSensor2.h"
#include "DX11ImageHelper.h"

class DX11RayCasting
{
	public:
		
		static HRESULT OnD3D11CreateDevice(ID3D11Device* pd3dDevice)
		{
			HRESULT hr = S_OK;

			V_RETURN(initialize(pd3dDevice));
			
			return  hr;
		}

		static void OnD3D11DestroyDevice()
		{
			destroy();
		}

		static HRESULT render(ID3D11DeviceContext* context, ID3D11ShaderResourceView* voxelBuffer, D3DXVECTOR3* gridPosition, int3* gridDimensions, D3DXVECTOR3* voxelExtends, const mat4f* lastRigidTransform, unsigned int imageWidth, unsigned int imageHeight) 
		{
			HRESULT hr = S_OK;

			// Initialize constant buffer
			D3D11_MAPPED_SUBRESOURCE mappedResource;
			V_RETURN(context->Map(m_constantBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource));

				CBuffer *cbuffer = (CBuffer*)mappedResource.pData;
				cbuffer->imageWidth = imageWidth;
				cbuffer->imageHeight = imageHeight;
				mat4f worldToLastKinectSpace = lastRigidTransform->getInverse();
				memcpy(&cbuffer->rigidTransform, lastRigidTransform, sizeof(mat4f));
				memcpy(&cbuffer->viewMat, &worldToLastKinectSpace, sizeof(mat4f));
				memcpy(&cbuffer->gridPosition, gridPosition, sizeof(D3DXVECTOR3));
				memcpy(&cbuffer->voxelExtends, voxelExtends, sizeof(D3DXVECTOR3));
				memcpy(&cbuffer->gridDimensions, gridDimensions, sizeof(int3));
				
			context->Unmap(m_constantBuffer, 0);

			// Setup pipeline
			context->CSSetShaderResources(0, 1, &voxelBuffer);
			context->CSSetUnorderedAccessViews(0, 1, &m_pOutputImage2DUAV, 0);
			context->CSSetUnorderedAccessViews(1, 1, &s_pNormalsUAV, 0);
			context->CSSetUnorderedAccessViews(2, 1, &s_pColorsUAV, 0);
			context->CSSetConstantBuffers(0, 1, &m_constantBuffer);
			ID3D11Buffer* CBGlobalAppState = GlobalAppState::getInstance().MapAndGetConstantBuffer(context);
			context->CSSetConstantBuffers(8, 1, &CBGlobalAppState);
			context->CSSetShader(m_pComputeShader, 0, 0);

			// Run compute shader
			unsigned int dimX = (unsigned int)ceil(((float)imageWidth)/m_blockSize);
			unsigned int dimY = (unsigned int)ceil(((float)imageHeight)/m_blockSize);

			// Start query for timing
			if(GlobalAppState::getInstance().s_timingsDetailledEnabled)
			{
				GlobalAppState::getInstance().WaitForGPU();
				m_timer.start();
			}

			context->Dispatch(dimX, dimY, 1);

			// Wait for query
			if(GlobalAppState::getInstance().s_timingsDetailledEnabled)
			{
				GlobalAppState::getInstance().WaitForGPU();
				TimingLog::totalTimeRayCast+=m_timer.getElapsedTimeMS();
				TimingLog::countRayCast++;
			}

			// Cleanup
			ID3D11ShaderResourceView* nullSRV[1] = {NULL};
			ID3D11UnorderedAccessView* nullUAV[1] = {NULL};
			ID3D11Buffer* nullB[1] = {NULL};

			context->CSSetShaderResources(0, 1, nullSRV);
			context->CSSetUnorderedAccessViews(0, 1, nullUAV, 0);
			context->CSSetUnorderedAccessViews(1, 1, nullUAV, 0);
			context->CSSetUnorderedAccessViews(2, 1, nullUAV, 0);
			context->CSSetConstantBuffers(0, 1, nullB);
			context->CSSetConstantBuffers(8, 1, nullB);
			context->CSSetShader(0, 0, 0);


			// Output
			DX11ImageHelper::applyCP(context, m_pOutputImage2DSRV, s_pPositionsUAV, imageWidth, imageHeight);
			DX11ImageHelper::applyNC(context, s_pPositionsSRV, s_pNormalsUAV, imageWidth, imageHeight);
			// Color has to be filled!!!

			return hr;
		}
		
		// Depth
		static ID3D11ShaderResourceView* getDepthImageSRV()
		{
			return m_pOutputImage2DSRV;
		}

		// Position
		static ID3D11UnorderedAccessView* getPositonsImageUAV()
		{
			return s_pPositionsUAV;
		}

		static ID3D11ShaderResourceView* getPositonsImageSRV()
		{
			return s_pPositionsSRV;
		}

		// Normals
		static ID3D11UnorderedAccessView* getNormalsImageUAV()
		{
			return s_pNormalsUAV;
		}

		static ID3D11ShaderResourceView* getNormalsImageSRV()
		{
			return s_pNormalsSRV;
		}

		// Color
		static ID3D11UnorderedAccessView* getColorsImageUAV()
		{
			return s_pColorsUAV;
		}

		static ID3D11ShaderResourceView* getColorsImageSRV()
		{
			return s_pColorsSRV;
		}

	private:

		static HRESULT initialize(ID3D11Device* pd3dDevice)
		{
			HRESULT hr = S_OK;

			char BLOCK_SIZE[5];
			sprintf_s(BLOCK_SIZE, "%d", m_blockSize);

			D3D_SHADER_MACRO shaderDefines[] = {{"groupthreads", BLOCK_SIZE}, {0}};

			ID3DBlob* pBlob = NULL;
	
			V_RETURN(CompileShaderFromFile(L"Shaders\\RayCasting.hlsl", "renderCS", "cs_5_0", &pBlob, shaderDefines));
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

			//create shader resource views
			V_RETURN(pd3dDevice->CreateShaderResourceView(m_pOutputImage2D, NULL, &m_pOutputImage2DSRV));

			//create unordered access views
			V_RETURN(pd3dDevice->CreateUnorderedAccessView(m_pOutputImage2D, NULL, &m_pOutputImage2DUAV));


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
						
			return  hr;
		}
				
		static void destroy()
		{
			SAFE_RELEASE(m_pComputeShader);
			SAFE_RELEASE(m_constantBuffer);

			SAFE_RELEASE(m_pOutputImage2D);
			SAFE_RELEASE(m_pOutputImage2DSRV);
			SAFE_RELEASE(m_pOutputImage2DUAV);

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
		}
		
		
		// State
		struct CBuffer
		{
			// Rendering
			unsigned int imageWidth;
			unsigned int imageHeight;
			unsigned int align0;
			unsigned int align1;

			float rigidTransform[16];
			float viewMat[16];

			// Grid
			float3 gridPosition;
			float align2;

			int3 gridDimensions;
			int align3;

			D3DXVECTOR3 voxelExtends;
			float align4;
		};
										
		static unsigned int m_blockSize;

		static ID3D11ComputeShader* m_pComputeShader;
		static ID3D11Buffer* m_constantBuffer;

		static ID3D11Texture2D* m_pOutputImage2D;
		static ID3D11ShaderResourceView* m_pOutputImage2DSRV;
		static ID3D11UnorderedAccessView* m_pOutputImage2DUAV;

		// Output
		static ID3D11Texture2D* s_pColors;
		static ID3D11ShaderResourceView* s_pColorsSRV;
		static ID3D11UnorderedAccessView* s_pColorsUAV;

		static ID3D11Texture2D* s_pPositions;
		static ID3D11ShaderResourceView* s_pPositionsSRV;
		static ID3D11UnorderedAccessView* s_pPositionsUAV;

		static ID3D11Texture2D* s_pNormals;
		static ID3D11ShaderResourceView* s_pNormalsSRV;
		static ID3D11UnorderedAccessView* s_pNormalsUAV;

		static Timer m_timer;
};
