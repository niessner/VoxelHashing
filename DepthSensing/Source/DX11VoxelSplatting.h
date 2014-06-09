#pragma once

#include <D3D11.h>
#include "DX11Utils.h"
#include <stdio.h>
#include "Timer.h"
#include <iostream>
#include "matrix4x4.h"
#include "DX11QuadDrawer.h"
#include "DXUTcamera.h"
#include "DXUT.h"
#include "GlobalAppState.h"
#include "DX11ImageHelper.h"

#include "Eigen.h"

#define SPLAT_RENDER_GEOMETRYSHADER 0
#define SPLAT_RENDER_TRILIST 1
#define SPLAT_RENDER_SINGLEPOINT 2
#define SPLAT_RENDER_SINGLEPOINT_WITHSPLATS 3

class DX11VoxelSplatting
{
	public:
		static HRESULT OnD3D11CreateDevice(ID3D11Device* pd3dDevice);

		static void OnD3D11DestroyDevice();

		static void Render(ID3D11DeviceContext* context, ID3D11ShaderResourceView* voxelHash, const mat4f* lastRigidTransform, unsigned int voxelBufferSize, float virtualVoxelSize, float splatSize) 
		{

			HRESULT hr = S_OK;

			// Initialize Constant Buffers
			D3D11_MAPPED_SUBRESOURCE mappedResource;
			V(context->Map(s_ConstantBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource));
			cbConstant *cbufferConstant = (cbConstant*)mappedResource.pData;
			cbufferConstant->virtualVoxelSize = virtualVoxelSize;	
			cbufferConstant->splatSize = splatSize;
			cbufferConstant->imageWidth = DXUTGetWindowWidth();
			cbufferConstant->imageHeight = DXUTGetWindowHeight();
			mat4f worldToLastKinectSpace = lastRigidTransform->getInverse();
			memcpy(&cbufferConstant->viewMat, &worldToLastKinectSpace, sizeof(mat4f));
			context->Unmap(s_ConstantBuffer, 0);
							
			RenderSplatDepth(context, voxelHash, voxelBufferSize);
			RenderSplatBlend(context, voxelHash, voxelBufferSize);
			if (s_SplatRenderMode == SPLAT_RENDER_SINGLEPOINT_WITHSPLATS)	{
				RenderNormalizeSplats(context, s_pColorsNormRTV);
				//now re-splat everything
				ReSplatDepth(context);
				ReSplatBlend(context);
			}

			if (s_SplatRenderMode == SPLAT_RENDER_SINGLEPOINT) {
				//TODO fix this
				assert(GlobalAppState::getInstance().s_windowHeight == GlobalAppState::getInstance().s_windowHeight && GlobalAppState::getInstance().s_windowWidth == GlobalAppState::getInstance().s_windowWidth);
				RenderNormalizeSplats(context, s_pColorsNormRTV);

				DX11ImageHelper::applyCS(context, s_pPositionsNormSRV, s_pDepthKinectResUAV, GlobalAppState::getInstance().s_windowWidth, GlobalAppState::getInstance().s_windowHeight);
				DX11ImageHelper::applyBilateralCompletionWithColor(context, s_pDepthKinectResSRV, s_pDepthKinectResFilteredUAV, s_pColorsNormSRV, s_pColorsCompletedUAV, GlobalAppState::getInstance().s_windowWidth, GlobalAppState::getInstance().s_windowHeight, 2.0f, 0.1f);
				DX11ImageHelper::applyCP(context, s_pDepthKinectResFilteredSRV, s_pPositionsKinectResUAV, GlobalAppState::getInstance().s_windowWidth, GlobalAppState::getInstance().s_windowHeight);
				
				ID3D11RenderTargetView* rtv = DXUTGetD3D11RenderTargetView();
				ID3D11DepthStencilView* dsv = DXUTGetD3D11DepthStencilView();
				context->OMSetRenderTargets(1, &rtv, NULL);
				DX11QuadDrawer::RenderQuad(context, s_pColorsCompletedSRV);
				context->OMSetRenderTargets(1, &rtv, dsv);
			} else {
				RenderNormalizeSplats(context);
				RenderDownSample(context);
			}


			////Fill holes using bilateral completion
			//if (s_SplatRenderMode == SPLAT_RENDER_SINGLEPOINT) {
			//	DX11ImageHelper::applyCS(context, s_pPositionsKinectResSRV, s_pDepthKinectResUAV, GlobalAppState::getInstance().s_windowWidth, GlobalAppState::getInstance().s_windowHeight);
			//	DX11ImageHelper::applyBilateralCompletion(context, s_pDepthKinectResSRV, s_pDepthKinectResFilteredUAV, GlobalAppState::getInstance().s_windowWidth, GlobalAppState::getInstance().s_windowHeight, 2.0f, 0.1f);
			//	DX11ImageHelper::applyCP(context, s_pDepthKinectResFilteredSRV, s_pPositionsKinectResUAV, GlobalAppState::getInstance().s_windowWidth, GlobalAppState::getInstance().s_windowHeight);
			//}

			//Filter depth positions here... just a try
			if (s_bFilterPositions) {	
				DX11ImageHelper::applyCS(context, s_pPositionsKinectResSRV, s_pDepthKinectResUAV, GlobalAppState::getInstance().s_windowWidth, GlobalAppState::getInstance().s_windowHeight);
				DX11ImageHelper::applyBF(context, s_pDepthKinectResSRV, s_pDepthKinectResFilteredUAV, GlobalAppState::getInstance().s_windowWidth, GlobalAppState::getInstance().s_windowHeight, 5.0f, 0.2f);
				DX11ImageHelper::applyCP(context, s_pDepthKinectResFilteredSRV, s_pPositionsKinectResUAV, GlobalAppState::getInstance().s_windowWidth, GlobalAppState::getInstance().s_windowHeight);
			}

			RenderComputeDownSampledNormals(context);


			//CreateAndCopyToDebugTexture2D(DXUTGetD3D11Device(), DXUTGetD3D11DeviceContext(), s_pPositionsNorm);
			//CreateAndCopyToDebugTexture2D(DXUTGetD3D11Device(), DXUTGetD3D11DeviceContext(), s_pColorsNorm);

			// De-Initialize Pipeline
			ID3D11ShaderResourceView* nullSAV[1] = { NULL };
			ID3D11UnorderedAccessView* nullUAV[1] = { NULL };
			ID3D11Buffer* nullCB[1] = { NULL };

			context->GSSetConstantBuffers(0, 1, nullCB);
			context->PSSetConstantBuffers(1, 1, nullCB);

			context->VSSetShader(0, 0, 0);
			context->GSSetShader(0, 0, 0);
			context->PSSetShader(0, 0, 0);
		}


		static ID3D11ShaderResourceView*	GetDepthStencilSRV() {
			return s_pDepthStencilSRV;
		}

		static ID3D11ShaderResourceView*	GetColorsSRV() {
			return s_pColorsCompletedSRV;
		}

		static ID3D11ShaderResourceView*	GetPositionsSRV() {
			return s_pPositionsSRV;
		}

		static ID3D11ShaderResourceView*	GetPositionsNormSRV() {
			return s_pPositionsNormSRV;
		}
		
		static ID3D11ShaderResourceView*	GetPositionsKinectResSRV() {
			return s_pPositionsKinectResSRV;
		}

		static ID3D11ShaderResourceView*	GetNormalsKinectResSRV() {
			return s_pNormalsKinectResSRV;
		}

		static ID3D11ShaderResourceView*	GetDepthKinectResSRV() {
			return s_pDepthKinectResSRV;
		}

		static ID3D11ShaderResourceView*	GetDepthKinectResFilteredSRV() {
			return s_pDepthKinectResFilteredSRV;
		}

		static void ChangeSplatRenderMode() {
			s_SplatRenderMode = (s_SplatRenderMode + 1)%4;
			if (s_SplatRenderMode == SPLAT_RENDER_GEOMETRYSHADER)			std::cout << "SplatRenderMode now GeometryShader" << std::endl; 
			if (s_SplatRenderMode == SPLAT_RENDER_TRILIST)					std::cout << "SplatRenderMode now TriList" << std::endl; 
			if (s_SplatRenderMode == SPLAT_RENDER_SINGLEPOINT)				std::cout << "SplatRenderMode now SinglePoint" << std::endl; 
			if (s_SplatRenderMode == SPLAT_RENDER_SINGLEPOINT_WITHSPLATS)	std::cout << "SplatRenderMode now SinglePointWithSplats" << std::endl; 
		}
		
		static void ToggleFilterPositions() {
			s_bFilterPositions = !s_bFilterPositions;
		}
	private:

		static void RenderSplatDepth( ID3D11DeviceContext* context, ID3D11ShaderResourceView* voxelHash, unsigned int voxelBufferSize)
		{
			//ID3D11RenderTargetView* rtv = DXUTGetD3D11RenderTargetView();
			ID3D11DepthStencilView* dsv = DXUTGetD3D11DepthStencilView();

			context->ClearDepthStencilView(s_pDepthStencilDSV, D3D11_CLEAR_DEPTH, 1.0f, 0);
			context->OMSetRenderTargets(0, 0, s_pDepthStencilDSV);

			// Setup Pipeline
			unsigned int stride = 0;
			unsigned int offset = 0;
			context->IASetVertexBuffers(0, 0, NULL, &stride, &offset);

			if (s_SplatRenderMode == SPLAT_RENDER_GEOMETRYSHADER) {
				context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_POINTLIST);		
				context->IASetInputLayout(NULL);
				context->VSSetShader(s_pVertexShader, 0, 0);

				context->GSSetConstantBuffers(0, 1, &s_ConstantBuffer);
				context->GSSetShaderResources(0, 1, &voxelHash);			
				context->GSSetShader(s_pGeometryShader, 0, 0);

				context->PSSetConstantBuffers(0, 1, &s_ConstantBuffer);
				context->PSSetShader(s_pPixelShader, 0, 0);

				context->Draw(voxelBufferSize, 0);
			} else if (s_SplatRenderMode == SPLAT_RENDER_TRILIST) {		
				context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);		
				context->IASetInputLayout(NULL);

				context->VSSetConstantBuffers(0, 1, &s_ConstantBuffer);
				context->VSSetShaderResources(0, 1, &voxelHash);	
				context->VSSetShader(s_pVertexShaderSprite, 0, 0);

				context->PSSetConstantBuffers(0, 1, &s_ConstantBuffer);
				context->PSSetShader(s_pPixelShader, 0, 0);

				context->Draw(6 * voxelBufferSize, 0);

			} else if (s_SplatRenderMode == SPLAT_RENDER_SINGLEPOINT || s_SplatRenderMode == SPLAT_RENDER_SINGLEPOINT_WITHSPLATS) {
				context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_POINTLIST);		
				context->IASetInputLayout(NULL);

				context->VSSetConstantBuffers(0, 1, &s_ConstantBuffer);
				context->VSSetShaderResources(0, 1, &voxelHash);	
				context->VSSetShader(s_pVertexShaderSinglePoint, 0, 0);		

				context->PSSetConstantBuffers(0, 1, &s_ConstantBuffer);
				context->PSSetShader(s_pPixelShader, 0, 0);

				context->Draw(voxelBufferSize, 0);
			}



			ID3D11RenderTargetView* rtv = DXUTGetD3D11RenderTargetView();
			//ID3D11DepthStencilView* dsv = DXUTGetD3D11DepthStencilView();
			context->OMSetRenderTargets(1, &rtv, dsv);

			ID3D11ShaderResourceView* nullSRV[] = { NULL };
			ID3D11UnorderedAccessView* nullUAV[] = { NULL };
			ID3D11Buffer* nullCB[] = { NULL, NULL };

			context->VSSetConstantBuffers(0, 2, nullCB);
			context->GSSetConstantBuffers(0, 2, nullCB);
			context->PSSetConstantBuffers(0, 2, nullCB);

			context->VSSetShaderResources(0, 1, nullSRV);
			context->GSSetShaderResources(0, 1, nullSRV);
			context->PSSetShaderResources(0, 1, nullSRV);

			context->VSSetShader(0, 0, 0);
			context->GSSetShader(0, 0, 0);
			context->PSSetShader(0, 0, 0);

		}

		static void ReSplatDepth( ID3D11DeviceContext* context)
		{
			ID3D11DepthStencilView* dsv = DXUTGetD3D11DepthStencilView();

			context->ClearDepthStencilView(s_pDepthStencilDSV, D3D11_CLEAR_DEPTH, 1.0f, 0);
			context->OMSetRenderTargets(0, 0, s_pDepthStencilDSV);

			// Setup Pipeline
			unsigned int stride = 0;
			unsigned int offset = 0;
			context->IASetVertexBuffers(0, 0, NULL, &stride, &offset);

			if (true || s_SplatRenderMode == SPLAT_RENDER_GEOMETRYSHADER) {
				context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_POINTLIST);		
				context->IASetInputLayout(NULL);
				context->VSSetShader(s_pVertexShader, 0, 0);

				context->GSSetConstantBuffers(0, 1, &s_ConstantBuffer);
				//context->GSSetShaderResources(0, 1, &voxelHash);	
				context->GSSetShaderResources(2, 1, &s_pColorsNormSRV);
				context->GSSetShaderResources(3, 1, &s_pPositionsNormSRV);
				context->GSSetShader(s_pGeometryShaderReSplat, 0, 0);

				context->PSSetConstantBuffers(0, 1, &s_ConstantBuffer);
				context->PSSetShader(s_pPixelShader, 0, 0);

				context->Draw(DXUTGetWindowWidth() * DXUTGetWindowHeight(), 0);
			}

			ID3D11RenderTargetView* rtv = DXUTGetD3D11RenderTargetView();
			//ID3D11DepthStencilView* dsv = DXUTGetD3D11DepthStencilView();
			context->OMSetRenderTargets(1, &rtv, dsv);

			ID3D11ShaderResourceView* nullSRV[] = { NULL, NULL, NULL, NULL };
			ID3D11UnorderedAccessView* nullUAV[] = { NULL };
			ID3D11Buffer* nullCB[] = { NULL, NULL };

			context->VSSetConstantBuffers(0, 2, nullCB);
			context->GSSetConstantBuffers(0, 2, nullCB);
			context->PSSetConstantBuffers(0, 2, nullCB);

			context->VSSetShaderResources(0, 4, nullSRV);
			context->GSSetShaderResources(0, 4, nullSRV);
			context->PSSetShaderResources(0, 4, nullSRV);

			context->VSSetShader(0, 0, 0);
			context->GSSetShader(0, 0, 0);
			context->PSSetShader(0, 0, 0);

		}

		static void RenderSplatBlend( ID3D11DeviceContext* context, ID3D11ShaderResourceView* voxelHash, unsigned int voxelBufferSize ) 
		{
			context->OMSetBlendState(s_pBlendStateAdditive, NULL, 0xffffffff);

			static float ClearColor[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
			context->ClearRenderTargetView(s_pColorsRTV,	ClearColor);
			context->ClearRenderTargetView(s_pPositionsRTV, ClearColor);
			ID3D11RenderTargetView* rtvs[] = {s_pColorsRTV, s_pPositionsRTV};
			context->OMSetRenderTargets(2, rtvs, NULL);

			// Setup Pipeline
			unsigned int stride = 0;
			unsigned int offset = 0;


			context->IASetVertexBuffers(0, 0, NULL, &stride, &offset);		
			context->IASetInputLayout(NULL);


			if (s_SplatRenderMode == SPLAT_RENDER_GEOMETRYSHADER) {
				context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_POINTLIST);
				context->VSSetShader(s_pVertexShader, 0, 0);

				context->GSSetConstantBuffers(0, 1, &s_ConstantBuffer);
				context->GSSetShaderResources(0, 1, &voxelHash);		
				context->GSSetShaderResources(1, 1, &s_pDepthStencilSRV);
				context->GSSetShader(s_pGeometryShaderBlend, 0, 0);

				context->PSSetConstantBuffers(0, 1, &s_ConstantBuffer);
				context->PSSetShaderResources(1, 1, &s_pDepthStencilSRV);
				context->PSSetShader(s_pPixelShaderBlend, 0, 0);

				context->Draw(voxelBufferSize, 0);
			} else if (s_SplatRenderMode == SPLAT_RENDER_TRILIST) {
				context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);		

				context->VSSetConstantBuffers(0, 1, &s_ConstantBuffer);
				context->VSSetShaderResources(0, 1, &voxelHash);	
				context->VSSetShader(s_pVertexShaderSprite, 0, 0);

				context->PSSetConstantBuffers(0, 1, &s_ConstantBuffer);
				context->PSSetShaderResources(1, 1, &s_pDepthStencilSRV);
				//context->PSSetShader(s_pPixelShaderSpriteBlend, 0, 0);
				context->PSSetShader(s_pPixelShaderBlend, 0, 0);

				context->Draw(6 * voxelBufferSize, 0);
			} else if (s_SplatRenderMode == SPLAT_RENDER_SINGLEPOINT || s_SplatRenderMode == SPLAT_RENDER_SINGLEPOINT_WITHSPLATS) {
				context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_POINTLIST);		
				context->IASetInputLayout(NULL);

				context->VSSetConstantBuffers(0, 1, &s_ConstantBuffer);
				context->VSSetShaderResources(0, 1, &voxelHash);	
				context->VSSetShader(s_pVertexShaderSinglePoint, 0, 0);		

				context->PSSetConstantBuffers(0, 1, &s_ConstantBuffer);
				context->PSSetShaderResources(1, 1, &s_pDepthStencilSRV);
				//context->PSSetShader(s_pPixelShaderSpriteBlend, 0, 0);
				context->PSSetShader(s_pPixelShaderBlend, 0, 0);

				context->Draw(voxelBufferSize, 0);
			}

			ID3D11RenderTargetView* rtv = DXUTGetD3D11RenderTargetView();
			ID3D11DepthStencilView* dsv = DXUTGetD3D11DepthStencilView();
			context->OMSetRenderTargets(1, &rtv, dsv);
			context->OMSetBlendState(s_pBlendStateDefault, NULL, 0xffffffff);

			ID3D11ShaderResourceView* nullSRV[] = { NULL, NULL };
			ID3D11UnorderedAccessView* nullUAV[] = { NULL };
			ID3D11Buffer* nullCB[] = { NULL, NULL };

			context->GSSetConstantBuffers(0, 2, nullCB);
			context->PSSetConstantBuffers(0, 2, nullCB);

			context->GSSetShaderResources(0, 2, nullSRV);
			context->PSSetShaderResources(0, 2, nullSRV);

			context->VSSetShader(0, 0, 0);
			context->GSSetShader(0, 0, 0);
			context->PSSetShader(0, 0, 0);
		}

		//! splats all 3d points in the current render target buffer
		static void ReSplatBlend(ID3D11DeviceContext* context) {

			context->OMSetBlendState(s_pBlendStateAdditive, NULL, 0xffffffff);

			static float ClearColor[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
			context->ClearRenderTargetView(s_pColorsRTV,	ClearColor);
			context->ClearRenderTargetView(s_pPositionsRTV, ClearColor);
			ID3D11RenderTargetView* rtvs[] = {s_pColorsRTV, s_pPositionsRTV};
			context->OMSetRenderTargets(2, rtvs, NULL);

			// Setup Pipeline
			unsigned int stride = 0;
			unsigned int offset = 0;

			context->IASetVertexBuffers(0, 0, NULL, &stride, &offset);		
			context->IASetInputLayout(NULL);

			if (true || s_SplatRenderMode == SPLAT_RENDER_GEOMETRYSHADER) {
				context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_POINTLIST);
				context->VSSetShader(s_pVertexShader, 0, 0);

				context->GSSetConstantBuffers(0, 1, &s_ConstantBuffer);
				//context->GSSetShaderResources(0, 1, &voxelHash);		
				context->GSSetShaderResources(1, 1, &s_pDepthStencilSRV);
				context->GSSetShaderResources(2, 1, &s_pColorsNormSRV);
				context->GSSetShaderResources(3, 1, &s_pPositionsNormSRV);
				context->GSSetShader(s_pGeometryShaderReSplatBlend, 0, 0);

				context->PSSetConstantBuffers(0, 1, &s_ConstantBuffer);
				context->PSSetShaderResources(1, 1, &s_pDepthStencilSRV);
				context->PSSetShader(s_pPixelShaderBlend, 0, 0);

				context->Draw(DXUTGetWindowWidth() * DXUTGetWindowHeight(), 0);
			} 
			//else if (m_SplatRenderMode == SPLAT_RENDER_TRILIST) {
			//	context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);		

			//	context->VSSetConstantBuffers(0, 1, &m_constantBufferConstant);
			//	context->VSSetConstantBuffers(1, 1, &m_constantBufferPerFrame);
			//	context->VSSetShaderResources(0, 1, &voxelHash);	
			//	context->VSSetShader(m_pVertexShaderSprite, 0, 0);

			//	context->PSSetConstantBuffers(0, 1, &m_constantBufferConstant);
			//	context->PSSetConstantBuffers(1, 1, &m_constantBufferPerFrame);
			//	context->PSSetShaderResources(1, 1, &m_pDepthStencilSRV);
			//	context->PSSetShader(m_pPixelShaderSpriteBlend, 0, 0);

			//	context->Draw(6 * voxelBufferSize, 0);
			//}

			ID3D11RenderTargetView* rtv = DXUTGetD3D11RenderTargetView();
			ID3D11DepthStencilView* dsv = DXUTGetD3D11DepthStencilView();
			context->OMSetRenderTargets(1, &rtv, dsv);
			context->OMSetBlendState(s_pBlendStateDefault, NULL, 0xffffffff);

			ID3D11ShaderResourceView* nullSRV[] = { NULL, NULL, NULL, NULL };
			ID3D11UnorderedAccessView* nullUAV[] = { NULL };
			ID3D11Buffer* nullCB[] = { NULL, NULL };

			context->GSSetConstantBuffers(0, 2, nullCB);
			context->PSSetConstantBuffers(0, 2, nullCB);

			context->GSSetShaderResources(0, 4, nullSRV);
			context->PSSetShaderResources(0, 4, nullSRV);

			context->VSSetShader(0, 0, 0);
			context->GSSetShader(0, 0, 0);
			context->PSSetShader(0, 0, 0);
		}


		static void RenderNormalizeSplats( ID3D11DeviceContext* context, ID3D11RenderTargetView* colorTarget = NULL) 
		{
			ID3D11RenderTargetView* rtv = DXUTGetD3D11RenderTargetView();
			ID3D11DepthStencilView* dsv = DXUTGetD3D11DepthStencilView();
			ID3D11RenderTargetView* rtvs[] = {rtv, s_pPositionsNormRTV};
			if (colorTarget)	rtvs[0] = colorTarget;	//in this case do not use the render buffer
			context->OMSetRenderTargets(2, rtvs, NULL);

			ID3D11ShaderResourceView* srvs[] = {s_pColorsSRV, s_pPositionsSRV};
			DX11QuadDrawer::RenderQuad(context, s_pPixelShaderNorm, srvs, 2);

			context->OMSetRenderTargets(1, &rtv, dsv);
		}

		//! copies depth data into the kinect-sized texture
		static void RenderDownSample( ID3D11DeviceContext* context ) 
		{
			if(DXUTGetWindowWidth() == 2*GlobalAppState::getInstance().s_windowWidth && DXUTGetWindowHeight() == 2*GlobalAppState::getInstance().s_windowHeight)
			{
				DX11ImageHelper::applyDS(context, s_pPositionsNormSRV, s_pPositionsKinectResUAV, GlobalAppState::getInstance().s_windowWidth, GlobalAppState::getInstance().s_windowHeight);
			}
			else if(DXUTGetWindowWidth() == GlobalAppState::getInstance().s_windowWidth && DXUTGetWindowHeight() == GlobalAppState::getInstance().s_windowHeight)
			{
				DX11ImageHelper::applyCopy(context, s_pPositionsNormSRV, s_pPositionsKinectResUAV, GlobalAppState::getInstance().s_windowWidth, GlobalAppState::getInstance().s_windowHeight);
			}
			else
			{
				std::cout << "Invalid Screen Size" << std::endl;
			}
		} 

		static void RenderComputeDownSampledNormals( ID3D11DeviceContext* context) 
		{
			DX11ImageHelper::applyNC(context, s_pPositionsKinectResSRV, s_pNormalsKinectResUAV, GlobalAppState::getInstance().s_windowWidth, GlobalAppState::getInstance().s_windowHeight);
		}




		/////////////////////////////////////////////////////
		// Voxel Splatting
		/////////////////////////////////////////////////////

		struct cbConstant
		{
			float virtualVoxelSize;
			float splatSize;
			unsigned int imageWidth;
			unsigned int imageHeight;
			D3DXMATRIX viewMat;
		};


		static ID3D11Buffer* s_ConstantBuffer;

		static ID3D11VertexShader* s_pVertexShaderSinglePoint;

		static ID3D11GeometryShader*	s_pGeometryShaderReSplat;
		static ID3D11GeometryShader*	s_pGeometryShaderReSplatBlend;

		static ID3D11VertexShader*		s_pVertexShaderSprite;
		//static ID3D11PixelShader*		s_pPixelShaderSpriteBlend;
	
		static ID3D11VertexShader*		s_pVertexShader;
		static ID3D11GeometryShader*	s_pGeometryShader;
		static ID3D11PixelShader*		s_pPixelShader;
		static ID3D11GeometryShader*	s_pGeometryShaderBlend;
		static ID3D11PixelShader*		s_pPixelShaderBlend;

		static ID3D11PixelShader*		s_pPixelShaderNorm;

		//for first pass (1280x960): depth/visibility pass
		static ID3D11Texture2D*				s_pDepthStencil;
		static ID3D11DepthStencilView*		s_pDepthStencilDSV;
		static ID3D11ShaderResourceView*	s_pDepthStencilSRV;

		//for second pass (1280x960): Gaussian splat pass
		static ID3D11Texture2D*				s_pPositions;
		static ID3D11RenderTargetView*		s_pPositionsRTV;
		static ID3D11ShaderResourceView*	s_pPositionsSRV;
		static ID3D11Texture2D*				s_pColors;
		static ID3D11RenderTargetView*		s_pColorsRTV;
		static ID3D11ShaderResourceView*	s_pColorsSRV;
		static ID3D11BlendState*			s_pBlendStateAdditive;
		static ID3D11BlendState*			s_pBlendStateDefault;

		//for third pass (1280x960): normalize
		static ID3D11Texture2D*				s_pPositionsNorm;
		static ID3D11RenderTargetView*		s_pPositionsNormRTV;
		static ID3D11ShaderResourceView*	s_pPositionsNormSRV;
		static ID3D11Texture2D*				s_pColorsNorm;
		static ID3D11RenderTargetView*		s_pColorsNormRTV;
		static ID3D11ShaderResourceView*	s_pColorsNormSRV;
		static ID3D11Texture2D*				s_pColorsCompleted;
		static ID3D11ShaderResourceView*	s_pColorsCompletedSRV;
		static ID3D11UnorderedAccessView*	s_pColorsCompletedUAV;
		
		//for fourth pass (640x480)
		static ID3D11Texture2D*				s_pPositionsKinectRes;
		static ID3D11ShaderResourceView*	s_pPositionsKinectResSRV;
		static ID3D11UnorderedAccessView*	s_pPositionsKinectResUAV;
		static ID3D11Texture2D*				s_pNormalsKinectRes;
		static ID3D11ShaderResourceView*	s_pNormalsKinectResSRV;
		static ID3D11UnorderedAccessView*	s_pNormalsKinectResUAV;
		static ID3D11Texture2D*				s_pDepthKinectRes;
		static ID3D11ShaderResourceView*	s_pDepthKinectResSRV;
		static ID3D11UnorderedAccessView*	s_pDepthKinectResUAV;
		static ID3D11Texture2D*				s_pDepthKinectResFiltered;
		static ID3D11ShaderResourceView*	s_pDepthKinectResFilteredSRV;
		static ID3D11UnorderedAccessView*	s_pDepthKinectResFilteredUAV;

		static unsigned int					s_SplatRenderMode;
		static bool							s_bFilterPositions;
};
