#pragma once

/************************************************************************/
/* Ray casting an signed distance function (including voxel hashes)     */
/************************************************************************/

#include "stdafx.h"

#include <D3D11.h>
#include "DXUT.h"
#include "DX11Utils.h"
#include "TimingLog.h"
#include "DX11RayMarchingStepsSplatting.h"
#include "DX11PhongLighting.h"

class DX11RayCastingHashSDF
{
	public:
		
		static HRESULT OnD3D11CreateDevice(ID3D11Device* pd3dDevice);

		static void OnD3D11DestroyDevice();

		static HRESULT Render(ID3D11DeviceContext* context, ID3D11ShaderResourceView* hash, ID3D11ShaderResourceView* hashCompact, ID3D11ShaderResourceView* sdfBlocksSDF, ID3D11ShaderResourceView* sdfBlocksRGBW, unsigned int hashNumValidBuckets, unsigned int renderTargetWidth, unsigned int renderTargetHeight, const mat4f* lastRigidTransform, ID3D11Buffer* CBsceneRepSDF);

		static HRESULT RenderStereo(ID3D11DeviceContext* context, ID3D11ShaderResourceView* hash, ID3D11ShaderResourceView* hashCompact, ID3D11ShaderResourceView* sdfBlocksSDF, ID3D11ShaderResourceView* sdfBlocksRGBW, unsigned int hashNumValidBuckets, unsigned int renderTargetWidth, unsigned int renderTargetHeight, const mat4f* lastRigidTransform, ID3D11Buffer* CBsceneRepSDF);

		static HRESULT RenderToTexture( ID3D11DeviceContext* context, ID3D11ShaderResourceView* hash, ID3D11ShaderResourceView* hashCompact, ID3D11ShaderResourceView* sdfBlocksSDF, ID3D11ShaderResourceView* sdfBlocksRGBW, unsigned int hashNumValidBuckets, unsigned int renderTargetWidth, unsigned int renderTargetHeight, const mat4f* lastRigidTransform, ID3D11Buffer* CBsceneRepSDF,
										ID3D11ShaderResourceView* pDepthStencilSplattingMinSRV, ID3D11ShaderResourceView* pDepthStencilSplattingMaxSRV,
										ID3D11DepthStencilView* pDepthStencilSplattingMinDSV, ID3D11DepthStencilView* pDepthStencilSplattingMaxDSV,
										ID3D11ShaderResourceView* pOutputImage2DSRV, ID3D11UnorderedAccessView* pOutputImage2DUAV,
										ID3D11ShaderResourceView* pPositionsSRV, ID3D11UnorderedAccessView* pPositionsUAV,
										ID3D11UnorderedAccessView* pColorsUAV,
										ID3D11ShaderResourceView* pNormalsSRV, ID3D11UnorderedAccessView* pNormalsUAV,
										ID3D11ShaderResourceView* pSSAOMapSRV, ID3D11UnorderedAccessView* pSSAOMapUAV, ID3D11UnorderedAccessView* pSSAOMapFilteredUAV);

		// Depth
		static ID3D11ShaderResourceView* getDepthImageSRV()	{
			return m_pOutputImage2DSRV;
		}

		// SSAO
		static ID3D11ShaderResourceView* getSSAOMapSRV() {
			return m_pSSAOMapSRV;
		}

		static ID3D11ShaderResourceView* getSSAOMapFilteredSRV() {
			return m_pSSAOMapFilteredSRV;
		}

		// Position
		static ID3D11UnorderedAccessView* getPositonsImageUAV()	{
			return s_pPositionsUAV;
		}

		static ID3D11ShaderResourceView* getPositonsImageSRV()	{
			return s_pPositionsSRV;
		}

		// Normals
		static ID3D11UnorderedAccessView* getNormalsImageUAV()	{
			return s_pNormalsUAV;
		}

		static ID3D11ShaderResourceView* getNormalsImageSRV() {
			return s_pNormalsSRV;
		}

		// Color
		static ID3D11UnorderedAccessView* getColorsImageUAV() {
			return s_pColorsUAV;
		}

		static ID3D11ShaderResourceView* getColorsImageSRV() {
			return s_pColorsSRV;
		}

		// Ray Interval Splatting
		static ID3D11ShaderResourceView* getDepthSplattingMinSRV() {
			return s_pDepthStencilSplattingMinSRV;
		}

		static ID3D11ShaderResourceView* getDepthSplattingMaxSRV() {
			return s_pDepthStencilSplattingMaxSRV;
		}

	private:

		static HRESULT initialize(ID3D11Device* pd3dDevice);

		static HRESULT rayIntervalSplatting(ID3D11DeviceContext* context, ID3D11ShaderResourceView* hash, unsigned int hashNumValidBuckets, const mat4f* lastRigidTransform, unsigned int renderTargetWidth, unsigned int renderTargetHeight, ID3D11Buffer* CBsceneRepSDF);
	
		static HRESULT rayIntervalSplattingRenderToTexture(	ID3D11DeviceContext* context, ID3D11ShaderResourceView* hash, unsigned int hashNumValidBuckets, const mat4f* lastRigidTransform, unsigned int renderTargetWidth, unsigned int renderTargetHeight, ID3D11Buffer* CBsceneRepSDF,
															ID3D11DepthStencilView* pDepthStencilSplattingMinDSV, ID3D11DepthStencilView* pDepthStencilSplattingMaxDSV);
						
		static void destroy();
		 
		// Ray Interval	
		struct CBuffer
		{
			float m_ViewMat[16];
			float m_ViewMatInverse[16];
			unsigned int m_RenderTargetWidth;
			unsigned int m_RenderTargetHeight;
			unsigned int m_SplatMinimum;
			unsigned int align0;
		};
										
		static unsigned int m_blockSize;

		static ID3D11ComputeShader* m_pComputeShader;
		static ID3D11Buffer* m_constantBuffer;

		static ID3D11Texture2D* m_pOutputImage2D;
		static ID3D11ShaderResourceView* m_pOutputImage2DSRV;
		static ID3D11UnorderedAccessView* m_pOutputImage2DUAV;

		static ID3D11Texture2D* m_pSSAOMap;
		static ID3D11ShaderResourceView* m_pSSAOMapSRV;
		static ID3D11UnorderedAccessView* m_pSSAOMapUAV;

		static ID3D11Texture2D* m_pSSAOMapFiltered;
		static ID3D11ShaderResourceView* m_pSSAOMapFilteredSRV;
		static ID3D11UnorderedAccessView* m_pSSAOMapFilteredUAV;

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
		
		static ID3D11Buffer* s_ConstantBufferSplatting;

		static ID3D11VertexShader*s_pVertexShaderSplatting;
		static ID3D11GeometryShader* s_pGeometryShaderSplatting;
		static ID3D11PixelShader* s_pPixelShaderSplatting;

		static ID3D11Texture2D* s_pDepthStencilSplattingMin;
		static ID3D11DepthStencilView*	s_pDepthStencilSplattingMinDSV;
		static ID3D11ShaderResourceView* s_pDepthStencilSplattingMinSRV;

		static ID3D11Texture2D* s_pDepthStencilSplattingMax;
		static ID3D11DepthStencilView*	s_pDepthStencilSplattingMaxDSV;
		static ID3D11ShaderResourceView* s_pDepthStencilSplattingMaxSRV;

		static ID3D11DepthStencilState* s_pDepthStencilStateSplattingMin;
		static ID3D11DepthStencilState* s_pDepthStencilStateSplattingMax;

		static ID3D11BlendState* s_pBlendStateDefault;
		static ID3D11BlendState* s_pBlendStateColorWriteDisabled;

		static ID3D11RasterizerState* s_pRastState;
		
		// Stereo
		static ID3D11Texture2D* m_pOutputImage2DStereo;
		static ID3D11Texture2D* m_pSSAOMapStereo;
		static ID3D11Texture2D* m_pSSAOMapFilteredStereo;

		static ID3D11ShaderResourceView* m_pOutputImage2DStereoSRV;
		static ID3D11ShaderResourceView* m_pSSAOMapStereoSRV;
		static ID3D11ShaderResourceView* m_pSSAOMapFilteredStereoSRV;

		static ID3D11UnorderedAccessView* m_pOutputImage2DStereoUAV;
		static ID3D11UnorderedAccessView* m_pSSAOMapStereoUAV;
		static ID3D11UnorderedAccessView* m_pSSAOMapFilteredStereoUAV;

		static ID3D11Texture2D* s_pColorsStereo;
		static ID3D11ShaderResourceView* s_pColorsStereoSRV;
		static ID3D11UnorderedAccessView* s_pColorsStereoUAV;

		static ID3D11Texture2D* s_pPositionsStereo;
		static ID3D11ShaderResourceView* s_pPositionsStereoSRV;
		static ID3D11UnorderedAccessView* s_pPositionsStereoUAV;

		static ID3D11Texture2D* s_pNormalsStereo;
		static ID3D11ShaderResourceView* s_pNormalsStereoSRV;
		static ID3D11UnorderedAccessView* s_pNormalsStereoUAV;

		static ID3D11Texture2D* s_pDepthStencilSplattingMinStereo;
		static ID3D11DepthStencilView*	s_pDepthStencilSplattingMinDSVStereo;
		static ID3D11ShaderResourceView* s_pDepthStencilSplattingMinSRVStereo;

		static ID3D11Texture2D* s_pDepthStencilSplattingMaxStereo;
		static ID3D11DepthStencilView*	s_pDepthStencilSplattingMaxDSVStereo;
		static ID3D11ShaderResourceView* s_pDepthStencilSplattingMaxSRVStereo;

		static Timer m_timer;
};
