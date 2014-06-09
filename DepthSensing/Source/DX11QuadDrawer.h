#pragma once

/************************************************************************/
/* DirectX quad drawing class for visualizing textures                  */
/************************************************************************/

#include "DX11Utils.h"

struct SimpleVertex {
	float3 pos;
	float2 pex;
};

struct CB_QUAD {
	D3DXMATRIX mWorldViewProjection;
	UINT width;
	UINT height;
	float2 dummy;
};

class DX11QuadDrawer
{
public:
	DX11QuadDrawer();
	~DX11QuadDrawer();

	static HRESULT OnD3D11CreateDevice(ID3D11Device* pd3dDevice);
	static void OnD3D11DestroyDevice();

	static void RenderQuad(ID3D11DeviceContext* pd3dDeviceContext, ID3D11PixelShader* pixelShader, ID3D11ShaderResourceView** srvs, UINT numShaderResourceViews, float2 Pow2Ratios = float2(1.0f, 1.0f));
	static void RenderQuad(ID3D11DeviceContext* pd3dDeviceContext, ID3D11ShaderResourceView* srv, float scale = 1.0f, float2 Pow2Ratios = float2(1.0f, 1.0f), ID3D11PixelShader* pixelShader = NULL);

private:
	static ID3D11InputLayout*	s_VertexLayout;
	static ID3D11Buffer*		s_VertexBuffer;
	static ID3D11Buffer*		s_IndexBuffer;

	static ID3D11VertexShader*	s_VertexShader;
	static ID3D11PixelShader*	s_PixelShaderFloat;
	static ID3D11PixelShader*	s_PixelShaderRGBA;
	static ID3D11PixelShader*	s_PixelShader3;

	static ID3D11Buffer*		s_CBquad;

	static ID3D11SamplerState*	s_PointSampler;

	static ID3D11Buffer*		s_pcbVSPowTwoRatios;

};

