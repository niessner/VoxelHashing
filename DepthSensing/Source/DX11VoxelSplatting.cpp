#include "DX11VoxelSplatting.h"

/////////////////////////////////////////////////////
// Voxel Splatting
/////////////////////////////////////////////////////

ID3D11Buffer* DX11VoxelSplatting::s_ConstantBuffer = NULL;

ID3D11GeometryShader*		DX11VoxelSplatting::s_pGeometryShaderReSplat = NULL;
ID3D11GeometryShader*		DX11VoxelSplatting::s_pGeometryShaderReSplatBlend = NULL;

ID3D11VertexShader*			DX11VoxelSplatting::s_pVertexShaderSinglePoint = NULL;

ID3D11VertexShader*			DX11VoxelSplatting::s_pVertexShaderSprite = NULL;
//ID3D11PixelShader*			DX11VoxelSplatting::s_pPixelShaderSpriteBlend = NULL;

ID3D11VertexShader*			DX11VoxelSplatting::s_pVertexShader = NULL;
ID3D11GeometryShader*		DX11VoxelSplatting::s_pGeometryShader = NULL;
ID3D11PixelShader*			DX11VoxelSplatting::s_pPixelShader = NULL;
ID3D11GeometryShader*		DX11VoxelSplatting::s_pGeometryShaderBlend = NULL;
ID3D11PixelShader*			DX11VoxelSplatting::s_pPixelShaderBlend = NULL;
ID3D11PixelShader*			DX11VoxelSplatting::s_pPixelShaderNorm = NULL;

ID3D11Texture2D*			DX11VoxelSplatting::s_pDepthStencil = NULL;
ID3D11DepthStencilView*		DX11VoxelSplatting::s_pDepthStencilDSV = NULL;
ID3D11ShaderResourceView*	DX11VoxelSplatting::s_pDepthStencilSRV = NULL;

ID3D11Texture2D*			DX11VoxelSplatting::s_pPositions = NULL;
ID3D11RenderTargetView*		DX11VoxelSplatting::s_pPositionsRTV = NULL;
ID3D11ShaderResourceView*	DX11VoxelSplatting::s_pPositionsSRV = NULL;
ID3D11Texture2D*			DX11VoxelSplatting::s_pColors = NULL;
ID3D11RenderTargetView*		DX11VoxelSplatting::s_pColorsRTV = NULL;
ID3D11ShaderResourceView*	DX11VoxelSplatting::s_pColorsSRV = NULL;
ID3D11BlendState*			DX11VoxelSplatting::s_pBlendStateAdditive = NULL;
ID3D11BlendState*			DX11VoxelSplatting::s_pBlendStateDefault = NULL;

ID3D11Texture2D*			DX11VoxelSplatting::s_pPositionsNorm  = NULL;
ID3D11RenderTargetView*		DX11VoxelSplatting::s_pPositionsNormRTV = NULL;
ID3D11ShaderResourceView*	DX11VoxelSplatting::s_pPositionsNormSRV = NULL;
ID3D11Texture2D*			DX11VoxelSplatting::s_pColorsNorm  = NULL;
ID3D11RenderTargetView*		DX11VoxelSplatting::s_pColorsNormRTV = NULL;
ID3D11ShaderResourceView*	DX11VoxelSplatting::s_pColorsNormSRV = NULL;
ID3D11Texture2D*			DX11VoxelSplatting::s_pColorsCompleted  = NULL;
ID3D11ShaderResourceView*	DX11VoxelSplatting::s_pColorsCompletedSRV = NULL;
ID3D11UnorderedAccessView*	DX11VoxelSplatting::s_pColorsCompletedUAV = NULL;

ID3D11Texture2D*			DX11VoxelSplatting::s_pPositionsKinectRes = NULL;
ID3D11ShaderResourceView*	DX11VoxelSplatting::s_pPositionsKinectResSRV = NULL;
ID3D11UnorderedAccessView*	DX11VoxelSplatting::s_pPositionsKinectResUAV = NULL;
ID3D11Texture2D*			DX11VoxelSplatting::s_pNormalsKinectRes = NULL;
ID3D11ShaderResourceView*	DX11VoxelSplatting::s_pNormalsKinectResSRV = NULL;
ID3D11UnorderedAccessView*	DX11VoxelSplatting::s_pNormalsKinectResUAV = NULL;
ID3D11Texture2D*			DX11VoxelSplatting::s_pDepthKinectRes = NULL;
ID3D11ShaderResourceView*	DX11VoxelSplatting::s_pDepthKinectResSRV = NULL;
ID3D11UnorderedAccessView*	DX11VoxelSplatting::s_pDepthKinectResUAV = NULL;
ID3D11Texture2D*			DX11VoxelSplatting::s_pDepthKinectResFiltered = NULL;
ID3D11ShaderResourceView*	DX11VoxelSplatting::s_pDepthKinectResFilteredSRV = NULL;
ID3D11UnorderedAccessView*	DX11VoxelSplatting::s_pDepthKinectResFilteredUAV = NULL;

//unsigned int				DX11VoxelSplatting::s_SplatRenderMode = SPLAT_RENDER_GEOMETRYSHADER;
unsigned int				DX11VoxelSplatting::s_SplatRenderMode = SPLAT_RENDER_SINGLEPOINT;
bool						DX11VoxelSplatting::s_bFilterPositions = true;

HRESULT DX11VoxelSplatting::OnD3D11CreateDevice( ID3D11Device* pd3dDevice )
{
	HRESULT hr = S_OK;


	ID3DBlob* pBlob = NULL;

	hr = CompileShaderFromFile(L"Shaders\\VoxelSplatting.hlsl", "GSReSplat", "gs_5_0", &pBlob);
	if(FAILED(hr)) return hr;

	hr = pd3dDevice->CreateGeometryShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &s_pGeometryShaderReSplat);
	if(FAILED(hr)) return hr;
	SAFE_RELEASE(pBlob);


	hr = CompileShaderFromFile(L"Shaders\\VoxelSplatting.hlsl", "GSReSplatBlend", "gs_5_0", &pBlob);
	if(FAILED(hr)) return hr;

	hr = pd3dDevice->CreateGeometryShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &s_pGeometryShaderReSplatBlend);
	if(FAILED(hr)) return hr;
	SAFE_RELEASE(pBlob);


	hr = CompileShaderFromFile(L"Shaders\\VoxelSplatting.hlsl", "VS_SINGLE_POINT", "vs_5_0", &pBlob);
	if(FAILED(hr)) return hr;

	hr = pd3dDevice->CreateVertexShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &s_pVertexShaderSinglePoint);
	if(FAILED(hr)) return hr;
	SAFE_RELEASE(pBlob);


	hr = CompileShaderFromFile(L"Shaders\\VoxelSplatting.hlsl", "VS_SPRITE", "vs_5_0", &pBlob);
	if(FAILED(hr)) return hr;

	hr = pd3dDevice->CreateVertexShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &s_pVertexShaderSprite);
	if(FAILED(hr)) return hr;
	SAFE_RELEASE(pBlob);

	//hr = CompileShaderFromFile(L"Shaders\\VoxelSplatting.hlsl", "PS_SPRITE_Blend", "ps_5_0", &pBlob);
	//if(FAILED(hr)) return hr;

	//hr = pd3dDevice->CreatePixelShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &s_pPixelShaderSpriteBlend);
	//if(FAILED(hr)) return hr;
	//SAFE_RELEASE(pBlob);





	hr = CompileShaderFromFile(L"Shaders\\VoxelSplatting.hlsl", "VS", "vs_5_0", &pBlob);
	if(FAILED(hr)) return hr;

	hr = pd3dDevice->CreateVertexShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &s_pVertexShader);
	if(FAILED(hr)) return hr;
	SAFE_RELEASE(pBlob);
	

	hr = CompileShaderFromFile(L"Shaders\\VoxelSplatting.hlsl", "GS", "gs_5_0", &pBlob);
	if(FAILED(hr)) return hr;

	hr = pd3dDevice->CreateGeometryShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &s_pGeometryShader);
	if(FAILED(hr)) return hr;
	SAFE_RELEASE(pBlob);


	hr = CompileShaderFromFile(L"Shaders\\VoxelSplatting.hlsl", "PS", "ps_5_0", &pBlob);
	if(FAILED(hr)) return hr;

	hr = pd3dDevice->CreatePixelShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &s_pPixelShader);
	if(FAILED(hr)) return hr;
	SAFE_RELEASE(pBlob);


	hr = CompileShaderFromFile(L"Shaders\\VoxelSplatting.hlsl", "GSBlend", "gs_5_0", &pBlob);
	if(FAILED(hr)) return hr;

	hr = pd3dDevice->CreateGeometryShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &s_pGeometryShaderBlend);
	if(FAILED(hr)) return hr;
	SAFE_RELEASE(pBlob);


	hr = CompileShaderFromFile(L"Shaders\\VoxelSplatting.hlsl", "PSBlend", "ps_5_0", &pBlob);
	if(FAILED(hr)) return hr;

	hr = pd3dDevice->CreatePixelShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &s_pPixelShaderBlend);
	if(FAILED(hr)) return hr;
	SAFE_RELEASE(pBlob);


	hr = CompileShaderFromFile(L"Shaders\\VoxelSplatting.hlsl", "PSNorm", "ps_5_0", &pBlob);
	if(FAILED(hr)) return hr;

	hr = pd3dDevice->CreatePixelShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &s_pPixelShaderNorm);
	if(FAILED(hr)) return hr;
	SAFE_RELEASE(pBlob);

	// Create Constant Buffers
	D3D11_BUFFER_DESC cbDesc;
	ZeroMemory(&cbDesc, sizeof(D3D11_BUFFER_DESC));
	cbDesc.Usage = D3D11_USAGE_DYNAMIC;
	cbDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	cbDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	cbDesc.MiscFlags = 0;

	cbDesc.ByteWidth = sizeof(cbConstant);
	hr = pd3dDevice->CreateBuffer(&cbDesc, NULL, &s_ConstantBuffer);
	if(FAILED(hr)) return hr;


	//for first pass
	D3D11_TEXTURE2D_DESC descTex;
	ZeroMemory(&descTex, sizeof(D3D11_TEXTURE2D_DESC));
	descTex.Usage = D3D11_USAGE_DEFAULT;
	descTex.BindFlags = D3D11_BIND_DEPTH_STENCIL | D3D11_BIND_SHADER_RESOURCE;
	descTex.CPUAccessFlags = 0;
	descTex.MiscFlags = 0;
	descTex.SampleDesc.Count = 1;
	descTex.SampleDesc.Quality = 0;
	descTex.ArraySize = 1;
	descTex.MipLevels = 1;
	descTex.Format = DXGI_FORMAT_R32_TYPELESS;
	descTex.Width = GlobalAppState::getInstance().s_windowWidth;
	descTex.Height = GlobalAppState::getInstance().s_windowHeight;
	V_RETURN(pd3dDevice->CreateTexture2D(&descTex, NULL, &s_pDepthStencil));

	D3D11_DEPTH_STENCIL_VIEW_DESC descDSV;
	ZeroMemory( &descDSV, sizeof(D3D11_DEPTH_STENCIL_VIEW_DESC) );
	descDSV.Format = DXGI_FORMAT_D32_FLOAT;
	descDSV.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
	descDSV.Texture2D.MipSlice = 0;
	V_RETURN(pd3dDevice->CreateDepthStencilView(s_pDepthStencil, &descDSV, &s_pDepthStencilDSV));

	D3D11_SHADER_RESOURCE_VIEW_DESC descSRV;
	ZeroMemory(&descSRV, sizeof(D3D11_SHADER_RESOURCE_VIEW_DESC));
	descSRV.Format = DXGI_FORMAT_R32_FLOAT;
	descSRV.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
	descSRV.Texture2D.MipLevels = 1;
	descSRV.Texture2D.MostDetailedMip = 0;
	V_RETURN(pd3dDevice->CreateShaderResourceView(s_pDepthStencil, &descSRV, &s_pDepthStencilSRV));

	//for second pass
	descTex.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
	descTex.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
	V_RETURN(pd3dDevice->CreateTexture2D(&descTex, NULL, &s_pPositions));
	V_RETURN(pd3dDevice->CreateShaderResourceView(s_pPositions, NULL, &s_pPositionsSRV));
	V_RETURN(pd3dDevice->CreateRenderTargetView(s_pPositions, NULL, &s_pPositionsRTV));
	//descTex.Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
	V_RETURN(pd3dDevice->CreateTexture2D(&descTex, NULL, &s_pColors));
	V_RETURN(pd3dDevice->CreateShaderResourceView(s_pColors, NULL, &s_pColorsSRV));
	V_RETURN(pd3dDevice->CreateRenderTargetView(s_pColors, NULL, &s_pColorsRTV));

	//for third pass
	descTex.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
	V_RETURN(pd3dDevice->CreateTexture2D(&descTex, NULL, &s_pPositionsNorm));
	V_RETURN(pd3dDevice->CreateShaderResourceView(s_pPositionsNorm, NULL, &s_pPositionsNormSRV));
	V_RETURN(pd3dDevice->CreateRenderTargetView(s_pPositionsNorm, NULL, &s_pPositionsNormRTV));
	//descTex.Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
	descTex.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
	V_RETURN(pd3dDevice->CreateTexture2D(&descTex, NULL, &s_pColorsNorm));
	V_RETURN(pd3dDevice->CreateShaderResourceView(s_pColorsNorm, NULL, &s_pColorsNormSRV));
	V_RETURN(pd3dDevice->CreateRenderTargetView(s_pColorsNorm, NULL, &s_pColorsNormRTV));
	descTex.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
	V_RETURN(pd3dDevice->CreateTexture2D(&descTex, NULL, &s_pColorsCompleted));
	V_RETURN(pd3dDevice->CreateShaderResourceView(s_pColorsCompleted, NULL, &s_pColorsCompletedSRV));
	V_RETURN(pd3dDevice->CreateUnorderedAccessView(s_pColorsCompleted, NULL, &s_pColorsCompletedUAV));

	//for fourth pass
	descTex.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
	descTex.Width = GlobalAppState::getInstance().s_windowWidth;
	descTex.Height = GlobalAppState::getInstance().s_windowHeight;
	V_RETURN(pd3dDevice->CreateTexture2D(&descTex, NULL, &s_pPositionsKinectRes));
	V_RETURN(pd3dDevice->CreateShaderResourceView(s_pPositionsKinectRes, NULL, &s_pPositionsKinectResSRV));
	V_RETURN(pd3dDevice->CreateUnorderedAccessView(s_pPositionsKinectRes, NULL, &s_pPositionsKinectResUAV));
	V_RETURN(pd3dDevice->CreateTexture2D(&descTex, NULL, &s_pNormalsKinectRes));
	V_RETURN(pd3dDevice->CreateShaderResourceView(s_pNormalsKinectRes, NULL, &s_pNormalsKinectResSRV));
	V_RETURN(pd3dDevice->CreateUnorderedAccessView(s_pNormalsKinectRes, NULL, &s_pNormalsKinectResUAV));
	descTex.Format = DXGI_FORMAT_R32_FLOAT;
	V_RETURN(pd3dDevice->CreateTexture2D(&descTex, NULL, &s_pDepthKinectRes));
	V_RETURN(pd3dDevice->CreateShaderResourceView(s_pDepthKinectRes, NULL, &s_pDepthKinectResSRV));
	V_RETURN(pd3dDevice->CreateUnorderedAccessView(s_pDepthKinectRes, NULL, &s_pDepthKinectResUAV));
	V_RETURN(pd3dDevice->CreateTexture2D(&descTex, NULL, &s_pDepthKinectResFiltered));
	V_RETURN(pd3dDevice->CreateShaderResourceView(s_pDepthKinectResFiltered, NULL, &s_pDepthKinectResFilteredSRV));
	V_RETURN(pd3dDevice->CreateUnorderedAccessView(s_pDepthKinectResFiltered, NULL, &s_pDepthKinectResFilteredUAV));

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
	blendDesc.RenderTarget[1] = blendDesc.RenderTarget[0];

	pd3dDevice->CreateBlendState(&blendDesc, &s_pBlendStateDefault);

	blendDesc.RenderTarget[0].BlendEnable = true;
	blendDesc.RenderTarget[0].SrcBlend = D3D11_BLEND_ONE;
	blendDesc.RenderTarget[0].DestBlend = D3D11_BLEND_ONE;
	blendDesc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
	blendDesc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ONE;
	blendDesc.RenderTarget[1] = blendDesc.RenderTarget[0];

	pd3dDevice->CreateBlendState(&blendDesc, &s_pBlendStateAdditive);

	return  hr;
}

void DX11VoxelSplatting::OnD3D11DestroyDevice()
{

	SAFE_RELEASE(s_ConstantBuffer);

	SAFE_RELEASE(s_pVertexShaderSinglePoint);

	SAFE_RELEASE(s_pVertexShaderSprite);
	//SAFE_RELEASE(s_pPixelShaderSpriteBlend);

	SAFE_RELEASE(s_pGeometryShaderReSplat);
	SAFE_RELEASE(s_pGeometryShaderReSplatBlend);

	SAFE_RELEASE(s_pVertexShader);
	SAFE_RELEASE(s_pGeometryShader);
	SAFE_RELEASE(s_pPixelShader);
	SAFE_RELEASE(s_pGeometryShaderBlend);
	SAFE_RELEASE(s_pPixelShaderBlend);
	SAFE_RELEASE(s_pPixelShaderNorm);

	SAFE_RELEASE(s_pDepthStencil);
	SAFE_RELEASE(s_pDepthStencilDSV);
	SAFE_RELEASE(s_pDepthStencilSRV);

	SAFE_RELEASE(s_pPositions);
	SAFE_RELEASE(s_pPositionsRTV);
	SAFE_RELEASE(s_pPositionsSRV);
	SAFE_RELEASE(s_pColors);
	SAFE_RELEASE(s_pColorsRTV);
	SAFE_RELEASE(s_pColorsSRV);
	SAFE_RELEASE(s_pBlendStateDefault);
	SAFE_RELEASE(s_pBlendStateAdditive);

	SAFE_RELEASE(s_pPositionsNorm );
	SAFE_RELEASE(s_pPositionsNormRTV);
	SAFE_RELEASE(s_pPositionsNormSRV);
	SAFE_RELEASE(s_pColorsNorm );
	SAFE_RELEASE(s_pColorsNormRTV);
	SAFE_RELEASE(s_pColorsNormSRV);
	SAFE_RELEASE(s_pColorsCompleted);
	SAFE_RELEASE(s_pColorsCompletedSRV);
	SAFE_RELEASE(s_pColorsCompletedUAV);

	SAFE_RELEASE(s_pPositionsKinectRes);
	SAFE_RELEASE(s_pPositionsKinectResSRV);
	SAFE_RELEASE(s_pPositionsKinectResUAV);
	SAFE_RELEASE(s_pNormalsKinectRes);
	SAFE_RELEASE(s_pNormalsKinectResSRV);
	SAFE_RELEASE(s_pNormalsKinectResUAV);
	SAFE_RELEASE(s_pDepthKinectRes);
	SAFE_RELEASE(s_pDepthKinectResSRV);
	SAFE_RELEASE(s_pDepthKinectResUAV);
	SAFE_RELEASE(s_pDepthKinectResFiltered);
	SAFE_RELEASE(s_pDepthKinectResFilteredSRV);
	SAFE_RELEASE(s_pDepthKinectResFilteredUAV);
}
