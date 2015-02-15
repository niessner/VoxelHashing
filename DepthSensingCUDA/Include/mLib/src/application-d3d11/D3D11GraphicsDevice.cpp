
void ml::D3D11GraphicsDevice::init(const WindowWin32 &window)
{
    //
    // Instead of reconstructing the backbuffer everytime the window is resized, we construct a generic, large backbuffer.
    //
	  m_width = 2560;
    m_height = m_width * window.height() / window.width();

	UINT createDeviceFlags = 0;
#ifdef _DEBUG
	createDeviceFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

	DXGI_SWAP_CHAIN_DESC swapChainDesc;
	ZeroMemory( &swapChainDesc, sizeof( swapChainDesc ) );
	swapChainDesc.BufferCount = 1;
	swapChainDesc.BufferDesc.Width = m_width;
	swapChainDesc.BufferDesc.Height = m_height;
	swapChainDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	swapChainDesc.BufferDesc.RefreshRate.Numerator = 60;
	swapChainDesc.BufferDesc.RefreshRate.Denominator = 1;
	swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	swapChainDesc.OutputWindow = window.handle();
	swapChainDesc.SampleDesc.Count = 1;
	swapChainDesc.SampleDesc.Quality = 0;
	swapChainDesc.Windowed = TRUE;

	D3D_FEATURE_LEVEL featureLevels[] =
	{
		D3D_FEATURE_LEVEL_11_0,
		D3D_FEATURE_LEVEL_10_1,
		D3D_FEATURE_LEVEL_10_0,
	};
	UINT numFeatureLevels = ARRAYSIZE( featureLevels );

	D3D_VALIDATE(D3D11CreateDeviceAndSwapChain( nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, createDeviceFlags, featureLevels, numFeatureLevels,
				 D3D11_SDK_VERSION, &swapChainDesc, &m_swapChain, &m_device, &m_featureLevel, &m_context ));

    //
	// Create a render target view
    //
	ID3D11Texture2D* backBuffer = nullptr;
	D3D_VALIDATE(m_swapChain->GetBuffer( 0, __uuidof( ID3D11Texture2D ), ( LPVOID* )&backBuffer ));
	
	D3D_VALIDATE(m_device->CreateRenderTargetView( backBuffer, nullptr, &m_renderView ));
	backBuffer->Release();

    //
	// Setup the rasterizer state
    //
	m_rasterDesc.AntialiasedLineEnable = false;
	m_rasterDesc.CullMode = D3D11_CULL_NONE;
	m_rasterDesc.DepthBias = 0;
	m_rasterDesc.DepthBiasClamp = 0.0f;
	m_rasterDesc.DepthClipEnable = true;
	m_rasterDesc.FillMode = D3D11_FILL_SOLID;
	m_rasterDesc.FrontCounterClockwise = false;
	m_rasterDesc.MultisampleEnable = false;
	m_rasterDesc.ScissorEnable = false;
	m_rasterDesc.SlopeScaledDepthBias = 0.0f;

	D3D_VALIDATE(m_device->CreateRasterizerState(&m_rasterDesc, &m_rasterState));
	m_context->RSSetState(m_rasterState);

    //
    // Create the depth buffer
    //
    D3D11_TEXTURE2D_DESC depthDesc;
    depthDesc.Width = m_width;
    depthDesc.Height = m_height;
    depthDesc.MipLevels = 1;
    depthDesc.ArraySize = 1;
    depthDesc.Format = DXGI_FORMAT_D32_FLOAT;
    depthDesc.SampleDesc.Count = 1;
    depthDesc.SampleDesc.Quality = 0;
    depthDesc.Usage = D3D11_USAGE_DEFAULT;
    depthDesc.BindFlags = D3D11_BIND_DEPTH_STENCIL;
    depthDesc.CPUAccessFlags = 0;
    depthDesc.MiscFlags = 0;
    D3D_VALIDATE(m_device->CreateTexture2D( &depthDesc, nullptr, &m_depthBuffer ));

    //
    // Setup the depth state
    //
    D3D11_DEPTH_STENCIL_DESC depthStateDesc;
    depthStateDesc.DepthEnable = true;
    depthStateDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
    depthStateDesc.DepthFunc = D3D11_COMPARISON_LESS;
    depthStateDesc.StencilEnable = false;
    D3D_VALIDATE(m_device->CreateDepthStencilState(&depthStateDesc, &m_depthState));
    m_context->OMSetDepthStencilState(m_depthState, 1);

    //
    // Setup the depth view
    //
    D3D11_DEPTH_STENCIL_VIEW_DESC depthViewDesc;
    depthViewDesc.Flags = 0;
    depthViewDesc.Format = DXGI_FORMAT_D32_FLOAT;
    depthViewDesc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
    depthViewDesc.Texture2D.MipSlice = 0;

    // Create the depth stencil view
    D3D_VALIDATE(m_device->CreateDepthStencilView( m_depthBuffer, &depthViewDesc, &m_depthView ));

    m_context->OMSetRenderTargets( 1, &m_renderView, m_depthView );

    //
    // Setup the viewport
    //
    D3D11_VIEWPORT viewport;
    viewport.Width = (FLOAT)m_width;
    viewport.Height = (FLOAT)m_height;
    viewport.MinDepth = 0.0f;
    viewport.MaxDepth = 1.0f;
    viewport.TopLeftX = 0;
    viewport.TopLeftY = 0;
    m_context->RSSetViewports( 1, &viewport );

    //
    // Setup the sampler state
    //
    D3D11_SAMPLER_DESC samplerStateDesc;
    samplerStateDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    samplerStateDesc.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
    samplerStateDesc.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;
    samplerStateDesc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
    samplerStateDesc.MipLODBias = 0.0f;
    samplerStateDesc.MaxAnisotropy = 1;
    samplerStateDesc.ComparisonFunc = D3D11_COMPARISON_NEVER;
    samplerStateDesc.MinLOD = -FLT_MAX;
    samplerStateDesc.MaxLOD = FLT_MAX;
    D3D_VALIDATE(m_device->CreateSamplerState(&samplerStateDesc, &m_samplerState));
    m_context->PSSetSamplers(0, 1, &m_samplerState);

#ifdef MLIB_GRAPHICSDEVICE_TRANSPARENCY
    ID3D11BlendState* d3dBlendState;
    D3D11_BLEND_DESC omDesc;
    ZeroMemory(&omDesc, sizeof(D3D11_BLEND_DESC));
    omDesc.RenderTarget[0].BlendEnable = true;
    omDesc.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC_ALPHA;
    omDesc.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
    omDesc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
    omDesc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ZERO;
    omDesc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ZERO;
    omDesc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
    omDesc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
    D3D_VALIDATE(m_device->CreateBlendState(&omDesc, &d3dBlendState));
    m_context->OMSetBlendState(d3dBlendState, 0, 0xffffffff);
#endif

#ifdef _DEBUG
	m_device->QueryInterface(__uuidof(ID3D11Debug), reinterpret_cast<void**>(&m_debug));
#endif
}

void ml::D3D11GraphicsDevice::resize(UINT width, UINT height)
{
    
}

void ml::D3D11GraphicsDevice::registerAsset(GraphicsAsset *asset)
{
	m_assets.insert(asset);
	//if (std::find(m_assets.begin(), m_assets.end(), asset) != m_assets.end()) m_assets.push_back(asset);
}

void ml::D3D11GraphicsDevice::renderBeginFrame()
{
	float clearColor[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
	m_context->ClearRenderTargetView( m_renderView, clearColor );
    m_context->ClearDepthStencilView( m_depthView, D3D11_CLEAR_DEPTH, 1.0f, 0);
}

void ml::D3D11GraphicsDevice::bindRenderDepth()
{
    m_context->OMSetRenderTargets(1, &m_renderView, m_depthView);
    D3D11_VIEWPORT viewport;
    viewport.Width = (FLOAT)m_width;
    viewport.Height = (FLOAT)m_height;
    viewport.MinDepth = 0.0f;
    viewport.MaxDepth = 1.0f;
    viewport.TopLeftX = 0;
    viewport.TopLeftY = 0;
    m_context->RSSetViewports(1, &viewport);
}

void ml::D3D11GraphicsDevice::clear(const ml::vec4f &clearColor)
{
    m_context->ClearRenderTargetView( m_renderView, clearColor.array );
    m_context->ClearDepthStencilView( m_depthView, D3D11_CLEAR_DEPTH, 1.0f, 0);
}

void ml::D3D11GraphicsDevice::renderEndFrame()
{
	m_swapChain->Present( 1, 0 );
}

void ml::D3D11GraphicsDevice::toggleWireframe()
{
    m_context->RSGetState(&m_rasterState);
    m_rasterState->GetDesc(&m_rasterDesc);
    if(m_rasterDesc.FillMode == D3D11_FILL_SOLID)
        m_rasterDesc.FillMode = D3D11_FILL_WIREFRAME;
    else
        m_rasterDesc.FillMode = D3D11_FILL_SOLID;
    m_rasterState->Release();
    D3D_VALIDATE(m_device->CreateRasterizerState(&m_rasterDesc, &m_rasterState));
    m_context->RSSetState(m_rasterState);
}

void ml::D3D11GraphicsDevice::setCullMode(D3D11_CULL_MODE mode)
{
    m_context->RSGetState(&m_rasterState);
    m_rasterState->GetDesc(&m_rasterDesc);
    m_rasterDesc.CullMode = mode;
    m_rasterState->Release();
    D3D_VALIDATE(m_device->CreateRasterizerState(&m_rasterDesc, &m_rasterState));
    m_context->RSSetState(m_rasterState);
}

void ml::D3D11GraphicsDevice::toggleCullMode()
{
    m_context->RSGetState(&m_rasterState);
    m_rasterState->GetDesc(&m_rasterDesc);
    if (m_rasterDesc.CullMode == D3D11_CULL_NONE)
        m_rasterDesc.CullMode = D3D11_CULL_FRONT;
    else
        m_rasterDesc.CullMode = D3D11_CULL_NONE;
    m_rasterState->Release();
    D3D_VALIDATE(m_device->CreateRasterizerState(&m_rasterDesc, &m_rasterState));
    m_context->RSSetState(m_rasterState);
}

void ml::D3D11GraphicsDevice::captureBackBufferInternal(Bitmap &result)
{
    ID3D11Texture2D* frameBuffer;

    D3D_VALIDATE(m_swapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), (LPVOID*)&frameBuffer));

    D3D11_TEXTURE2D_DESC desc;
    frameBuffer->GetDesc(&desc);

    if(m_captureBuffer == nullptr)
    {
        desc.BindFlags = 0;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ | D3D11_CPU_ACCESS_WRITE;
        desc.Usage = D3D11_USAGE_STAGING;
        D3D_VALIDATE(m_device->CreateTexture2D(&desc, nullptr, &m_captureBuffer));
    }

    m_context->CopyResource( m_captureBuffer, frameBuffer );

    result.allocate(desc.Height, desc.Width);
    
    D3D11_MAPPED_SUBRESOURCE resource;
    UINT subresource = D3D11CalcSubresource( 0, 0, 0 );
    HRESULT hr = m_context->Map( m_captureBuffer, subresource, D3D11_MAP_READ_WRITE, 0, &resource );
    const BYTE *data = (BYTE *)resource.pData;
    //resource.pData; // TEXTURE DATA IS HERE

    for( UINT row = 0; row < desc.Height; row++ )
    {
        memcpy( &result(row, 0), data + resource.RowPitch * row, desc.Width * sizeof(RGBColor) );
    }

    m_context->Unmap(m_captureBuffer, subresource);
}