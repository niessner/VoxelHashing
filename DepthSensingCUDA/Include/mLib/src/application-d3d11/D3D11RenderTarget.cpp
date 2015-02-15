
namespace ml
{

void D3D11RenderTarget::load(GraphicsDevice &g, const UINT width, const UINT height)
{
    m_width = width;
    m_height = height;
    release(g);
    
    g.castD3D11().registerAsset(this);

    reset(g);
}

void D3D11RenderTarget::release(GraphicsDevice &g)
{
    SAFE_RELEASE(m_renderView);
    SAFE_RELEASE(m_depthView);
    SAFE_RELEASE(m_texture);
    SAFE_RELEASE(m_depthBuffer);
}

void D3D11RenderTarget::reset(GraphicsDevice &g)
{
    release(g);

    if (m_width == 0 || m_height == 0)
        return;

    auto &device = g.castD3D11().device();
    auto &context = g.castD3D11().context();

    //
    // Create the render target
    //
    D3D11_TEXTURE2D_DESC renderDesc;
    renderDesc.Width = m_width;
    renderDesc.Height = m_height;
    renderDesc.MipLevels = 0;
    renderDesc.ArraySize = 1;
    renderDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    renderDesc.SampleDesc.Count = 1;
    renderDesc.SampleDesc.Quality = 0;
    renderDesc.Usage = D3D11_USAGE_DEFAULT;
    renderDesc.BindFlags = D3D11_BIND_RENDER_TARGET;
    renderDesc.CPUAccessFlags = 0;
    renderDesc.MiscFlags = 0;

    D3D_VALIDATE(device.CreateTexture2D(&renderDesc, nullptr, &m_texture));
    
    //
    // Create the render target view
    //
    D3D_VALIDATE(device.CreateRenderTargetView(m_texture, nullptr, &m_renderView));
    
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
    D3D_VALIDATE(device.CreateTexture2D(&depthDesc, nullptr, &m_depthBuffer));

    //
    // Create the depth view
    //
    D3D11_DEPTH_STENCIL_VIEW_DESC depthViewDesc;
    depthViewDesc.Flags = 0;
    depthViewDesc.Format = DXGI_FORMAT_D32_FLOAT;
    depthViewDesc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
    depthViewDesc.Texture2D.MipSlice = 0;
    D3D_VALIDATE(device.CreateDepthStencilView(m_depthBuffer, nullptr, &m_depthView));

    //
    // Create the capture buffer
    //
    renderDesc.BindFlags = 0;
    renderDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ | D3D11_CPU_ACCESS_WRITE;
    renderDesc.Usage = D3D11_USAGE_STAGING;
    D3D_VALIDATE(device.CreateTexture2D(&renderDesc, nullptr, &m_captureTexture));
}

void D3D11RenderTarget::bind(GraphicsDevice &g)
{
    if (m_texture == nullptr)
        return;

    auto &context = g.castD3D11().context();
    context.OMSetRenderTargets(1, &m_renderView, m_depthView);

    D3D11_VIEWPORT viewport;
    viewport.Width = (FLOAT)m_width;
    viewport.Height = (FLOAT)m_height;
    viewport.MinDepth = 0.0f;
    viewport.MaxDepth = 1.0f;
    viewport.TopLeftX = 0;
    viewport.TopLeftY = 0;
    context.RSSetViewports(1, &viewport);
}

void D3D11RenderTarget::clear(GraphicsDevice &g, const ml::vec4f &clearColor)
{
    auto &context = g.castD3D11().context();
    context.ClearRenderTargetView(m_renderView, clearColor.array);
    context.ClearDepthStencilView(m_depthView, D3D11_CLEAR_DEPTH, 1.0f, 0);
}

void D3D11RenderTarget::captureDepthBuffer(GraphicsDevice &g, ColorImageR32 &result)
{
    std::cout << "captureDepthBuffer is not yet implemented" << std::endl;

    auto &context = g.castD3D11().context();
    context.CopyResource(m_captureTexture, m_texture);

    result.allocateToSize(m_height, m_width);

    D3D11_MAPPED_SUBRESOURCE resource;
    UINT subresource = D3D11CalcSubresource(0, 0, 0);
    HRESULT hr = context.Map(m_captureTexture, subresource, D3D11_MAP_READ, 0, &resource);
    const BYTE *data = (BYTE *)resource.pData;

    for (UINT row = 0; row < m_height; row++)
    {
        memcpy(&result(row, 0U), data + resource.RowPitch * row, m_width * sizeof(ml::vec4uc));
    }

    context.Unmap(m_captureTexture, subresource);
}

void D3D11RenderTarget::captureColorBuffer(GraphicsDevice &g, ColorImageR8G8B8A8 &result)
{
    auto &context = g.castD3D11().context();
    context.CopyResource(m_captureTexture, m_texture);

    result.allocateToSize(m_height, m_width);

    D3D11_MAPPED_SUBRESOURCE resource;
    UINT subresource = D3D11CalcSubresource(0, 0, 0);
    HRESULT hr = context.Map(m_captureTexture, subresource, D3D11_MAP_READ, 0, &resource);
    const BYTE *data = (BYTE *)resource.pData;

    for (UINT row = 0; row < m_height; row++)
    {
        memcpy(&result(row, 0U), data + resource.RowPitch * row, m_width * sizeof(ml::vec4uc));
    }

    context.Unmap(m_captureTexture, subresource);
}

void D3D11RenderTarget::captureBitmap(GraphicsDevice &g, Bitmap &result)
{
    auto &context = g.castD3D11().context();
    context.CopyResource(m_captureTexture, m_texture);

    result.allocate(m_height, m_width);

    D3D11_MAPPED_SUBRESOURCE resource;
    UINT subresource = D3D11CalcSubresource(0, 0, 0);
    HRESULT hr = context.Map(m_captureTexture, subresource, D3D11_MAP_READ, 0, &resource);
    const BYTE *data = (BYTE *)resource.pData;
    
    for (UINT row = 0; row < m_height; row++)
    {
        memcpy(&result(row, 0), data + resource.RowPitch * row, m_width * sizeof(RGBColor));
    }

    context.Unmap(m_captureTexture, subresource);
}

}