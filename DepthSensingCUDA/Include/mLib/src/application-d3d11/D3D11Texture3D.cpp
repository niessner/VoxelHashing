
namespace ml
{

void D3D11Texture3D::load(GraphicsDevice &g, const Grid3<RGBColor> &data)
{
    release(g);
    
    g.castD3D11().registerAsset(this);
    m_data = data;

    reset(g);
}

void D3D11Texture3D::release(GraphicsDevice &g)
{
    SAFE_RELEASE(m_texture);
    SAFE_RELEASE(m_view);
}

void D3D11Texture3D::reset(GraphicsDevice &g)
{
    release(g);

    if (m_data.dimX() == 0)
        return;

    auto &device = g.castD3D11().device();
    auto &context = g.castD3D11().context();

    D3D11_TEXTURE3D_DESC desc;
    desc.Width = (UINT)m_data.dimX();
    desc.Height = (UINT)m_data.dimY();
    desc.Depth = (UINT)m_data.dimZ();
    desc.MipLevels = 0;
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
    desc.CPUAccessFlags = 0;
    desc.MiscFlags = D3D11_RESOURCE_MISC_GENERATE_MIPS;

    D3D_VALIDATE(device.CreateTexture3D(&desc, nullptr, &m_texture));
    D3D_VALIDATE(device.CreateShaderResourceView(m_texture, nullptr, &m_view));

    context.UpdateSubresource(m_texture, 0, nullptr, m_data.ptr(), (UINT)m_data.dimX() * sizeof(RGBColor), (UINT)m_data.dimX() * (UINT)m_data.dimY() * sizeof(RGBColor));

    context.GenerateMips(m_view);
}

void D3D11Texture3D::bind(GraphicsDevice &g) const
{
    if (m_view == nullptr)
        return;
    auto &context = g.castD3D11().context();
    context.PSSetShaderResources(0, 1, &m_view);
}

}