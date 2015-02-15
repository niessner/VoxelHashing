
#ifndef APPLICATION_D3D11_D3D11TEXTURE_H_
#define APPLICATION_D3D11_D3D11TEXTURE_H_

namespace ml {

class D3D11Texture : public GraphicsAsset
{
public:
    D3D11Texture()
	{
        m_texture = nullptr;
        m_view = nullptr;
	}
    //
    // TODO: implement other copy constructors similar to D3D11Mesh
    //
    D3D11Texture(D3D11Texture &&t)
    {
        m_bmp = std::move(t.m_bmp);
        m_view = t.m_view; t.m_view = nullptr;
        m_texture = t.m_texture; t.m_texture = nullptr;
    }
    void operator = (D3D11Texture &&t)
    {
        m_bmp = std::move(t.m_bmp);
        m_view = t.m_view; t.m_view = nullptr;
        m_texture = t.m_texture; t.m_texture = nullptr;
    }
    ~D3D11Texture()
	{
        SAFE_RELEASE(m_texture);
        SAFE_RELEASE(m_view);

        // m_view does not seem to be a correctly reference-counted object.
        m_view = nullptr;
        
	}
    D3D11Texture(GraphicsDevice &g, const Bitmap &bmp)
    {
        m_texture = nullptr;
        m_view = nullptr;
        load(g, bmp);
    }
	void load(GraphicsDevice &g, const Bitmap &bmp);

	void release(GraphicsDevice &g);
	void reset(GraphicsDevice &g);

    void bind(GraphicsDevice &g) const;

    GraphicsAssetType type() const
    {
        return GraphicsAssetTexture;
    }

    const Bitmap& bmp() const
    {
        return m_bmp;
    }

private:
    Bitmap m_bmp;
    ID3D11Texture2D *m_texture;
    ID3D11ShaderResourceView *m_view;
};

}  // namespace ml

#endif  // APPLICATION_D3D11_D3D11TEXTURE_H_
