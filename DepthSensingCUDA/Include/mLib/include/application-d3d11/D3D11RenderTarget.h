
#ifndef APPLICATION_D3D11_D3D11RENDERTARGET_H_
#define APPLICATION_D3D11_D3D11RENDERTARGET_H_

namespace ml {

class D3D11RenderTarget : public GraphicsAsset
{
public:
    D3D11RenderTarget()
	{
        m_width = 0;
        m_height = 0;
        m_texture = nullptr;
        m_captureTexture = nullptr;
        m_depthBuffer = nullptr;
        m_renderView = nullptr;
        m_depthView = nullptr;
	}
    D3D11RenderTarget(D3D11RenderTarget &&t)
    {
        //
        // TODO: swap semantics would be cleaner here
        //
        m_width = t.m_width;
        m_height = t.m_height;
        m_texture = t.m_texture; t.m_texture = nullptr;
        m_captureTexture = t.m_captureTexture; t.m_captureTexture = nullptr;
        m_depthBuffer = t.m_depthBuffer; t.m_depthBuffer = nullptr;
        m_renderView = t.m_renderView; t.m_renderView = nullptr;
        m_depthView = t.m_depthView; t.m_depthView = nullptr;
    }
    void operator = (D3D11RenderTarget &&t)
    {
        m_width = t.m_width;
        m_height = t.m_height;
        m_texture = t.m_texture; t.m_texture = nullptr;
        m_captureTexture = t.m_captureTexture; t.m_captureTexture = nullptr;
        m_depthBuffer = t.m_depthBuffer; t.m_depthBuffer = nullptr;
        m_renderView = t.m_renderView; t.m_renderView = nullptr;
        m_depthView = t.m_depthView; t.m_depthView = nullptr;
    }
    ~D3D11RenderTarget()
	{
        SAFE_RELEASE(m_renderView);
        SAFE_RELEASE(m_depthView);
        SAFE_RELEASE(m_texture);
        SAFE_RELEASE(m_captureTexture);
        SAFE_RELEASE(m_depthBuffer);
	}
    D3D11RenderTarget(GraphicsDevice &g, const UINT width, const UINT height)
    {
        m_texture = nullptr;
        m_captureTexture = nullptr;
        m_depthBuffer = nullptr;
        m_renderView = nullptr;
        m_depthView = nullptr;
        load(g, width, height);
    }

    // create a new render target with given width and height. Also creates an equal-sized depth buffer.
    void load(GraphicsDevice &g, const UINT width, const UINT height);

	void release(GraphicsDevice &g);
	void reset(GraphicsDevice &g);

    // sets the render and depth buffers as the render target for the current device.
    // to return to the original graphics device render target, call bindRenderDepth() on the graphics device.
    void bind(GraphicsDevice &g);

    // clears the render and depth buffers
    void clear(GraphicsDevice &g, const ml::vec4f &clearColor);

    // save the render target data as a bitmap
    void captureBitmap(GraphicsDevice &g, Bitmap &result);
    
    void captureColorBuffer(GraphicsDevice &g, ColorImageR8G8B8A8 &result);
    void captureDepthBuffer(GraphicsDevice &g, ColorImageR32 &result);

    GraphicsAssetType type() const
    {
        return GraphicsAssetTexture;
    }

private:
    UINT m_width, m_height;
    ID3D11Texture2D *m_texture;
    ID3D11Texture2D *m_captureTexture;
    ID3D11Texture2D *m_depthBuffer;

    ID3D11RenderTargetView *m_renderView;
    ID3D11DepthStencilView *m_depthView;
};

}  // namespace ml

#endif  // APPLICATION_D3D11_D3D11RENDERTARGET_H_
