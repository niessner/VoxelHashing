
#ifndef APPLICATION_D3D11_D3D11PIXELSHADER_H_
#define APPLICATION_D3D11_D3D11PIXELSHADER_H_

namespace ml {

class D3D11PixelShader : public GraphicsAsset
{
public:
	D3D11PixelShader()
	{
		m_shader = nullptr;
		m_blob = nullptr;
	}
	~D3D11PixelShader()
	{
		SAFE_RELEASE(m_shader);
		SAFE_RELEASE(m_blob);
	}
	void load(GraphicsDevice &g, const std::string &filename);

	void release(GraphicsDevice &g);
	void reset(GraphicsDevice &g);

    void bind(GraphicsDevice &g) const;

	UINT64 hash64();

private:
	ID3D11PixelShader *m_shader;
	ID3DBlob *m_blob;
	std::string m_filename;
};

}  // namespace ml

#endif  // APPLICATION_D3D11_D3D11PIXELSHADER_H_
