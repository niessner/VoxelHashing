
#ifndef APPLICATION_D3D11_D3D11VERTEXSHADER_H_
#define APPLICATION_D3D11_D3D11VERTEXSHADER_H_

namespace ml {

class D3D11VertexShader : public GraphicsAsset
{
public:
	D3D11VertexShader()
	{
		m_shader = nullptr;
		m_blob = nullptr;
		m_standardLayout = nullptr;
	}
	~D3D11VertexShader()
	{
		SAFE_RELEASE(m_shader);
		SAFE_RELEASE(m_blob);
		SAFE_RELEASE(m_standardLayout);
	}
	void load(GraphicsDevice &g, const std::string &filename);

	void release(GraphicsDevice &g);
	void reset(GraphicsDevice &g);

    void bind(GraphicsDevice &g) const;

private:
	ID3D11VertexShader *m_shader;
	ID3DBlob *m_blob;
	std::string m_filename;

	ID3D11InputLayout *m_standardLayout;
};

}  // namespace ml

#endif  // APPLICATION_D3D11_D3D11VERTEXSHADER_H_
