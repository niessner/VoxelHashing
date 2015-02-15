
#ifndef APPLICATION_D3D11_D3D11CONSTANTBUFFER_H_
#define APPLICATION_D3D11_D3D11CONSTANTBUFFER_H_

namespace ml {

template<class T>
class D3D11ConstantBuffer : public GraphicsAsset
{
public:
	D3D11ConstantBuffer()
	{
		m_buffer = nullptr;
	}
	~D3D11ConstantBuffer()
	{
		SAFE_RELEASE(m_buffer);
	}
	void init(GraphicsDevice &g)
	{
		reset(g);
	}

	void release(GraphicsDevice &g)
	{
		SAFE_RELEASE(m_buffer);
	}

	void reset(GraphicsDevice &g)
	{
		release(g);

		D3D11_BUFFER_DESC desc;
		ZeroMemory( &desc, sizeof(desc) );
		desc.Usage = D3D11_USAGE_DEFAULT;
		desc.ByteWidth = sizeof(T);
		desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
		desc.CPUAccessFlags = 0;
		D3D_VALIDATE(g.castD3D11().device().CreateBuffer( &desc, nullptr, &m_buffer ));
	}

    void updateAndBind(GraphicsDevice &g, const T &data, UINT constantBufferIndex)
    {
        update(g, data);
        bindVertexShader(g, constantBufferIndex);
        bindPixelShader(g, constantBufferIndex);
    }

	void update(GraphicsDevice &g, const T &data)
	{
		g.castD3D11().context().UpdateSubresource( m_buffer, 0, nullptr, &data, 0, 0 );
	}

	void bindVertexShader(GraphicsDevice &g, UINT constantBufferIndex)
	{
		g.castD3D11().context().VSSetConstantBuffers( constantBufferIndex, 1, &m_buffer );
	}

	void bindPixelShader(GraphicsDevice &g, UINT constantBufferIndex)
	{
		g.castD3D11().context().PSSetConstantBuffers( constantBufferIndex, 1, &m_buffer );
	}

private:
	ID3D11Buffer *m_buffer;
};

}  // namespace ml

#endif  // APPLICATION_D3D11_D3D11CONSTANTBUFFER_H_
