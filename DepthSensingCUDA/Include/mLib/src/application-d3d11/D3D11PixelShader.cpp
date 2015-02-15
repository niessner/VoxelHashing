
void ml::D3D11PixelShader::load(GraphicsDevice &g, const std::string &filename)
{
	release(g);
	SAFE_RELEASE(m_blob);

	m_filename = filename;
	g.castD3D11().registerAsset(this);

	m_blob = D3D11Utility::CompileShader(m_filename, "pixelShaderMain", "ps_4_0");
	MLIB_ASSERT_STR(m_blob != nullptr, "CompileShader failed");

	reset(g);
}

void ml::D3D11PixelShader::release(GraphicsDevice &g)
{
	SAFE_RELEASE(m_shader);
}

void ml::D3D11PixelShader::reset(GraphicsDevice &g)
{
	release(g);

	auto &device = g.castD3D11().device();

	D3D_VALIDATE(device.CreatePixelShader(m_blob->GetBufferPointer(), m_blob->GetBufferSize(), nullptr, &m_shader));
}

void ml::D3D11PixelShader::bind(GraphicsDevice &g) const
{
	auto &context = g.castD3D11().context();
	context.PSSetShader( m_shader, nullptr, 0 );
}