
void ml::D3D11VertexShader::load(GraphicsDevice &g, const std::string &filename)
{
    if (!util::fileExists(filename))
    {
        std::cout << "file not found: " << filename << std::endl;
        return;
    }
	release(g);
	SAFE_RELEASE(m_blob);

	m_filename = filename;
	g.castD3D11().registerAsset(this);

	m_blob = D3D11Utility::CompileShader(m_filename, "vertexShaderMain", "vs_4_0");
	MLIB_ASSERT_STR(m_blob != nullptr, "CompileShader failed");

	reset(g);
}

void ml::D3D11VertexShader::release(GraphicsDevice &g)
{
	SAFE_RELEASE(m_shader);
	SAFE_RELEASE(m_standardLayout);
}

void ml::D3D11VertexShader::reset(GraphicsDevice &g)
{
	release(g);

	auto &device = g.castD3D11().device();
	
	D3D_VALIDATE(device.CreateVertexShader(m_blob->GetBufferPointer(), m_blob->GetBufferSize(), nullptr, &m_shader));

	device.CreateInputLayout( D3D11TriMesh::layout, D3D11TriMesh::layoutElementCount, m_blob->GetBufferPointer(), m_blob->GetBufferSize(), &m_standardLayout );
}

void ml::D3D11VertexShader::bind(GraphicsDevice &g) const
{
	auto &context = g.castD3D11().context();
	context.VSSetShader( m_shader, nullptr, 0 );
	context.IASetInputLayout( m_standardLayout );
}