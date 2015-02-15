
const D3D11_INPUT_ELEMENT_DESC ml::D3D11TriMesh::layout[layoutElementCount] =
{
	{ "position", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
	{ "normal", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
	{ "color", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
	{ "texCoord", 0, DXGI_FORMAT_R32G32_FLOAT, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
};

void ml::D3D11TriMesh::updateColors(GraphicsDevice &g, const std::vector<vec4f> &newValues)
{
    auto &vertices = m_triMesh.getVertices();
	if (newValues.size() != vertices.size())
        throw MLIB_EXCEPTION("vertex buffer size doesn't match");

    for (size_t i = 0; i < newValues.size(); i++)
		vertices[i].color = newValues[i];

    reset(g);
}


void ml::D3D11TriMesh::release(GraphicsDevice &g)
{
	SAFE_RELEASE(m_vertexBuffer);
	SAFE_RELEASE(m_indexBuffer);
}

void ml::D3D11TriMesh::reset(GraphicsDevice &g)
{
	release(g);
	initVB(g);
	initIB(g);
}

void ml::D3D11TriMesh::initVB(GraphicsDevice &g)
{
    if (m_triMesh.getVertices().size() == 0) return;
	auto &device = g.castD3D11().device();

	D3D11_BUFFER_DESC bufferDesc;
	ZeroMemory( &bufferDesc, sizeof(bufferDesc) );
	bufferDesc.Usage = D3D11_USAGE_DEFAULT;
    bufferDesc.ByteWidth = sizeof(TriMeshf::Vertex<float>) * (UINT)m_triMesh.getVertices().size();
	bufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	bufferDesc.CPUAccessFlags = 0;

	D3D11_SUBRESOURCE_DATA data;
	ZeroMemory( &data, sizeof(data) );
    data.pSysMem = &m_triMesh.getVertices()[0];

	D3D_VALIDATE(device.CreateBuffer( &bufferDesc, &data, &m_vertexBuffer ));
}

void ml::D3D11TriMesh::initIB(GraphicsDevice &g)
{
    auto &indices = m_triMesh.getIndices();
    if (indices.size() == 0) return;
	auto &device = g.castD3D11().device();

	D3D11_BUFFER_DESC bufferDesc;
	ZeroMemory( &bufferDesc, sizeof(bufferDesc) );
	bufferDesc.Usage = D3D11_USAGE_DEFAULT;
	bufferDesc.ByteWidth = sizeof( vec3ui ) * (UINT)indices.size();
	bufferDesc.BindFlags = D3D11_BIND_INDEX_BUFFER;
	bufferDesc.CPUAccessFlags = 0;

	D3D11_SUBRESOURCE_DATA data;
	ZeroMemory( &data, sizeof(data) );
    data.pSysMem = &indices[0];

	D3D_VALIDATE(device.CreateBuffer( &bufferDesc, &data, &m_indexBuffer ));
}

void ml::D3D11TriMesh::render(GraphicsDevice &g) const
{
    if (m_triMesh.getIndices().size() == 0) return;
	auto &context = g.castD3D11().context();

	context.IASetIndexBuffer( m_indexBuffer, DXGI_FORMAT_R32_UINT, 0 );

    UINT stride = sizeof(TriMeshf::Vertex<float>);
	UINT offset = 0;
	context.IASetVertexBuffers( 0, 1, &m_vertexBuffer, &stride, &offset );

	context.IASetPrimitiveTopology( D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST );

    context.DrawIndexed((UINT)m_triMesh.getIndices().size() * 3, 0, 0);
}
