
#ifndef APPLICATION_D3D11_D3D11TRIMESH_H_
#define APPLICATION_D3D11_D3D11TRIMESH_H_

namespace ml {

class D3D11TriMesh : public GraphicsAsset
{
public:
    static const UINT layoutElementCount = 4;
    static const D3D11_INPUT_ELEMENT_DESC layout[layoutElementCount];

	D3D11TriMesh()
	{
        m_device = nullptr;
		m_vertexBuffer = nullptr;
		m_indexBuffer = nullptr;
	}

	template<class T>
	D3D11TriMesh(GraphicsDevice &g, const MeshData<T>& meshData)
    {
		m_vertexBuffer = nullptr;
		m_indexBuffer = nullptr;
		load(g, meshData);
	}

	template<class T>
	D3D11TriMesh(GraphicsDevice &g, const TriMesh<T>& triMesh)
    {
		m_vertexBuffer = nullptr;
		m_indexBuffer = nullptr;
		load(g, triMesh);
	}

	~D3D11TriMesh()
	{
		SAFE_RELEASE(m_vertexBuffer);
		SAFE_RELEASE(m_indexBuffer);
	}

    void load(GraphicsDevice &g, const D3D11TriMesh& mesh)
    {
        m_triMesh = mesh.m_triMesh;
        reset(g);
    }

	template<class T>
	void load(GraphicsDevice &g, const TriMesh<T>& triMesh)
    {
        m_device = &g.castD3D11();
        m_triMesh = triMesh;
		reset(g);
	}

	template<class T>
	void load(GraphicsDevice &g, const MeshData<T>& meshData)
    {
        load(g, TriMesh<T>(meshData));
	}

	void release(GraphicsDevice &g);
	void reset(GraphicsDevice &g);

	void render(GraphicsDevice &g) const;

	void updateColors(GraphicsDevice &g, const std::vector<vec4f> &vertexColors);

    bbox3f boundingBox() const
    {
        return m_triMesh.getBoundingBox();
    }

    const TriMeshf& getTriMesh() const
    {
        return m_triMesh;
	}

	void getMeshData(MeshDataf& meshData) const
    {
        meshData = m_triMesh.getMeshData();
	}

	MeshDataf getMeshData() const {
		MeshDataf meshData;
		getMeshData(meshData);
		return meshData;
	}

    D3D11TriMesh(const D3D11TriMesh &t)
    {
        m_vertexBuffer = nullptr;
        m_indexBuffer = nullptr;
        load(*t.m_device, t);
    }
    D3D11TriMesh(D3D11TriMesh &&t)
    {
        m_device = t.m_device; t.m_device = nullptr;
        m_vertexBuffer = t.m_vertexBuffer; t.m_vertexBuffer = nullptr;
        m_indexBuffer = t.m_indexBuffer; t.m_indexBuffer = nullptr;
        m_triMesh = std::move(t.m_triMesh);
    }

    void operator = (const D3D11TriMesh& t)
    {
        m_vertexBuffer = nullptr;
        m_indexBuffer = nullptr;
        load(*t.m_device, t);
    }

    void operator = (D3D11TriMesh&& t)
    {
        m_device = t.m_device; t.m_device = nullptr;
        m_vertexBuffer = t.m_vertexBuffer; t.m_vertexBuffer = nullptr;
        m_indexBuffer = t.m_indexBuffer; t.m_indexBuffer = nullptr;
        m_triMesh = std::move(t.m_triMesh);
    }

private:
    void initVB(GraphicsDevice &g);
	void initIB(GraphicsDevice &g);

    D3D11GraphicsDevice *m_device;

	ID3D11Buffer *m_vertexBuffer;
	ID3D11Buffer *m_indexBuffer;
	
    TriMeshf m_triMesh;
};

}  // namespace ml

#endif  // APPLICATION_D3D11_D3D11TRIMESH_H_