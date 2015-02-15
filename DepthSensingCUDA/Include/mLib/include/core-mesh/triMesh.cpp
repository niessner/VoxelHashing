
#ifndef CORE_MESH_TRIMESH_INL_H_
#define CORE_MESH_TRIMESH_INL_H_

namespace ml {
	
	template<class FloatType>
	TriMesh<FloatType>::TriMesh( const MeshData<FloatType>& meshData )
	{

		m_Vertices.resize(meshData.m_Vertices.size());

		m_bHasNormals = meshData.m_Normals.size() > 0;
		m_bHasColors = meshData.m_Colors.size() > 0;
		m_bHasTexCoords = meshData.m_TextureCoords.size() > 0;

		for (size_t i = 0; i < m_Vertices.size(); i++) {
			m_Vertices[i].position = meshData.m_Vertices[i];
		}

		for (unsigned int i = 0; i < meshData.m_FaceIndicesVertices.size(); i++) {
			if (meshData.m_FaceIndicesVertices[i].size() == 3) {
				//we need to split vertices if the same vertex has different texcoords and/or normals
				bool bFaceHasNormals	= m_bHasNormals && meshData.getFaceIndicesNormals()[i].size() > 0;
				bool bFaceHasTexCoords	= m_bHasTexCoords && meshData.getFaceIndicesTexCoords()[i].size() > 0;
				bool bFaceHasColors		= m_bHasColors && meshData.getFaceIndicesColors()[i].size() > 0;

				if (bFaceHasNormals || bFaceHasTexCoords || bFaceHasColors) {
					vec3ui coords = vec3ui(0,0,0);
					for (unsigned int j = 0; j < 3; j++) {
						bool vertexSplit = false;
						if (bFaceHasNormals) { //split if normal is different than the one found before
							const point3d<FloatType>& n = meshData.m_Normals[meshData.getFaceIndicesNormals()[i][j]];
							if (m_Vertices[meshData.getFaceIndicesVertices()[i][j]].normal != point3d<FloatType>::origin && 
								m_Vertices[meshData.getFaceIndicesVertices()[i][j]].normal != n)	vertexSplit = true;
						}
						if (bFaceHasTexCoords) { //split if texcoord is different than the one found before
							const point2d<FloatType>& t = meshData.m_TextureCoords[meshData.getFaceIndicesTexCoords()[i][j]];
							if (m_Vertices[meshData.getFaceIndicesVertices()[i][j]].texCoord != point2d<FloatType>::origin && 
								m_Vertices[meshData.getFaceIndicesVertices()[i][j]].texCoord != t) vertexSplit = true;
						}
						if (bFaceHasColors) { //split if texcoord is different than the one found before
							const point4d<FloatType>& c = meshData.m_Colors[meshData.getFaceIndicesColors()[i][j]];
							if (m_Vertices[meshData.getFaceIndicesVertices()[i][j]].color != point4d<FloatType>::origin && 
								m_Vertices[meshData.getFaceIndicesVertices()[i][j]].color != c) vertexSplit = true;
						}

						if (vertexSplit) {
							MLIB_WARNING("vertex split untested");
							Vertex<FloatType> v = m_Vertices[meshData.getFaceIndicesVertices()[i][j]];
							if (bFaceHasNormals)		v.normal = meshData.m_Normals[meshData.getFaceIndicesNormals()[i][j]];
							if (bFaceHasTexCoords)	v.texCoord = meshData.m_TextureCoords[meshData.getFaceIndicesTexCoords()[i][j]];
							if (bFaceHasColors)		v.color = meshData.m_Colors[meshData.getFaceIndicesColors()[i][j]];
							m_Vertices.push_back(v);
							coords[j] = (unsigned int)m_Vertices.size() - 1;
						} else {
							if (bFaceHasNormals)		m_Vertices[meshData.getFaceIndicesVertices()[i][j]].normal = meshData.m_Normals[meshData.getFaceIndicesNormals()[i][j]];
							if (bFaceHasTexCoords)	m_Vertices[meshData.getFaceIndicesVertices()[i][j]].texCoord = meshData.m_TextureCoords[meshData.getFaceIndicesTexCoords()[i][j]];
							if (bFaceHasColors)		m_Vertices[meshData.getFaceIndicesVertices()[i][j]].color = meshData.m_Colors[meshData.getFaceIndicesColors()[i][j]];
							coords[j] = meshData.getFaceIndicesVertices()[i][j];
						}
					}
					m_Indices.push_back(coords);

					//m_Indices.push_back(vec3ui(meshData.m_FaceIndicesVertices[i][0], meshData.m_FaceIndicesVertices[i][1], meshData.m_FaceIndicesVertices[i][2]));
					//if (hasNormals) {
					//	//we are ignoring the fact that sometimes there might be vertex split required (if a vertex has two different normals)
					//	m_Vertices[m_Indices[i][0]].normal = meshData.m_Normals[meshData.m_FaceIndicesNormals[i][0]];
					//	m_Vertices[m_Indices[i][1]].normal = meshData.m_Normals[meshData.m_FaceIndicesNormals[i][1]];
					//	m_Vertices[m_Indices[i][2]].normal = meshData.m_Normals[meshData.m_FaceIndicesNormals[i][2]];
					//}
					//if (hasTexCoords) {
					//	//we are ignoring the fact that sometimes there might be vertex split required (if a vertex has two different normals)
					//	m_Vertices[m_Indices[i][0]].texCoord = meshData.m_TextureCoords[meshData.m_FaceIndicesTextureCoords[i][0]];
					//	m_Vertices[m_Indices[i][1]].texCoord = meshData.m_TextureCoords[meshData.m_FaceIndicesTextureCoords[i][1]];
					//	m_Vertices[m_Indices[i][2]].texCoord = meshData.m_TextureCoords[meshData.m_FaceIndicesTextureCoords[i][2]];
					//}
				} else {
					m_Indices.push_back(vec3ui(meshData.m_FaceIndicesVertices[i][0], meshData.m_FaceIndicesVertices[i][1], meshData.m_FaceIndicesVertices[i][2]));
				}
			} else {
				MLIB_WARNING("non triangle face found - ignoring it");
			}
		}
	}


	template<class FloatType>
	void TriMesh<FloatType>::computeNormals()
	{
		for (int i = 0; i < (int)m_Vertices.size(); i++) {
			m_Vertices[i].normal = point3d<FloatType>::origin;
		}

		for (int i = 0; i < (int)m_Indices.size(); i++) {
			point3d<FloatType> faceNormal = 
				(m_Vertices[m_Indices[i].y].position - m_Vertices[m_Indices[i].x].position)^(m_Vertices[m_Indices[i].z].position - m_Vertices[m_Indices[i].x].position);

			m_Vertices[m_Indices[i].x].normal += faceNormal;
			m_Vertices[m_Indices[i].y].normal += faceNormal;
			m_Vertices[m_Indices[i].z].normal += faceNormal;
		}
		for (int i = 0; i < (int)m_Vertices.size(); i++) {
			m_Vertices[i].normal.normalize();
		}

		m_bHasNormals = true;
	}

    template<class FloatType>
    TriMesh<FloatType> TriMesh<FloatType>::flatLoopSubdivision(UINT iterations, float minEdgeLength) const
    {
        TriMeshf result = *this;
        for (UINT i = 0; i < iterations; i++)
            result = result.flatLoopSubdivision(minEdgeLength);
        return result;
    }

    template<class FloatType>
    TriMesh<FloatType> TriMesh<FloatType>::flatLoopSubdivision(float minEdgeLength) const
    {
        struct Edge
        {
            Edge(UINT32 _v0, UINT32 _v1)
            {
                v0 = std::min(_v0, _v1);
                v1 = std::max(_v0, _v1);
            }

            union
            {
                struct {
                    UINT32 v0, v1;
                };
                UINT64 val;
            };
        };

        struct edgeCompare
        {
            bool operator() (const Edge &a, const Edge &b)
            {
                return a.val < b.val;
            }
        };
        
        map<Edge, UINT, edgeCompare> edgeToNewVertexMap;

        TriMesh<FloatType> result;
        
        result.m_Vertices = m_Vertices;
        result.m_Indices.reserve(m_Indices.size() * 4);

        for (const vec3ui &tri : m_Indices)
        {
            /*bool subdivide = true;
            for (UINT eIndex = 0; eIndex < 3; eIndex++)
            {
                const vec3f &v0 = m_Vertices[tri[eIndex]].position;
                const vec3f &v1 = m_Vertices[tri[(eIndex + 1) % 3]].position;
                float edgeLength = vec3f::dist(v0, v1);
                if (edgeLength < minEdgeLength)
                    subdivide = false;
            }*/
            bool subdivide = math::triangleArea(m_Vertices[tri[0]].position, m_Vertices[tri[1]].position, m_Vertices[tri[2]].position) >= (minEdgeLength * minEdgeLength);

            if (subdivide)
            {
                UINT edgeMidpoints[3];

                for (UINT eIndex = 0; eIndex < 3; eIndex++)
                {
                    const UINT v0 = tri[eIndex];
                    const UINT v1 = tri[(eIndex + 1) % 3];
                    Edge e = Edge(v0, v1);
                    if (edgeToNewVertexMap.count(e) == 0)
                    {
                        edgeToNewVertexMap[e] = (UINT)result.m_Vertices.size();
                        result.m_Vertices.push_back((m_Vertices[v0] + m_Vertices[v1]) * (FloatType)0.5);
                    }

                    edgeMidpoints[eIndex] = edgeToNewVertexMap[e];
                }

                result.m_Indices.push_back(vec3ui(tri[0], edgeMidpoints[0], edgeMidpoints[2]));
                result.m_Indices.push_back(vec3ui(edgeMidpoints[0], tri[1], edgeMidpoints[1]));
                result.m_Indices.push_back(vec3ui(edgeMidpoints[2], edgeMidpoints[1], tri[2]));
                result.m_Indices.push_back(vec3ui(edgeMidpoints[2], edgeMidpoints[0], edgeMidpoints[1]));
            }
            else
            {
                result.m_Indices.push_back(tri);
            }
        }

        return result;
    }


}

#endif // CORE_MESH_TRIMESH_INL_H_