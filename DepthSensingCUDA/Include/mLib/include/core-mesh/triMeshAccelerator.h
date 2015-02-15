
#ifndef _COREMESH_TRIMESHACCELERATOR_H_
#define _COREMESH_TRIMESHACCELERATOR_H_

namespace ml {

//////////////////////////////////////////////////////////////////////////
// Interface for TriMesh Acceleration Structures
//////////////////////////////////////////////////////////////////////////

template<class FloatType>
class TriMeshAccelerator 
{

public:

	//////////////////////////////////////////////////////////////////////////
	// API helpers
	//////////////////////////////////////////////////////////////////////////

	void build(const TriMesh<FloatType> &mesh, bool storeLocalCopy = false) {
		std::vector<const TriMesh<FloatType>* > meshes;
		meshes.push_back(&mesh);
		build(meshes, storeLocalCopy);
	}

	void build(const std::vector<const TriMesh<FloatType>* >& triMeshes, bool storeLocalCopy = false) {
		destroy();
		std::vector<const std::vector<typename TriMesh<FloatType>::Vertex<FloatType>>*> vertices(triMeshes.size());
		std::vector<const std::vector<vec3ui>*> indices(triMeshes.size());

		if (storeLocalCopy) {
			m_VerticesCopy.resize(triMeshes.size());
			for (size_t i = 0; i < triMeshes.size(); i++) {
				m_VerticesCopy[i] = triMeshes[i]->getVertices();
				vertices[i] = &m_VerticesCopy[i];
				indices[i] = &triMeshes[i]->getIndices();
			}
		} else {
			for (size_t i = 0; i < triMeshes.size(); i++) {
				vertices[i] = &triMeshes[i]->getVertices();
				indices[i] = &triMeshes[i]->getIndices();
			}
		}
		createTrianglePointers(vertices, indices);

		buildInternal();	//construct the acceleration structure
	}

	void build(const TriMesh<FloatType>& mesh, const Matrix4x4<FloatType>& transform) {
		std::vector< std::pair<const TriMesh<FloatType>*, Matrix4x4<FloatType>> > meshes;
		meshes.push_back(std::make_pair(&mesh, transform));
		build(meshes);
	}

	//! constructs the acceleration structure; always generates an internal copy
	void build(const std::vector<std::pair<const TriMesh<FloatType>*, Matrix4x4<FloatType>>>& triMeshPairs) {
		destroy();

		std::vector<const std::vector<typename TriMesh<FloatType>::Vertex<FloatType>>*> vertices(triMeshPairs.size());
		std::vector<const std::vector<vec3ui>*> indices(triMeshPairs.size());

		m_VerticesCopy.resize(triMeshPairs.size());
		for (size_t i = 0; i < triMeshPairs.size(); i++) {
			m_VerticesCopy[i] = triMeshPairs[i].first->getVertices();
			//apply the transform locally
			for (auto& v : m_VerticesCopy[i]) {
				v.position = triMeshPairs[i].second * v.position;
			}
			vertices[i] = &m_VerticesCopy[i];
			indices[i] = &triMeshPairs[i].first->getIndices();
		}
		createTrianglePointers(vertices, indices);

		buildInternal();	//construct the acceleration structure
	}

	size_t triangleCount() const
	{
		return m_Triangles.size();
	}

protected:

	std::vector<std::vector<typename TriMesh<FloatType>::Vertex<FloatType>> >			m_VerticesCopy;
	std::vector<typename TriMesh<FloatType>::Triangle<FloatType>>						m_Triangles;
	std::vector<typename TriMesh<FloatType>::Triangle<FloatType>*>						m_TrianglePointers;

private:

	//! takes a vector of meshes: including vertices and indices
	void createTrianglePointers(
		const std::vector<const std::vector<typename TriMesh<FloatType>::Vertex<FloatType>>*>& verticesVec,
		const std::vector<const std::vector<vec3ui>*>& indicesVec) 
	{
		//reserver memory
		m_Triangles.clear();
		size_t numTris = 0;
		for (size_t i = 0; i < indicesVec.size(); i++) {
			numTris += indicesVec[i]->size();
		}
		m_Triangles.reserve(numTris);

		//loop over meshes
		for (size_t m = 0; m < indicesVec.size(); m++) {
			const auto& indices = *indicesVec[m];
			const auto& vertices = *verticesVec[m];
			//loop over tris within a mesh
			for (size_t i = 0; i < indices.size(); i++) {
				//generate triangle with triangle and mesh index
				m_Triangles.push_back(typename TriMesh<FloatType>::Triangle<FloatType>(&vertices[indices[i].x], &vertices[indices[i].y], &vertices[indices[i].z], (unsigned int)i, (unsigned int)m));
			}
		}

		//create triangle pointers
		m_TrianglePointers.resize(m_Triangles.size());
		for (size_t i = 0; i < m_Triangles.size(); i++) {
			m_TrianglePointers[i] = &m_Triangles[i];
		}
	}

	void destroy() {
		m_Triangles.clear();
		m_TrianglePointers.clear();
		m_VerticesCopy.clear();
	}

	//////////////////////////////////////////////////////////////////////////
	// Interface Definition
	//////////////////////////////////////////////////////////////////////////

	//! given protected data above filed, the data structure is constructed
	virtual void buildInternal() = 0;
};

} // namespace ml

#endif
