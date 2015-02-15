#ifndef CORE_MESH_MESHSHAPES_H_
#define CORE_MESH_MESHSHAPES_H_

namespace ml {

namespace shapes {

TriMeshf plane(const vec3f &start, const vec3f &end, const vec3f &normal);

TriMeshf box(float xDim, float yDim, float zDim, const vec4f& color = ml::vec4f(1, 1, 1, 1));

TriMeshf box(const OBBf &obb, const vec4f& color = ml::vec4f(1, 1, 1, 1));

TriMeshf box(const bbox3f &bbox, const vec4f& color = ml::vec4f(1, 1, 1, 1));

inline TriMeshf box(float dim = 1, const vec4f& color = vec4f(1, 1, 1, 1)) { return box(dim, dim, dim, color); }

TriMeshf cylinder(float radius, float height, UINT stacks, UINT slices, const vec4f& color = ml::vec4f(1, 1, 1, 1));

TriMeshf cylinder(const vec3f &p0, const vec3f &p1, float radius, UINT stacks, UINT slices, const vec4f& color = ml::vec4f(1, 1, 1, 1));

TriMeshf torus(const vec3f &center, float majorRadius, float minorRadius, UINT stacks, UINT slices, const vec4f& color = ml::vec4f(1, 1, 1, 1));

TriMeshf torus(const vec3f &center, float majorRadius, float minorRadius, UINT stacks, UINT slices, const std::function<vec4f(unsigned int)> &stackIndexToColor);

inline TriMeshf line(const vec3f& p0, const vec3f& p1, const vec4f& color, const float thickness) {
  return cylinder(p0, p1, thickness, 2, 9, color);
}

TriMeshf wireframeBox(float dimension, const vec4f& color, float thickness);

TriMeshf wireframeBox(const mat4f& unitCubeToWorld, const vec4f& color, float thickness);

TriMeshf sphere(const float radius, const ml::vec3f& pos, const size_t stacks = 10, const size_t slices = 10, const ml::vec4f& color = ml::vec4f(1,1,1,1));

template<class FloatType>
MeshData<FloatType> toMeshData(const BoundingBox3<FloatType>& s, const point4d<FloatType>& color = vec4f(1,1,1,1), bool bottomPlaneOnly = false) {
	MeshData<FloatType> meshData;	std::vector<vec3ui> indices;
	if (bottomPlaneOnly) {
		s.makeTriMeshBottomPlane(meshData.m_Vertices, indices, meshData.m_Normals);
	} else {
		s.makeTriMesh(meshData.m_Vertices, indices, meshData.m_Normals);
	}
	//meshData.m_FaceIndicesVertices.resize(indices.size(), std::vector<unsigned int>(3));
	meshData.m_FaceIndicesVertices.resize(indices.size(), 3);
	for (size_t i = 0; i < indices.size(); i++) {
		meshData.m_FaceIndicesVertices[i][0] = indices[i].x;
		meshData.m_FaceIndicesVertices[i][1] = indices[i].y;
		meshData.m_FaceIndicesVertices[i][2] = indices[i].z;
	}
	meshData.m_Colors.resize(meshData.m_Vertices.size(), color);
	return meshData;
}

}	// namespace shapes

}  // namespace ml

#endif  // INCLUDE_CORE_MESH_MESHSHAPES_H_
