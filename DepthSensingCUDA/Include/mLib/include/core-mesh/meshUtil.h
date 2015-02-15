#ifndef CORE_MESH_MESHUTIL_H_
#define CORE_MESH_MESHUTIL_H_

namespace ml {
namespace meshutil {

TriMeshf createPointCloudTemplate(const TriMeshf& templateMesh, const std::vector<vec3f>& points, const std::vector<vec4f>& colors = std::vector<vec4f>());
TriMeshf createUnifiedMesh(const std::vector< std::pair<const TriMeshf*, mat4f> >& meshes);
TriMeshf createUnifiedMesh(const std::vector<std::pair<TriMeshf, mat4f>>& meshes);
TriMeshf createUnifiedMesh(const std::vector<TriMeshf>& meshes);

} // namespace meshutil
} // namespace ml

#endif  // CORE_MESH_MESHUTIL_H_
