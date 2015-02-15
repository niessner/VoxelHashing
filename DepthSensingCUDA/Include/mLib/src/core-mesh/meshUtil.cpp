
namespace ml {

    //
    // TODO: really, the TriMeshf version should be the one that calls the TriMesh* version!!!
    //
    TriMeshf meshutil::createUnifiedMesh(const std::vector< std::pair<const TriMeshf*, mat4f> >& meshes) {
        std::vector< std::pair<TriMeshf, mat4f> > meshList;

        for (const auto &m : meshes)
        {
            meshList.push_back(std::make_pair(*m.first, m.second));
        }

        return createUnifiedMesh(meshList);
    }

TriMeshf meshutil::createUnifiedMesh(const std::vector< std::pair<TriMeshf, mat4f> >& meshes) {
  auto lambdaVertices = [](size_t total, const std::pair<TriMeshf, mat4f>& t) { return t.first.getVertices().size() + total; };
  const size_t totalPoints = std::accumulate(meshes.begin(), meshes.end(), static_cast<size_t>(0), lambdaVertices);
  auto lambdaIndices = [](size_t total, const std::pair<TriMeshf, mat4f>& t) { return t.first.getIndices().size() + total; };
  const size_t totalIndices = std::accumulate(meshes.begin(), meshes.end(), static_cast<size_t>(0), lambdaIndices);

  std::vector<TriMeshf::Vertexf> vertices(totalPoints);
  std::vector<vec3ui> indices(totalIndices);

  UINT vIndex = 0, iIndex = 0;
  for (const auto& m : meshes) {
    const UINT baseVertexIndex = vIndex;

    for (UINT vertexIndex = 0; vertexIndex < m.first.getVertices().size(); vertexIndex++) {
      TriMeshf::Vertexf& v = vertices[vIndex++];
      v = m.first.getVertices()[vertexIndex];
      v.position =  m.second * v.position;
    }
    for (UINT indexIndex = 0; indexIndex < m.first.getIndices().size(); indexIndex++) {
      indices[iIndex++] = m.first.getIndices()[indexIndex] + vec3ui(baseVertexIndex,baseVertexIndex,baseVertexIndex);
    }
  }

  return TriMeshf(vertices, indices);
}

TriMeshf meshutil::createUnifiedMesh(const std::vector<TriMeshf>& meshes) {
  auto lambdaVertices = [](size_t total, const TriMeshf& t) { return t.getVertices().size() + total; };
  const size_t totalPoints = std::accumulate(meshes.begin(), meshes.end(), static_cast<size_t>(0), lambdaVertices);
  auto lambdaIndices = [](size_t total, const TriMeshf& t) { return t.getIndices().size() + total; };
  const size_t totalIndices = std::accumulate(meshes.begin(), meshes.end(), static_cast<size_t>(0), lambdaIndices);

  std::vector<TriMeshf::Vertexf> vertices(totalPoints);
  std::vector<vec3ui> indices(totalIndices);

  UINT vIndex = 0, iIndex = 0;
  for (const auto& m : meshes) {
    const UINT baseVertexIndex = vIndex;

    for (UINT vertexIndex = 0; vertexIndex < m.getVertices().size(); vertexIndex++) {
      TriMeshf::Vertexf& v = vertices[vIndex++];
      v = m.getVertices()[vertexIndex];
    }
    for (UINT indexIndex = 0; indexIndex < m.getIndices().size(); indexIndex++) {
      indices[iIndex++] = m.getIndices()[indexIndex] + vec3ui(baseVertexIndex,baseVertexIndex,baseVertexIndex);
    }
  }

  return TriMeshf(vertices, indices, false, false, false, true);
}

TriMeshf meshutil::createPointCloudTemplate(const TriMeshf& templateMesh,
                                           const std::vector<vec3f>& points,
                                           const std::vector<vec4f>& colors) {
  const UINT64 pointCount = points.size();
  const UINT64 tVertices = templateMesh.getVertices().size();
  const UINT64 tIndices = templateMesh.getIndices().size();
  const vec4f defaultColor(1.f, 0.f, 0.f, 1.0f);

  std::vector<TriMeshf::Vertexf> vertices(pointCount * tVertices);
  std::vector<vec3ui> indices(pointCount * tIndices);

  for (UINT pointIndex = 0; pointIndex < points.size(); pointIndex++) {
    const vec3f& p = points[pointIndex];
    const vec4f& c = colors.empty() ? defaultColor : colors[pointIndex];
    const UINT64 baseVertexIndex = pointIndex * tVertices;

    for (UINT vertexIndex = 0; vertexIndex < tVertices; vertexIndex++) {
      TriMeshf::Vertexf& v = vertices[baseVertexIndex + vertexIndex];
      v = templateMesh.getVertices()[vertexIndex];
      v.position += p;
      v.color = c;
    }
    for (UINT indexIndex = 0; indexIndex < tIndices; indexIndex++) {
      indices[pointIndex * tIndices + indexIndex] = templateMesh.getIndices()[indexIndex] + vec3ui(pointIndex * (UINT)tVertices);
    }
  }

  return TriMeshf(vertices, indices);
}

}  // namespace ml
