namespace ml {

namespace shapes {

float cubeVData[8][3] = {
	{1.0f, 1.0f, 1.0f}, { -1.0f, 1.0f, 1.0f}, { -1.0f, -1.0f, 1.0f},
	{1.0f, -1.0f, 1.0f}, {1.0f, 1.0f, -1.0f}, { -1.0f, 1.0f, -1.0f},
	{ -1.0f, -1.0f, -1.0f}, {1.0f, -1.0f, -1.0f}
};

int cubeIData[12][3] = {
	{1, 2, 3}, {1, 3, 0}, {0, 3, 7}, {0, 7, 4}, {3, 2, 6},
	{3, 6, 7}, {1, 6, 2}, {1, 5, 6}, {0, 5, 1}, {0, 4, 5},
	{6, 5, 4}, {6, 4, 7}
};

int cubeEData[12][2] = {
	{0, 1}, {1, 2}, {2, 3}, {3, 0},
	{4, 5}, {5, 6}, {6, 7}, {7, 4},
	{0, 4}, {1, 5}, {2, 6}, {3, 7}
};

TriMeshf plane(const vec3f &start, const vec3f &end, const vec3f &normal)
{
    std::vector<TriMeshf::Vertexf> vertices(4);
    std::vector<UINT> indices(6);

    for (auto &v : vertices)
    {
        v.color = vec4f(1.0f, 1.0f, 1.0f, 1.0f);
        v.normal = normal;
    }

    vertices[0].position = vec3f(start.x, start.y, 0.0f);
    vertices[1].position = vec3f(end.x, start.y, 0.0f);
    vertices[2].position = vec3f(end.x, end.y, 0.0f);
    vertices[3].position = vec3f(start.x, end.y, 0.0f);

    indices[0] = 0; indices[1] = 1; indices[2] = 2;
    indices[3] = 0; indices[4] = 2; indices[5] = 3;

    return TriMeshf(vertices, indices);
}

TriMeshf box(const bbox3f &bbox, const vec4f& color)
{
    auto extent = bbox.getExtent();
    auto center = bbox.getCenter();
    TriMeshf result = box(extent.x, extent.y, extent.z, color);
    for (auto &v : result.getVertices())
    {
        v.position += center;
    }
    return result;
}

TriMeshf box(const OBBf &obb, const vec4f& color)
{
    TriMeshf result = box(bbox3f(vec3f::origin, vec3f(1.0f, 1.0f, 1.0f)), color);

    result.transform(obb.getOBBToWorld());

    return result;
}

TriMeshf box(float xDim, float yDim, float zDim, const vec4f& color) {
	std::vector<TriMeshf::Vertexf> vv(8);
	std::vector<UINT> vi(12 * 3);

	// Vertices
	for (int i = 0; i < 8; i++) {
		vv[i].position = vec3f(cubeVData[i][0], cubeVData[i][1], cubeVData[i][2]);
		vv[i].normal = vec3f(1.0f, 0.0f, 0.0f);  // TODO(ms): write and call generateNormals() function
		vv[i].color = color;
	}

	// Triangles
	for (int i = 0; i < 12; i++) {
		vi[i * 3 + 0] = cubeIData[i][0];
		vi[i * 3 + 1] = cubeIData[i][1];
		vi[i * 3 + 2] = cubeIData[i][2];
	}

	TriMeshf mesh(vv, vi);
	mesh.scale(vec3f(0.5f * xDim, 0.5f * yDim, 0.5f * zDim));

	return mesh;
}

TriMeshf cylinder(float radius, float height, UINT stacks, UINT slices, const vec4f& color) {
	std::vector<TriMeshf::Vertexf> vertices((stacks + 1) * slices);
	std::vector<UINT> indices(stacks * slices * 6);

	UINT vIndex = 0;
	for (UINT i = 0; i <= stacks; i++)
		for (UINT i2 = 0; i2 < slices; i2++)
		{
			auto& vtx = vertices[vIndex++];
			float theta = float(i2) * 2.0f * math::PIf / float(slices);
			vtx.position = vec3f(radius * cosf(theta), radius * sinf(theta), height * float(i) / float(stacks));
			vtx.normal = vec3f(1.0f, 0.0f, 0.0f);  // TODO(ms): write and call generateNormals() function
			vtx.color = color;
		}

		UINT iIndex = 0;
		for (UINT i = 0; i < stacks; i++)
			for (UINT i2 = 0; i2 < slices; i2++)
			{
				int i2p1 = (i2 + 1) % slices;

				indices[iIndex++] = (i + 1) * slices + i2;
				indices[iIndex++] = i * slices + i2;
				indices[iIndex++] = i * slices + i2p1;


				indices[iIndex++] = (i + 1) * slices + i2;
				indices[iIndex++] = i * slices + i2p1;
				indices[iIndex++] = (i + 1) * slices + i2p1;
			}

			return TriMeshf(vertices, indices, true);
}

TriMeshf cylinder(const vec3f& p0, const vec3f& p1, float radius, UINT stacks, UINT slices, const vec4f& color) {
	float height = (p1 - p0).length();

	TriMeshf result = shapes::cylinder(radius, height, stacks, slices, color);
	result.transform(mat4f::translation(p0) * mat4f::face(vec3f::eZ, p1 - p0));
	return result;
}

TriMeshf torus(const vec3f &center, float majorRadius, float minorRadius, UINT stacks, UINT slices, const vec4f& color)
{
	return torus(center, majorRadius, minorRadius, stacks, slices, [&](unsigned int stackIndex) { return color; });
}

TriMeshf torus(const vec3f &center, float majorRadius, float minorRadius, UINT stacks, UINT slices, const std::function<vec4f(unsigned int)> &stackIndexToColor)
{
	std::vector<TriMeshf::Vertexf> vertices(slices * stacks);
	std::vector<UINT> indices(stacks * slices * 6);

	UINT vIndex = 0;
  // initial theta faces y front
  float baseTheta = ml::math::PIf/2.0f; 
	for (UINT i = 0; i < stacks; i++)
	{
		float theta = float(i) * 2.0f * ml::math::PIf / float(stacks) + baseTheta;
		auto color = stackIndexToColor(i);
		float sinT = sinf(theta);
		float cosT = cosf(theta);
		ml::vec3f t0(cosT * majorRadius, sinT * majorRadius, 0.0f);
		for(UINT i2 = 0; i2 < slices; i2++)
		{
			auto& vtx = vertices[vIndex++];

			float phi = float(i2) * 2.0f * ml::math::PIf / float(slices);
			float sinP = sinf(phi);
			vtx.position = ml::vec3f(minorRadius * cosT * sinP, minorRadius * sinT * sinP, minorRadius * cosf(phi)) + t0;
			vtx.color = color;
		}
	}

	UINT iIndex = 0;
	for(UINT i = 0; i < stacks; i++)
	{
		UINT ip1 = (i + 1) % stacks;
		for(UINT i2 = 0; i2 < slices; i2++)
		{
			UINT i2p1 = (i2 + 1) % slices;

			indices[iIndex++] = ip1 * slices + i2;
			indices[iIndex++] = i * slices + i2;
			indices[iIndex++] = i * slices + i2p1;

			indices[iIndex++] = ip1 * slices + i2;
			indices[iIndex++] = i * slices + i2p1;
			indices[iIndex++] = ip1 * slices + i2p1;
		}
	}

	return TriMeshf(vertices, indices, true);
}

TriMeshf wireframeBox(float dim, const vec4f& color, float thickness) {
	std::vector<ml::TriMeshf> meshes;
	ml::vec3f v[8];  std::memmove(v, cubeVData, sizeof(v[0]) * 8);
	for (uint i = 0; i < 12; i++) {
		meshes.push_back(line(dim * v[cubeEData[i][0]], dim * v[cubeEData[i][1]], color, thickness));
	}
	return meshutil::createUnifiedMesh(meshes);
}

TriMeshf wireframeBox(const mat4f& xf, const vec4f& color, float thickness) {
	std::vector<ml::TriMeshf> meshes;
	ml::vec3f v[8];  std::memmove(v, cubeVData, sizeof(v[0]) * 8);
	for (uint i = 0; i < 8; i++) { v[i] = xf * v[i]; }
	for (unsigned int i = 0; i < 12; i++) {
		const ml::vec3f& p0 = v[cubeEData[i][0]];
		const ml::vec3f& p1 = v[cubeEData[i][1]];
		meshes.push_back(line(p0, p1, color, thickness));
	}
	return meshutil::createUnifiedMesh(meshes);
}

TriMeshf sphere(const float radius, const ml::vec3f& pos, const size_t stacks /*= 10*/, const size_t slices /*= 10*/, const ml::vec4f& color /*= ml::vec4f(1,1,1,1) */) {
	MeshDataf meshdata;
	auto& V = meshdata.m_Vertices;
	auto& I = meshdata.m_FaceIndicesVertices;
	auto& N = meshdata.m_Normals;
	auto& C = meshdata.m_Colors;
	const float thetaDivisor = 1.0f / stacks * ml::math::PIf;
	const float phiDivisor = 1.0f / slices * 2.0f * ml::math::PIf; 
	for (size_t t = 0; t < stacks; t++) { // stacks increment elevation (theta)
		float theta1 = t * thetaDivisor;
		float theta2 = (t + 1) * thetaDivisor;

		for (size_t p = 0; p < slices; p++) { // slices increment azimuth (phi)
			float phi1 = p * phiDivisor;
			float phi2 = (p + 1) * phiDivisor;

			const auto sph2xyz = [&](float r, float theta, float phi) {
				const float sinTheta = sinf(theta), sinPhi = sinf(phi), cosTheta = cosf(theta), cosPhi = cosf(phi);
				return ml::vec3f(r * sinTheta * cosPhi, r * sinTheta * sinPhi, r * cosTheta);
			};

			// phi2   phi1
			//  |      |
			//  2------1 -- theta1
			//  |\ _   |
			//  |    \ |
			//  3------4 -- theta2
			//  
			// Points
			const ml::vec3f c1 = pos + sph2xyz(radius, theta1, phi1),
				c2 = pos + sph2xyz(radius, theta1, phi2),
				c3 = pos + sph2xyz(radius, theta2, phi2),
				c4 = pos + sph2xyz(radius, theta2, phi1);
			V.push_back(c1);
			V.push_back(c2);
			V.push_back(c3);
			V.push_back(c4);

			// Colors
			for (int i = 0; i < 4; i++) {
				C.push_back(color);
			}

			// Normals
			N.push_back(c1.getNormalized());
			N.push_back(c2.getNormalized());
			N.push_back(c3.getNormalized());
			N.push_back(c4.getNormalized());

			const UINT baseIdx = static_cast<UINT>(t * slices * 4 + p * 4);

			// Indices
			std::vector<unsigned int> indices;
			if ( t == 0 ) {  // top cap -- t1p1, t2p2, t2p1
				indices.push_back(baseIdx + 0);
				indices.push_back(baseIdx + 2);
				indices.push_back(baseIdx + 3);
				I.push_back(indices);
			}
			else if ( t + 1 == stacks ) {  // bottom cap -- t2p2, t1p1, t1p2
				indices.push_back(baseIdx + 2);
				indices.push_back(baseIdx + 0);
				indices.push_back(baseIdx + 1);
				I.push_back(indices);
			}
			else {  // regular piece
				indices.push_back(baseIdx + 0);
				indices.push_back(baseIdx + 1);
				indices.push_back(baseIdx + 3);
				I.push_back(indices);
				indices.clear();
				indices.push_back(baseIdx + 1);
				indices.push_back(baseIdx + 2);
				indices.push_back(baseIdx + 3);
				I.push_back(indices);
			}
		}
	}
	//meshdata.mergeCloseVertices(0.00001f, true);
	return TriMeshf(meshdata);
}

}  // namespace shapes

}  // namespace ml