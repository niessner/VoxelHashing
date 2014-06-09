
#ifndef CORE_MESH_MESHDATA_INL_H_
#define CORE_MESH_MESHDATA_INL_H_

namespace ml {

template<class FloatType>
std::ostream& operator<<(std::ostream& os, const MeshData<FloatType>& meshData) {
	os	<< "MeshData:\n" 
		<< "\tVertices:  " << meshData.m_Vertices.size() << "\n" 
		<< "\tColors:    " << meshData.m_Colors.size() << "\n"
		<< "\tNormals:   " << meshData.m_Normals.size() << "\n" 
		<< "\tTexCoords: " << meshData.m_TextureCoords.size() << "\n" 
		<< std::endl;

	return os;
}

static inline bool FaceLess(const std::vector<unsigned int>& t0_, const std::vector<unsigned int>& t1_)
{
	if (t0_.size() != t1_.size())	return t0_.size() < t1_.size();
	else {
		std::vector<unsigned int> t0 = t0_;
		std::vector<unsigned int> t1 = t1_;

		std::sort(t0.begin(), t0.end());
		std::sort(t1.begin(), t1.end());
		for (size_t i = 0; i < t0.size(); i++) {
			if (t0[i] == t1[i]) continue;
			return t0[i] < t1[i];
		}
	}
	return false;
}




template <class FloatType>
unsigned int MeshData<FloatType>::removeDuplicateFaces()
{
	//template<>
	//struct std::hash<std::vector<unsigned int>> : public std::unary_function<vec3i, size_t> {
	//	size_t operator()(const vec3i& v) const {
	//		//TODO larger prime number (64 bit) to match size_t
	//		const size_t p0 = 73856093;
	//		const size_t p1 = 19349669;
	//		const size_t p2 = 83492791;
	//		const size_t res = ((size_t)v.x * p0)^((size_t)v.y * p1)^((size_t)v.z * p2);
	//		return res;
	//	}
	//};

	struct vecHash {
		size_t operator()(const std::vector<unsigned int>& v) const {
			//TODO larger prime number (64 bit) to match size_t
			const size_t p[] = {73856093, 19349669, 83492791};
			size_t res = 0;
			for (unsigned int i : v) {
				res = res ^ (size_t)i * p[i%3];
			}
			return res;
			//const size_t res = ((size_t)v.x * p0)^((size_t)v.y * p1)^((size_t)v.z * p2);
		}
	};

	size_t numFaces = m_FaceIndicesVertices.size();
	std::vector<std::vector<unsigned int>> faces_new;
	faces_new.reserve(numFaces);

	std::unordered_set<std::vector<unsigned int>, vecHash> _set;
	for (size_t i = 0; i < numFaces; i++) {
		std::vector<unsigned int> f = m_FaceIndicesVertices[i];
		std::sort(f.begin(), f.end());
		if (_set.find(f) == _set.end()) {
			//not found yet
			_set.insert(f);
			faces_new.push_back(m_FaceIndicesVertices[i]);	//inserted the unsorted one
		}
	}
	if (m_FaceIndicesVertices.size() != faces_new.size()) {
		m_FaceIndicesVertices = std::vector<std::vector<unsigned int>>(faces_new.begin(), faces_new.end());
	}

	//std::cout << "Removed " << numFaces-faces_new.size() << " duplicate faces of " << numFaces << " faces" << std::endl;

	return (unsigned int)faces_new.size();
}




template <class FloatType>
static inline bool VertexLess(const point3d<FloatType>& v0, const point3d<FloatType>& v1)
{
	if (v0[0] < v1[0]) return true;
	if (v0[0] > v1[0]) return false;
	if (v0[1] < v1[1]) return true;
	if (v0[1] > v1[1]) return false;
	if (v0[2] < v1[2]) return true;

	return false;
}


template <class FloatType>
unsigned int MeshData<FloatType>::removeDuplicateVertices() {
	unsigned int numV = (unsigned int)m_Vertices.size();
	//int numT = (int)tris.size();

	std::map<point3d<FloatType>, unsigned int, bool(*)(const point3d<FloatType>&, const point3d<FloatType>&)> pts(VertexLess);

	std::vector<unsigned int> vertexLookUp;	vertexLookUp.resize(numV);
	std::vector<point3d<FloatType>> new_verts; new_verts.reserve(numV);
	std::vector<point4d<FloatType>> new_color;		if (hasPerVertexColors())		new_color.reserve(m_Colors.size());
	std::vector<point3d<FloatType>> new_normals;	if (hasPerVertexNormals())		new_normals.reserve(m_Normals.size());
	std::vector<point2d<FloatType>> new_tex;		if (hasPerVertexTexCoords())	new_tex.reserve(m_TextureCoords.size());

	unsigned int cnt = 0;
	for (size_t i1 = 0; i1 < numV; i1++) {
		const point3d<FloatType>& pt = m_Vertices[i1];

		std::map<point3d<FloatType>, unsigned int, bool(*)(const point3d<FloatType>&, const point3d<FloatType>&)>::iterator it = pts.find(pt);

		if (it != pts.end()) {
			vertexLookUp[i1] = it->second;
		} else {
			pts.insert(std::make_pair(pt, cnt));
			new_verts.push_back(pt);
			vertexLookUp[i1] = cnt;
			cnt++;
			if (hasPerVertexColors())		new_color.push_back(m_Colors[i1]);
			if (hasPerVertexNormals())		new_normals.push_back(m_Normals[i1]);
			if (hasPerVertexTexCoords())	new_tex.push_back(m_TextureCoords[i1]);
		}
	}

	// Update faces
	for (std::vector<std::vector<unsigned int>>::iterator it = m_FaceIndicesVertices.begin(); it != m_FaceIndicesVertices.end(); it++) {
		for (std::vector<unsigned int>::iterator idx = it->begin(); idx != it->end(); idx++) {
			*idx = vertexLookUp[*idx];
		}
		//*it = vertexLookUp[*it];
	}

	//std::cout << "Removed " << numV-cnt << " duplicate vertices of " << numV << " vertices" << std::endl;

	if (m_Vertices != new_verts) {
		m_Vertices = std::vector<point3d<FloatType>>(new_verts.begin(), new_verts.end());
		if (hasPerVertexColors())		m_Colors = std::vector<point4d<FloatType>>(new_color.begin(), new_color.end());
		if (hasPerVertexNormals())		m_Normals = std::vector<point3d<FloatType>>(new_normals.begin(), new_normals.end());
		if (hasPerVertexTexCoords())	m_TextureCoords = std::vector<point2d<FloatType>>(new_tex.begin(), new_tex.end());
	}

	return cnt;
}



template <class FloatType>
unsigned int MeshData<FloatType>::hasNearestNeighbor( const vec3i& coord, SparseGrid3D<std::list<std::pair<point3d<FloatType>,unsigned int>>> &neighborQuery, const point3d<FloatType>& v, FloatType thresh )
{
	FloatType threshSq = thresh*thresh;
	for (int i = -1; i <= 1; i++) {
		for (int j = -1; j <= 1; j++) {
			for (int k = -1; k <= 1; k++) {
				vec3i c = coord + vec3i(i,j,k);
				if (neighborQuery.exists(c)) {
					for (const std::pair<point3d<FloatType>, unsigned int>& n : neighborQuery[c]) {
						if (point3d<FloatType>::distSq(v,n.first) < threshSq) {
							return n.second;
						}
					}
				}
			}
		}
	}
	return (unsigned int)-1;
}

template <class FloatType>
unsigned int MeshData<FloatType>::hasNearestNeighborApprox(const vec3i& coord, SparseGrid3D<unsigned int> &neighborQuery, FloatType thresh ) {
	FloatType threshSq = thresh*thresh;

	for (int i = -1; i <= 1; i++) {
		for (int j = -1; j <= 1; j++) {
			for (int k = -1; k <= 1; k++) {
				vec3i c = coord + vec3i(i,j,k);
				if (neighborQuery.exists(c)) {
					return neighborQuery[c];
				}
			}
		}
	}
	return (unsigned int)-1;
}



template <class FloatType>
unsigned int MeshData<FloatType>::mergeCloseVertices(FloatType thresh, bool approx)
{
	if (thresh <= (FloatType)0)	throw MLIB_EXCEPTION("invalid thresh " + std::to_string(thresh));	
	unsigned int numV = (unsigned int)m_Vertices.size();

	std::vector<unsigned int> vertexLookUp;	vertexLookUp.resize(numV);
	std::vector<point3d<FloatType>> new_verts; new_verts.reserve(numV);
	std::vector<point4d<FloatType>> new_color;		if (hasPerVertexColors())		new_color.reserve(m_Colors.size());
	std::vector<point3d<FloatType>> new_normals;	if (hasPerVertexNormals())		new_normals.reserve(m_Normals.size());
	std::vector<point2d<FloatType>> new_tex;		if (hasPerVertexTexCoords())	new_tex.reserve(m_TextureCoords.size());

	unsigned int cnt = 0;
	if (approx) {
		SparseGrid3D<unsigned int> neighborQuery(0.6f, numV*2);
		for (unsigned int v = 0; v < numV; v++) {

			const point3d<FloatType>& vert = m_Vertices[v];
			vec3i coord = toVirtualVoxelPos(vert, thresh);		
			unsigned int nn = hasNearestNeighborApprox(coord, neighborQuery, thresh);

			if (nn == (unsigned int)-1) {
				neighborQuery[coord] = cnt;
				new_verts.push_back(vert);
				vertexLookUp[v] = cnt;
				cnt++;
				if (hasPerVertexColors())		new_color.push_back(m_Colors[v]);
				if (hasPerVertexNormals())		new_normals.push_back(m_Normals[v]);
				if (hasPerVertexTexCoords())	new_tex.push_back(m_TextureCoords[v]);
			} else {
				vertexLookUp[v] = nn;
			}
		}
	} else {
		SparseGrid3D<std::list<std::pair<point3d<FloatType>, unsigned int>>> neighborQuery(0.6f, numV*2);
		for (unsigned int v = 0; v < numV; v++) {

			const point3d<FloatType>& vert = m_Vertices[v];
			vec3i coord = toVirtualVoxelPos(vert, thresh);		
			unsigned int nn = hasNearestNeighbor(coord, neighborQuery, vert, thresh);

			if (nn == (unsigned int)-1) {
				neighborQuery[coord].push_back(std::make_pair(vert,cnt));
				new_verts.push_back(vert);
				vertexLookUp[v] = cnt;
				cnt++;
				if (hasPerVertexColors())		new_color.push_back(m_Colors[v]);
				if (hasPerVertexNormals())		new_normals.push_back(m_Normals[v]);
				if (hasPerVertexTexCoords())	new_tex.push_back(m_TextureCoords[v]);
			} else {
				vertexLookUp[v] = nn;
			}
		}
	}

	// Update faces
	for (std::vector<std::vector<unsigned int>>::iterator it = m_FaceIndicesVertices.begin(); it != m_FaceIndicesVertices.end(); it++) {
		for (std::vector<unsigned int>::iterator idx = it->begin(); idx != it->end(); idx++) {
			*idx = vertexLookUp[*idx];
		}
	}

	if (m_Vertices.size() != new_verts.size()) {
		m_Vertices = std::vector<point3d<FloatType>>(new_verts.begin(), new_verts.end());

		if (hasPerVertexColors())		m_Colors = std::vector<point4d<FloatType>>(new_color.begin(), new_color.end());
		if (hasPerVertexNormals())		m_Normals = std::vector<point3d<FloatType>>(new_normals.begin(), new_normals.end());
		if (hasPerVertexTexCoords())	m_TextureCoords = std::vector<point2d<FloatType>>(new_tex.begin(), new_tex.end());
	}

	removeDegeneratedFaces();
	//std::cout << "Merged " << numV-cnt << " of " << numV << " vertices" << std::endl;
	return cnt;
}



template <class FloatType>
unsigned int MeshData<FloatType>::removeDegeneratedFaces()
{
	std::vector<std::vector<unsigned int>> newFacesIndicesVertices;

	for (size_t i = 0; i < m_FaceIndicesVertices.size(); i++) {
		std::unordered_set<unsigned int> _set(m_FaceIndicesVertices[i].size());
		bool foundDuplicate = false;
		for (unsigned int idx : m_FaceIndicesVertices[i]) {
			if (_set.find(idx) != _set.end()) {
				foundDuplicate = true;
				break;
			} else {
				_set.insert(idx);
			}
		}
		if (!foundDuplicate) {
			newFacesIndicesVertices.push_back(m_FaceIndicesVertices[i]);
		}
	}
	if (m_FaceIndicesVertices.size() != newFacesIndicesVertices.size()) {
		m_FaceIndicesVertices = newFacesIndicesVertices;
	}

	return (unsigned int)m_FaceIndicesVertices.size();
}




template <class FloatType>
unsigned int MeshData<FloatType>::removeIsolatedVertices()
{
	unsigned int numV = (unsigned int)m_Vertices.size();
	std::vector<unsigned int> vertexLookUp;	vertexLookUp.resize(numV);
	std::vector<point3d<FloatType>> new_verts; new_verts.reserve(numV);
	std::vector<point4d<FloatType>> new_color;		if (hasPerVertexColors())		new_color.reserve(m_Colors.size());
	std::vector<point3d<FloatType>> new_normals;	if (hasPerVertexNormals())		new_normals.reserve(m_Normals.size());
	std::vector<point2d<FloatType>> new_tex;		if (hasPerVertexTexCoords())	new_tex.reserve(m_TextureCoords.size());

	std::unordered_map<unsigned int, unsigned int> _map(m_Vertices.size());
	unsigned int cnt = 0;
	for (auto& face : m_FaceIndicesVertices) {
		for (auto& idx : face) {
			if (_map.find(idx) != _map.end()) {
				idx = _map[idx];	//set to new idx, which already exists
			} else {
				_map[idx] = cnt;
				new_verts.push_back(m_Vertices[idx]);
				if (hasPerVertexColors())		new_color.push_back(m_Colors[idx]);
				if (hasPerVertexNormals())		new_normals.push_back(m_Normals[idx]);
				if (hasPerVertexTexCoords())	new_tex.push_back(m_TextureCoords[idx]);

				idx = cnt;
				cnt++;
			}
		}
	}

	if (m_Vertices.size() != new_verts.size()) {
		m_Vertices = std::vector<point3d<FloatType>>(new_verts.begin(), new_verts.end());

		if (hasPerVertexColors())		m_Colors = std::vector<point4d<FloatType>>(new_color.begin(), new_color.end());
		if (hasPerVertexNormals())		m_Normals = std::vector<point3d<FloatType>>(new_normals.begin(), new_normals.end());
		if (hasPerVertexTexCoords())	m_TextureCoords = std::vector<point2d<FloatType>>(new_tex.begin(), new_tex.end());
	}

	return (unsigned int)m_Vertices.size();
}


template <class FloatType>
unsigned int MeshData<FloatType>::removeVerticesBehindPlane( const Plane<FloatType>& plane, FloatType thresh )
{
	unsigned int numV = (unsigned int)m_Vertices.size();
	unsigned int numF = (unsigned int)m_FaceIndicesVertices.size();

	std::vector<unsigned int> vertexLookUp;	vertexLookUp.resize(numV);
	std::vector<point3d<FloatType>> new_verts;	new_verts.reserve(numV);
	std::vector<std::vector<unsigned int>> new_faces;	new_faces.reserve(numF);
	std::vector<point4d<FloatType>> new_color;		if (hasPerVertexColors())		new_color.reserve(m_Colors.size());
	std::vector<point3d<FloatType>> new_normals;	if (hasPerVertexNormals())		new_normals.reserve(m_Normals.size());
	std::vector<point2d<FloatType>> new_tex;		if (hasPerVertexTexCoords())	new_tex.reserve(m_TextureCoords.size());

	std::unordered_map<unsigned int, unsigned int> _map(m_Vertices.size());
	unsigned int cnt = 0;
	for (auto& face : m_FaceIndicesVertices) {
		bool keepFace = true;
		for (auto& idx : face) {
			if (plane.distanceToPoint(m_Vertices[idx]) < thresh) {
				keepFace = false;
				break;
			}
		}
		if (keepFace) {
			for (auto& idx : face) {

				if (_map.find(idx) != _map.end()) {
					idx = _map[idx];	//set to new idx, which already exists
				} else {
					_map[idx] = cnt;
					new_verts.push_back(m_Vertices[idx]);
					if (hasPerVertexColors())		new_color.push_back(m_Colors[idx]);
					if (hasPerVertexNormals())		new_normals.push_back(m_Normals[idx]);
					if (hasPerVertexTexCoords())	new_tex.push_back(m_TextureCoords[idx]);

					idx = cnt;
					cnt++;
				}
			}
			new_faces.push_back(face);
		}
	}

	if (m_Vertices.size() != new_verts.size()) {
		m_Vertices = std::vector<point3d<FloatType>>(new_verts.begin(), new_verts.end());

		if (hasPerVertexColors())		m_Colors = std::vector<point4d<FloatType>>(new_color.begin(), new_color.end());
		if (hasPerVertexNormals())		m_Normals = std::vector<point3d<FloatType>>(new_normals.begin(), new_normals.end());
		if (hasPerVertexTexCoords())	m_TextureCoords = std::vector<point2d<FloatType>>(new_tex.begin(), new_tex.end());
	}
	if (m_FaceIndicesVertices.size() != new_faces.size()) {
		m_FaceIndicesVertices = std::vector<std::vector<unsigned int>>(new_faces.begin(), new_faces.end());
	}

	return (unsigned int)m_Vertices.size();
}


template <class FloatType>
void MeshData<FloatType>::merge( const MeshData<FloatType>& other )
{
    if (other.isEmpty()) {
        return;
    }
	if (isEmpty()) {
		*this = other;
		return;
	}

	////TODO just delete if non existent in other mesh
	//assert(
	//	hasNormals() == other.hasNormals() &&
	//	hasColors() == other.hasColors() &&
	//	hasTexCoords() == other.hasTexCoords() &&
	//	hasPerVertexNormals() == other.hasPerVertexNormals() &&
	//	hasPerVertexTexCoords() == other.hasPerVertexTexCoords() &&
	//	hasPerVertexColors() == other.hasPerVertexColors() &&
	//	hasVertexIndices() == other.hasVertexIndices() &&
	//	hasColorIndices() == other.hasColorIndices() &&
	//	hasTexCoordsIndices() == other.hasTexCoordsIndices()
	//);

	if (hasVertexIndices() != other.hasVertexIndices()) throw MLIB_EXCEPTION("invalid mesh conversion");

	if (hasNormals() != other.hasNormals() || hasNormalIndices() != other.hasNormalIndices()) {
		MLIB_WARNING("normals deleted");
		m_Normals.clear();
		m_FaceIndicesNormals.clear();
	}
	if (hasColors() != other.hasColors() || hasColorIndices() != other.hasColorIndices()) {
		MLIB_WARNING("colors deleted");
		m_Colors.clear();
		m_FaceIndicesColors.clear();
	}
	if (hasTexCoords() != other.hasTexCoords() || hasTexCoordsIndices() != other.hasTexCoordsIndices()) {
		MLIB_WARNING("texcoords deleted");
		m_TextureCoords.clear();
		m_FaceIndicesTextureCoords.clear();
	}

	size_t vertsBefore = m_Vertices.size();
	size_t normsBefore = m_Normals.size();
	size_t colorBefore = m_Colors.size();
	size_t texCoordsBefore = m_TextureCoords.size();
	m_Vertices.insert(m_Vertices.end(), other.m_Vertices.begin(), other.m_Vertices.end());
	if (hasColors())	m_Colors.insert(m_Colors.end(), other.m_Colors.begin(), other.m_Colors.end());
	if (hasNormals())	m_Normals.insert(m_Normals.end(), other.m_Normals.begin(), other.m_Normals.end());
	if (hasTexCoords())	m_TextureCoords.insert(m_TextureCoords.end(), other.m_TextureCoords.begin(), other.m_TextureCoords.end());

	if (hasVertexIndices()) {
		size_t indicesBefore = m_FaceIndicesVertices.size();
		m_FaceIndicesVertices.insert(m_FaceIndicesVertices.end(), other.m_FaceIndicesVertices.begin(), other.m_FaceIndicesVertices.end());
		for (size_t i = indicesBefore; i < m_FaceIndicesVertices.size(); i++) {
			for (auto& idx : m_FaceIndicesVertices[i]) idx += (unsigned int)vertsBefore;
		}
	}
	if (hasNormalIndices()) {
		size_t indicesBefore = m_FaceIndicesNormals.size();
		m_FaceIndicesNormals.insert(m_FaceIndicesNormals.end(), other.m_FaceIndicesNormals.begin(), other.m_FaceIndicesNormals.end());
		for (size_t i = indicesBefore; i < m_FaceIndicesNormals.size(); i++) {
			for (auto& idx : m_FaceIndicesNormals[i]) idx +=  (unsigned int)normsBefore;
		}
	}
	if (hasColorIndices()) {
		size_t indicesBefore = m_FaceIndicesColors.size();
		m_FaceIndicesColors.insert(m_FaceIndicesColors.end(), other.m_FaceIndicesColors.begin(), other.m_FaceIndicesColors.end());
		for (size_t i = indicesBefore; i < m_FaceIndicesColors.size(); i++) {
			for (auto& idx : m_FaceIndicesColors[i]) idx +=  (unsigned int)colorBefore;
		}
	}
	if (hasTexCoordsIndices()) {
		size_t indicesBefore = m_FaceIndicesTextureCoords.size();
		m_FaceIndicesTextureCoords.insert(m_FaceIndicesTextureCoords.end(), other.m_FaceIndicesTextureCoords.begin(), other.m_FaceIndicesTextureCoords.end());
		for (size_t i = indicesBefore; i < m_FaceIndicesTextureCoords.size(); i++) {
			for (auto& idx : m_FaceIndicesTextureCoords[i]) idx +=  (unsigned int)texCoordsBefore;
		}
	}
}


template <class FloatType>
void MeshData<FloatType>::subdivideFacesMidpoint()
{
	m_Vertices.reserve(m_Vertices.size() + m_FaceIndicesVertices.size());	//there will be 1 new vertex per face
	if (hasPerVertexColors())		m_Colors.reserve(m_Colors.size() + m_FaceIndicesVertices.size());
	if (hasPerVertexNormals())		m_Normals.reserve(m_Normals.size() + m_FaceIndicesVertices.size());
	if (hasPerVertexTexCoords())	m_TextureCoords.reserve(m_TextureCoords.size() + m_FaceIndicesVertices.size());

	std::vector<std::vector<unsigned int>> newFaces;
	for (auto& face : m_FaceIndicesVertices) {
		point3d<FloatType> centerP = point3d<FloatType>(0,0,0);
		for (auto& idx : face) {
			centerP += m_Vertices[idx];
		}
		centerP /= (FloatType)face.size();
		m_Vertices.push_back(centerP);

		if (hasPerVertexColors()) {
			point4d<FloatType> centerC = point4d<FloatType>(0,0,0,0);
			for (auto& idx : face) {
				centerC += m_Colors[idx];
			}
			centerC /= (FloatType)face.size();
			m_Colors.push_back(centerC);
		}
		if (hasPerVertexNormals()) {
			point3d<FloatType> centerN = point3d<FloatType>(0,0,0);
			for (auto& idx : face) {
				centerN += m_Normals[idx];
			}
			centerN /= (FloatType)face.size();
			m_Normals.push_back(centerN);
		}
		if (hasPerVertexTexCoords()) {
			point2d<FloatType> centerT = point2d<FloatType>(0,0);
			for (auto& idx : face) {
				centerT += m_TextureCoords[idx];
			}
			centerT /= (FloatType)face.size();
			m_TextureCoords.push_back(centerT);
		}


		unsigned int newIdx = (unsigned int)m_Vertices.size() - 1;
		for (size_t i = 0; i < face.size(); i++) {
			newFaces.push_back(std::vector<unsigned int>(3));
			newFaces[newFaces.size()-1][0] = face[i];
			newFaces[newFaces.size()-1][1] = face[(i+1)%face.size()];
			newFaces[newFaces.size()-1][2] = newIdx;
		}
	}

	m_FaceIndicesVertices = newFaces;
}


template <class FloatType>
FloatType MeshData<FloatType>::subdivideFacesLoop( float edgeThresh /*= 0.0f*/ )
{
	m_Vertices.reserve(m_Vertices.size() + m_FaceIndicesVertices.size());	//there will be 1 new vertex per face (TODO FIX)
	if (hasPerVertexColors())		m_Colors.reserve(m_Colors.size() + m_FaceIndicesVertices.size());
	if (hasPerVertexNormals())		m_Normals.reserve(m_Normals.size() + m_FaceIndicesVertices.size());
	if (hasPerVertexTexCoords())	m_TextureCoords.reserve(m_TextureCoords.size() + m_FaceIndicesVertices.size());


	struct Edge {
		Edge(unsigned int _v0, unsigned int _v1) {
			if (_v0 < _v1) {
				v0 = _v0;
				v1 = _v1;
			} else {
				v1 = _v0;
				v0 = _v1;
			}
		}
		bool operator==(const Edge& other) const {
			return v0==other.v0 && v1==other.v1;
		}

		bool needEdgeVertex(float thresh, const std::vector<point3d<FloatType>>& vertices) const {
			if (thresh == 0.0f) return true;
			else {
				return ((vertices[v0] - vertices[v1]).lengthSq() > thresh*thresh);
			}
		}

		float edgeLength(const std::vector<point3d<FloatType>>& vertices) const {
			return (vertices[v0] - vertices[v1]).length();
		} 

		unsigned int v0;
		unsigned int v1;
	};

	struct EdgeHash {
		size_t operator()(const Edge& e) const {
			//TODO larger prime number (64 bit) to match size_t
			const size_t p[] = {73856093, 19349669};
			return e.v0*p[0] ^ e.v1*p[1];
			//const size_t res = ((size_t)v.x * p0)^((size_t)v.y * p1)^((size_t)v.z * p2);
		}
	};

	FloatType maxEdgeLen = 0.0f;

	//maps edges to new vertex indices
	std::unordered_map<Edge, unsigned int, EdgeHash> edgeMap;
	for (const std::vector<unsigned int>& face : m_FaceIndicesVertices) {

		for (unsigned int i = 0; i < face.size(); i++) {
			Edge e(face[i], face[(i+1)%face.size()]);

			FloatType edgeLen = e.edgeLength(m_Vertices);
			if (edgeLen > maxEdgeLen) maxEdgeLen = edgeLen;

			if (e.needEdgeVertex(edgeThresh, m_Vertices)) {
				if (edgeMap.find(e) == edgeMap.end()) {
					m_Vertices.push_back((FloatType)0.5*(m_Vertices[e.v0] + m_Vertices[e.v1]));
					if (hasPerVertexColors()) m_Colors.push_back((FloatType)0.5*(m_Colors[e.v0] + m_Colors[e.v1]));
					if (hasPerVertexNormals()) m_Normals.push_back((FloatType)0.5*(m_Normals[e.v0] + m_Normals[e.v1]));
					if (hasPerVertexTexCoords()) m_TextureCoords.push_back((FloatType)0.5*(m_TextureCoords[e.v0] + m_TextureCoords[e.v1]));
					unsigned int idx = (unsigned int)m_Vertices.size() - 1;
					edgeMap[e] = idx;
				}
			}
		}

	}

	std::vector<std::vector<unsigned int>> newFaces;    newFaces.reserve(m_FaceIndicesVertices.size() * 4);
	for (const std::vector<unsigned int>& face : m_FaceIndicesVertices) {
		bool allEdgesExist = true;
		bool noneEdgesExist = true;
		for (unsigned int i = 0; i < face.size(); i++) {
			Edge e(face[i], face[(i+1)%face.size()]);
			if (edgeMap.find(e) == edgeMap.end())   {
				allEdgesExist = false;
			} else {
				noneEdgesExist = false;
			}
		}

		if (allEdgesExist) {
			std::vector<unsigned int> centerFace(face.size());
			for (unsigned int i = 0; i < face.size(); i++) {
				Edge ePrev(face[i], face[(i+1)%face.size()]);
				Edge eNext(face[(i+1)%face.size()], face[(i+2)%face.size()]);
				newFaces.push_back(std::vector<unsigned int>(3));
				newFaces.back()[0] = edgeMap[ePrev];
				newFaces.back()[1] = face[(i+1)%face.size()];
				newFaces.back()[2] = edgeMap[eNext];

				centerFace[i] = newFaces.back()[0];
			}
			newFaces.push_back(centerFace);

		} 
		else if (noneEdgesExist) {
			newFaces.push_back(face);
		}
		else {
			std::vector<unsigned int> cFace;
			for (unsigned int i = 0; i < face.size(); i++) {
				cFace.push_back(face[i]);             
				Edge e(face[i], face[(i+1)%face.size()]);
				if (edgeMap.find(e) != edgeMap.end())   cFace.push_back(edgeMap[e]);
			}

			//centroid based vertex insertion
			point3d<FloatType> centerP = point3d<FloatType>(0,0,0);
			for (auto& idx : face) {
				centerP += m_Vertices[idx];
			}
			centerP /= (FloatType)face.size();
			m_Vertices.push_back(centerP);

			if (hasPerVertexColors()) {
				point4d<FloatType> centerC = point4d<FloatType>(0,0,0,0);
				for (auto& idx : face) {
					centerC += m_Colors[idx];
				}
				centerC /= (FloatType)face.size();
				m_Colors.push_back(centerC);
			}
			if (hasPerVertexNormals()) {
				point3d<FloatType> centerN = point3d<FloatType>(0,0,0);
				for (auto& idx : face) {
					centerN += m_Normals[idx];
				}
				centerN /= (FloatType)face.size();
				m_Normals.push_back(centerN);
			}
			if (hasPerVertexTexCoords()) {
				point2d<FloatType> centerT = point2d<FloatType>(0,0);
				for (auto& idx : face) {
					centerT += m_TextureCoords[idx];
				}
				centerT /= (FloatType)face.size();
				m_TextureCoords.push_back(centerT);
			}


			unsigned int newIdx = (unsigned int)m_Vertices.size() - 1;
			for (size_t i = 0; i < cFace.size(); i++) {
				newFaces.push_back(std::vector<unsigned int>(3));
				newFaces[newFaces.size()-1][0] = cFace[i];
				newFaces[newFaces.size()-1][1] = cFace[(i+1)%cFace.size()];
				newFaces[newFaces.size()-1][2] = newIdx;
			}
		} 
	}

	m_FaceIndicesVertices = std::vector<std::vector<unsigned int>>(newFaces.begin(), newFaces.end());
	return maxEdgeLen;
}

}  // namespace ml
#endif
