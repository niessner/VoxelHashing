#ifndef CORE_MESH_MESHDATA_H_
#define CORE_MESH_MESHDATA_H_

namespace ml {



//! raw mesh data could be also a point cloud
template <class FloatType>
class MeshData {
public:

	class Indices {
	public:

		class Face {
		public:
			Face() : indices(nullptr), valence(0) {}
			Face(unsigned int* i, unsigned int v) : indices(i) , valence(v) {}

			unsigned int& operator[](unsigned int i) {
				return indices[i];
			}
			const unsigned int& operator[](unsigned int i) const {
				return indices[i];
			}
			bool operator==(const Face& other) const {
				if (valence != other.valence) return false;
				for (unsigned int i = 0; i < valence; i++) {
					if (indices[i] != other.indices[i]) return false;
				}
				return true;
			}
			bool operator!=(const Face& other) const {
				return !(*this==other);
			}

			unsigned int size() const {
				return valence;
			}

			void setSize(unsigned int v) {
				valence = v;
			}

			void setPtr(unsigned int* i) {
				indices = i;
			}

			//iterator over all indices within a face
			template<bool is_const_iterator = true>
			class const_noconst_iterator : public std::iterator<std::random_access_iterator_tag, unsigned int> {
			public:
				typedef typename std::conditional<is_const_iterator, const Indices*, Indices*>::type IndicesPtr;
				typedef typename std::conditional<is_const_iterator, const Indices&, Indices&>::type IndicesRef;
				typedef typename std::conditional<is_const_iterator, const Face*, Face*>::type FacePtr;
				typedef typename std::conditional<is_const_iterator, const Face&, Face&>::type FaceRef;
				typedef typename std::conditional<is_const_iterator, const unsigned int*, unsigned int*>::type uintPtr;
				typedef typename std::conditional<is_const_iterator, const unsigned int&, unsigned int&>::type uintRef;

				const_noconst_iterator(FacePtr f, unsigned int c = 0) {
					face = f;
					curr = c;
				}
				const_noconst_iterator(const_noconst_iterator<false>& other) {
					face = other.getFace();
					curr = other.getCurr();
				}
				const_noconst_iterator& operator++() {
					curr++;
					return *this;
				}
				const_noconst_iterator operator++(int) {
					const_noconst_iterator tmp(*this);
					operator++();
					return tmp;
				}
				const_noconst_iterator& operator--() {
					curr--;
					return *this;
				}
				const_noconst_iterator operator--(int) {
					const_noconst_iterator tmp(*this);
					operator--();
					return tmp;
				}

				void     operator+=(const std::size_t& n)  {curr += (unsigned int)n;}
				void     operator+=(const const_noconst_iterator& other) {curr += other.curr;}
				const_noconst_iterator operator+ (const std::size_t& n)  {const_noconst_iterator tmp(*this); tmp += n; return tmp;}
				const_noconst_iterator operator+ (const const_noconst_iterator& other) {const_noconst_iterator tmp(*this); tmp += other; return tmp;}

				void        operator-=(const std::size_t& n)  {curr -= (unsigned int)n;}
				void        operator-=(const const_noconst_iterator& other) {curr -= other.curr;}
				const_noconst_iterator    operator- (const std::size_t& n)  {const_noconst_iterator tmp(*this); tmp -= n; return tmp;}
				std::size_t operator- (const const_noconst_iterator& other) {return curr - other.curr;}

				bool operator< (const const_noconst_iterator& other) {return (curr-other.curr)< 0;}
				bool operator<=(const const_noconst_iterator& other) {return (curr-other.curr)<=0;}
				bool operator> (const const_noconst_iterator& other) {return (curr-other.curr)> 0;}
				bool operator>=(const const_noconst_iterator& other) {return (curr-other.curr)>=0;}

				uintRef operator*() {
					return (*face)[curr];
				}
				const unsigned int& operator*() const {
					return (*face)[curr];
				}

				bool operator==(const const_noconst_iterator& other) const {
					assert(face == other.face);
					return curr == other.curr;
				}
				bool operator!=(const const_noconst_iterator& other) const {
					return !operator==(other);
				}

				unsigned int getCurr() const {
					return curr;
				}

				FacePtr getFace() {
					return face;
				}
				const Face* getFace() const {
					return face;
				}
			private:
				unsigned int curr;
				FacePtr face;
			};

			typedef const_noconst_iterator<false> iterator;
			typedef const_noconst_iterator<true> const_iterator;

			iterator begin() {
				return iterator(this);
			}
			iterator end() {
				return iterator(this, size());
			}
			const_iterator begin() const {
				return const_iterator(this);
			}
			const_iterator end() const {
				return const_iterator(this, size());
			}

			const unsigned int* getIndices() const {
				return indices;
			}
		private:
			unsigned int* indices;
			unsigned int valence;
		};


		Indices() {}
		Indices(const Indices& i) {
			m_Indices = i.m_Indices;
			m_Faces = i.m_Faces;
			recomputeFacePtr();
		}
		Indices(Indices&& i) {
			m_Indices = std::move(i.m_Indices);
			m_Faces = std::move(i.m_Faces);
			recomputeFacePtr();
		}
		void operator=(Indices&& i) {
			m_Indices = std::move(i.m_Indices);
			m_Faces = std::move(i.m_Faces);
			recomputeFacePtr();
		}

		void operator=(const Indices& i) {
			m_Indices = i.m_Indices;
			m_Faces = i.m_Faces;
			recomputeFacePtr();
		}

		void reserve(size_t numFaces, unsigned int avgFaceValence = 3) {
			m_Indices.reserve(numFaces*avgFaceValence);
			m_Faces.reserve(numFaces);
		}

		unsigned int size() const {
			return (unsigned int)m_Faces.size();
		}

		void clear() {
			m_Indices.clear();
			m_Faces.clear();
		}

		//! resizes the index vector with the same valence
		void resize(size_t numFaces, unsigned int faceValence = 3) {
			if (size() != 0)	throw MLIB_EXCEPTION("not supported yet");
			m_Indices.resize(numFaces*faceValence);
			m_Faces.reserve(numFaces);	//TODO use a lambda function to initialize
			for (size_t i = 0; i < numFaces; i++) {
				m_Faces.push_back(Face(&m_Indices[faceValence*i], faceValence));
			}

		}

		void push_back(const Face& f) {
			addFace(f.getIndices(), f.size());
		}
		void push_back(const std::vector<unsigned int>& indices) {
			addFace(indices);
		}

		void addFace(const std::vector<unsigned int>& indices) {
			addFace(&indices[0], (unsigned int)indices.size());
		}
		void addFace(const unsigned int* indices, unsigned int count) {
			if (count > 0) {
				unsigned int* ptrBefore = m_Indices.size() > 0 ? &m_Indices[0] : nullptr;
				m_Indices.resize(m_Indices.size() + count);
				unsigned int* ptr = &m_Indices[m_Indices.size() - count];
				memcpy(ptr, indices, count*sizeof(unsigned int));
				m_Faces.push_back(Face(ptr, count));
				if (ptrBefore != &m_Indices[0])	recomputeFacePtr();
			} else {
				//allow to insert empty faces
				m_Faces.push_back(Face(nullptr, 0));
			}
		}

		unsigned int getFaceValence(size_t i) const {
			return m_Faces[i].size();
		}

		Face& getFace(size_t i) {
			return m_Faces[i];
		}		
		Face& operator[](size_t i) {
			return getFace(i);
		}
		const Face& getFace(size_t i) const {
			return m_Faces[i];
		}		
		const Face& operator[](size_t i) const {
			return getFace(i);
		}


		//iterator over all faces
		template<bool is_const_iterator = true>
		class const_noconst_iterator : public std::iterator<std::random_access_iterator_tag, Face> {
		public:
			typedef typename std::conditional<is_const_iterator, const Indices*, Indices*>::type IndicesPtr;
			typedef typename std::conditional<is_const_iterator, const Indices&, Indices&>::type IndicesRef;
			typedef typename std::conditional<is_const_iterator, const Face*, Face*>::type FacePtr;
			typedef typename std::conditional<is_const_iterator, const Face&, Face&>::type FaceRef;

			const_noconst_iterator(IndicesPtr i, size_t c = 0) {
				curr = c;
				indices = i;
			}
			const_noconst_iterator(const_noconst_iterator<false>& other) {
				curr = other.getCurr();
				indices = other.getIndices();
			}
			const_noconst_iterator& operator++() {
				curr++;
				return *this;
			}
			const_noconst_iterator operator++(int) {
				const_noconst_iterator tmp(*this);
				operator++();
				return tmp;
			}
			const_noconst_iterator& operator--() {
				curr--;
				return *this;
			}
			const_noconst_iterator operator--(int) {
				const_noconst_iterator tmp(*this);
				operator--();
				return tmp;
			}

			void     operator+=(const std::size_t& n)  {curr += n;}
			void     operator+=(const const_noconst_iterator& other) {curr += other.curr;}
			const_noconst_iterator operator+ (const std::size_t& n)  {const_noconst_iterator tmp(*this); tmp += n; return tmp;}
			const_noconst_iterator operator+ (const const_noconst_iterator& other) {const_noconst_iterator tmp(*this); tmp += other; return tmp;}

			void        operator-=(const std::size_t& n)  {curr -= n;}
			void        operator-=(const const_noconst_iterator& other) {curr -= other.curr;}
			const_noconst_iterator    operator- (const std::size_t& n)  {const_noconst_iterator tmp(*this); tmp -= n; return tmp;}
			std::size_t operator- (const const_noconst_iterator& other) {return curr - other.curr;}

			bool operator< (const const_noconst_iterator& other) {return (curr-other.curr)< 0;}
			bool operator<=(const const_noconst_iterator& other) {return (curr-other.curr)<=0;}
			bool operator> (const const_noconst_iterator& other) {return (curr-other.curr)> 0;}
			bool operator>=(const const_noconst_iterator& other) {return (curr-other.curr)>=0;}

			Face operator*() {
				return (*indices)[curr];
			}

			Face* operator->() {
				return &(*indices)[curr];
			}
	

			bool operator==(const const_noconst_iterator& other) const {
				assert(indices == other.indices);
				return curr == other.curr;
			}
			bool operator!=(const const_noconst_iterator& other) const {
				return !operator==(other);
			}

			size_t getCurr() const {
				return curr;
			}
			IndicesPtr getIndices() {
				return indices;
			}

			const Indices* getIndices() const {
				return indices;
			}
		private:
			size_t curr;
			IndicesPtr indices;
		};

		typedef const_noconst_iterator<false> iterator;
		typedef const_noconst_iterator<true> const_iterator;

		iterator begin() {
			return iterator(this);
		}
		iterator end() {
			return iterator(this, size());
		}
		const_iterator begin() const {
			return const_iterator(this);
		}
		const_iterator end() const {
			return const_iterator(this, size());
		}

		Face& back() {
			return m_Faces.back();
		}
		Face& back() const {
			return m_Faces.back();
		}

		void append(const Indices& other) {
			m_Indices.insert(m_Indices.end(), other.m_Indices.begin(), other.m_Indices.end());
			m_Faces.insert(m_Faces.end(), other.m_Faces.begin(), other.m_Faces.end());
			recomputeFacePtr();

		}


		bool operator==(const Indices& other) const {
			if (size() != other.size())	return false;
			if (m_Indices.size() != other.m_Indices.size()) return false;
			for (size_t i = 0; i < m_Indices.size(); i++) {
				if (m_Indices[i] != other.m_Indices[i])	return false;
			}
			return true;
		}

		bool operator!=(const Indices& other) const {
			return operator!=(other);
		}

		void operator=(const std::vector<std::vector<unsigned int>>& other) {
			MLIB_WARNING("inefficient function; should not actually been sued");
			clear();
			for (auto& f : other) {
				addFace(&f[0], (unsigned int)f.size());
			} 
		}
	private:
		void recomputeFacePtr()
		{
			size_t offset = 0;
			for (size_t i = 0; i < m_Faces.size(); i++) {
				if (m_Faces[i].size()) {
					//ignore empty faces
					m_Faces[i].setPtr(&m_Indices[offset]);
					offset += m_Faces[i].size();
				}
			}
		}

		std::vector<unsigned int>	m_Indices;
		std::vector<Face>			m_Faces;

	};



	MeshData() {}
	MeshData(MeshData&& d) {
		m_Vertices = std::move(d.m_Vertices);
		m_Normals = std::move(d.m_Normals);
		m_TextureCoords = std::move(d.m_TextureCoords);
		m_Colors = std::move(d.m_Colors);
		m_FaceIndicesVertices = std::move(d.m_FaceIndicesVertices);
		m_FaceIndicesNormals = std::move(d.m_FaceIndicesNormals);
		m_FaceIndicesTextureCoords = std::move(d.m_FaceIndicesTextureCoords);
		m_FaceIndicesColors = std::move(d.m_FaceIndicesColors);
		m_MaterialFile = std::move(d.m_MaterialFile);
		m_MaterialIndices = std::move(d.m_MaterialIndices);
	}
	void operator=(MeshData&& d) {
		m_Vertices = std::move(d.m_Vertices);
		m_Normals = std::move(d.m_Normals);
		m_TextureCoords = std::move(d.m_TextureCoords);
		m_Colors = std::move(d.m_Colors);
		m_FaceIndicesVertices = std::move(d.m_FaceIndicesVertices);
		m_FaceIndicesNormals = std::move(d.m_FaceIndicesNormals);
		m_FaceIndicesTextureCoords = std::move(d.m_FaceIndicesTextureCoords);
		m_FaceIndicesColors = std::move(d.m_FaceIndicesColors);
		m_MaterialFile = std::move(d.m_MaterialFile);
		m_MaterialIndices = std::move(d.m_MaterialIndices);
	}
	void clear() {
		m_Vertices.clear();
		m_Normals.clear();
		m_Colors.clear();
		m_FaceIndicesVertices.clear();
		m_FaceIndicesNormals.clear();
		m_FaceIndicesTextureCoords.clear();
		m_FaceIndicesColors.clear();
		m_MaterialFile.clear();
		m_MaterialIndices.clear();
	}

	bool isConsistent(bool detailedCheck = false) const {

		bool consistent = true;
		if (m_FaceIndicesNormals.size() > 0			&& m_FaceIndicesVertices.size() != m_FaceIndicesNormals.size())			consistent = false;
		if (m_FaceIndicesTextureCoords.size() > 0	&& m_FaceIndicesVertices.size() != m_FaceIndicesTextureCoords.size())	consistent = false;
		if (m_FaceIndicesColors.size() > 0			&& m_FaceIndicesColors.size() != m_FaceIndicesColors.size())			consistent = false;

		if (hasPerVertexNormals() && m_Vertices.size() != m_Normals.size())			consistent = false;
		if (hasPerVertexTexCoords() && m_Vertices.size() != m_TextureCoords.size()) consistent = false;
		if (hasPerVertexColors() && m_Vertices.size() != m_Colors.size())			consistent = false;

		if (detailedCheck) {
			//make sure no index is out of bounds
			for (auto& face : m_FaceIndicesVertices) {
				for (auto idx : face) {
					if (idx >= m_Vertices.size())	consistent = false;
				}
			}
			for (auto& face : m_FaceIndicesColors) {
				for (auto idx : face) {
					if (idx >= m_Colors.size())	consistent = false;
				}
			}
			for (auto& face : m_FaceIndicesNormals) {
				for (auto idx : face) {
					if (idx >= m_FaceIndicesNormals.size())	consistent = false;
				}
			}
		}
		return consistent;
	}

	void applyTransform(const Matrix4x4<FloatType>& t) {
		for (size_t i = 0; i < m_Vertices.size(); i++) {
			m_Vertices[i] = t*m_Vertices[i];
		}
		Matrix4x4<FloatType> invTrans = t.getInverse().getTranspose();
		for (size_t i = 0; i < m_Normals.size(); i++) {
			m_Normals[i] = invTrans*m_Normals[i];
		}
	}

	BoundingBox3<FloatType> getBoundingBox() const {
		BoundingBox3<FloatType> bb;
		for (size_t i = 0; i < m_Vertices.size(); i++) {
			bb.include(m_Vertices[i]);
		}
		return bb;
	}

	const Indices& getFaceIndicesVertices() const {
		return m_FaceIndicesVertices;
	}
	const Indices& getFaceIndicesNormals() const {
		if (!hasNormals())	throw MLIB_EXCEPTION("mesh does not have normals");
		else if (m_FaceIndicesNormals.size() > 0)			return m_FaceIndicesNormals;
		else if (m_Vertices.size() == m_Normals.size())		return m_FaceIndicesVertices;
		else throw MLIB_EXCEPTION("vertex/normal mismatch");
	}
	const Indices& getFaceIndicesTexCoords() const {
		if (!hasTexCoords())	throw MLIB_EXCEPTION("mesh does not have texcoords");
		else if (m_FaceIndicesTextureCoords.size() > 0)			return m_FaceIndicesTextureCoords;
		else if (m_Vertices.size() == m_TextureCoords.size())	return m_FaceIndicesVertices;
		else throw MLIB_EXCEPTION("vertex/texcoord mismatch");
	}
	const Indices& getFaceIndicesColors() const {
		if (!hasColors())	throw MLIB_EXCEPTION("mesh does not have colors");
		else if (m_FaceIndicesColors.size() > 0)		return m_FaceIndicesColors;
		else if (m_Vertices.size() == m_Colors.size())	return m_FaceIndicesVertices;
		else throw MLIB_EXCEPTION("vertex/color mismatch");
	}

	bool hasNormals() const { return m_Normals.size() > 0; }
	bool hasColors() const { return m_Colors.size() > 0; }
	bool hasTexCoords() const { return m_TextureCoords.size() > 0; }

	bool hasPerVertexNormals()	const	{ return hasNormals() && m_FaceIndicesNormals.size() == 0; }
	bool hasPerVertexColors()	const	{ return hasColors() && m_FaceIndicesColors.size() == 0; }
	bool hasPerVertexTexCoords() const	{ return hasTexCoords() && m_FaceIndicesTextureCoords.size() == 0; }

	bool hasVertexIndices() const { return m_FaceIndicesVertices.size() > 0; }
	bool hasColorIndices() const { return m_FaceIndicesColors.size() > 0; }
	bool hasNormalIndices() const { return m_FaceIndicesNormals.size() > 0; }
	bool hasTexCoordsIndices() const { return m_FaceIndicesTextureCoords.size() > 0; }

	//! todo check this
	bool isEmpty() const {
		return m_Vertices.size() == 0;
	}

	//! merges two meshes (assumes the same memory layout/type)
	void merge(const MeshData<FloatType>& other);
	unsigned int removeDuplicateVertices();
	unsigned int removeDuplicateFaces();
	unsigned int mergeCloseVertices(FloatType thresh, bool approx = false);
	unsigned int removeDegeneratedFaces();

	//! also removes isolated normals, colors, etc.
	unsigned int removeIsolatedVertices();

	//! removes all the vertices that are behind a plane (faces with one or more of those vertices are being deleted as well)
	//! larger thresh removes less / negative thresh removes more
	unsigned int removeVerticesInFrontOfPlane(const Plane<FloatType>& plane, FloatType thresh = 0.0f);

	//! removes all the faces that are fully behind a plane
	//! larger thresh removes less / negative thresh removes more
	unsigned int removeFacesInFrontOfPlane(const Plane<FloatType>& plane, FloatType thresh = 0.0f);

	std::vector<point3d<FloatType>>	m_Vertices;			//vertices are indexed (see below)
	std::vector<point3d<FloatType>>	m_Normals;			//normals are indexed (see below/or per vertex)
	std::vector<point2d<FloatType>>	m_TextureCoords;	//tex coords are indexed (see below/or per vertex)
	std::vector<point4d<FloatType>>	m_Colors;			//colors are not indexed (see below/or per vertex) 
	//std::vector<std::vector<unsigned int>>	m_FaceIndicesVertices;		//indices in face array
	//std::vector<std::vector<unsigned int>>	m_FaceIndicesNormals;		//indices in normal array (if size==0, indicesVertices is used)
	//std::vector<std::vector<unsigned int>>	m_FaceIndicesTextureCoords;	//indices in texture array (if size==0, indicesVertices is used)
	//std::vector<std::vector<unsigned int>>	m_FaceIndicesColors;		//indices in color array (if size==0, indicesVertices is used)
	Indices	m_FaceIndicesVertices;		//indices in face array
	Indices	m_FaceIndicesNormals;		//indices in normal array (if size==0, indicesVertices is used)
	Indices	m_FaceIndicesTextureCoords;	//indices in texture array (if size==0, indicesVertices is used)
	Indices	m_FaceIndicesColors;		//indices in color array (if size==0, indicesVertices is used)

	std::string		m_MaterialFile;	// in case of objs, refers to the filename
	struct MaterialIndex {
		MaterialIndex() {}
		MaterialIndex(unsigned int s, unsigned int e, const std::string& name) : start(s), end(e), materialName(s) {}
		unsigned int start;
		unsigned int end;	//end index NOT included (similar to iterators)
		std::string materialName;
	};
	std::vector<MaterialIndex>	m_MaterialIndices;	//active material for indices; from - to (in case of objcs)


	std::vector< std::pair <MeshData, Material<FloatType> > > splitByMaterial(const std::string& overwriteMTLPath = "") const {
		std::vector<std::pair<MeshData, Material<FloatType> > > res;
		splitByMaterial(res, overwriteMTLPath);
		return res;
	}

	void splitByMaterial(std::vector<std::pair<MeshData, Material<FloatType> > >& res, const std::string& overwriteMTLPath = "") const {
		res.clear();
		std::vector<Material<FloatType>> mats;
		if (overwriteMTLPath != "")
			Material<FloatType>::loadFromMTL(overwriteMTLPath, mats);
		else 
			Material<FloatType>::loadFromMTL(m_MaterialFile, mats);
		
		res.resize(m_MaterialIndices.size());

		for (size_t i = 0; i < m_MaterialIndices.size(); i++) {
			Material<FloatType>* m = nullptr;
			for (size_t j = 0; j < mats.size(); j++) {
				if (m_MaterialIndices[i].materialName == mats[j].m_name) {
					m = &mats[j];
					break;
				}
			}
			if (!m) throw MLIB_EXCEPTION("material not found");

			res[i].second = *m;
			MeshData& meshData = res[i].first;
			meshData.clear();

			//vertices
			{
				std::unordered_map<unsigned int, unsigned int> _map(m_Vertices.size());
				unsigned int cnt = 0;
				for (size_t j = m_MaterialIndices[i].start; j < m_MaterialIndices[i].end; j++) {
					meshData.m_FaceIndicesVertices.push_back(getFaceIndicesVertices()[j]);
					Indices::Face& face = meshData.m_FaceIndicesVertices.back();				
					for (auto& idx : face) {
						if (_map.find(idx) != _map.end()) {
							idx = _map[idx];	//set to new idx, which already exists
						} else {
							_map[idx] = cnt;
							meshData.m_Vertices.push_back(m_Vertices[idx]);
							if (hasPerVertexColors())		meshData.m_Colors.push_back(m_Colors[idx]);
							if (hasPerVertexNormals())		meshData.m_Normals.push_back(m_Normals[idx]);
							if (hasPerVertexTexCoords())	meshData.m_TextureCoords.push_back(m_TextureCoords[idx]);
							//TODO HANDLE COLOR NORMAL TEX COORD
							idx = cnt;
							cnt++;
						}
					}
				}				
			}
			if (hasColorIndices()) {
				std::unordered_map<unsigned int, unsigned int> _map(m_Colors.size());
				unsigned int cnt = 0;
				for (size_t j = m_MaterialIndices[i].start; j < m_MaterialIndices[i].end; j++) {
					meshData.m_FaceIndicesColors.push_back(getFaceIndicesColors()[j]);
					Indices::Face& face = meshData.m_FaceIndicesColors.back();				
					for (auto& idx : face) {
						if (_map.find(idx) != _map.end()) {
							idx = _map[idx];	//set to new idx, which already exists
						} else {
							_map[idx] = cnt;
							meshData.m_Colors.push_back(m_Colors[idx]);
							idx = cnt;
							cnt++;
						}
					}
				}	
			}
			if (hasNormalIndices()) {
				std::unordered_map<unsigned int, unsigned int> _map(m_Normals.size());
				unsigned int cnt = 0;
				for (size_t j = m_MaterialIndices[i].start; j < m_MaterialIndices[i].end; j++) {
					meshData.m_FaceIndicesNormals.push_back(getFaceIndicesNormals()[j]);
					Indices::Face& face = meshData.m_FaceIndicesNormals.back();				
					for (auto& idx : face) {
						if (_map.find(idx) != _map.end()) {
							idx = _map[idx];	//set to new idx, which already exists
						} else {
							_map[idx] = cnt;
							meshData.m_Normals.push_back(m_Normals[idx]);
							idx = cnt;
							cnt++;
						}
					}
				}	
			}
			if (hasTexCoordsIndices()) {
				std::unordered_map<unsigned int, unsigned int> _map(m_TextureCoords.size());
				unsigned int cnt = 0;
				for (size_t j = m_MaterialIndices[i].start; j < m_MaterialIndices[i].end; j++) {
					meshData.m_FaceIndicesTextureCoords.push_back(getFaceIndicesTexCoords()[j]);
					Indices::Face& face = meshData.m_FaceIndicesTextureCoords.back();				
					for (auto& idx : face) {
						if (_map.find(idx) != _map.end()) {
							idx = _map[idx];	//set to new idx, which already exists
						} else {
							_map[idx] = cnt;
							meshData.m_TextureCoords.push_back(m_TextureCoords[idx]);
							idx = cnt;
							cnt++;
						}
					}
				}	
			}

			meshData.deleteRedundantIndices();


		}
	}
	
	//! Debug print with all details
	void print() const {
		std::cout << "Vertices:\n";
		std::cout << m_Vertices << std::endl;
		std::cout << "Faces:\n";
		std::cout << m_FaceIndicesVertices << std::endl;
	}


	//! computes per vertex normals
	void computeVertexNormals() {
		MLIB_WARNING("untested");
		if (m_FaceIndicesVertices.size() == 0) throw MLIB_EXCEPTION("must be an indexed face set");
		m_Normals.clear();
		m_FaceIndicesNormals.clear();

		m_Normals.resize(m_Vertices.size(), point3d<FloatType>(0,0,0));
		for (const auto& face : m_FaceIndicesVertices) {

			point3d<FloatType> n = point3d<FloatType>(0,0,0);
			unsigned int first = face[0];
			for (unsigned int i = 1; i < face.size() - 1; i++) {
				n += (m_Vertices[face[i]] - m_Vertices[first]) ^ (m_Vertices[face[i+1]] - m_Vertices[first]);
			}
			n.getNormalized();

			for (auto idx : face) {
				m_Normals[idx] += n;
			}
		}
		for (auto& n : m_Normals) {
			n.normalize();
		}
	}
	//! inserts a midpoint into every faces; and triangulates the result
	void subdivideFacesMidpoint();

    //! inserts a midpoint into every faces; and triangulates the result
    FloatType subdivideFacesLoop(float edgeThresh = 0.0f);


	void deleteRedundantIndices() {
		if (m_FaceIndicesVertices == m_FaceIndicesNormals)			m_FaceIndicesNormals.clear();
		if (m_FaceIndicesVertices == m_FaceIndicesColors)			m_FaceIndicesColors.clear();
		if (m_FaceIndicesVertices == m_FaceIndicesTextureCoords)	m_FaceIndicesTextureCoords.clear();
	}
private:
	inline vec3i toVirtualVoxelPos(const point3d<FloatType>& v, FloatType voxelSize) {
		return vec3i(v/voxelSize+(FloatType)0.5*point3d<FloatType>(math::sign(v)));
	} 
	//! returns -1 if there is no vertex closer to 'v' than thresh; otherwise the vertex id of the closer vertex is returned
	unsigned int hasNearestNeighbor(const vec3i& coord, SparseGrid3<std::list<std::pair<point3d<FloatType>,unsigned int> > > &neighborQuery, const point3d<FloatType>& v, FloatType thresh );

	//! returns -1 if there is no vertex closer to 'v' than thresh; otherwise the vertex id of the closer vertex is returned (manhattan distance)
	unsigned int hasNearestNeighborApprox(const vec3i& coord, SparseGrid3<unsigned int> &neighborQuery, FloatType thresh );
};

typedef MeshData<float>		MeshDataf;
typedef MeshData<double>	MeshDatad;


} // namespace ml

#include "meshData.cpp"

#endif  // CORE_MESH_MESHDATA_H_

