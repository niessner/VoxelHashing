
#ifndef CORE_BASE_BINARY_GRID3D_H_
#define CORE_BASE_BINARY_GRID3D_H_

namespace ml {


	class BinaryGrid3 {
	public:
		BinaryGrid3() {
			m_depth = m_height = m_width = 0;
			m_data = nullptr;
		}
		BinaryGrid3(size_t width, size_t height, size_t depth) {
			m_data = nullptr;
			allocate(width,height,depth);
			clear();
		}

		BinaryGrid3(const vec3ui& dim) {
			m_data = nullptr;
			allocate(dim);
			clear();
		}

		BinaryGrid3(const BinaryGrid3& other) {
      m_data = nullptr;
      if (other.m_data != nullptr) {
        allocate(other.m_width, other.m_height, other.m_depth);
        memcpy(m_data, other.m_data, getNumUInts()*sizeof(unsigned int));
      } else {
        m_data = nullptr;
        m_width = other.m_width;
        m_height = other.m_height;
        m_depth = other.m_depth;
      }
		}

		BinaryGrid3(BinaryGrid3&& other) {
			m_height = other.m_height;
			m_width = other.m_width;
			m_depth = other.m_depth;

			m_data = other.m_data;

			other.m_width = 0;
			other.m_height = 0;
			other.m_depth = 0;

			other.m_data = nullptr;
		}

		~BinaryGrid3() {
			SAFE_DELETE_ARRAY(m_data)
		}

		inline void allocate(size_t width, size_t height, size_t depth) {
			SAFE_DELETE_ARRAY(m_data);
			m_width = width;
			m_height = height;
			m_depth = depth;

      size_t dataSize = getNumUInts();
			m_data = new unsigned int[dataSize];
		}

		inline void allocate(const vec3ul& dim) {
			allocate(dim.x, dim.y, dim.z);
		}

		inline BinaryGrid3& operator=(const BinaryGrid3& other) {
      if (this != &other) {
        if (other.m_data != nullptr) {
          allocate(other.m_width, other.m_height, other.m_depth);
          memcpy(m_data, other.m_data, getNumUInts()*sizeof(unsigned int));
        } else {
          SAFE_DELETE_ARRAY(m_data);
          m_data = nullptr;
          m_width = other.m_width;
          m_height = other.m_height;
          m_depth = other.m_depth;
        }
      }
      return *this;
		}

		inline BinaryGrid3& operator=(BinaryGrid3&& other) {
      if (this != &other) {
        SAFE_DELETE_ARRAY(m_data);

        m_width = other.m_width;
        m_height = other.m_height;
        m_depth = other.m_depth;
        m_data = other.m_data;

        other.m_width = 0;
        other.m_height = 0;
        other.m_depth = 0;
        other.m_data = nullptr;
      }
      return *this;
		}

		inline bool operator==(const BinaryGrid3& other) const {
			if (m_width != other.m_width ||
				m_height != other.m_height ||
				m_depth != other.m_depth)	return false;

			size_t numUInts = getNumUInts();
			for (size_t i = 0; i < numUInts; i++) {
				if (m_data[i] != other.m_data[i])	return false;
			}

			return true;
		}

		inline bool operator!=(const BinaryGrid3& other) const {
			return !(*this == other);
		}

		//! clears all voxels
		inline void clear() {
			size_t numUInts = getNumUInts();
			for (size_t i = 0; i < numUInts; i++) {
				m_data[i] = 0;
			}
		}

		inline bool isVoxelSet(size_t x, size_t y, size_t z) const {
			size_t linIdx = z*m_height*m_width + x*m_height + y;
			size_t baseIdx = linIdx / bitsPerUInt;
			size_t localIdx = linIdx % bitsPerUInt;
			return (m_data[baseIdx] & (1 << localIdx)) != 0;
		}

		inline bool isVoxelSet(const vec3ul& v) const {
			return isVoxelSet(v.x, v.y, v.z);
		}

		inline void setVoxel(size_t x, size_t y, size_t z) {
			size_t linIdx = z*m_height*m_width + x*m_height + y;
			size_t baseIdx = linIdx / bitsPerUInt;
			size_t localIdx = linIdx % bitsPerUInt;
			m_data[baseIdx] |= (1 << localIdx);
		}

		inline void setVoxel(const vec3ul& v) {
			setVoxel(v.x, v.y, v.z);
		}

		inline void clearVoxel(size_t x, size_t y, size_t z) {
			size_t linIdx = z*m_height*m_width + x*m_height + y;
			size_t baseIdx = linIdx / bitsPerUInt;
			size_t localIdx = linIdx % bitsPerUInt;
			m_data[baseIdx] &= ~(1 << localIdx);
		}

		inline void clearVoxel(const vec3ul& v) {
			clearVoxel(v.x, v.y, v.z);
		}

		inline void toggleVoxel(size_t x, size_t y, size_t z) {
			size_t linIdx = z*m_height*m_width + x*m_height + y;
			size_t baseIdx = linIdx / bitsPerUInt;
			size_t localIdx = linIdx % bitsPerUInt;
			m_data[baseIdx] ^= (1 << localIdx);
		}

		inline void toggleVoxel(const vec3ul& v) {
			toggleVoxel(v.x, v.y, v.z);
		}

		inline void toggleVoxelAndBehindRow(size_t x, size_t y, size_t z) {
			for (size_t i = x; i < m_width; i++) {
				toggleVoxel(i, y, z);
			}
		}
		inline void toggleVoxelAndBehindRow(const vec3ul& v) {
			toggleVoxelAndBehindRow(v.x, v.y, v.z);
		}

		inline void toggleVoxelAndBehindSlice(size_t x, size_t y, size_t z) {
			for (size_t i = z; i < m_depth; i++) {
				toggleVoxel(x, y, i);
			}
		}
		inline void toggleVoxelAndBehindSlice(const vec3ul& v) {
			toggleVoxelAndBehindSlice(v.x, v.y, v.z);
		}

		inline void print() const {
			for (size_t z = 0; z < m_depth; z++) {
				std::cout << "slice0" << std::endl;
				for (size_t y = 0; y < m_height; y++) {
					for (size_t x = 0; x < m_width; x++) {
						if (isVoxelSet(x,y,z)) {
							std::cout << "1";
						} else {
							std::cout << "0";
						}

					}
					std::cout << "\n";
				}
			}
		}

		inline size_t dimX() const {
			return m_width;
		}
		inline size_t dimY() const {
			return m_height;
		}
		inline size_t dimZ() const {
			return m_depth;
		}

		inline vec3ul getDimensions() const {
			return vec3ul(dimX(), dimY(), dimZ());
		}

		inline size_t getNumTotalEntries() const {
			return m_width*m_height*m_depth;
		}

		inline bool isValidCoordinate(size_t x, size_t y, size_t z) const
		{
			return (x < m_width && y < m_height && z < m_depth);
		}

		inline bool isValidCoordinate(const vec3ul& v) const
		{
			return isValidCoordinate(v.x, v.y, v.z);
		}

		inline size_t getNumOccupiedEntries() const {
			size_t numOccupiedEntries = 0;
			size_t numUInts = getNumUInts();
			for (size_t i = 0; i < numUInts; i++) {
				numOccupiedEntries += math::numberOfSetBits(m_data[i]);
			}
			return numOccupiedEntries;
		}

	private:
    // boost archive serialization functions
    friend class boost::serialization::access;
    template <class Archive>
    void save(Archive& ar, const unsigned int version) const {
      ar << m_width << m_height << m_depth << boost::serialization::make_array(m_data, getNumUInts());
    }
    template<class Archive>
    void load(Archive& ar, const unsigned int version) {
      ar >> m_width >> m_height >> m_depth;
      allocate(m_width, m_height, m_depth);
      ar >> boost::serialization::make_array(m_data, getNumUInts());
    }
    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
      boost::serialization::split_member(ar, *this, version);
    }

		inline size_t getNumUInts() const {
			size_t numEntries = getNumTotalEntries();
			return (numEntries + bitsPerUInt - 1) / bitsPerUInt;
		}

    static const unsigned int bitsPerUInt = sizeof(unsigned int)*8;
		size_t			m_width, m_height, m_depth;
		unsigned int*	m_data;
	};


}

#endif