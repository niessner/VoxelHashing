
#ifndef CORE_BASE_GRID3D_H_
#define CORE_BASE_GRID3D_H_

namespace ml
{

template <class T> class Grid3
{
public:
	Grid3();
	Grid3(size_t dimX, size_t dimY, size_t dimZ);
    Grid3(size_t dimX, size_t dimY, size_t dimZ, const T &clearValue);
	Grid3(const Grid3<T> &G);
	Grid3(Grid3<T> &&G);

	~Grid3();

	//
	// Memory
	//
	void deleteMemory();
	Grid3<T>& operator = (const Grid3<T> &G);
	Grid3<T>& operator = (Grid3<T> &&G);

  void allocate(size_t dimX, size_t dimY, size_t dimZ);
  void allocate(size_t dimX, size_t dimY, size_t dimZ, const T &clearValue);

	inline Grid3<T>& operator += (const Grid3<T> &right)
	{
    MLIB_ASSERT_STR(m_dimX == right.m_dimX && m_dimY == right.m_dimY && m_dimZ == right.m_dimZ, "grid dimensions must be equal");
		for (size_t i = 0; i < getNumTotalEntries(); i++) {
			m_data[i] += right.m_data[i];
		}
		return *this;
	}
	inline Grid3<T>& operator *= (T right)
	{
		for (size_t i = 0; i < getNumTotalEntries(); i++) {
			m_data[i] *= right.m_data[i];
		}
		return *this;
	}

	inline Grid3<T> operator * (T x)
	{
		Grid3<T> result(m_dimX, m_dimY, m_dimZ);
		for (size_t i = 0; i < getNumTotalEntries(); i++) {
			result.m_data =  m_data * x;
		}
		return result;
	}

	//
	// Accessors
	//
	inline T& operator() (size_t x, size_t y, size_t z)
	{
#if defined(MLIB_BOUNDS_CHECK) || defined(_DEBUG)
		MLIB_ASSERT_STR( (x < m_dimX) && (y < m_dimY) && (z < m_dimZ), "Out-of-bounds grid access");
#endif
		return m_data[z*m_dimY*m_dimX + x*m_dimY + y];
	}
	inline const T& operator() (size_t dimX, size_t dimY, size_t slice) const
	{
#if defined(MLIB_BOUNDS_CHECK) || defined(_DEBUG)
		MLIB_ASSERT_STR( (dimX < m_dimX) && (dimY < m_dimY) && (slice < m_dimZ), "Out-of-bounds grid access");
#endif
		return m_data[slice*m_dimY*m_dimX + dimX*m_dimY + dimY];
	}
	inline size_t dimX() const
	{
		return m_dimX;
	}
	inline size_t dimY() const
	{
		return m_dimY;
	}
	inline size_t dimZ() const 
	{
		return m_dimZ;
	}

	//inline vec3ul getDimensions() const {
	//	return vec3ul(dimX(), dimY(), dimZ());
	//}

	inline bool square() const
	{
		return (m_dimX == m_dimY && m_dimY == m_dimZ);
	}
	inline T* ptr()
	{
		return m_data;
	}
	inline const T* ptr() const
	{
		return m_data;
	}

	//
	// Query
	//
	inline bool isValidCoordinate(size_t x, size_t y, size_t z ) const
	{
		return (x < m_dimX && y < m_dimY && z < m_dimZ);
	}

	//
	// Modifiers
	//
	void clear(const T &clearValue);

    void fill(const std::function<T(size_t x, size_t y, size_t z)> &fillFunction)
    {
        for (UINT xIndex = 0; xIndex < m_dimX; xIndex++)
            for (UINT yIndex = 0; yIndex < m_dimY; yIndex++)
                for (UINT zIndex = 0; zIndex < m_dimZ; zIndex++)
                {
                    (*this)(xIndex, yIndex, zIndex) = fillFunction(xIndex, yIndex, zIndex);
                }
    }

	size_t getNumTotalEntries() const {
		return m_dimX * m_dimY * m_dimZ;
	}
protected:
	T *m_data;
	size_t m_dimX, m_dimY, m_dimZ;
};

template <class T> inline bool operator == (const Grid3<T> &a, const Grid3<T> &b)
{
	if(a.dimX() != b.dimX() || a.dimY() != b.dimY() || a.dimZ() != b.dimZ()) return false;
	const size_t totalEntries = a.getNumTotalEntries();
	for (size_t i = 0; i < totalEntries; i++) {
		if (a.ptr()[i] != b.ptr()[i])	return false;
	}
	return true;
}

template <class T> inline bool operator != (const Grid3<T> &a, const Grid3<T> &b)
{
	return !(a == b);
}

typedef Grid3<float> Grid3f;
typedef Grid3<double> Grid3d;

}  // namespace ml

#include "grid3.cpp"

#endif  // CORE_BASE_GRID3D_H_
