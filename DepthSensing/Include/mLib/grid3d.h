
#ifndef CORE_BASE_GRID3D_H_
#define CORE_BASE_GRID3D_H_

namespace ml
{

template <class T> class Grid3D
{
public:
	Grid3D();
	Grid3D(UINT rows, UINT cols, UINT slices);
	Grid3D(UINT rows, UINT cols, UINT slices, const T &clearValue);
	Grid3D(const Grid3D<T> &G);
	Grid3D(Grid3D<T> &&G);

	~Grid3D();

	//
	// Memory
	//
	void deleteMemory();
	Grid3D<T>& operator = (const Grid3D<T> &G);
	Grid3D<T>& operator = (Grid3D<T> &&G);

	void allocate(UINT rows, UINT cols, UINT slices);
	void allocate(UINT rows, UINT cols, UINT slices, const T &clearValue);

	inline Grid3D<T>& operator += (const Grid3D<T> &right)
	{
		MLIB_ASSERT_STR(m_rows == right.m_rows && m_cols == right.m_cols, "grid dimensions must be equal");
		for (UINT i = 0; i < getNumTotalEntries(); i++) {
			m_data[i] += right.m_data[i];
		}
		return *this;
	}
	inline Grid3D<T>& operator *= (T right)
	{
		for (UINT i = 0; i < getNumTotalEntries(); i++) {
			m_data[i] *= right.m_data[i];
		}
		return *this;
	}

	inline Grid3D<T> operator * (T x)
	{
		Grid3D<T> result(m_rows, m_cols, m_slices);
		for (UINT i = 0; i < getNumTotalEntries(); i++) {
			result.m_data =  m_data * x;
		}
		return result;
	}

	//
	// Accessors
	//
	inline T& operator() (UINT row, UINT col, UINT slice)
	{
#if defined(MLIB_BOUNDS_CHECK) || defined(_DEBUG)
		MLIB_ASSERT_STR( (row < m_rows) && (col < m_cols) && (slice < m_slices), "Out-of-bounds grid access");
#endif
		return m_data[slice*m_cols*m_rows + row*m_cols + col];
	}
	inline const T& operator() (UINT row, UINT col, UINT slice) const
	{
#if defined(MLIB_BOUNDS_CHECK) || defined(_DEBUG)
		MLIB_ASSERT_STR( (row < m_rows) && (col < m_cols) && (slice < m_slices), "Out-of-bounds grid access");
#endif
		return m_data[slice*m_cols*m_rows + row*m_cols + col];
	}
	inline UINT rows() const
	{
		return m_rows;
	}
	inline UINT cols() const
	{
		return m_cols;
	}
	inline UINT slices() const 
	{
		return m_slices;
	}
	inline bool square() const
	{
		return (m_rows == m_cols && m_cols == m_slices);
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
	inline bool validCoordinates(UINT row, UINT col, UINT slices ) const
	{
		return (row < m_rows && col < m_cols && slices < m_slices);
	}

	//
	// Modifiers
	//
	void clear(const T &clearValue);

	UINT getNumTotalEntries() const {
		return m_rows * m_cols * m_slices;
	}
protected:
	T *m_data;
	UINT m_rows, m_cols, m_slices;
};

template <class T> inline bool operator == (const Grid3D<T> &a, const Grid3D<T> &b)
{
	if(a.rows() != b.rows() || a.cols() != b.cols() || a.slices() != b.slices()) return false;
	const UINT totalEntries = a.getNumTotalEntries();
	for (UINT i = 0; i < totalEntries; i++) {
		if (a.ptr()[i] != b.ptr()[i])	return false;
	}
	return true;
}

template <class T> inline bool operator != (const Grid3D<T> &a, const Grid3D<T> &b)
{
	return !(a == b);
}

}  // namespace ml

#include "grid3d.cpp"

#endif  // CORE_BASE_GRID3D_H_
