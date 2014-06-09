
#ifndef CORE_BASE_GRID2D_H_
#define CORE_BASE_GRID2D_H_

namespace ml
{

template <class T> class Grid2D
{
public:
	Grid2D();
	Grid2D(UINT rows, UINT cols);
	Grid2D(UINT rows, UINT cols, const T &clearValue);
	Grid2D(const Grid2D<T> &G);
	Grid2D(Grid2D<T> &&G);

	~Grid2D();

	//
	// Memory
	//
	void deleteMemory();
	Grid2D<T>& operator = (const Grid2D<T> &G);
	Grid2D<T>& operator = (Grid2D<T> &&G);

	void allocate(UINT rows, UINT cols);
	void allocate(UINT rows, UINT cols, const T &clearValue);

	inline Grid2D<T>& operator += (const Grid2D<T> &right)
	{
		MLIB_ASSERT_STR(m_rows == right.m_rows && m_cols == right.m_cols, "grid dimensions must be equal");
		for (UINT r = 0; r < m_rows; r++)
			for (UINT c = 0; c < m_cols; c++)
				m_data[r * m_cols + c] += right(r,c);
		return *this;
	}
	inline Grid2D<T>& operator *= (T right)
	{
		for (UINT r = 0; r < m_rows; r++)
			for (UINT c = 0; c < m_cols; c++)
				m_data[r * m_cols + c] *= right;
		return *this;
	}

	inline Grid2D<T> operator * (T x)
	{
		Grid2D<T> result(m_rows, m_cols);
		for (UINT r = 0; r < m_rows; r++)
			for (UINT c = 0; c < m_cols; c++)
				result(r,c) = m_data[r * m_cols + c] * x;
		return result;
	}

	//
	// Accessors
	//
	inline T& operator() (UINT row, UINT col)
	{
#if defined(MLIB_BOUNDS_CHECK) || defined(_DEBUG)
		MLIB_ASSERT_STR( (row < m_rows) && (col < m_cols), "Out-of-bounds grid access");
#endif
		return m_data[row * m_cols + col];
	}
	inline const T& operator() (UINT row, UINT col) const
	{
#if defined(MLIB_BOUNDS_CHECK) || defined(_DEBUG)
		MLIB_ASSERT_STR( (row < m_rows) && (col < m_cols), "Out-of-bounds grid access");
#endif
		return m_data[row * m_cols + col];
	}
	inline UINT rows() const
	{
		return m_rows;
	}
	inline UINT cols() const
	{
		return m_cols;
	}
	inline std::pair<UINT, UINT> dimensions() const
	{
		return std::make_pair(m_rows, m_cols);
	}
	inline bool square() const
	{
		return (m_rows == m_cols);
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
	inline bool validCoordinates(int row, int col) const
	{
		return (row >= 0 && row < int(m_rows) && col >= 0 && col < int(m_cols));
	}

	void setRow(UINT row, const std::vector<T> &values)
	{
		for(UINT col = 0; col < m_cols; col++) m_data[row * m_cols + col] = values[col];
	}

	void setCol(UINT col, const std::vector<T> &values)
	{
		for(UINT row = 0; row < m_rows; row++) m_data[row * m_cols + col] = values[row];
	}

	std::vector<T> getRow(UINT row) const
	{
		std::vector<T> result(m_cols);
		const T *CPtr = m_data;
		for(UINT col = 0; col < m_cols; col++)
		{
			result[col] = CPtr[row * m_cols + col];
		}
		return result;
	}

	std::vector<T> getCol(UINT col) const
	{
		std::vector<T> result(m_rows);
		const T *CPtr = m_data;
		for(UINT row = 0; row < m_rows; row++)
		{
			result[col] = CPtr[row * m_cols + col];
		}
		return result;
	}

	std::pair<UINT, UINT> maxIndex() const;
	const T& maxValue() const;
	std::pair<UINT, UINT> minIndex() const;
	const T& minValue() const;

	//
	// Modifiers
	//
	void clear(const T &clearValue);

protected:
	T *m_data;
	UINT m_rows, m_cols;
};

template <class T> inline bool operator == (const Grid2D<T> &a, const Grid2D<T> &b)
{
	if(a.rows() != b.rows() || a.cols() != b.cols()) return false;
	for(UINT row = 0; row < a.rows(); row++)
		for(UINT col = 0; col < a.cols(); col++)
			if(a(row, col) != b(row, col))
				return false;
	return true;
}

template <class T> inline bool operator != (const Grid2D<T> &a, const Grid2D<T> &b)
{
	return !(a == b);
}

}

#include "grid2d.cpp"

#endif  // CORE_BASE_GRID2D_H_
