
#ifndef CORE_BASE_GRID2D_H_
#define CORE_BASE_GRID2D_H_

namespace ml
{

template <class T> class Grid2
{
public:
	Grid2();
	Grid2(size_t dimX, size_t dimY);
  Grid2(size_t dimX, size_t dimY, const T &clearValue);
	Grid2(const Grid2<T> &G);
	Grid2(Grid2<T> &&G);

	~Grid2();

	//
	// Memory
	//
	void deleteMemory();
	Grid2<T>& operator = (const Grid2<T> &G);
	Grid2<T>& operator = (Grid2<T> &&G);

  void allocate(size_t dimX, size_t dimY);
  void allocate(size_t dimX, size_t dimY, const T &clearValue);

	inline Grid2<T>& operator += (const Grid2<T> &right)
	{
		MLIB_ASSERT_STR(m_dimX == right.m_dimX && m_dimY == right.m_dimY, "grid dimensions must be equal");
		for (size_t r = 0; r < m_dimX; r++)
			for (size_t c = 0; c < m_dimY; c++)
				m_data[r * m_dimY + c] += right(r,c);
		return *this;
	}
	inline Grid2<T>& operator *= (T right)
	{
		for (size_t r = 0; r < m_dimX; r++)
			for (size_t c = 0; c < m_dimY; c++)
				m_data[r * m_dimY + c] *= right;
		return *this;
	}

	inline Grid2<T> operator * (T x)
	{
		Grid2<T> result(m_dimX, m_dimY);
		for (size_t r = 0; r < m_dimX; r++)
			for (size_t c = 0; c < m_dimY; c++)
				result(r,c) = m_data[r * m_dimY + c] * x;
		return result;
	}

	//
	// Accessors
	//
	inline T& operator() (size_t row, size_t col)
	{
#if defined(MLIB_BOUNDS_CHECK) || defined(_DEBUG)
		MLIB_ASSERT_STR( (row < m_dimX) && (col < m_dimY), "Out-of-bounds grid access");
#endif
		return m_data[row * m_dimY + col];
	}
	inline const T& operator() (size_t row, size_t col) const
	{
#if defined(MLIB_BOUNDS_CHECK) || defined(_DEBUG)
		MLIB_ASSERT_STR( (row < m_dimX) && (col < m_dimY), "Out-of-bounds grid access");
#endif
		return m_data[row * m_dimY + col];
	}
	inline size_t rows() const
	{
		return m_dimX;
	}
	inline size_t cols() const
	{
		return m_dimY;
	}

	inline size_t dimX() const {
		return m_dimX;
	}

	inline size_t dimY() const {
		return m_dimY;
	}

	inline std::pair<size_t, size_t> dimensions() const
	{
		return std::make_pair(m_dimX, m_dimY);
	}

	//inline vec2ul getDimensions() const {
	//	return vec2ul(m_rows, m_cols);
	//}

	inline bool square() const
	{
		return (m_dimX == m_dimY);
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
	inline bool isValidCoordinate(int row, int col) const
	{
		return (row >= 0 && row < int(m_dimX) && col >= 0 && col < int(m_dimY));
	}

	void setRow(size_t row, const std::vector<T> &values)
	{
		for(size_t col = 0; col < m_dimY; col++) m_data[row * m_dimY + col] = values[col];
	}

	void setCol(size_t col, const std::vector<T> &values)
	{
		for(size_t row = 0; row < m_dimX; row++) m_data[row * m_dimY + col] = values[row];
	}

	std::vector<T> getRow(size_t row) const
	{
		std::vector<T> result(m_dimY);
		const T *CPtr = m_data;
		for(size_t col = 0; col < m_dimY; col++)
		{
			result[col] = CPtr[row * m_dimY + col];
		}
		return result;
	}

	std::vector<T> getCol(size_t col) const
	{
		std::vector<T> result(m_dimX);
		const T *CPtr = m_data;
		for(size_t row = 0; row < m_dimX; row++)
		{
			result[row] = CPtr[row * m_dimY + col];
		}
		return result;
	}

	std::pair<size_t, size_t> maxIndex() const;
	const T& maxValue() const;
	std::pair<size_t, size_t> minIndex() const;
	const T& minValue() const;

	//
	// Modifiers
	//
	void clear(const T &clearValue);

    const T* begin() const
    {
        return m_data;
    }
    T* begin()
    {
        return m_data;
    }
    const T* end() const
    {
        return m_data + m_dimX * m_dimY;
    }
    T* end()
    {
        return m_data + m_dimX * m_dimY;
    }

protected:
	T *m_data;
	size_t m_dimX, m_dimY;
};

template <class T> inline bool operator == (const Grid2<T> &a, const Grid2<T> &b)
{
	if(a.dimX() != b.dimX() || a.dimY() != b.dimY()) return false;
	for(size_t row = 0; row < a.dimX(); row++)
		for(size_t col = 0; col < a.dimY(); col++)
			if(a(row, col) != b(row, col))
				return false;
	return true;
}

template <class T> inline bool operator != (const Grid2<T> &a, const Grid2<T> &b)
{
	return !(a == b);
}

typedef Grid2<float> Grid2f;
typedef Grid2<double> Grid2d;

}

#include "grid2.cpp"

#endif  // CORE_BASE_GRID2D_H_
