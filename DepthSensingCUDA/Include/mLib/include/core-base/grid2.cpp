
#ifndef CORE_BASE_GRID2D_H_INL_
#define CORE_BASE_GRID2D_H_INL_

namespace ml
{

template <class T> Grid2<T>::Grid2()
{
	m_dimX = 0;
	m_dimY = 0;
	m_data = nullptr;
}

template <class T> Grid2<T>::Grid2(size_t dimX, size_t dimY)
{
	m_dimX = dimX;
	m_dimY = dimY;
	m_data = new T[dimX * dimY];
}

template <class T> Grid2<T>::Grid2(size_t dimX, size_t dimY, const T &clearValue)
{
	m_dimX = dimX;
	m_dimY = dimY;
	m_data = new T[dimX * dimY];
	clear(clearValue);
}

template <class T> Grid2<T>::Grid2(const Grid2<T> &G)
{
	m_dimX = G.m_dimX;
	m_dimY = G.m_dimY;

	const size_t totalEntries = m_dimX * m_dimY;
	m_data = new T[totalEntries];
	for(size_t i = 0; i < totalEntries; i++)
	{
		m_data[i] = G.m_data[i];
	}
}

template <class T> Grid2<T>::Grid2(Grid2<T> &&G)
{
	m_dimX = G.m_dimX;
	m_dimY = G.m_dimY;
	m_data = G.m_data;

	G.m_dimX = 0;
	G.m_dimY = 0;
	G.m_data = nullptr;
}

template <class T> Grid2<T>::~Grid2()
{
	deleteMemory();
}

template <class T> void Grid2<T>::deleteMemory()
{
	m_dimX = 0;
	m_dimY = 0;
	if(m_data != nullptr)
	{
		delete[] m_data;
		m_data = nullptr;
	}
}

template <class T> Grid2<T>& Grid2<T>::operator = (const Grid2<T> &G)
{
	if(m_data) delete[] m_data;
	m_dimX = G.m_dimX;
	m_dimY = G.m_dimY;

	const size_t totalEntries = m_dimX * m_dimY;
	m_data = new T[totalEntries];
	for(size_t i = 0; i < totalEntries; i++) m_data[i] = G.m_data[i];

	return *this;
}

template <class T> Grid2<T>& Grid2<T>::operator = (Grid2<T> &&G)
{
	std::swap(m_dimX, G.m_dimX);
	std::swap(m_dimY, G.m_dimY);
	std::swap(m_data, G.m_data);
	return *this;
}

template <class T> void Grid2<T>::allocate(size_t dimX, size_t dimY)
{
	m_dimX = dimX;
	m_dimY = dimY;
	if(m_data) delete[] m_data;
	m_data = new T[dimX * dimY];
}

template <class T> void Grid2<T>::allocate(size_t dimX, size_t dimY, const T &clearValue)
{
	allocate(dimX, dimY);
	clear(clearValue);
}

template <class T> void Grid2<T>::clear(const T &clearValue)
{
	const size_t totalEntries = m_dimX * m_dimY;
	for(size_t i = 0; i < totalEntries; i++) m_data[i] = clearValue;
}

template <class T> std::pair<size_t, size_t> Grid2<T>::maxIndex() const
{
	std::pair<size_t, size_t> maxIndex(0, 0);
	const T *maxValue = m_data;
	for(size_t rowIndex = 0; rowIndex < m_dimX; rowIndex++)
		for(size_t colIndex = 0; colIndex < m_dimY; colIndex++)
		{
			const T *curValue = &m_data[rowIndex * m_dimY + colIndex];
			if(*curValue > *maxValue)
			{
				maxIndex = std::make_pair(rowIndex, colIndex);
				maxValue = curValue;
			}
		}
	return maxIndex;
}

template <class T> const T& Grid2<T>::maxValue() const
{
	std::pair<size_t, size_t> index = maxIndex();
	return m_data[index.first * m_dimY + index.second];
}

template <class T> std::pair<size_t, size_t> Grid2<T>::minIndex() const
{
	std::pair<size_t, size_t> minIndex(0, 0);
	const T *minValue = &m_data[0];
	for(size_t rowIndex = 0; rowIndex < m_dimX; rowIndex++)
	{
		for(size_t colIndex = 0; colIndex < m_dimY; colIndex++)
		{
			const T *curValue = &m_data[rowIndex * m_dimY + colIndex];
			if(*curValue < *minValue)
			{
				minIndex = std::make_pair(rowIndex, colIndex);
				minValue = curValue;
			}
		}
	}
	return minIndex;
}

template <class T> const T& Grid2<T>::minValue() const
{
	std::pair<size_t, size_t> index = minIndex();
	return m_data[index.first * m_dimY + index.second];
}

}  // namespace ml

#endif  // CORE_BASE_GRID2D_H_INL_
