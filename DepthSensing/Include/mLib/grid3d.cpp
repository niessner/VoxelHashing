
#ifndef CORE_BASE_GRID3D_INL_H_
#define CORE_BASE_GRID3D_INL_H_

namespace ml
{

template <class T> Grid3D<T>::Grid3D()
{
	m_rows = 0;
	m_cols = 0;
	m_slices = 0;
	m_data = NULL;
}

template <class T> Grid3D<T>::Grid3D(UINT rows, UINT cols, UINT slices)
{
	m_rows = rows;
	m_cols = cols;
	m_slices = slices;
	m_data = new T[rows * cols * slices];
}

template <class T> Grid3D<T>::Grid3D(UINT rows, UINT cols, UINT slices, const T &clearValue)
{
	m_rows = rows;
	m_cols = cols;
	m_slices = slices;
	m_data = new T[rows * cols * slices];
	clear(clearValue);
}

template <class T> Grid3D<T>::Grid3D(const Grid3D<T> &G)
{
	m_rows = G.m_rows;
	m_cols = G.m_cols;
	m_slices = G.m_slices;

	const UINT totalEntries = getNumTotalEntries();
	m_data = new T[totalEntries];
	for(UINT i = 0; i < totalEntries; i++) {
		m_data[i] = G.m_data[i];
	}
}

template <class T> Grid3D<T>::Grid3D(Grid3D<T> &&G)
{
	m_cols = G.m_cols;
	m_rows = G.m_rows;
	m_slices = G.m_slices;

	m_data = G.m_data;

	G.m_rows = 0;
	G.m_cols = 0;
	G.m_slices = 0;

	G.m_data = NULL;
}

template <class T> Grid3D<T>::~Grid3D()
{
	deleteMemory();
}

template <class T> void Grid3D<T>::deleteMemory()
{
	m_rows = 0;
	m_cols = 0;
	m_slices = 0;
	if(m_data != NULL)
	{
		delete[] m_data;
		m_data = NULL;
	}
}

template <class T> Grid3D<T>& Grid3D<T>::operator = (const Grid3D<T> &G)
{
	if(m_data) delete[] m_data;
	m_rows = G.m_rows;
	m_cols = G.m_cols;
	m_slices = G.m_slices;

	const UINT totalEntries = getNumTotalEntries();
	m_data = new T[totalEntries];
	for (UINT i = 0; i < totalEntries; i++) {
		m_data[i] = G.m_data[i];
	}

	return *this;
}

template <class T> Grid3D<T>& Grid3D<T>::operator = (Grid3D<T> &&G)
{
	std::swap(m_rows, G.m_rows);
	std::swap(m_cols, G.m_cols);
	std::swap(m_slices, G.m_slices);
	std::swap(m_data, G.m_data);
	return *this;
}

template <class T> void Grid3D<T>::allocate(UINT rows, UINT cols, UINT slices)
{
	m_rows = rows;
	m_cols = cols;
	m_slices = slices;
	if(m_data) delete[] m_data;
	m_data = new T[rows * cols * slices];
}

template <class T> void Grid3D<T>::allocate(UINT rows, UINT cols, UINT slices, const T &clearValue)
{
	allocate(rows, cols, slices);
	clear(clearValue);
}

template <class T> void Grid3D<T>::clear(const T &clearValue)
{
	const UINT totalEntries = getNumTotalEntries();
	for(UINT i = 0; i < totalEntries; i++) m_data[i] = clearValue;
}

}  // namespace ml

#endif  // CORE_BASE_GRID3D_INL_H_