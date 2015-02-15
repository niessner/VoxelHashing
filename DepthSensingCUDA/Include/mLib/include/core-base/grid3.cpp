
#ifndef CORE_BASE_GRID3D_INL_H_
#define CORE_BASE_GRID3D_INL_H_

namespace ml
{

template <class T> Grid3<T>::Grid3()
{
	m_dimX = 0;
	m_dimY = 0;
	m_dimZ = 0;
	m_data = nullptr;
}

template <class T> Grid3<T>::Grid3(size_t dimX, size_t dimY, size_t dimZ)
{
	m_dimX = dimX;
	m_dimY = dimY;
	m_dimZ = dimZ;
	m_data = new T[dimX * dimY * dimZ];
}

template <class T> Grid3<T>::Grid3(size_t dimX, size_t dimY, size_t dimZ, const T &clearValue)
{
	m_dimX = dimX;
	m_dimY = dimY;
	m_dimZ = dimZ;
	m_data = new T[dimX * dimY * dimZ];
	clear(clearValue);
}

template <class T> Grid3<T>::Grid3(const Grid3<T> &G)
{
  m_dimX = G.m_dimX;
  m_dimY = G.m_dimY;
  m_dimZ = G.m_dimZ;

	const size_t totalEntries = getNumTotalEntries();
	m_data = new T[totalEntries];
	for(size_t i = 0; i < totalEntries; i++) {
		m_data[i] = G.m_data[i];
	}
}

template <class T> Grid3<T>::Grid3(Grid3<T> &&G)
{
  m_dimX = G.m_dimX;
  m_dimY = G.m_dimY;
  m_dimZ = G.m_dimZ;

	m_data = G.m_data;

  G.m_dimX = 0;
  G.m_dimY = 0;
  G.m_dimZ = 0;

	G.m_data = nullptr;
}

template <class T> Grid3<T>::~Grid3()
{
	deleteMemory();
}

template <class T> void Grid3<T>::deleteMemory()
{
	m_dimX = 0;
	m_dimY = 0;
	m_dimZ = 0;
	if(m_data != nullptr)
	{
		delete[] m_data;
		m_data = nullptr;
	}
}

template <class T> Grid3<T>& Grid3<T>::operator = (const Grid3<T> &G)
{
	if(m_data) delete[] m_data;
  m_dimX = G.m_dimX;
  m_dimY = G.m_dimY;
  m_dimZ = G.m_dimZ;

	const size_t totalEntries = getNumTotalEntries();
	m_data = new T[totalEntries];
	for (size_t i = 0; i < totalEntries; i++) {
		m_data[i] = G.m_data[i];
	}

	return *this;
}

template <class T> Grid3<T>& Grid3<T>::operator = (Grid3<T> &&G)
{
  std::swap(m_dimX, G.m_dimX);
  std::swap(m_dimY, G.m_dimY);
  std::swap(m_dimZ, G.m_dimZ);
	std::swap(m_data, G.m_data);
	return *this;
}

template <class T> void Grid3<T>::allocate(size_t dimX, size_t dimY, size_t dimZ)
{
	m_dimX = dimX;
	m_dimY = dimY;
	m_dimZ = dimZ;
	if(m_data) delete[] m_data;
	m_data = new T[dimX * dimY * dimZ];
}

template <class T> void Grid3<T>::allocate(size_t dimX, size_t dimY, size_t dimZ, const T &clearValue)
{
	allocate(dimX, dimY, dimZ);
	clear(clearValue);
}

template <class T> void Grid3<T>::clear(const T &clearValue)
{
	const size_t totalEntries = getNumTotalEntries();
	for (size_t i = 0; i < totalEntries; i++) m_data[i] = clearValue;
}

}  // namespace ml

#endif  // CORE_BASE_GRID3D_INL_H_