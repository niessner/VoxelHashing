
#ifndef CORE_MATH_DENSEMATRIX_H_
#define CORE_MATH_DENSEMATRIX_H_

namespace ml {

template <class T> class DenseMatrix
{
public:
	DenseMatrix()
	{
		m_rows = 0;
		m_cols = 0;
	}

	DenseMatrix(const DenseMatrix<T>& s)
	{
		m_rows = s.m_rows;
		m_cols = s.m_cols;
		m_data = s.m_data;
		m_dataPtr = &m_data[0];
	}

    DenseMatrix(DenseMatrix<T> &&s)
	{
		m_rows = s.m_rows;
		m_cols = s.m_cols;
		s.m_rows = 0;
		s.m_cols = 0;
		m_data = std::move(s.m_data);
		m_dataPtr = &m_data[0];
	}

    explicit DenseMatrix(UINT squareDimension)
	{
		m_rows = squareDimension;
		m_cols = squareDimension;
		m_data.resize(m_rows * m_cols);
		m_dataPtr = &m_data[0];
	}

	explicit DenseMatrix(const MathVector<T> &diagonal)
	{
		m_rows = (UINT)diagonal.size();
		m_cols = (UINT)diagonal.size();
		m_data.resize(m_rows * m_cols);
		m_dataPtr = &m_data[0];
		for(UINT row = 0; row < m_rows; row++)
		{
			for(UINT col = 0; col < m_cols; col++)
				(*this)(row, col) = 0.0;
			(*this)(row, row) = diagonal[row];
		}

	}

	DenseMatrix(UINT rows, UINT cols, T clearValue = (T)0.0)
	{
		m_rows = rows;
		m_cols = cols;
        m_data.resize(m_rows * m_cols, clearValue);
		m_dataPtr = &m_data[0];
	}

	DenseMatrix(const std::string &s, MatrixStringFormat format)
	{
		if(format == MatrixStringFormatMathematica)
		{
			//
			// this is really a dense format and should be loaded as such, then cast into a SparseMatrix
			//
			std::vector<std::string> data = ml::util::split(s,"},{");
			m_rows = (UINT)data.size();
			m_cols = (UINT)ml::util::split(data[0], ",").size();
			m_data.resize(m_rows * m_cols);
			m_dataPtr = &m_data[0];

			for(UINT row = 0; row < m_rows; row++)
			{
				std::vector<std::string> values = ml::util::split(data[row], ",");
				for(UINT col = 0; col < values.size(); col++)
				{
					const std::string s = ml::util::replace(ml::util::replace(values[col], "{",""), "}","");
					(*this)(row, col) = (T)std::stod(s);
				}
			}
		}
		else
		{
			MLIB_ERROR("invalid matrix string format");
		}
	}

    DenseMatrix(const Matrix4x4<T> &m)
    {
        m_rows = 4;
        m_cols = 4;
        m_data.resize(16);
        m_dataPtr = &m_data[0];
        for (unsigned int element = 0; element < m_data.size(); element++)
            m_data[element] = m[element];
    }

	DenseMatrix(const Matrix3x3<T> &m)
	{
		m_rows = 3;
		m_cols = 3;
		m_data.resize(9);
		m_dataPtr = &m_data[0];
		for (unsigned int element = 0; element < m_data.size(); element++)
			m_data[element] = m[element];
	}

	DenseMatrix(const Matrix2x2<T> &m)
	{
		m_rows = 2;
		m_cols = 2;
		m_data.resize(4);
		m_dataPtr = &m_data[0];
		for (unsigned int element = 0; element < m_data.size(); element++)
			m_data[element] = m[element];
	}

	void resize(UINT rows, UINT cols, T clearValue = (T)0.0)
	{
		m_rows = rows;
		m_cols = cols;
		m_data.resize(m_rows * m_cols, clearValue);
		m_dataPtr = &m_data[0];
	}


	void operator=(const DenseMatrix<T>& s)
	{
		m_rows = s.m_rows;
		m_cols = s.m_cols;
		m_data = s.m_data;
        m_dataPtr = &m_data[0];
	}

	void operator=(DenseMatrix<T>&& s)
	{
		m_rows = s.m_rows;
		m_cols = s.m_cols;
		s.m_rows = 0;
		s.m_cols = 0;
		m_data = std::move(s.m_data);
		m_dataPtr = &m_data[0];
	}

	//
	// Accessors
	//
	T& operator()(UINT row, UINT col)
	{
		return m_dataPtr[row * m_cols + col];
	}

	T operator()(UINT row, UINT col) const
	{
		return m_dataPtr[row * m_cols + col];
	}

	UINT rows() const
	{
		return m_rows;
	}

	UINT cols() const
	{
		return m_cols;
	}

	bool square() const
	{
		return (m_rows == m_cols);
	}

	//! Access i-th element of the Matrix for constant access
	inline T operator[] (unsigned int i) const {
		assert(i < m_cols*m_rows);
		return m_dataPtr[i];
	}

	//! Access i-th element of the Matrix
	inline  T& operator[] (unsigned int i) {
		assert(i < m_cols*m_rows);
		return m_dataPtr[i];
	}

	std::vector<T> diagonal() const
	{
		MLIB_ASSERT_STR(square(), "diagonal called on non-square matrix");
		std::vector<T> result(m_rows);
		for(UINT row = 0; row < m_rows; row++)
			result[row] = m_data[row * m_cols + row];
		return result;
	}

    const T* ptr() const
    {
        return m_dataPtr;
    }


	//
	// math functions
	//
	DenseMatrix<T> transpose() const;
	T maxMagnitude() const;
	DenseMatrix<T> inverse();
	void invertInPlace();
	bool valid() const;

	//
	// overloaded operator helpers
	//
	static DenseMatrix<T> add(const DenseMatrix<T> &A, const DenseMatrix<T> &B);
	static DenseMatrix<T> subtract(const DenseMatrix<T> &A, const DenseMatrix<T> &B);
	static DenseMatrix<T> multiply(const DenseMatrix<T> &A, T c);
	static std::vector<T> multiply(const DenseMatrix<T> &A, const std::vector<T> &v);
	static DenseMatrix<T> multiply(const DenseMatrix<T> &A, const DenseMatrix<T> &B);

	//
	// common matrices
	//
	static DenseMatrix<T> identity(int n)
	{
		return DenseMatrix<T>(MathVector<T>(n, (T)1.0));
	}

	unsigned int rank(T eps = (T)0.00001) const {
		if (!square())	throw MLIB_EXCEPTION("");
		return util::rank<DenseMatrix<T>, T>(*this, m_rows, eps);
	} 

	// checks whether the matrix is symmetric
	bool isSymmetric(T eps = (T)0.00001) const {
		if (!square())	return false;
		for (unsigned int i = 1; i < m_rows; i++) {
			for (unsigned int j = 0; j < m_cols/2; j++) {
				if (!math::floatEqual((*this)(i, j), (*this)(j, i), eps)) return false;
			}
		}
		return true;
	}

	EigenSystem<T> eigenSystem() const {
		return EigenSolver<T>::solve<EigenSolver<T>::TYPE_DEFAULT>(*this);
	}

    //
    // in-place operators
    //
    void operator /= (T x)
    {
        T rcp = (T)1.0 / x;
        for (T &e : m_data)
            e *= rcp;
    }

private:
	UINT m_rows, m_cols;
	T* m_dataPtr;
    std::vector< T > m_data;
};

template<class T>
DenseMatrix<T> operator + (const DenseMatrix<T> &A, const DenseMatrix<T> &B)
{
	return DenseMatrix<T>::add(A, B);
}

template<class T>
DenseMatrix<T> operator - (const DenseMatrix<T> &A, const DenseMatrix<T> &B)
{
	return DenseMatrix<T>::subtract(A, B);
}

template<class T>
DenseMatrix<T> operator * (const DenseMatrix<T> &A, const DenseMatrix<T> &B)
{
	return DenseMatrix<T>::multiply(A, B);
}

template<class T>
std::vector<T> operator * (const DenseMatrix<T> &A, const MathVector<T> &B)
{
	return DenseMatrix<T>::multiply(A, B);
}

template<class T>
DenseMatrix<T> operator * (const DenseMatrix<T> &A, T val)
{
	return DenseMatrix<T>::multiply(A, val);
}

//! writes to a stream
template <class T> 
inline std::ostream& operator<<(std::ostream& s, const DenseMatrix<T>& m)
{ 
	for (unsigned int i = 0; i < m.rows(); i++) {
		for (unsigned int j = 0; j < m.cols(); j++) {
			s << m(i,j) << " ";
		}
		std::cout << std::endl;
	}
	return s;
}

typedef DenseMatrix<float> DenseMatrixf;
typedef DenseMatrix<double> DenseMatrixd;

}  // namespace ml

#include "denseMatrix.cpp"

#endif  // CORE_MATH_DENSEMATRIX_H_