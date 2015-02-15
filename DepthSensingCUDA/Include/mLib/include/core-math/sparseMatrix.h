
#ifndef CORE_MATH_SPARSEMATRIX_H_
#define CORE_MATH_SPARSEMATRIX_H_

namespace ml {

enum MatrixStringFormat
{
	MatrixStringFormatMathematica,
};

template <class FloatType> 
struct SparseRowEntry
{
	SparseRowEntry() {}
	SparseRowEntry(UINT _col, FloatType _val)
	{
		col = _col;
		val = _val;
	}
	UINT col;
	FloatType val;
};

template <class FloatType>
struct SparseRow
{
	FloatType& operator()(UINT col)
	{
		for(SparseRowEntry<FloatType> &e : entries)
		{
			if(e.col == col) return e.val;
		}
		entries.push_back(SparseRowEntry<FloatType>(col, 0.0));
		return entries.back().val;
	}
	FloatType operator()(UINT col) const
	{
		for(const SparseRowEntry<FloatType> &e : entries)
		{
			if(e.col == col) return e.val;
		}
		return 0.0;
	}
	std::vector< SparseRowEntry<FloatType> > entries;
};

template<class FloatType, class = typename std::enable_if<std::is_arithmetic<FloatType>::value, FloatType>::type>
class SparseMatrix;

template<class FloatType>
class SparseMatrix<FloatType>
{
public:
	SparseMatrix()
	{
		m_rows = 0;
		m_cols = 0;
	}

	SparseMatrix(const SparseMatrix<FloatType>& s)
	{
		m_rows = s.m_rows;
		m_cols = s.m_cols;
		m_data = s.m_data;
	}

	SparseMatrix(SparseMatrix &&s)
	{
		m_rows = s.m_rows;
		m_cols = s.m_cols;
		s.m_rows = 0;
		s.m_cols = 0;
		m_data = std::move(s.m_data);
	}

	explicit SparseMatrix(UINT squareDimension)
	{
		m_rows = squareDimension;
		m_cols = squareDimension;
		m_data.allocate(m_rows);
	}

	SparseMatrix(UINT rows, UINT cols)
	{
		m_rows = rows;
		m_cols = cols;
		m_data.resize(m_rows);
	}

	SparseMatrix(const std::string &s, MatrixStringFormat format)
	{
		if(format == MatrixStringFormatMathematica)
		{
			//
			// this is really a dense format and should be loaded as such, then cast into a SparseMatrix
			//
			std::vector<std::string> data = ml::util::split(s, "},{");
			m_rows = (UINT)data.size();
			//m_cols = (UINT)data[0].split(",").size();
			m_cols = (UINT)ml::util::split(data[0], ",").size();
			m_data.resize(m_rows);

			for(UINT row = 0; row < m_rows; row++)
			{
				std::vector<std::string> values = ml::util::split(data[row], ",");
				for(UINT col = 0; col < values.size(); col++)
				{
					const std::string s = ml::util::replace(ml::util::replace(values[col], "{",""), "}","");
					(*this)(row, col) = (FloatType)std::stod(s);
				}
			}
		}
		else
		{
			MLIB_ERROR("invalid matrix string format");
		}
	}

	void operator=(const SparseMatrix<FloatType>& s)
	{
		m_rows = s.m_rows;
		m_cols = s.m_cols;
		m_data = s.m_data;
	}

	void operator=(SparseMatrix<FloatType>&& s)
	{
		m_rows = s.m_rows;
		m_cols = s.m_cols;
		s.m_rows = 0;
		s.m_cols = 0;
		m_data = std::move(s.m_data);
	}

	//
	// Accessors
	//
	FloatType& operator()(UINT row, UINT col)
	{
		return m_data[row](col);
	}
	FloatType operator()(UINT row, UINT col) const
	{
		return m_data[row](col);
	}
	UINT rows() const
	{
		return m_rows;
	}
	UINT cols() const
	{
		return m_cols;
	}
	const SparseRow<FloatType>& sparseRow(UINT row) const
	{
		return m_data[row];
	}
	const MathVector<FloatType> denseRow(UINT row) const
	{
		MathVector<FloatType> result(m_cols);
		for(UINT col = 0; col < m_cols; col++)
			result[col] = (*this)(row, col);
		return result;
	}
	const MathVector<FloatType> denseCol(UINT col) const
	{
		MathVector<FloatType> result(m_rows);
		for(UINT row = 0; row < m_rows; row++)
			result[row] = (*this)(row, col);
		return result;
	}
	MathVector<FloatType> diagonal() const
	{
		MLIB_ASSERT_STR(square(), "diagonal called on non-square matrix");
		MathVector<FloatType> result(m_rows);
		for(UINT row = 0; row < m_rows; row++)
			result[row] = m_data[row](row);
		return result;
	}

	//
	// math functions
	//
	SparseMatrix<FloatType> transpose() const;
	FloatType maxMagnitude() const;
	bool square() const
	{
		return (m_rows == m_cols);
	}
	void invertInPlace();

	//
	// overloaded operator helpers
	//
	static SparseMatrix<FloatType> add(const SparseMatrix<FloatType> &A, const SparseMatrix<FloatType> &B);
	static SparseMatrix<FloatType> subtract(const SparseMatrix<FloatType> &A, const SparseMatrix<FloatType> &B);
	static SparseMatrix<FloatType> multiply(const SparseMatrix<FloatType> &A, FloatType c);
	static MathVector<FloatType> multiply(const SparseMatrix<FloatType> &A, const MathVector<FloatType> &v);
	static SparseMatrix<FloatType> multiply(const SparseMatrix<FloatType> &A, const SparseMatrix<FloatType> &B);
	
	// returns the scalar v^T A v
	static FloatType quadratic(const SparseMatrix<FloatType> &A, const MathVector<FloatType> &v)
	{
		return MathVector<FloatType>::dot(v, multiply(A, v));
	}

private:
	UINT m_rows, m_cols;
    std::vector< SparseRow<FloatType> > m_data;

	// set is a more efficient version of operator() that assumes the entry
	// does not exist.
	void insert(UINT row, UINT col, double val)
	{
		m_data[row].entries.push_back(SparseRowEntry<FloatType>(col, val));
	}
};

template<class FloatType>
SparseMatrix<FloatType> operator + (const SparseMatrix<FloatType> &A, const SparseMatrix<FloatType> &B)
{
	return SparseMatrix<FloatType>::add(A, B);
}

template<class FloatType>
SparseMatrix<FloatType> operator - (const SparseMatrix<FloatType> &A, const SparseMatrix<FloatType> &B)
{
	return SparseMatrix<FloatType>::subtract(A, B);
}

template<class FloatType>
SparseMatrix<FloatType> operator * (const SparseMatrix<FloatType> &A, const SparseMatrix<FloatType> &B)
{
	return SparseMatrix<FloatType>::multiply(A, B);
}

template<class FloatType>
MathVector<FloatType> operator * (const SparseMatrix<FloatType> &A, const MathVector<FloatType> &B)
{
	return SparseMatrix<FloatType>::multiply(A, B);
}

template<class FloatType>
SparseMatrix<FloatType> operator * (const SparseMatrix<FloatType> &A, FloatType val)
{
	return SparseMatrix<FloatType>::multiply(A, val);
}

//typedef SparseMatrix<float> SparseMatrixf;
//typedef SparseMatrix<double> SparseMatrixd;

}  // namespace ml

#include "sparseMatrix.cpp"

#endif  // CORE_MATH_SPARSEMATRIX_H_
