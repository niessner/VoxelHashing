
#ifndef CORE_MATH_DENSEMATRIX_INL_H_
#define CORE_MATH_DENSEMATRIX_INL_H_

namespace ml {

template<class FloatType>
FloatType DenseMatrix<FloatType>::maxMagnitude() const
{
	MLIB_ASSERT_STR(valid(), "dense matrix has invalid entries");
	double result = 0.0;
	for(UINT row = 0; row < m_rows; row++)
		for(UINT col = 0; col < m_cols; col++)
			result = std::max(result, fabs(m_dataPtr[row * m_cols + col]));
	return result;
}

template<class FloatType>
bool DenseMatrix<FloatType>::valid() const
{
	for(UINT row = 0; row < m_rows; row++)
		for(UINT col = 0; col < m_cols; col++)
		{
			double v = m_dataPtr[row * m_cols + col];
			if(v != v) return false;
		}
	return true;
}

template<class FloatType>
DenseMatrix<FloatType> DenseMatrix<FloatType>::transpose() const
{
    DenseMatrix<FloatType> result(m_cols, m_rows);
    for(UINT row = 0; row < m_rows; row++)
        for(UINT col = 0; col < m_cols; col++)
            result.m_dataPtr[col * m_rows + row] = m_dataPtr[row * m_cols + col];
    return result;
}

template<class FloatType>
DenseMatrix<FloatType> DenseMatrix<FloatType>::multiply(const DenseMatrix<FloatType> &A, FloatType val)
{
	DenseMatrix<FloatType> result(A.m_rows, A.m_cols);
	for(UINT row = 0; row < A.m_rows; row++)
		for(UINT col = 0; col < A.m_cols; col++)
			result.m_dataPtr[row * A.m_cols + col] = A.m_dataPtr[row * A.m_cols + col] * val;
	return result;
}

template<class FloatType>
DenseMatrix<FloatType> DenseMatrix<FloatType>::add(const DenseMatrix<FloatType> &A, const DenseMatrix<FloatType> &B)
{
	MLIB_ASSERT_STR(A.rows() == B.rows() && A.cols() == B.cols(), "invalid matrix dimensions");
	
	const UINT rows = A.m_rows;
	const UINT cols = A.m_cols;

	DenseMatrix<FloatType> result(A.m_rows, A.m_cols);
	for(UINT row = 0; row < rows; row++)
		for(UINT col = 0; col < cols; col++)
			result.m_dataPtr[row * cols + col] = A.m_dataPtr[row * cols + col] + B.m_dataPtr[row * cols + col];
	return result;
}

template<class FloatType>
DenseMatrix<FloatType> DenseMatrix<FloatType>::subtract(const DenseMatrix<FloatType> &A, const DenseMatrix<FloatType> &B)
{
	MLIB_ASSERT_STR(A.rows() == B.rows() && A.cols() == B.cols(), "invalid matrix dimensions");

	const UINT rows = A.m_rows;
	const UINT cols = A.m_cols;

	DenseMatrix<FloatType> result(A.m_rows, A.m_cols);
	for(UINT row = 0; row < rows; row++)
		for(UINT col = 0; col < cols; col++)
			result.m_dataPtr[row * cols + col] = A.m_dataPtr[row * cols + col] - B.m_dataPtr[row * cols + col];
	return result;
}

template<class FloatType>
std::vector<FloatType> DenseMatrix<FloatType>::multiply(const DenseMatrix<FloatType> &A, const std::vector<FloatType> &B)
{
	MLIB_ASSERT_STR(A.cols() == B.size(), "invalid dimensions");
	const int rows = A.m_rows;
	const UINT cols = A.m_cols;
	std::vector<FloatType> result(rows);
//#ifdef MLIB_OPENMP
//#pragma omp parallel for
//#endif
	for(int row = 0; row < rows; row++)
	{
		FloatType val = 0.0;
		for(UINT col = 0; col < cols; col++)
			val += A.m_dataPtr[row * cols + col] * B[col];
		result[row] = val;
	}
	return result;
}

template<class FloatType>
DenseMatrix<FloatType> DenseMatrix<FloatType>::multiply(const DenseMatrix<FloatType> &A, const DenseMatrix<FloatType> &B)
{
	MLIB_ASSERT_STR(A.cols() == B.rows(), "invalid dimensions");

	const UINT rows = A.rows();
	const UINT cols = B.cols();
	const UINT innerCount = A.cols();

	DenseMatrix<FloatType> result(rows, cols);
	
	for(UINT row = 0; row < rows; row++)
		for(UINT col = 0; col < cols; col++)
		{
			FloatType sum = 0.0;
			for(UINT inner = 0; inner < innerCount; inner++)
				sum += A(row, inner) * B(inner, col);
			result(row, col) = sum;
		}

	return result;
}

template<class FloatType>
DenseMatrix<FloatType> DenseMatrix<FloatType>::inverse()
{
	DenseMatrix<FloatType> result = *this;
	result.invertInPlace();
	return result;
}

template<class FloatType>
void DenseMatrix<FloatType>::invertInPlace()
{
	MLIB_ASSERT_STR(square(), "DenseMatrix<D>::invertInPlace called on non-square matrix");
	for (UINT i = 1; i < m_rows; i++)
	{
		(*this)(0, i) /= (*this)(0, 0);
	}

	for (UINT i = 1; i < m_rows; i++)
	{
		//
		// do a column of L
		//
		for (UINT j = i; j < m_rows; j++)
		{
			FloatType sum = 0;
			for (UINT k = 0; k < i; k++)  
			{
				sum += (*this)(j, k) * (*this)(k, i);
			}
			(*this)(j, i) -= sum;
		}
		if (i == m_rows - 1)
		{
			continue;
		}

		//
		// do a row of U
		//
		for (UINT j = i + 1; j < m_rows; j++)
		{
			FloatType sum = 0;
			for (UINT k = 0; k < i; k++)
				sum += (*this)(i, k) * (*this)(k, j);
			(*this)(i, j) = ((*this)(i, j) - sum) / (*this)(i, i);
		}
	}

	//
	// invert L
	//
	for (UINT i = 0; i < m_rows; i++)
		for (UINT j = i; j < m_rows; j++)
		{
			FloatType sum = (FloatType)1.0;
			if ( i != j )
			{
				sum = 0;
				for (UINT k = i; k < j; k++ ) 
				{
					sum -= (*this)(j, k) * (*this)(k, i);
				}
			}
			(*this)(j, i) = sum / (*this)(j, j);
		}

		//
		// invert U
		//
		for (UINT i = 0; i < m_rows; i++)
			for (UINT j = i; j < m_rows; j++)
			{
				if ( i == j )
				{
					continue;
				}
				FloatType sum = 0;
				for (UINT k = i; k < j; k++)
				{
					FloatType val = (FloatType)1.0;
					if(i != k)
					{
						val = (*this)(i, k);
					}
					sum += (*this)(k, j) * val;
				}
				(*this)(i, j) = -sum;
			}

			//
			// final inversion
			//
			for (UINT i = 0; i < m_rows; i++)
			{
				for (UINT j = 0; j < m_rows; j++)
				{
					FloatType sum = 0;
					UINT larger = j;
					if(i > j)
					{
						larger = i;
					}
					for (UINT k = larger; k < m_rows; k++)
					{
						FloatType val = (FloatType)1.0;
						if(j != k)
						{
							val = (*this)(j, k);
						}
						sum += val * (*this)(k, i);
					}
					(*this)(j, i) = sum;
				}
			}
			//Assert(ElementsValid(), "Degenerate Matrix inversion.");
}

}  // namespace ml

#endif  // CORE_MATH_DENSEMATRIX_INL_H_