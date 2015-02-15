
#ifndef CORE_MATH_SPARSEMATRIX_INL_H_
#define CORE_MATH_SPARSEMATRIX_INL_H_

namespace ml {

template <class D>
D SparseMatrix<D>::maxMagnitude() const
{
	double result = 0.0;
	for(UINT row = 0; row < m_rows; row++)
		for(const SparseRowEntry<D> &e : m_data[row].entries)
			result = std::max(result, fabs(e.val));
	return result;
}

template <class D> 
SparseMatrix<D> SparseMatrix<D>::transpose() const
{
    SparseMatrix<D> result(m_cols, m_rows);
    for(UINT row = 0; row < m_rows; row++)
        for(const SparseRowEntry<D> &e : m_data[row].entries)
            result.insert(e.col, row, e.val);
    return result;
}

template <class D> 
SparseMatrix<D> SparseMatrix<D>::multiply(const SparseMatrix<D> &A, D val)
{
	SparseMatrix<D> result = A;
	for(UINT row = 0; row < A.m_rows; row++)
		for(SparseRowEntry<D> &e : result.m_data[row].entries)
			e.val *= val;
	return result;
}

template <class D> 
SparseMatrix<D> SparseMatrix<D>::add(const SparseMatrix<D> &A, const SparseMatrix<D> &B)
{
	MLIB_ASSERT_STR(A.rows() == B.rows() && A.cols() == B.cols(), "invalid matrix dimensions");
	SparseMatrix<D> result = A;
	for(UINT row = 0; row < B.m_rows; row++)
		for(const SparseRowEntry<D> &e : B.m_data[row].entries)
			result(row, e.col) += e.val;
	return result;
}

template <class D> 
SparseMatrix<D> SparseMatrix<D>::subtract(const SparseMatrix<D> &A, const SparseMatrix<D> &B)
{
	MLIB_ASSERT_STR(A.rows() == B.rows() && A.cols() == B.cols(), "invalid matrix dimensions");
	SparseMatrix<D> result = A;
	for(UINT row = 0; row < B.m_rows; row++)
		for(const SparseRowEntry<D> &e : B.m_data[row].entries)
			result(row, e.col) -= e.val;
	return result;
}

template <class D> 
MathVector<D> SparseMatrix<D>::multiply(const SparseMatrix<D> &A, const MathVector<D> &B)
{
	MLIB_ASSERT_STR(A.cols() == B.size(), "invalid dimensions");
	const int rows = A.m_rows;
	MathVector<D> result(rows);

	const D* BPtr = &B[0];
	D* resultPtr = &result[0];

#ifdef MLIB_OPENMP
#pragma omp parallel for
#endif
	for(int row = 0; row < rows; row++)
	{
		D val = 0.0;
		for(const SparseRowEntry<D> &e : A.m_data[row].entries) {
			val += e.val * BPtr[e.col];
		}
		resultPtr[row] = val;
	}
	return result;
}

template <class D> 
SparseMatrix<D> SparseMatrix<D>::multiply(const SparseMatrix<D> &A, const SparseMatrix<D> &B)
{
	MLIB_ASSERT_STR(A.cols() == B.rows(), "invalid dimensions");

	const UINT rows = A.rows();
	SparseMatrix<D> result(rows, B.cols());
	
	for(UINT row = 0; row < rows; row++)
	{
		for(const SparseRowEntry<D> &eA : A.m_data[row].entries)
		{
			for(const SparseRowEntry<D> &eB : B.m_data[eA.col].entries)
			{
				result(row, eB.col) += eA.val * eB.val;
			}
		}
	}
	return result;
}

template <class D> 
void SparseMatrix<D>::invertInPlace()
{
	MLIB_ASSERT_STR(square(), "SparseMatrix<D>::invertInPlace called on non-square matrix");
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
			D sum = 0;
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
			D sum = 0;
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
			D sum = (D)1.0;
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
				D sum = 0;
				for (UINT k = i; k < j; k++)
				{
					D val = (D)1.0;
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
					D sum = 0;
					UINT larger = j;
					if(i > j)
					{
						larger = i;
					}
					for (UINT k = larger; k < m_rows; k++)
					{
						D val = (D)1.0;
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

#endif  // CORE_MATH_SPARSEMATRIX_INL_H_
