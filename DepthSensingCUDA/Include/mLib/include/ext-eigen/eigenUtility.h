#ifndef EXT_EIGENUTILITY_H_
#define EXT_EIGENUTILITY_H_

namespace ml
{

namespace eigenutil
{
	//
	// this doesn't actually need to make a separate copy, but this is much simpler to write. It should eventually
	// be converted to a proper iterator.
	//
	template<class D>
	static std::vector< Eigen::Triplet<D> > makeEigenTriplets(const SparseMatrix<D> &M)
	{
		std::vector< Eigen::Triplet<D> > triplets;
		for(UINT rowIndex = 0; rowIndex < M.rows(); rowIndex++)
		{
			const SparseRow<D> &row = M.sparseRow(rowIndex);
			for(const SparseRowEntry<D> &e : row.entries)
				triplets.push_back(Eigen::Triplet<D>(rowIndex, e.col, e.val));
		}
		return triplets;
	}

	//
	// Does not return by-value because it is not clear if Eigen supports move semantics.
	//
	template<class D>
	static void makeEigenMatrix(const SparseMatrix<D> &M, Eigen::SparseMatrix<D> &result)
	{
		result.resize(M.rows(), M.cols());
		auto triplets = makeEigenTriplets(M);
		result.setFromTriplets(triplets.begin(), triplets.end());
	}

	template<class D>
	static void makeEigenMatrix(const DenseMatrix<D> &M, Eigen::MatrixXd &result) 
	{
		result.resize(M.rows(), M.cols());
		for (unsigned int i = 0; i < M.rows(); i++) {
			for (unsigned int j = 0; j < M.cols(); j++) {
				result(i, j) = M(i, j);
			}
		}
	}


	template<class D>
	static Eigen::VectorXd makeEigenVector(const MathVector<D> &v)
	{
		const UINT n = (UINT)v.size();
		Eigen::VectorXd result(n);
		for(UINT i = 0; i < n; i++) result[i] = v[i];
		return result;
	}

	template<class D>
	static MathVector<D> dumpEigenVector(const Eigen::VectorXd &v)
	{
		const UINT n = (UINT)v.size();
		MathVector<double> result(n);
		for(UINT i = 0; i < n; i++) result[i] = v[i];
		return result;
	}

	template<class D>
	static void dumpEigenMatrix(const Eigen::MatrixXd &m, DenseMatrix<D>& result)
	{
		//DenseMatrix<D> result(m.rows(), m.cols());
		result.resize((UINT)m.rows(), (UINT)m.cols());
		for (unsigned int i = 0; i < m.rows(); i++) {
			for (unsigned int j = 0; j < m.cols(); j++) {
				result(i, j) = (D)m(i, j);
			}
		}
	}

}

}  // namespace ml

#endif  // EXT_EIGENUTILITY_H_
