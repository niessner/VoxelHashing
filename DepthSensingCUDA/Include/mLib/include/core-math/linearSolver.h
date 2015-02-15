
#ifndef CORE_MATH_LINEARSOLVER_H_
#define CORE_MATH_LINEARSOLVER_H_

namespace ml {

template<class FloatType> class LinearSolver
{
public:
	virtual MathVector<FloatType> solve(const SparseMatrix<FloatType> &A, const MathVector<FloatType> &b) = 0;
	virtual MathVector<FloatType> solveLeastSquares(const SparseMatrix<FloatType> &A, const MathVector<FloatType> &b) = 0;
	static FloatType solveError(const SparseMatrix<FloatType> &A, const MathVector<FloatType> &x, const MathVector<FloatType> &b)
	{
		//.map([](T n) {return std::string(n);})
		//return (A * x - b).map([](D x) {return fabs(x);}).maxValue();
		FloatType res = (FloatType)0.0;
		std::vector<FloatType> Ax = A*x;
		for (size_t i = 0; i < Ax.size(); i++) {
			res +=  (Ax[i] - b[i])*(Ax[i] - b[i]);
		} 
		return std::sqrt(res);
	}
};

template<class FloatType> class LinearSolverConjugateGradient : public LinearSolver<FloatType>
{
public:
	LinearSolverConjugateGradient(UINT maxIterations = 10000, FloatType tolerance = 1e-10)
	{
		m_maxIterations = maxIterations;
		m_tolerance = tolerance;
	}

	MathVector<FloatType> solve(const SparseMatrix<FloatType> &A, const MathVector<FloatType> &b)
	{
		MLIB_ASSERT_STR(A.square() && b.size() == A.rows(), "invalid solve dimensions");
		const UINT n = (UINT)b.size();

		//std::vector<D> dInverse = A.diagonal().map([](D x) {return (D)1.0 / x;});
		MathVector<FloatType> dInverse = A.diagonal();
		auto invert = [=] (FloatType& x) { x = (FloatType)1.0/x; };
		for_each(dInverse.begin(), dInverse.end(), invert);

		MathVector<FloatType> x(n, 0.0);
		MathVector<FloatType> r = b - A * x;
		MathVector<FloatType> z = dInverse * r;
		MathVector<FloatType> p = z;

		for(UINT iteration = 0; iteration < m_maxIterations; iteration++)
		{
			FloatType gamma = r | z;
			if(fabs(gamma) < 1e-20) break;
			FloatType alpha = gamma / SparseMatrix<FloatType>::quadratic(A, p);
			x = x + alpha * p;
			r = r - alpha * (A * p);

			if (*std::max_element(r.begin(), r.end()) <= m_tolerance && *std::min_element(r.begin(), r.end()) >= -m_tolerance)	break;

			z = dInverse * r;
			FloatType beta = (z | r) / gamma;
			p = z + beta * p;
		}
		return x;
	}

	MathVector<FloatType> solveLeastSquares(const SparseMatrix<FloatType> &A, const MathVector<FloatType> &b)
	{
		auto Atranspose = A.transpose();
		return solve(Atranspose * A, Atranspose * b);
	}

private:
	UINT m_maxIterations;
	FloatType m_tolerance;
};

}  // namespace ml

#endif  // CORE_MATH_LINEARSOLVER_H_
