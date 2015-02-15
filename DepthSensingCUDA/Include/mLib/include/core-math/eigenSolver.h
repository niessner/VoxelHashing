
#ifndef CORE_MATH_EIGENSOLVER_H_
#define CORE_MATH_EIGENSOLVER_H_

namespace ml {

//
// All eigensystem code is generally stateless and could be done with static functions. Virtual classes
// are used instead of static functions to enforce a common interface.
//

template<class FloatType>
struct EigenSystem
{
    EigenSystem(int n)
    {
        eigenvectors = DenseMatrix<FloatType>(n, n);
        eigenvalues.resize(n);
    }
	EigenSystem(EigenSystem&& e) {
		eigenvectors = std::move(e.eigenvectors);
		eigenvalues = std::move(e.eigenvalues);
	}
	void operator=(EigenSystem&& e) {
		eigenvectors = std::move(e.eigenvectors);
		eigenvalues = std::move(e.eigenvalues);
	}

    std::vector<FloatType*> eigenvectorList()
    {
        std::vector<FloatType*> result(eigenvectors.rows());
        for (unsigned int row = 0; row < eigenvectors.rows(); row++)
            result[row] = &eigenvectors(row, 0);
        return result;
    }

	//! sorting by absolute eigenvalues (biggest first)
	void sortByAbsValue() {

		//simple selection sort
		for (unsigned int i = 0; i < (UINT)eigenvalues.size(); i++) {
			FloatType currMax = (FloatType)0;
			unsigned int currMaxIdx = (unsigned int)-1;
			for (unsigned int j = i; j < (UINT)eigenvalues.size(); j++) {
				if (std::abs(eigenvalues[j]) > currMax) {
					currMax = std::abs(eigenvalues[j]);
					currMaxIdx = j;
				}
			}

			if (currMaxIdx != i && currMaxIdx != (unsigned int)-1) {
				std::swap(eigenvalues[i], eigenvalues[currMaxIdx]);
				for (unsigned int j = 0; j < (UINT)eigenvalues.size(); j++) {
					std::swap(eigenvectors(i,j), eigenvectors(currMaxIdx,j));
				}
			}
		}

	}

    DenseMatrix<FloatType> eigenvectors;
    MathVector<FloatType> eigenvalues;
};

typedef EigenSystem<float> EigenSystemf;
typedef EigenSystem<double> EigenSystemd;

template<class FloatType>
std::ostream& operator<<(std::ostream& s, const EigenSystem<FloatType>& e) {
	s << e.eigenvectors << std::endl << e.eigenvalues << std::endl;
	return s;
}

template<class FloatType> class EigenSolverVTK;
template<class FloatType> class EigenSolverNR;
template<class FloatType> class EigenSolverEigen;	//warning: only works if eigen is included

template<class FloatType> class EigenSolver
{
public:
	enum SolverType {
		TYPE_DEFAULT = 0,
		TYPE_VTK = 1,
		TYPE_NR = 2,
		TYPE_EIGEN = 3	//warning: only works if eigen is included
	};
    //type 0 -> VTK; type 1 -> NR
	//TODO VS 2013
	//template<SolverType solverType = TYPE_DEFAULT>
	template<SolverType solverType>
	static EigenSystem<FloatType> solve(const DenseMatrix<FloatType> &M) {
		// (the tuple maps to indices)
		std::tuple_element<solverType, std::tuple<EigenSolverVTK<FloatType>, EigenSolverVTK<FloatType>, EigenSolverNR<FloatType>, EigenSolverEigen<FloatType> > >::type solver;
		EigenSystem<FloatType> system = solver.eigenSystem(M);
		if (solverType == TYPE_NR) {
			system.sortByAbsValue();
		}
		return system;
	}

    EigenSystem<FloatType> eigenSystem(const DenseMatrix<FloatType> &M) const
    {
        EigenSystem<FloatType> result(M.rows());
        eigenSystemInternal(M, &result.eigenvectorList()[0], &result.eigenvalues[0]);
        return result;
    }
    void eigenSystem(const DenseMatrix<FloatType> &M, FloatType **eigenvectors, FloatType *eigenvalues) const
    {
        eigenSystemInternal(M, eigenvalues, eigenvectors);
    }
    void eigenTest(const DenseMatrix<FloatType> &M) const
    {
        EigenSystem<FloatType> result(M.rows());
        eigenSystemInternal(M, &result.eigenvectorList()[0], &result.eigenvalues[0]);
        eigenTest(M, &result.eigenvectorList()[0], &result.eigenvalues[0]);
    }
    void eigenTest(const DenseMatrix<FloatType> &M, FloatType **eigenvectors, FloatType *eigenvalues) const
    {
        const unsigned int n = M.rows();
        std::vector<FloatType> eigenvector(n);
        double maxError = 0.0;
        for (UINT eigenIndex = 0; eigenIndex < n; eigenIndex++)
        {
            for (UINT element = 0; element < n; element++)
                eigenvector[element] = eigenvectors[element][eigenIndex];

            std::vector<FloatType> result = DenseMatrix<FloatType>::multiply(M, eigenvector);
            
            double error = 0.0;
            FloatType eigenvalue = eigenvalues[eigenIndex];
            for (UINT element = 0; element < n; element++)
                error += fabs(eigenvector[element] * eigenvalue - result[element]);
            
            maxError = std::max(maxError, error);
        }
        std::cout << "max eigenvector error: " << maxError << std::endl;
    }

private:
    virtual void eigenSystemInternal(const DenseMatrix<FloatType> &M, FloatType **eigenvectors, FloatType *eigenvalues) const = 0;
};

typedef EigenSolver<float> EigenSolverf;
typedef EigenSolver<double> EigenSolverd;


template<class FloatType> class EigenSolverVTK : public EigenSolver<FloatType>
{
public:
    void eigenSystemInternal(const DenseMatrix<FloatType> &M, FloatType **eigenvectors, FloatType *eigenvalues) const;
};

template<class FloatType> class EigenSolverNR : public  EigenSolver<FloatType>
{
public:
	void eigenSystemInternal(const DenseMatrix<FloatType> &M, FloatType **eigenvectors, FloatType *eigenvalues) const {

		MLIB_ASSERT(M.square());

		const unsigned int n = M.rows();
		const FloatType* matrix = M.ptr();

		// Use jacobi's method:
		// Build matrix NR-style:
		FloatType** CV = new FloatType*[n+1];
		for (unsigned int i1 = 0; i1 < n+1; i1++)	CV[i1] = new FloatType[n+1];

		FloatType* lambda = new FloatType[n+1];

		FloatType** v = new FloatType*[n+1];
		for(unsigned int i1 = 0; i1 < n+1; i1++)	v[i1] = new FloatType[n+1];


		for (unsigned int i = 0; i < n; i++)
			for (unsigned int j = 0; j < n; j++)
				CV[i+1][j+1] = (FloatType)matrix[i+n*j];

		int num_of_required_jabobi_rotations;

		if (!jacobi(CV, n, lambda, v, &num_of_required_jabobi_rotations)) {
			for(unsigned int i1 = 0; i1 < n+1; i1++) {
				delete[] v[i1];
				delete[] CV[i1];
			}
			delete[] v;
			delete[] CV;
			delete[] lambda;
			//return false;
		}


		for (unsigned int i = 0; i < n; i++) {
			eigenvalues[i] = lambda[i+1];
			for (unsigned int j = 0; j < n; j++) {
				eigenvectors[i][j] = v[i+1][j+1];
			}
		}

		//point3d<FloatType> vec1(v[1][1], v[2][1], v[3][1]);
		//point3d<FloatType> vec2(v[1][2], v[2][2], v[3][2]);
		//point3d<FloatType> vec3(v[1][3], v[2][3], v[3][3]);

		//// Sort eigenvectors such the ev[0] is the smallest...
		//if (fabs(lambda[1]) < fabs(lambda[2]) && fabs(lambda[1]) < fabs(lambda[3])) {
		//	ev2 = vec1;
		//	lambda_2 = lambda[1];
		//	if (fabs(lambda[2]) < fabs(lambda[3])) {
		//		ev1 = vec2;
		//		ev0 = vec3;
		//		lambda_1 = lambda[2];
		//		lambda_0 = lambda[3];
		//	} else {
		//		ev0 = vec2;
		//		ev1 = vec3;
		//		lambda_1 = lambda[3];
		//		lambda_0 = lambda[2];
		//	}
		//} else if (fabs(lambda[2]) < fabs(lambda[1]) && fabs(lambda[2]) < fabs(lambda[3])) {
		//	ev2 = vec2;
		//	lambda_2 = lambda[2];
		//	if (fabs(lambda[1]) < fabs(lambda[3])) {
		//		ev1 = vec1;
		//		ev0 = vec3;
		//		lambda_1 = lambda[1];
		//		lambda_0 = lambda[3];
		//	} else {
		//		ev0 = vec1;
		//		ev1 = vec3;
		//		lambda_1 = lambda[3];
		//		lambda_0 = lambda[1];
		//	}
		//} else { // lambda[3] smallest!
		//	ev2 = vec3;
		//	lambda_2 = lambda[3];
		//	if (fabs(lambda[1]) < fabs(lambda[2])) {
		//		ev1 = vec1;
		//		ev0 = vec2;
		//		lambda_1 = lambda[1];
		//		lambda_0 = lambda[2];
		//	} else {
		//		ev0 = vec1;
		//		ev1 = vec2;
		//		lambda_1 = lambda[2];
		//		lambda_0 = lambda[1];
		//	}
		//}


		for(unsigned int i1 = 0; i1 < n+1; i1++) {
			delete[] v[i1];
			delete[] CV[i1];
		}
		delete[] v;
		delete[] CV;
		delete[] lambda;

		//return true;	// Use jacobi's method
	}
};

}  // namespace ml

#include "eigenSolver.inl"

#endif  // CORE_MATH_EIGENSOLVER_H_
