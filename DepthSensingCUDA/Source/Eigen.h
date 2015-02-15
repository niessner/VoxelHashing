#pragma once

#ifndef _EIGEN_
#define _EIGEN_

#include <Eigen/Dense>
#include <Eigen/StdVector>

typedef Eigen::Matrix<float, 6, 7> Matrix6x7f;
typedef Eigen::Matrix<float, 6, 6> Matrix6x6f;
typedef Eigen::Matrix<float, 4, 4> Matrix4x4f;
typedef Eigen::Matrix<float, 3, 3> Matrix3x3f;

typedef Eigen::Matrix<float, 6, 1> Vector6f;
typedef Eigen::Matrix<float, 4, 1> Vector4f;

EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Vector3f)

using namespace Eigen;

template<typename T,unsigned int n,unsigned m>
std::ostream &operator<<(std::ostream &out, Matrix<T,n,m> &other)
{
	for(int i=0; i<other.rows(); i++) {
		out << other(i,0);
		for(int j=1; j<other.cols(); j++) {
			out << "\t" << other(i,j);
		}
		out << std::endl;
	}
	return out;
}

#endif // _EIGEN_
