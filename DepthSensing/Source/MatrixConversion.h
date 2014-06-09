#pragma once

/*********************************************************/
/* Helper functions from eigen to and from our format    */
/*********************************************************/

#include "stdafx.h"

#include "Eigen.h"
#include "d3dx9math.h"

namespace MatrixConversion
{
	static mat4f EigToMat(const Eigen::Matrix4f& mat)
	{
		return mat4f(mat.data()).getTranspose();
	}

	static Eigen::Matrix4f MatToEig(const mat4f& mat)
	{
		return Eigen::Matrix4f(mat.ptr()).transpose();
	}

	static Eigen::Vector4f VecH(const Eigen::Vector3f& v)
	{
		return Eigen::Vector4f(v[0], v[1], v[2], 1.0);
	}

	static Eigen::Vector3f VecDH(const Eigen::Vector4f& v)
	{
		return Eigen::Vector3f(v[0]/v[3], v[1]/v[3], v[2]/v[3]);
	}

	static Eigen::Vector3f VecToEig(const vec3f& v)
	{
		return Eigen::Vector3f(v[0], v[1], v[2]);
	}

	static vec3f EigToVec(const Eigen::Vector3f& v)
	{
		return vec3f(v[0], v[1], v[2]);
	}

	static vec3f EigToVec(const D3DXMATRIX v)
	{
		return vec3f(v[0], v[1], v[2]);
	}
}
