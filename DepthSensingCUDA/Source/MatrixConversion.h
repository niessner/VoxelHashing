#pragma once

#include "Eigen.h"
#include "d3dx9math.h"
#include "cuda_SimpleMatrixUtil.h"

namespace MatrixConversion
{
	//static D3DXMATRIX EigToMat(const Eigen::Matrix4f& mat)
	//{
	//	D3DXMATRIX m(mat.data());
	//	D3DXMATRIX res; D3DXMatrixTranspose(&res, &m);

	//	return res;
	//}
	//static Eigen::Matrix4f MatToEig(const D3DXMATRIX& mat)
	//{
	//	return Eigen::Matrix4f((float*)mat.m).transpose();
	//}

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
	


	// dx/cuda conversion
	static vec3f toMlib(const D3DXVECTOR3& v) {
		return vec3f(v.x, v.y, v.z);
	}
	static vec4f toMlib(const D3DXVECTOR4& v) {
		return vec4f(v.x, v.y, v.z, v.w);
	}
	static mat4f toMlib(const D3DXMATRIX& m) {
		mat4f c((const float*)&m);
		return c.getTranspose();
	}
	static D3DXVECTOR3 toDX(const vec3f& v) {
		return D3DXVECTOR3(v.x, v.y, v.z);
	}
	static D3DXVECTOR4 toDX(const vec4f& v) {
		return D3DXVECTOR4(v.x, v.y, v.z, v.w);
	}
	static D3DXMATRIX toDX(const mat4f& m) {
		D3DXMATRIX c((const float*)m.ptr());
		D3DXMatrixTranspose(&c, &c);
		return c;
	}

	static mat4f toMlib(const float4x4& m) {
		return mat4f(m.ptr());
	}
	static vec4f toMlib(const float4& v) {
		return vec4f(v.x, v.y, v.z, v.w);
	}
	static vec3f toMlib(const float3& v) {
		return vec3f(v.x, v.y, v.z);
	}
	static vec4i toMlib(const int4& v) {
		return vec4i(v.x, v.y, v.z, v.w);
	}
	static vec3i toMlib(const int3& v) {
		return vec3i(v.x, v.y, v.z);
	}
	static float4x4 toCUDA(const mat4f& m) {
		return float4x4(m.ptr());
	}
	static float4x4 toCUDA(const Eigen::Matrix4f& mat) {
		return float4x4(mat.data()).getTranspose();
	}

	static float4 toCUDA(const vec4f& v) {
		return make_float4(v.x, v.y, v.z, v.w);
	}
	static float3 toCUDA(const vec3f& v) {
		return make_float3(v.x, v.y, v.z);
	}
	static int4 toCUDA(const vec4i& v) {
		return make_int4(v.x, v.y, v.z, v.w);
	}
	static int3 toCUDA(const vec3i& v) {
		return make_int3(v.x, v.y, v.z);
	}


	//doesnt really work
	//template<class FloatType>
	//point3d<FloatType> eulerAngles(const Matrix3x3<FloatType>& r) {
	//	point3d<FloatType> res;

	//	//check for gimbal lock
	//	if (r(0,2) == (FloatType)-1.0) {
	//		FloatType x = 0; //gimbal lock, value of x doesn't matter
	//		FloatType y = (FloatType)M_PI / 2;
	//		FloatType z = x + atan2(r(1,0), r(2,0));
	//		res = point3d<FloatType>(x, y, z);
	//	} else if (r(0,2) == (FloatType)1.0) {
	//		FloatType x = 0;
	//		FloatType y = -(FloatType)M_PI / 2;
	//		FloatType z = -x + atan2(-r(1,0), -r(2,0));
	//		res = point3d<FloatType>(x, y, z);
	//	} else { //two solutions exist
	//		FloatType x1 = -asin(r(0,2));
	//		FloatType x2 = (FloatType)M_PI - x1;

	//		FloatType y1 = atan2(r(1,2) / cos(x1), r(2,2) / cos(x1));
	//		FloatType y2 = atan2(r(1,2) / cos(x2), r(2,2) / cos(x2));

	//		FloatType z2 = atan2(r(0,1) / cos(x2), r(0,0) / cos(x2));
	//		FloatType z1 = atan2(r(0,1) / cos(x1), r(0,0) / cos(x1));

	//		//choose one solution to return
	//		//for example the "shortest" rotation
	//		if ((std::abs(x1) + std::abs(y1) + std::abs(z1)) <= (std::abs(x2) + std::abs(y2) + std::abs(z2))) {
	//			res = point3d<FloatType>(x1, y1, z1);
	//		} else {
	//			res = point3d<FloatType>(x2, y2, z2);
	//		}
	//	}

	//	res.x = math::radiansToDegrees(-res.x);
	//	res.y = math::radiansToDegrees(-res.y);
	//	res.z = math::radiansToDegrees(-res.z);

	//	return res;
	//}
}
