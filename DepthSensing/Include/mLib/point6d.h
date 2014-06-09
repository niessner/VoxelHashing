
#ifndef CORE_MATH_POINT6D_H_
#define CORE_MATH_POINT6D_H_

#include "point4d.h"

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cassert>

namespace ml {

//! 6D vector.
template <class T>
class point6d : public BinaryDataSerialize< point6d<T> >
{
public:
	explicit point6d(T v) {
		array[0] = array[1] = array[2] = array[3] = array[4] = array[5] = v;
	}

	point6d() {
		array[0] = array[1] = array[2] = array[3] = array[4] = array[5] = 0;
	}

	point6d(T x, T y, T z, T xx, T yy, T zz) {
		array[0] = x;
		array[1] = y;
		array[2] = z;
		array[3] = xx;
		array[4] = yy;
		array[5] = zz;
	}

	template <class U>
	point6d(const point6d<U>& other) {
		array[0] = (T)other.array[0];
		array[1] = (T)other.array[1];
		array[2] = (T)other.array[2];
		array[3] = (T)other.array[3];
		array[4] = (T)other.array[4];
		array[5] = (T)other.array[5];
	}

	point6d(const point6d& other) {
		array[0] = other.array[0];
		array[1] = other.array[1];
		array[2] = other.array[2];
		array[3] = other.array[3];
		array[4] = other.array[4];
		array[5] = other.array[5];
	}

	point6d(const T* other) {
		array[0] = other[0];
		array[1] = other[1];
		array[2] = other[2];
		array[3] = other[3];
		array[4] = other[4];
		array[5] = other[5];
	}

	inline const point6d<T>& operator=(const point6d& other) {
		array[0] = other.array[0];
		array[1] = other.array[1];
		array[2] = other.array[2];
		array[3] = other.array[3];
		array[4] = other.array[4];
		array[5] = other.array[5];
		return *this;
	}

	inline const point6d<T>& operator=(T other) {
		array[0] = other;
		array[1] = other;
		array[2] = other;
		array[3] = other;
		array[4] = other;
		array[5] = other;
		return *this;
	}

	inline bool operator!=(const point6d& other) const {
		return !(*this == other);
	}

	inline point6d<T> operator-() const {
		return point6d<T>(-array[0], -array[1], -array[2], -array[3], -array[4], -array[5]);
	}

	inline point6d<T> operator+(const point6d& other) const {
		return point6d<T>(array[0]+other.array[0], array[1]+other.array[1], array[2]+other.array[2],
						  array[3]+other.array[3], array[4]+other.array[4], array[5]+other.array[5]);
	}

	inline point6d<T> operator+(T val) const {
		return point6d<T>(array[0]+val, array[1]+val, array[2]+val, array[3]+val, array[4]+val, array[5]+val);
	}

	inline void operator+=(const point6d& other) {
		array[0] += other.array[0];
		array[1] += other.array[1];
		array[2] += other.array[2];
		array[3] += other.array[3];
		array[4] += other.array[4];
		array[5] += other.array[5];
	}

	inline void operator-=(const point6d& other) {
		array[0] -= other.array[0];
		array[1] -= other.array[1];
		array[2] -= other.array[2];
		array[3] -= other.array[3];
		array[4] -= other.array[4];
		array[5] -= other.array[5];
	}

	inline void operator+=(T val) {
		array[0] += val;
		array[1] += val;
		array[2] += val;
		array[3] += val;
		array[4] += val;
		array[5] += val;
	}

	inline void operator-=(T val) {
		array[0] -= val;
		array[1] -= val;
		array[2] -= val;
		array[3] -= val;
		array[4] += val;
		array[5] += val;
	}

	inline void operator*=(T val) {
		array[0] *= val;
		array[1] *= val;
		array[2] *= val;
		array[3] *= val;
		array[4] *= val;
		array[5] *= val;
	}

	inline void operator/=(T val) {

		T inv_val = ((T)1)/(val);

		array[0] *= inv_val;
		array[1] *= inv_val;
		array[2] *= inv_val;
		array[3] *= inv_val;
		array[4] *= inv_val;
		array[5] *= inv_val;
	}

	inline point6d<T> operator*(T val) const {
		return point6d<T>(array[0]*val, array[1]*val, array[2]*val, array[3]*val, array[4]*val, array[5]*val);
	}

	inline point6d<T> operator/(T val) const {
		return point6d<T>(array[0]/val, array[1]/val, array[2]/val, array[3]/val, array[4]/val, array[5]/val);
	}

	//! dot product
	inline T operator|(const point6d& other) const {
		return (array[0]*other.array[0] + array[1]*other.array[1] + array[2]*other.array[2] + array[3]*other.array[3] + array[4]*other.array[4] + array[5]*other.array[5]);
	}

	inline point6d<T> operator-(const point6d& other) const {
		return point6d<T>(array[0]-other.array[0], array[1]-other.array[1], array[2]-other.array[2], array[3]-other.array[3], array[4]-other.array[4], array[5]-other.array[5]);
	}

	inline point6d<T> operator-(T val) const {
		return point6d<T>(array[0]-val, array[1]-val, array[2]-val, array[3]-val, array[4]-val, array[5]-val);
	}

	inline bool operator==(const point6d& other) const {
		if ((array[0] == other.array[0]) && (array[1] == other.array[1]) && (array[2] == other.array[2])
			&& (array[3] == other.array[3]) && (array[4] == other.array[4]) && (array[5] == other.array[5]))
			return true;

		return false;
	}

	inline T lengthSq() const {
		return (array[0]*array[0] + array[1]*array[1] + array[2]*array[2] + array[3]*array[3] + array[4]*array[4] + array[5]*array[5]);
	}

	inline T length() const {
		return sqrt(lengthSq());
	}

	static T distSq(const point6d& v0, const point6d& v1) {
		return (v0-v1).lengthSq();
	}

	static T dist(const point6d& v0, const point6d& v1) {
		return (v0-v1).length();
	}

	~point6d(void) {};

	inline void print() const {
		Console::log() << "(" << array[0] << " / " << array[1] << " / " << array[2] << " / " << array[3] << " / " << array[4] << " / " << array[5] << ")" << std::endl;
	}


	const T& operator[](unsigned int i) const {
		assert(i < 6);
		return array[i];
	}

	T& operator[](unsigned int i) {
		assert(i < 6);
		return array[i];

	}

	inline void normalize() {
		T inv = (T)1/length();
		array[0] *= inv;
		array[1] *= inv;
		array[2] *= inv;
		array[3] *= inv;
		array[4] *= inv;
		array[5] *= inv;
	}

	inline point6d<T> getNormalized() const {
		T val = (T)1.0 / length();
		return point6d<T>(array[0] * val, array[1] * val, array[2] * val, array[3] * val, array[4] * val, array[5] * val);
	}

	inline T* ptr() {
		return &array[0];
	}

	inline std::vector<T> toStdVector() const {
		std::vector<T> result(6);
		result[0] = x;
		result[1] = y;
		result[2] = z;
		result[3] = xx;
		result[4] = yy;
		result[5] = zz;
		return result;
	}

	inline point1d<T> getPoint1d() const {
		return point1d<T>(x);
	}
	inline point2d<T> getPoint2d() const {
		return point2d<T>(x,y);
	}
	inline point3d<T> getPoint3d() const {
		return point3d<T>(x,y,z);
	}
	inline point4d<T> getPoint4d() const {
		return point4d<T>(x,y,z,w);
	}

	union {
		struct {
			T x,y,z, xx, yy, zz;	// standard names for components
		};
		T array[6];					// array access
	};
};

//! operator for scalar * vector
template <class T>
inline point6d<T> operator*(T s, const point6d<T>& v)
{
	return v * s;
}
template <class T>
inline point6d<T> operator/(T s, const point6d<T>& v)
{
	return v / s;
}
template <class T>
inline point6d<T> operator+(T s, const point6d<T>& v)
{
	return v + s;
}
template <class T>
inline point6d<T> operator-(T s, const point6d<T>& v)
{
	return v - s;
}

namespace math {
	template<class T>
	inline point6d<int> sign(const point6d<T>& v) {
		return point6d<int>(sign(v.x), sign(v.y), sign(v.z), sign(v.xx), sign(v.yy), sign(v.zz));
	}
}

//! write a point6d to a stream
template <class T> 
inline std::ostream& operator<<(std::ostream& s, const point6d<T>& v)
{ return (s << v[0] << " / " << v[1] << " / " << v[2] << " / " << v[3] << " / " << v[4] << " / " << v[5]);}

//! read a point6d from a stream
template <class T> 
inline std::istream& operator>>(std::istream& s, point6d<T>& v)
{ return (s >> v[0] >> v[1] >> v[2] >> v[3] >> v[4] >> v[5]); }


typedef point6d<double> vec6d;
typedef point6d<float> vec6f;
typedef point6d<int> vec6i;
typedef point6d<unsigned int> vec6ui;
typedef point6d<unsigned char> vec6uc;

}  // namespace ml

#endif  // CORE_MATH_POINT6D_H_
