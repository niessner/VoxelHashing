
#ifndef CORE_MATH_POINT2D_H_
#define CORE_MATH_POINT2D_H_

#include "point1d.h"

#include <iostream>
#include <cmath>
#include <iostream>
#include <cassert>

namespace ml {

//! 2D vector
template <class T>
class point2d : public BinaryDataSerialize< point2d<T> >
{
public:
	explicit point2d(T v) {
		array[0] = array[1] = v;
	}

	point2d() {
		array[0] = array[1] = 0;
	}

	point2d(T x, T y) {
		array[0] = x;
		array[1] = y;
	}

	template <class U>
	point2d(const point2d<U>& other) {
		array[0] = (T)other.array[0];
		array[1] = (T)other.array[1];
	}

	point2d(const point2d& other) {
		array[0] = other.array[0];
		array[1] = other.array[1];
	}

	point2d(const T* other) {
		array[0] = other[0];
		array[1] = other[1];
	}

	inline const point2d<T>& operator=(const point2d& other) {
		array[0] = other.array[0];
		array[1] = other.array[1];
		return *this;
	}

	inline const point2d<T>& operator=(T other) {
		array[0] = other;
		array[1] = other;
		return *this;
	}

	inline point2d<T> operator-() const {
		return point2d<T>(-array[0], -array[1]);
	}

	inline point2d<T> operator+(const point2d& other) const {
		return point2d<T>(array[0]+other.array[0], array[1]+other.array[1]);
	}

	inline point2d<T> operator+(T val) const {
		return point2d<T>(array[0]+val, array[1]+val);
	}

	inline void operator+=(const point2d& other) {
		array[0] += other.array[0];
		array[1] += other.array[1];
	}

	inline void operator-=(const point2d& other) {
		array[0] -= other.array[0];
		array[1] -= other.array[1];
	}

	inline void operator+=(T val) {
		array[0] += val;
		array[1] += val;
	}

	inline void operator-=(T val) {
		array[0] -= val;
		array[1] -= val;
	}

	inline void operator*=(T val) {
		array[0] *= val;
		array[1] *= val;
	}

	inline void operator/=(T val) {
		T inv = (T)1/val;
		array[0] *= inv;
		array[1] *= inv;
	}

	inline point2d<T> operator*(T val) const {
		return point2d<T>(array[0]*val, array[1]*val);
	}

	inline point2d<T> operator/(T val) const {
		T inv = (T)1/val;
		return point2d<T>(array[0]*inv, array[1]*inv);
	}

	inline point2d<T> operator-(const point2d& other) const {
		return point2d<T>(array[0]-other.array[0], array[1]-other.array[1]);
	}

	inline point2d<T> operator-(T val) const {
		return point2d<T>(array[0]-val, array[1]-val);
	}

	inline bool operator==(const point2d& other) const {
		if ((array[0] == other.array[0]) && (array[1] == other.array[1]))
			return true;

		return false;
	}

	inline bool operator!=(const point2d& other) const {
		return !(*this == other);
	}

	//! dot product
	inline T operator|(const point2d& other) const {
		return (array[0]*other.array[0] + array[1]*other.array[1]);
	}

	inline T& operator[](unsigned int i) {
		assert(i < 2);
		return array[i];
	}

	inline const T& operator[](unsigned int i) const {
		assert(i < 2);
		return array[i];
	}

	~point2d(void) {};

	inline bool operator<(const point2d& other) const {
		if ((x < other.x) && (y < other.y))
			return true;

		return false;
	}

	inline T lengthSq() const {
		return (array[0]*array[0] + array[1]*array[1]);
	}

	inline T length() const {
		return sqrt(array[0]*array[0] + array[1]*array[1]);
	}

	static T distSq(const point2d& v0, const point2d& v1) {
		return ((v0.array[0]-v1.array[0])*(v0.array[0]-v1.array[0]) + (v0.array[1]-v1.array[1])*(v0.array[1]-v1.array[1]));
	}

	static T dist(const point2d& v0, const point2d& v1) {
		return sqrt((v0.array[0]-v1.array[0])*(v0.array[0]-v1.array[0]) + (v0.array[1]-v1.array[1])*(v0.array[1]-v1.array[1]));
	}

	inline void normalize() {
		T val = (T)1.0 / length();
		array[0] *= val;
		array[1] *= val;
	}

	inline point2d<T> getNormalized() const {
		T val = (T)1.0 / length();
		return point2d<T>(array[0] * val, array[1] * val);
	}

	inline void print() const {
		Console::log() << "(" << array[0] << " / " << array[1] << ")" << std::endl;
	}

	inline T* ptr() {
		return &array[0];
	}

    inline std::vector<T> toStdVector() const {
        std::vector<T> result(2);
        result[0] = x;
        result[1] = y;
        return result;
    }

	static const point2d<T> origin;
	static const point2d<T> eX;
	static const point2d<T> eY;

	inline point1d<T> getPoint1d() const {
		return point1d<T>(x);
	}

	union {
		struct {
			T x,y;      // standard names for components
		};
		struct {
			T r,g;		// colors
		};
		T array[2];     // array access
	};
};

//! operator for scalar * vector
template <class T>
inline point2d<T> operator*(T s, const point2d<T>& v)
{
	return v * s;
}
template <class T>
inline point2d<T> operator/(T s, const point2d<T>& v)
{
	return v / s;
}
template <class T>
inline point2d<T> operator+(T s, const point2d<T>& v)
{
	return v + s;
}
template <class T>
inline point2d<T> operator-(T s, const point2d<T>& v)
{
	return v - s;
}

namespace math {
	template<class T>
	inline point2d<int> sign(const point2d<T>& v) {
		return point2d<int>(sign(v.x), sign(v.y));
	}
}

//! write a point2d to a stream
template <class T> inline std::ostream& operator<<(std::ostream& s, const point2d<T>& v)
{ return (s << v[0] << " / " << v[1]);}

//! read a point2d from a stream
template <class T> inline std::istream& operator>>(std::istream& s, point2d<T>& v)
{ return (s >> v[0] >> v[1]); }


typedef point2d<double> vec2d;
typedef point2d<float> vec2f;
typedef point2d<int> vec2i;
typedef point2d<unsigned int> vec2ui;
typedef point2d<unsigned char> vec2uc;


template<> const vec2f vec2f::origin(0.0f, 0.0f);
template<> const vec2f vec2f::eX(1.0f, 0.0f);
template<> const vec2f vec2f::eY(0.0f, 1.0f);
					
template<> const vec2d vec2d::origin(0.0, 0.0);
template<> const vec2d vec2d::eX(1.0, 0.0);
template<> const vec2d vec2d::eY(0.0, 1.0);

}  // namespace ml

#endif  // CORE_MATH_POINT2D_H_
