
#ifndef CORE_MATH_POINT3D_H_
#define CORE_MATH_POINT3D_H_


#include <iostream>
#include <cmath>

#include "point2d.h"

namespace ml {

//! 3D vector.
template <class T>
class point3d : public BinaryDataSerialize< point3d<T> >
{
public:
	explicit point3d(T v) {
		array[0] = array[1] = array[2] = v;
	}

	point3d() {
		array[0] = array[1] = array[2] = 0;
	}

	point3d(T x, T y, T z) {
		array[0] = x;
		array[1] = y;
		array[2] = z;
	}

	template <class U>
	point3d(const point3d<U>& other) {
		array[0] = (T)other.array[0];
		array[1] = (T)other.array[1];
		array[2] = (T)other.array[2];
	}

    explicit point3d(const point2d<T>& other, T z) {
		array[0] = other.array[0];
		array[1] = other.array[1];
		array[2] = z;
	}

	point3d(const point3d& other) {
		array[0] = other.array[0];
		array[1] = other.array[1];
		array[2] = other.array[2];
	}

	point3d(const T* other) {
		array[0] = other[0];
		array[1] = other[1];
		array[2] = other[2];
	}

	inline const point3d<T>& operator=(const point3d& other) {
		array[0] = other.array[0];
		array[1] = other.array[1];
		array[2] = other.array[2];
		return *this;
	}

	inline const point3d<T>& operator=(T other) {
		array[0] = other;
		array[1] = other;
		array[2] = other;
		return *this;
	}

	inline point3d<T> operator-() const {
		return point3d<T>(-array[0], -array[1], -array[2]);
	}

	inline point3d<T> operator+(const point3d& other) const {
		return point3d<T>(array[0]+other.array[0], array[1]+other.array[1], array[2]+other.array[2]);
	}

	inline point3d<T> operator+(T val) const {
		return point3d<T>(array[0]+val, array[1]+val, array[2]+val);
	}

	inline void operator+=(const point3d& other) {
		array[0] += other.array[0];
		array[1] += other.array[1];
		array[2] += other.array[2];
	}

	inline void operator-=(const point3d& other) {
		array[0] -= other.array[0];
		array[1] -= other.array[1];
		array[2] -= other.array[2];
	}

	inline void operator+=(T val) {
		array[0] += val;
		array[1] += val;
		array[2] += val;
	}

	inline void operator-=(T val) {
		array[0] -= val;
		array[1] -= val;
		array[2] -= val;
	}

	inline void operator*=(T val) {
		array[0] *= val;
		array[1] *= val;
		array[2] *= val;
	}

	inline void operator/=(T val) {
		T inv = (T)1 / val;
		array[0] *= inv;
		array[1] *= inv;
		array[2] *= inv;
	}

	inline point3d<T> operator*(T val) const {
		return point3d<T>(array[0]*val, array[1]*val, array[2]*val);
	}

	inline point3d<T> operator/(T val) const {
		return point3d<T>(array[0]/val, array[1]/val, array[2]/val);
	}

	//! Cross product
	inline point3d<T> operator^(const point3d& other) const {
		return point3d<T>(array[1]*other.array[2] - array[2]*other.array[1], array[2]*other.array[0] - array[0]*other.array[2], array[0]*other.array[1] - array[1]*other.array[0]);
	}

	//! Dot product
	inline T operator|(const point3d& other) const {
		return (array[0]*other.array[0] + array[1]*other.array[1] + array[2]*other.array[2]);
	}

	static inline T dot(const point3d& l, const point3d& r) {
		return(l.array[0] * r.array[0] + l.array[1] * r.array[1] + l.array[2] * r.array[2]);
	}

    static inline point3d cross(const point3d& l, const point3d& r) {
        return (l ^ r);
    }

	inline point3d<T> operator-(const point3d& other) const {
		return point3d<T>(array[0]-other.array[0], array[1]-other.array[1], array[2]-other.array[2]);
	}

	inline point3d<T> operator-(T val) const {
		return point3d<T>(array[0]-val, array[1]-val, array[2]-val);
	}

	inline bool operator==(const point3d& other) const {
		if ((array[0] == other.array[0]) && (array[1] == other.array[1]) && (array[2] == other.array[2]))
			return true;

		return false;
	}

	inline bool operator!=(const point3d& other) const {
		return !(*this == other);
	}


	inline T lengthSq() const {
		return (array[0]*array[0] + array[1]*array[1] + array[2]*array[2]);
	}

	inline T length() const {
		return sqrt(lengthSq());
	}

	static T distSq(const point3d& v0, const point3d& v1) {
		return ((v0.array[0]-v1.array[0])*(v0.array[0]-v1.array[0]) + (v0.array[1]-v1.array[1])*(v0.array[1]-v1.array[1]) + (v0.array[2]-v1.array[2])*(v0.array[2]-v1.array[2]));
	}

	static T dist(const point3d& v0, const point3d& v1) {
		return sqrt((v0.array[0]-v1.array[0])*(v0.array[0]-v1.array[0]) + (v0.array[1]-v1.array[1])*(v0.array[1]-v1.array[1]) + (v0.array[2]-v1.array[2])*(v0.array[2]-v1.array[2]));
	}


	inline operator T*() {
		return array;
	}
	
	inline operator const T*() const {
		return array;
	}

	inline void print() const {
		Console::log() << "(" << array[0] << " / " << array[1] << " / " << array[2] << ")" << std::endl;
	}

	const T& operator[](int i) const {
		assert(i < 3);
		return array[i];
	}

	T& operator[](int i) {
		assert(i < 3);
		return array[i];
	}

	inline void normalize() {
		T val = (T)1.0 / length();
		array[0] *= val;
		array[1] *= val;
		array[2] *= val;
	}

	inline point3d<T> getNormalized() const {
		T val = (T)1.0 / length();
		return point3d<T>(array[0] * val, array[1] * val, array[2] * val);
	}

    // returns the angle between two vectors *in degrees*
    static T angleBetween(const point3d<T> &v0, const point3d<T> &v1) {
        T l0 = v0.length();
        T l1 = v1.length();
        if(l0 <= 0.0f || l1 <= 0.0f)
            return 0.0f;
        else
            return math::radiansToDegrees(acosf(math::clamp(point3d<T>::dot(v0, v1) / l0 / l1, -1.0f, 1.0f)));
    }

	inline T* ptr() {
		return &array[0];
	}

	inline std::vector<T> toStdVector() const {
		std::vector<T> result(3);
		result[0] = x;
		result[1] = y;
		result[2] = z;
		return result;
	}

	static const point3d<T> origin;
	static const point3d<T> eX;
	static const point3d<T> eY;
	static const point3d<T> eZ;

	inline point1d<T> getPoint1d() const {
		return point1d<T>(x);
	}
	inline point2d<T> getPoint2d() const {
		return point2d<T>(x,y);
	}

	union {
		struct {
			T x,y,z;    // standard names for components
		};
		struct {
			T r,g,b;	// colors
		};
		T array[3];     // array access
	};
};

//! operator for scalar * vector
template <class T>
inline point3d<T> operator*(T s, const point3d<T>& v) {
	return v * s;
}
template <class T>
inline point3d<T> operator/(T s, const point3d<T>& v)
{
	return point3d<T>(s/v.x, s/v.y, s/v.z);
}
template <class T>
inline point3d<T> operator+(T s, const point3d<T>& v)
{
	return v + s;
}
template <class T>
inline point3d<T> operator-(T s, const point3d<T>& v)
{
	return point3d<T>(s-v.x, s-v.y, s-v.z);
}
 
namespace math {
	template<class T>
	inline point3d<int> sign(const point3d<T>& v) {
		return point3d<int>(sign(v.x), sign(v.y), sign(v.z));
	}
}


//! write a point3d to a stream
template <class T> 
inline std::ostream& operator<<(std::ostream& s, const point3d<T>& v) {
  return (s << v[0] << " " << v[1] << " " << v[2]);
}

//! read a point3d from a stream
template <class T> 
inline std::istream& operator>>(std::istream& s, point3d<T>& v) {
  return (s >> v[0] >> v[1] >> v[2]);
}

typedef point3d<double> vec3d;
typedef point3d<float> vec3f;
typedef point3d<int> vec3i;
typedef point3d<short> vec3s;
typedef point3d<unsigned int> vec3ui;
typedef point3d<unsigned char> vec3uc;
typedef point3d<UINT64> vec3ul;
typedef point3d<INT64> vec3l;


template<> const vec3f vec3f::origin(0.0f, 0.0f, 0.0f);
template<> const vec3f vec3f::eX(1.0f, 0.0f, 0.0f);
template<> const vec3f vec3f::eY(0.0f, 1.0f, 0.0f);
template<> const vec3f vec3f::eZ(0.0f, 0.0f, 1.0f);

template<> const vec3d vec3d::origin(0.0, 0.0, 0.0);
template<> const vec3d vec3d::eX(1.0, 0.0, 0.0);
template<> const vec3d vec3d::eY(0.0, 1.0, 0.0);
template<> const vec3d vec3d::eZ(0.0, 0.0, 1.0);


namespace math {
	template<class T>
	inline vec3i round(const point3d<T>& f) {
		return vec3i(round(f.x), round(f.y), round(f.z));
	}
	template<class T>
	inline vec3i ceil(const point3d<T>& f) {
		return vec3i(ceil(f.x), ceil(f.y), ceil(f.z));
	}
	template<class T>
	inline vec3i floor(const point3d<T>& f) {
		return vec3i(floor(f.x), floor(f.y), floor(f.z));
	}
	template<class T>
	inline point3d<T> abs(const point3d<T>& p) {
		return point3d<T>(abs(p.x), abs(p.y), abs(p.z));
	}
    template<class T>
    inline point3d<T> max(const point3d<T>& p, T v) {
        return point3d<T>( std::max(p.x, v),
                           std::max(p.y, v),
                           std::max(p.z, v));
    }
	template<class T>
	inline point3d<T> max(const point3d<T>& p, const point3d<T>& v) {
		return point3d<T>(
			std::max(p.x, v.x),
			std::max(p.y, v.y),
			std::max(p.z, v.z));
	}
	template<class T>
	inline point3d<T> min(const point3d<T>& p, T v) {
		return point3d<T>(	std::min(p.x, v),
							std::min(p.y, v),
							std::min(p.z, v));
	}
	template<class T>
	inline point3d<T> min(const point3d<T>& p, const point3d<T>& v) {
		return point3d<T>(
			std::min(p.x, v.x),
			std::min(p.y, v.y),
			std::min(p.z, v.z));
	}


    template<class T>
    inline T triangleArea(const point3d<T>& v0, const point3d<T>& v1, const point3d<T>& v2) {
        return ((v1 - v0) ^ (v2 - v0)).length() * (T)0.5;
    }

    template<class T>
    inline T triangleArea(const point2d<T>& v0, const point2d<T>& v1, const point2d<T>& v2) {
        return triangleArea(point3d<T>(v0, 0.0f),
                            point3d<T>(v1, 0.0f),
                            point3d<T>(v2, 0.0f));
    }

    template<class T>
    inline point3d<T> triangleNormal(const point3d<T>& v0, const point3d<T>& v1, const point3d<T>& v2) {
        return ((v1 - v0) ^ (v2 - v0)).getNormalized();
    }
}

}  // namespace ml

#endif  // CORE_MATH_POINT3D_H_
