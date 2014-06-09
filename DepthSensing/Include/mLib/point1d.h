
#ifndef CORE_MATH_POINT1D_H_
#define CORE_MATH_POINT1D_H_

#include <iostream>
#include <cmath>
#include <iostream>
#include <cassert>

namespace ml {

//! 1D vector (I know it's a joke, but we need it for compatibility reasons)
template <class T>
class point1d : public BinaryDataSerialize< point1d<T> >
{
public:
	point1d(T v) {
		array[0] = v;
	}

	point1d() {
		array[0] = 0;
	}

	template <class U>
	point1d(const point1d<U>& other) {
		array[0] = (T)other.array[0];
	}

	point1d(const point1d& other) {
		array[0] = other.array[0];
	}

	inline const point1d<T>& operator=(const point1d& other) {
		array[0] = other.array[0];
		return *this;
	}

	inline point1d<T> operator-() const {
		return point1d<T>(-array[0]);
	}

	inline point1d<T> operator+(const point1d& other) const {
		return point1d<T>(array[0]+other.array[0]);
	}

	inline point1d<T> operator+(T val) const {
		return point1d<T>(array[0]+val);
	}

	inline void operator+=(const point1d& other) {
		array[0] += other.array[0];
	}

	inline void operator-=(const point1d& other) {
		array[0] -= other.array[0];
	}

	inline void operator+=(T val) {
		array[0] += val;
	}

	inline void operator-=(T val) {
		array[0] -= val;
	}

	inline void operator*=(T val) {
		array[0] *= val;
	}

	inline void operator/=(T val) {
		array[0] /= val;
	}

	inline point1d<T> operator*(T val) const {
		return point1d<T>(array[0]*val);
	}

	inline point1d<T> operator/(T val) const {
		return point1d<T>(array[0]/val);
	}

	inline point1d<T> operator-(const point1d& other) const {
		return point1d<T>(array[0]-other.array[0]);
	}

	inline point1d<T> operator-(T val) const {
		return point1d<T>(array[0]-val);
	}

	inline bool operator==(const point1d& other) const {
		if ((array[0] == other.array[0]))
			return true;

		return false;
	}

	inline bool operator!=(const point1d& other) const {
		return !(*this == other);
	}


	//! dot product
	inline T operator|(const point1d& other) const {
		return (array[0]*other.array[0]);
	}

	inline T& operator[](unsigned int i) {
		assert(i < 1);
		return array[i];
	}

	inline const T& operator[](unsigned int i) const {
		assert(i < 1);
		return array[i];
	}

	~point1d(void) {};

	inline T lengthSq() const {
		return (array[0]*array[0]);
	}

	inline T length() const {
		return array[0];
	}

	static T distSq(const point1d& v0, const point1d& v1) {
		return (v0.array[0] - v1.array[1])*(v0.array[0] - v1.array[1]);
	}

	static T dist(const point1d& v0, const point1d& v1) {
		return std::abs(v0.array[0] - v1.array[1]);
	}

	inline point1d getNormalized() const {
		return point1d<T>();
	}

	inline void normalize() const {
		array[0] /= length();
	}

	inline void print() const {
		Console::log() << "(" << array[0] << ")" << std::endl;
	}

	inline T* ptr() {
		return &array[0];
	}

	inline std::vector<T> toStdVector() const {
		std::vector<T> result(1);
		result[0] = x;
		return result;
	}

	static const point1d<T> origin;
	static const point1d<T> eX;
	static const point1d<T> eY;

	union {
		struct {
			T x;        // standard names for components
		};
		struct {
			T r;		// colors
		};
		T array[1];     // array access
	};
};

//! operator for scalar * vector
template <class T>
inline point1d<T> operator*(T s, const point1d<T>& v)
{
	return v * s;
}
template <class T>
inline point1d<T> operator/(T s, const point1d<T>& v)
{
	return v / s;
}
template <class T>
inline point1d<T> operator+(T s, const point1d<T>& v)
{
	return v + s;
}
template <class T>
inline point1d<T> operator-(T s, const point1d<T>& v)
{
	return v - s;
}

namespace math {
	template<class T>
	inline point1d<int> sign(const point1d<T>& v) {
		return point1d<int>(sign(v.x));
	}
}


//! write a point1d to a stream
template <class T> inline std::ostream& operator<<(std::ostream& s, const point1d<T>& v)
{ return (s << v[0]);}

//! read a point1d from a stream
template <class T> inline std::istream& operator>>(std::istream& s, point1d<T>& v)
{ return (s >> v[0]); }


typedef point1d<double> vec1d;
typedef point1d<float> vec1f;
typedef point1d<int> vec1i;
typedef point1d<unsigned int> vec1ui;
typedef point1d<unsigned char> vec1uc;


template<> const vec1f vec1f::origin(0.0f);
template<> const vec1f vec1f::eX(1.0f);

template<> const vec1d vec1d::origin(0.0);
template<> const vec1d vec1d::eX(1.0);

}  // namespace ml

#endif  // CORE_MATH_POINT2D_H_
