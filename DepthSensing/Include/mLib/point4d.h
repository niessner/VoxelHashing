
#ifndef CORE_MATH_POINT4D_H_
#define CORE_MATH_POINT4D_H_

#include "point3d.h"

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cassert>

namespace ml {

//! 4D vector.
template <class T>
class point4d : public BinaryDataSerialize< point4d<T> >
{
    public:
        explicit point4d(T v) {
            array[0] = array[1] = array[2] = array[3] = v;
        }

        point4d() {
            array[0] = array[1] = array[2] = array[3] = 0;
        }

        point4d(T x, T y, T z, T w) {
            array[0] = x;
            array[1] = y;
            array[2] = z;
            array[3] = w;
        }
		
		template <class U>
		point4d(const point4d<U>& other) {
			array[0] = (T)other.array[0];
			array[1] = (T)other.array[1];
			array[2] = (T)other.array[2];
			array[3] = (T)other.array[3];
		}

        point4d(const point3d<T>& other, T w = (T)1) {
            array[0] = other.array[0];
            array[1] = other.array[1];
            array[2] = other.array[2];
            array[3] = w;
        }

        point4d(const point4d& other) {
            array[0] = other.array[0];
            array[1] = other.array[1];
            array[2] = other.array[2];
            array[3] = other.array[3];
        }

		point4d(const T* other) {
			array[0] = other[0];
			array[1] = other[1];
			array[2] = other[2];
			array[3] = other[3];
		}

        inline const point4d<T>& operator=(const point4d& other) {
            array[0] = other.array[0];
            array[1] = other.array[1];
            array[2] = other.array[2];
            array[3] = other.array[3];
            return *this;
        }

        inline const point4d<T>& operator=(T other) {
            array[0] = other;
            array[1] = other;
            array[2] = other;
            array[3] = other;
            return *this;
        }


        inline point4d<T> operator-() const {
            return point4d<T>(-array[0], -array[1], -array[2], -array[3]);
        }

        inline point4d<T> operator+(const point4d& other) const {
            return point4d<T>(array[0] + other.array[0], array[1] + other.array[1],
                              array[2] + other.array[2], array[3] + other.array[3]);
        }

		inline point4d<T> operator+(T val) const {
			return point4d<T>(array[0]+val, array[1]+val, array[2]+val, array[3]+val);
		}

        inline void operator+=(const point4d& other) {
            array[0] += other.array[0];
            array[1] += other.array[1];
            array[2] += other.array[2];
            array[3] += other.array[3];
        }

        inline void operator-=(const point4d& other) {
            array[0] -= other.array[0];
            array[1] -= other.array[1];
            array[2] -= other.array[2];
            array[3] -= other.array[3];
        }

		inline void operator+=(T val) {
			array[0] += val;
			array[1] += val;
			array[2] += val;
			array[3] += val;
		}

		inline void operator-=(T val) {
			array[0] -= val;
			array[1] -= val;
			array[2] -= val;
			array[3] -= val;
		}

        inline void operator*=(T val) {
            array[0] *= val;
            array[1] *= val;
            array[2] *= val;
            array[3] *= val;
        }

        inline void operator/=(T val) {
            T inv = (T)1 / val;
            array[0] *= inv;
            array[1] *= inv;
            array[2] *= inv;
            array[3] *= inv;
        }

        inline point4d<T> operator*(T val) const {
            return point4d<T>(array[0] * val, array[1]*val, array[2]*val, array[3]*val);
        }

        inline point4d<T> operator/(T val) const {
            T inv = (T)1 / val;
            return point4d<T>(array[0]*inv, array[1]*inv, array[2]*inv, array[3]*inv);
        }

        //! cross product (of .xyz)
        inline point4d<T> operator^(const point4d& other) const {
            return point4d<T>(array[1] * other.array[2] - array[2] * other.array[1],
                              array[2] * other.array[0] - array[0] * other.array[2],
                              array[0] * other.array[1] - array[1] * other.array[0], T(1));
        }

        //! dot product
        inline T operator|(const point4d& other) const {
            return (array[0] * other.array[0] + array[1] * other.array[1] + array[2] *
                    other.array[2] + array[3] * other.array[3]);
        }

        inline point4d<T> operator-(const point4d& other) const {
            return point4d<T>(array[0]-other.array[0], array[1]-other.array[1], array[2]-other.array[2], array[3]-other.array[3]);
        }

		inline point4d<T> operator-(T val) const {
			return point4d<T>(array[0]-val, array[1]-val, array[2]-val, array[3]-val);
		}

        inline bool operator==(const point4d& other) const {
            if ((array[0] == other.array[0]) && (array[1] == other.array[1]) &&
                (array[2] == other.array[2]) && (array[3] == other.array[3]))
            { return true; }

            return false;
        }

		inline bool operator!=(const point4d& other) const {
			return !(*this == other);
		}

        inline T lengthSq() const {
            return (array[0]*array[0] + array[1]*array[1] + array[2]*array[2] + array[3]*array[3]);
        }

        inline T length() const {
            return sqrt(lengthSq());
        }

        static T distSq(const point4d& v0, const point4d& v1) {
            return (
                       (v0.array[0] - v1.array[0]) * (v0.array[0] - v1.array[0]) +
                       (v0.array[1] - v1.array[1]) * (v0.array[1] - v1.array[1]) +
                       (v0.array[2] - v1.array[2]) * (v0.array[2] - v1.array[2]) +
                       (v0.array[3] - v1.array[3]) * (v0.array[3] - v1.array[3])
                   );
        }

        static T dist(const point4d& v0, const point4d& v1)  {
            return (v0 - v1).length();
        }

        ~point4d(void) {};

        void print() const
        {
            Console::log() << "(" << array[0] << " / " << array[1] << " / " << array[2] <<
                           " / " << array[3] << " ) " << std::endl;
        }

        inline const T& operator[](int i) const
        {
            assert(i < 4);
            return array[i];
        }

        inline T& operator[](int i)
        {
            assert(i < 4);
            return array[i];
        }

        inline void normalize()
        {
            T val = (T)1.0 / length();
            array[0] *= val;
            array[1] *= val;
            array[2] *= val;
            array[3] *= val;
        }

        inline point4d<T> getNormalized() const
        {
            T val = (T)1.0 / length();
            return point4d<T>(array[0] * val, array[1] * val, array[2] * val,
                              array[3] * val);
        }

        inline void dehomogenize()
        {
            array[0] /= array[3];
            array[1] /= array[3];
            array[2] /= array[3];
            array[3] /= array[3];
        }


        inline bool isLinearDependent(const point4d& other) const
        {
            T factor = x / other.x;

            if ((std::fabs(x / factor - other.x) + std::fabs(y / factor - other.y) +
                    std::fabs(z / factor - other.z) + std::fabs(w / factor - other.w)) < 0.00001)
            { return true; }
            else
            { return false; }
        }

		inline T* ptr() {
			return &array[0];
		}

		inline std::vector<T> toStdVector() const {
			std::vector<T> result(4);
			result[0] = x;
			result[1] = y;
			result[2] = z;
			result[3] = w;
			return result;
		}

		static const point4d<T> origin;
		static const point4d<T> eX;
		static const point4d<T> eY;
		static const point4d<T> eZ;
		static const point4d<T> eW;

		inline point1d<T> getPoint1d() const {
			return point1d<T>(x);
		}
		inline point2d<T> getPoint2d() const {
			return point2d<T>(x,y);
		}
		inline point3d<T> getPoint3d() const {
			return point3d<T>(x,y,z);
		}

        union
        {
            struct
            {
                T x, y, z, w; // standard names for components
            };
            //struct {
            //  T r,g,b,w;  // colors
            //};
            T array[4];     // array access
        };
};

//! operator for scalar * vector
template <class T>
inline point4d<T> operator*(T s, const point4d<T>& v)
{
    return v * s;
}
template <class T>
inline point4d<T> operator/(T s, const point4d<T>& v)
{
	return v / s;
}
template <class T>
inline point4d<T> operator+(T s, const point4d<T>& v)
{
	return v + s;
}
template <class T>
inline point4d<T> operator-(T s, const point4d<T>& v)
{
	return v - s;
}

namespace math {
	template<class T>
	inline point4d<int> sign(const point4d<T>& v) {
		return point4d<int>(sign(v.x), sign(v.y), sign(v.z), sign(v.w));
	}
}


//! write a point4d to a stream (should be the inverse of read operator; with " ")
template <class T>
inline std::ostream& operator<<(std::ostream& s, const point4d<T>& v)
{ return (s << v[0] << " / " << v[1] << " / " << v[2] << " / " << v[3]);}

//! read a point4d from a stream
template <class T>
inline std::istream& operator>>(std::istream& s, point4d<T>& v)
{ return (s >> v[0] >> v[1] >> v[2] >> v[3]); }


typedef point4d<double> vec4d;
typedef point4d<float> vec4f;
typedef point4d<int> vec4i;
typedef point4d<unsigned int> vec4ui;
typedef point4d<unsigned char> vec4uc;


template<> const vec4f vec4f::origin(0.0f, 0.0f, 0.0f, 0.0f);
template<> const vec4f vec4f::eX(1.0f, 0.0f, 0.0f, 0.0f);
template<> const vec4f vec4f::eY(0.0f, 1.0f, 0.0f, 0.0f);
template<> const vec4f vec4f::eZ(0.0f, 0.0f, 1.0f, 0.0f);
template<> const vec4f vec4f::eW(0.0f, 0.0f, 0.0f, 1.0f);

template<> const vec4d vec4d::origin(0.0, 0.0, 0.0, 0.0);
template<> const vec4d vec4d::eX(1.0, 0.0, 0.0, 0.0);
template<> const vec4d vec4d::eY(0.0, 1.0, 0.0, 0.0);
template<> const vec4d vec4d::eZ(0.0, 0.0, 1.0, 0.0);
template<> const vec4d vec4d::eW(0.0, 0.0, 0.0, 1.0);

}  // namespace ml

#endif  // CORE_MATH_POINT4D_H_
