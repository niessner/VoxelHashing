
#ifndef CORE_MATH_MATHVECTOR_H_
#define CORE_MATH_MATHVECTOR_H_

namespace ml {

/************************************************************************/
/* math vector class (used for instance with sparse matrices)           */
/************************************************************************/

template<class T, class = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
class MathVector;

template<class T>
class MathVector<T> : public std::vector<T>
{
public:
	MathVector() {
	}
	MathVector(size_t size) {
		std::vector<T>::resize(size);
	}
	MathVector(size_t size, const T& val) {
		std::vector<T>::resize(size, val);
	}
	MathVector(const MathVector& v) : std::vector<T>(v) {
	}
	MathVector(MathVector&& v) : std::vector<T>(v) {
	}


	void operator=(const MathVector& v) {
		std::vector<T>::operator =(v);
	}
	void operator=(MathVector&& v) {
		std::vector<T>::operator =(v);
	}

	void operator+=(const MathVector& other) {
		for (size_t i = 0; i < std::vector<T>::size(); i++) {
			(*this)[i] += other[i];
		}
	}
	void operator-=(const MathVector& other) {
		for (size_t i = 0; i < std::vector<T>::size(); i++) {
			(*this)[i] -= other[i];
		}
	}
	void operator*=(const MathVector& other) {
		for (size_t i = 0; i < std::vector<T>::size(); i++) {
			(*this)[i] *= other[i];
		}
	}
	void operator/=(const MathVector& other) {
		for (size_t i = 0; i < std::vector<T>::size(); i++) {
			(*this)[i] /= other[i];
		}
	}
	void operator*=(const T& other) {
		for (size_t i = 0; i < std::vector<T>::size(); i++) {
			(*this)[i] *= other;
		}
	}
	void operator/=(const T& other) {
		for (size_t i = 0; i < std::vector<T>::size(); i++) {
			(*this)[i] /= other;
		}
	}


	MathVector operator+(const MathVector& other) const {
		MathVector res(this->size());
		for (size_t i = 0; i < res.size(); i++) {
			res[i] = (*this)[i] + other[i];
		}
		return res;
	}
	MathVector operator-(const MathVector& other) const {
		MathVector res(this->size());
		for (size_t i = 0; i < res.size(); i++) {
			res[i] = (*this)[i] - other[i];
		}
		return res;
	}
	MathVector operator*(const MathVector& other) const {
		MathVector res(this->size());
		for (size_t i = 0; i < res.size(); i++) {
			res[i] = (*this)[i] * other[i];
		}
		return res;
	}
	MathVector operator/(const MathVector& other) const {
		MathVector res(this->size());
		for (size_t i = 0; i < res.size(); i++) {
			res[i] = (*this)[i] / other[i];
		}
		return res;
	}
	MathVector operator*(const T& other) const {
		MathVector res(this->size());
		for (size_t i = 0; i < res.size(); i++) {
			res[i] = (*this)[i] * other;
		}
		return res;
	}
	MathVector operator/(const T& other) const {
		MathVector res(this->size());
		for (size_t i = 0; i < res.size(); i++) {
			res[i] = (*this)[i] / other;
		}
		return res;
	}



	//! dot product
	T operator|(const MathVector& other) const {
		return dot(*this, other);
	} 

	static T dot(const MathVector& v0, const MathVector& v1) {
		T res = (T)0;
		for (size_t i = 0; i < v0.size(); i++) {
			res += v0[i] * v1[i];
		}
		return res;
	}
};


template<class T>
inline MathVector<T> operator*(const T& v, const MathVector<T>& other)  {
	MathVector<T> res(other.size());
	for (size_t i = 0; i < res.size(); i++) {
		res[i] = other[i] * v;
	}
	return res;
}

template<class T>
inline MathVector<T> operator/(const T& v, const MathVector<T>& other)  {
	MathVector<T> res(other.size());
	for (size_t i = 0; i < res.size(); i++) {
		res[i] = other[i] * v;
	}
	return res;
}

template<class T>
inline std::ostream& operator<<(std::ostream& s, const MathVector<T>& v) {
	s << "vector size " << v.size() << "\n";
	for (size_t i = 0; i < v.size(); i++) {
		s << '\t' << v[i];
		if (i != v.size() - 1) s << '\n';
	}
	return s;
}

}  // namespace ml



#endif  // CORE_MATH_MATHVECTOR_H_
