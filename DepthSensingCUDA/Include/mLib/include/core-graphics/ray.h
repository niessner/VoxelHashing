
#ifndef CORE_GRAPHICS_RAY_H_
#define CORE_GRAPHICS_RAY_H_

namespace ml {


template<class FloatType>
class Ray
{
public:

    Ray()
    {

    }

	inline Ray(const point3d<FloatType> &o, const point3d<FloatType> &d) {
		m_Origin = o;
		m_Direction = d.getNormalized();
        m_InverseDirection = point3d<FloatType>((FloatType)1.0 / m_Direction.x, (FloatType)1.0 / m_Direction.y, (FloatType)1.0 / m_Direction.z);

		m_Sign.x = (m_InverseDirection.x < (FloatType)0);
		m_Sign.y = (m_InverseDirection.y < (FloatType)0);
		m_Sign.z = (m_InverseDirection.z < (FloatType)0);
	}

	inline point3d<FloatType> getHitPoint(FloatType t) const {
		return m_Origin + t * m_Direction;
	}

	inline const point3d<FloatType>& origin() const {
		return m_Origin;
	}

	inline const point3d<FloatType>& direction() const {
		return m_Direction;
	}

	inline const point3d<FloatType>& inverseDirection() const {
		return m_InverseDirection;
	}

	inline const vec3i& sign() const {
		return m_Sign;
	}

	inline void transform(const Matrix4x4<FloatType>& m) {
		*this = Ray(m * m_Origin,  m.transformNormalAffine(m_Direction));
	}

	inline void rotate(const Matrix3x3<FloatType>& m) {
		*this = Ray(m_Origin, m * m_Direction);
	}

	inline void translate(const point3d<FloatType>& p) {
		*this = Ray(m_Origin + p, m_Direction);
	}
private:
	point3d<FloatType> m_Direction;
	point3d<FloatType> m_InverseDirection;
	point3d<FloatType> m_Origin;

	vec3i m_Sign;
};

template<class FloatType>
Ray<FloatType> operator*(const Matrix4x4<FloatType>& m, const Ray<FloatType>& r) {
	Ray<FloatType> res = r; 
	res.transform(m);
	return res;
}

template<class FloatType>
std::ostream& operator<<(std::ostream& os, const Ray<FloatType>& r) {
	os << r.origin() << " | " << r.direction();
	return os;
}

typedef Ray<float> Rayf;
typedef Ray<double> Rayd;

}  // namespace ml

#endif  // CORE_GRAPHICS_RAY_H_
