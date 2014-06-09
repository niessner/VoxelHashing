
#ifndef CORE_GRAPHICS_RAY_H_
#define CORE_GRAPHICS_RAY_H_

namespace ml {


template<class FloatType>
class Ray
{
public:

	Ray(const point3d<FloatType> &o, const point3d<FloatType> &d) {
		m_Origin = o;
		m_Direction = d;
		m_InverseDirection = point3d<FloatType>((FloatType)1.0/d.x, (FloatType)1.0/d.y, (FloatType)1.0/d.z);

		m_Sign.x = (m_InverseDirection.x < (FloatType)0);
		m_Sign.y = (m_InverseDirection.y < (FloatType)0);
		m_Sign.z = (m_InverseDirection.z < (FloatType)0);
	}

	~Ray() {
	}

	point3d<FloatType> getHitPoint(FloatType t) const {
		return m_Origin + t * m_Direction;
	}

	const point3d<FloatType>& origin() const {
		return m_Origin;
	}

	const point3d<FloatType>& direction() const {
		return m_Direction;
	}

	const point3d<FloatType>& inverseDirection() const {
		return m_InverseDirection;
	}

	const vec3i& sign() const {
		return m_Sign;
	}

private:
	point3d<FloatType> m_Direction;
	point3d<FloatType> m_InverseDirection;
	point3d<FloatType> m_Origin;

	vec3i m_Sign;
};

template<class FloatType>
std::ostream& operator<<(std::ostream& os, const Ray<FloatType>& r) {
	os << r.origin() << " | " << r.direction();
	return os;
}

typedef Ray<float> Rayf;
typedef Ray<double> Rayd;

}  // namespace ml

#endif  // CORE_GRAPHICS_RAY_H_
