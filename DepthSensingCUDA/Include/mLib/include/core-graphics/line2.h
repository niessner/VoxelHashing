
#ifndef CORE_GRAPHICS_LINE2D_H_
#define CORE_GRAPHICS_LINE2D_H_

namespace ml {

template<class T>
class Line2
{
public:

    Line2(const LineSegment2<T> &segment)
    {
        m_p0 = segment.p0();
        m_dir = segment.delta();
    }
    Line2(const point2d<T> &p0, const point2d<T> &p1)
    {
        m_p0 = p0;
        m_dir = p1 - p0;
	}

	const point2d<T>& p0() const
    {
        return m_p0;
	}

	const point2d<T>& dir() const
    {
        return m_dir;
	}

private:
	point2d<T> m_p0;
	point2d<T> m_dir;
};

typedef Line2<float> Line2f;
typedef Line2<double> Line2d;

}  // namespace ml

#endif  // CORE_GRAPHICS_LINE2D_H_
