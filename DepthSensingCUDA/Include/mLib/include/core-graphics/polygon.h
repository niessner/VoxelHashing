
#ifndef CORE_GRAPHICS_POLYGON_H_
#define CORE_GRAPHICS_POLYGON_H_

namespace ml {

template<class T>
struct Polygon
{
    static Polygon<T> clip(const Polygon<T> &sourcePoly, const Polygon<T> &clipPoly);
    static Polygon<T> clip(const Polygon<T> &sourcePoly, const Line2<T> &clipLine, const point2d<T> &clipCentroid);

    vector< LineSegment2<T> > segments() const
    {
        vector< LineSegment2<T> > result;
        if (points.size() <= 1)
            return result;
        for (UINT pointIndex = 0; pointIndex < points.size() - 1; pointIndex++)
            result.push_back(LineSegment2<T>(points[pointIndex], points[pointIndex + 1]));
        result.push_back(LineSegment2<T>(points.back(), points[0]));
        return result;
    }

    point2d<T> centroid() const
    {
        point2d<T> result;
        for (const auto &p : points)
            result += p;
        return result / (T)points.size();
    }

    void translate(const point2d<T> &v)
    {
        for (point2d<T> &p : points)
            p += v;
    }

    void scale(float s)
    {
        for (point2d<T> &p : points)
            p *= s;
    }

    T convexArea() const
    {
        if (points.size() <= 2)
            return 0.0;
        T sum = 0.0;
        for (size_t v1 = 1; v1 < points.size() - 1; v1++)
            sum += math::triangleArea(points[0], points[v1], points[v1 + 1]);
        return sum;
    }

    vector< point2d<T> > points;
};

typedef Polygon<float> Polygonf;
typedef Polygon<double> Polygond;

}  // namespace ml

#include "polygon.cpp"

#endif  // CORE_GRAPHICS_POLYGON_H_
