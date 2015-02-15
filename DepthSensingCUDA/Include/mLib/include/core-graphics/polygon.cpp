
#ifndef CORE_GRAPHICS_POLYGON_INL_H_
#define CORE_GRAPHICS_POLYGON_INL_H_

namespace ml {

//
// Sutherland-Hodgman Clipping
// http://gamedevelopment.tutsplus.com/tutorials/understanding-sutherland-hodgman-clipping-for-physics-engines--gamedev-11917
// http://www.cc.gatech.edu/grads/h/Hao-wei.Hsieh/Haowei.Hsieh/code2.html
//
template<class T>
Polygon<T> Polygon<T>::clip(const Polygon<T> &sourcePoly, const Line2<T> &clipLine, const point2d<T> &clipCentroid)
{
    Polygon<T> output;

    if (sourcePoly.points.size() == 0)
        return output;

    //
    // find the normal of the line segment pointing towards clipCentroid
    //
    point2d<T> normal = clipLine.dir();
    normal = point2d<T>(-normal.y, normal.x);
    if (((clipCentroid - clipLine.p0()) | normal) < 0.0f)
        normal = -normal;

    auto sideTest = [&](const point2d<T> &pt)
    {
        return ((pt - clipLine.p0()) | normal) >= 0.0f;
    };

    point2d<T> startPoint = sourcePoly.points.back();
    for (const point2d<T> &endPoint : sourcePoly.points)
    {
        bool startSide = sideTest(startPoint);
        bool endSide = sideTest(endPoint);

        if (startSide && endSide)
        {
            output.points.push_back(endPoint);
        }
        if (startSide && !endSide)
        {
            point2d<T> intersection = startPoint;
            intersection::intersectLine2Line2(clipLine, Line2<T>(startPoint, endPoint), intersection);
            output.points.push_back(intersection);
        }
        if (!startSide && endSide)
        {
            point2d<T> intersection = startPoint;
            intersection::intersectLine2Line2(clipLine, Line2<T>(startPoint, endPoint), intersection);
            output.points.push_back(intersection);
            output.points.push_back(endPoint);
        }

        startPoint = endPoint;
    }

    return output;
}

template<class T>
Polygon<T> Polygon<T>::clip(const Polygon<T> &sourcePoly, const Polygon<T> &clipPoly)
{
    Polygon<T> output = sourcePoly;
    point2d<T> clipCentroid = clipPoly.centroid();

    for (const LineSegment2<T> &clipSegment : clipPoly.segments())
        output = clip(output, Line2<T>(clipSegment), clipCentroid);

    return output;
}

}  // namespace ml

#endif  // CORE_GRAPHICS_POLYGON_INL_H_
