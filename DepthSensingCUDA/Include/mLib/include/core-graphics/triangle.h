#pragma once

#ifndef _TRIANGLE_H_
#define _TRIANGLE_H_

namespace ml {

template<class T>
struct Triangle
{
    Triangle() {}
    Triangle(const point3d<T> &v0, const point3d<T> &v1, const point3d<T> &v2)
    {
        vertices[0] = v0;
        vertices[1] = v1;
        vertices[2] = v2;
    }
    point3d<T> getNormal() const
    {
        return ml::math::triangleNormal(vertices[0], vertices[1], vertices[2]);
    }

    point3d<T> vertices[3];

};

typedef Triangle<float> Trianglef;
typedef Triangle<double> Triangled;

} //namespace ml


#endif
