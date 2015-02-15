#ifndef _COREMESH_TRIMESHRAYACCELERATOR_INL_
#define _COREMESH_TRIMESHRAYACCELERATOR_INL_

namespace ml {
/*
template<class FloatType>
template<class Accelerator>
bool TriMeshRayAccelerator<FloatType>::getFirstIntersection(const Ray<FloatType>& ray,
                                    const std::vector< Accelerator > &objectAccelerators,
                                    Intersection &intersect,
                                    UINT &objectIndex)
{
    UINT curObjectIndex = 0;
    intersect.dist = std::numeric_limits<float>::max();
    for (const auto &accelerator : objectAccelerators)
    {
        ml::TriMeshRayAcceleratorf::Intersection curIntersection;
        if (accelerator->intersect(ray, curIntersection))
        {
            if (curIntersection.dist < intersect.dist)
            {
                intersect = curIntersection;
                objectIndex = curObjectIndex;
            }
        }
        curObjectIndex++;
    }
    return (intersect.dist != std::numeric_limits<float>::max());
}

template<class FloatType>
template<class Accelerator>
bool TriMeshRayAccelerator<FloatType>::getFirstIntersectionDirect(const Ray<FloatType>& ray,
    const std::vector< Accelerator > &objectAccelerators,
    Intersection &intersect,
    UINT &objectIndex)
{
    UINT curObjectIndex = 0;
    intersect.dist = std::numeric_limits<float>::max();
    for (const auto &accelerator : objectAccelerators)
    {
        TriMeshRayAccelerator<FloatType>::Intersection curIntersection;
        if (accelerator.intersect(ray, curIntersection))
        {
            if (curIntersection.dist < intersect.dist)
            {
                intersect = curIntersection;
                objectIndex = curObjectIndex;
            }
        }
        curObjectIndex++;
    }
    return (intersect.dist != std::numeric_limits<float>::max());
}

template<class FloatType>
bool TriMeshRayAcceleratorBruteForce<FloatType>::Triangle::intersect(const Ray<FloatType> &r, FloatType& _t, FloatType& _u, FloatType& _v, FloatType tmin = (FloatType)0, FloatType tmax = std::numeric_limits<FloatType>::max()) const
{
    const point3d<FloatType> &d = r.direction();
    const point3d<FloatType> &p = r.origin();

    point3d<FloatType> e1 = pos[1] - pos[0];
    point3d<FloatType> e2 = pos[2] - pos[0];

    point3d<FloatType> h = d ^ e2;
    FloatType a = e1 | h;

    if (a == (FloatType)0.0 || a == -(FloatType)0.0) return false;

    FloatType f = (FloatType)1.0 / a;
    point3d<FloatType> s = p - pos[0];
    FloatType u = f * (s | h);

    if (u < (FloatType)0.0 || u >(FloatType)1.0) return false;

    point3d<FloatType> q = s ^ e1;
    FloatType v = f * (d | q);

    if (v < (FloatType)0.0 || u + v >(FloatType)1.0) return false;

    // at this stage we can compute t to find out where the intersection point is on the line
    FloatType t = f * (e2 | q);

    if (t <= tmax && t >= tmin)
    {
        _t = t;
        _u = u;
        _v = v;
        return true;
    }
    return false;
}

template<class FloatType>
void TriMeshRayAcceleratorBruteForce<FloatType>::initInternal(const std::vector< std::pair<const TriMesh<FloatType> *, mat4f> > &meshes, bool storeLocalCopy)
{
    m_bbox.reset();
    for (const auto &p : meshes)
    {
        ml::ObjectOrientedBoundingBox<FloatType> oobb = p.second * ml::ObjectOrientedBoundingBox<FloatType>(p.first->getBoundingBox());
        m_bbox.include(oobb.getVertices());
    }

    if (!storeLocalCopy)
    {
        m_meshes = meshes;
        return;
    }
        
    size_t triCount = 0;
    for (const auto &p : meshes)
        triCount += p.first->getIndices().size();

    m_tris.clear();
    m_tris.reserve(triCount);

    int meshIndex = 0;
    for (const auto &p : meshes)
    {
        for (const auto &indices : p.first->getIndices())
        {
            Triangle tri;
                
            tri.meshIndex = meshIndex;

            for (int i = 0; i < 3; i++)
                tri.pos[i] = p.second * p.first->getVertices()[indices[i]].position;

            m_tris.push_back(tri);
        }
        meshIndex++;
    }
}

template<class FloatType>
bool TriMeshRayAcceleratorBruteForce<FloatType>::intersect(const Ray<FloatType> &ray, TriMeshRayAccelerator<FloatType>::Intersection &result) const
{
    result.dist = std::numeric_limits<float>::max();
    result.meshIndex = -1;
    result.triangleIndex = -1;

    if (!m_bbox.intersect(ray, 0.0f, std::numeric_limits<float>::max()))
        return false;

    if (m_tris.size() > 0)
    {
        int triangleIndex = 0;
        for (const Triangle &tri : m_tris)
        {
            FloatType dist, u, v;
            if (tri.intersect(ray, dist, u, v) && dist < result.dist)
            {
                result.dist = dist;
                result.meshIndex = tri.meshIndex;
                result.triangleIndex = triangleIndex;
                result.uv.x = u;
                result.uv.y = v;
                result.pos = tri.getPos(result.uv);
                result.normal = tri.normal();
            }
            triangleIndex++;
        }
    }
    else if (m_meshes.size() > 0)
    {
        int meshIndex = 0, triangleIndex = 0;
        for (const auto &p : m_meshes)
        {
            for (const auto &indices : p.first->getIndices())
            {
                Triangle tri;

                tri.meshIndex = meshIndex;
                for (int i = 0; i < 3; i++)
                    tri.pos[i] = p.second * p.first->getVertices()[indices[i]].position;

                FloatType dist, u, v;
                if (tri.intersect(ray, dist, u, v) && dist < result.dist)
                {
                    result.dist = dist;
                    result.meshIndex = tri.meshIndex;
                    result.triangleIndex = triangleIndex;
                    result.uv.x = u;
                    result.uv.y = v;
                    result.pos = tri.getPos(result.uv);
                    result.normal = tri.normal();
                }

                triangleIndex++;
            }
            meshIndex++;
        }
    }
        
    return (result.meshIndex != -1);
}
*/
} // ml

#endif