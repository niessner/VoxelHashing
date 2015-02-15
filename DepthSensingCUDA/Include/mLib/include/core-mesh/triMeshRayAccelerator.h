
#ifndef _COREMESH_TRIMESHRAYACCELERATOR_H_
#define _COREMESH_TRIMESHRAYACCELERATOR_H_

namespace ml {

//////////////////////////////////////////////////////////////////////////
// Interface for TriMesh-Ray Acceleration Structures
//////////////////////////////////////////////////////////////////////////

template<class FloatType>
class TriMeshRayAccelerator : virtual public TriMeshAccelerator<FloatType>
{
public:
    struct Intersection
    {
		Intersection() : triangle(nullptr) {}

        bool valid() const {
            return triangle != nullptr;
        }

		bool isValid() const {
			return triangle != nullptr;
		}
		
		typename TriMesh<FloatType>::Vertex<FloatType> getSurfaceVertex() const {
			return triangle->getSurfaceVertex(u,v);
		}
		point3d<FloatType> getSurfacePosition() const {
			return triangle->getSurfacePosition(u,v);
		}
		point4d<FloatType> getSurfaceColor() const {
			return triangle->getSurfaceColor(u,v);
		}
		point3d<FloatType> getSurfaceNormal() const {
			return triangle->getSurfaceNormal(u,v);
		}
		point2d<FloatType> getSurfaceTexCoord() const {
			return triangle->getSurfaceTexCoord(u,v); 
		}

		unsigned int getTriangleIndex() const {
			return triangle->getIndex();
		}
		unsigned int getMeshIndex() const {
			return triangle->getMeshIndex();
		}

		FloatType t, u, v;	
		const typename TriMesh<FloatType>::Triangle<FloatType>* triangle;
    };


	typename const TriMesh<FloatType>::Triangle<FloatType>* intersect(const Ray<FloatType>& r, FloatType& t, FloatType& u, FloatType& v, FloatType tmin = (FloatType)0, FloatType tmax = std::numeric_limits<FloatType>::max(), bool onlyFrontFaces = false) const {
		return intersectInternal(r, t, u, v, tmin, tmax, onlyFrontFaces);
	}

	bool intersect(const Ray<FloatType>& r, Intersection& i, FloatType tmin = (FloatType)0, FloatType tmax = std::numeric_limits<FloatType>::max(), bool onlyFrontFaces = false) const {
		i.triangle = intersectInternal(r, i.t, i.u, i.v, tmin, tmax, onlyFrontFaces);
		return i.isValid();
	}

	Intersection intersect(const Ray<FloatType> &r, FloatType tmin = (FloatType)0, FloatType tmax = std::numeric_limits<FloatType>::max(), bool onlyFrontFaces = false) const {
		Intersection i;
		i.triangle = intersectInternal(r, i.t, i.u, i.v, tmin, tmax, onlyFrontFaces);
		return i;
	}


	template<class Accelerator>
	static Intersection getFirstIntersection(
		const Ray<FloatType>& ray,  
		const std::vector< Accelerator* >& objectAccelerators,
		UINT &objectIndex) 
	{
		Intersection intersect;

		UINT curObjectIndex = 0;
		intersect.t = std::numeric_limits<float>::max();
		for (const auto &accelerator : objectAccelerators)
		{
			Intersection curIntersection;
			if (accelerator->intersect(ray, curIntersection))
			{
				if (curIntersection.t < intersect.t)
				{
					intersect = curIntersection;
					objectIndex = curObjectIndex;
				}
			}
			curObjectIndex++;
		}
		return intersect;
		//return (intersect.t != std::numeric_limits<float>::max());
	}

    //
    // ml::mat4f is the inverse of the "accelerator to world" matrix!
    // TODO: this returns the closest interesction, but the "t" parameter is broken because it is in the space of the transformed object, not the original world-space ray.
    //
    template<class Accelerator>
    static Intersection getFirstIntersectionTransform(
        const Ray<FloatType>& ray,
        const std::vector< std::pair<const Accelerator*, ml::mat4f> >& invTransformedAccelerators,
        UINT &objectIndex)
    {
        Intersection intersect;

        UINT curObjectIndex = 0;
        intersect.t = std::numeric_limits<float>::max();
        float bestDistSqToOrigin = std::numeric_limits<float>::max();

        for (const auto &accelerator : invTransformedAccelerators)
        {
            Intersection curIntersection;
            if (accelerator.first->intersect(accelerator.second * ray, curIntersection))
            {
                float curDistSqToOrigin = (float)distSq(accelerator.second.getInverse() * curIntersection.getSurfacePosition(), ray.origin());
                if (curDistSqToOrigin < bestDistSqToOrigin)
                {
                    bestDistSqToOrigin = curDistSqToOrigin;
                    intersect = curIntersection;
                    objectIndex = curObjectIndex;
                }
            }
            curObjectIndex++;
        }

        return intersect;
    }


	template<class Accelerator>
	static bool getFirstIntersection(
		const Ray<FloatType>& ray,  
		const std::vector< Accelerator* >& objectAccelerators,
		Intersection& i,
		UINT &objectIndex) 
	{
		i = getFirstIntersection(ray, objectAccelerators, objectIndex);
		return i.isValid();
	}

    //
    // ml::mat4f  is the inverse of the "accelerator to world" matrix!
    //
    template<class Accelerator>
    static bool getFirstIntersectionTransform(
        const Ray<FloatType>& ray,
        const std::vector< std::pair<const Accelerator*, ml::mat4f> >& invTransformedAccelerators,
        Intersection& i,
        UINT &objectIndex)
    {
        i = getFirstIntersectionTransform(ray, invTransformedAccelerators, objectIndex);
        return i.isValid();
    }

private:

	virtual typename const TriMesh<FloatType>::Triangle<FloatType>* intersectInternal(const Ray<FloatType>& r, FloatType& t, FloatType& u, FloatType& v, FloatType tmin = (FloatType)0, FloatType tmax = std::numeric_limits<FloatType>::max(), bool onlyFrontFaces = false) const = 0;

};

typedef TriMeshRayAccelerator<float> TriMeshRayAcceleratorf;
typedef TriMeshRayAccelerator<double> TriMeshRayAcceleratord;

} // namespace ml

/*

template<class FloatType>
class TriMeshRayAcceleratorBruteForce : public TriMeshRayAccelerator<FloatType>
{
public:

    struct Triangle
    {
        point3d<FloatType> pos[3];
        UINT meshIndex;

        point3d<FloatType> getPos(const point2d<FloatType> &uv) const
        {
            return (pos[0] + (pos[1] - pos[0]) * uv.x + (pos[2] - pos[0]) * uv.y);
        }
        point3d<FloatType> normal() const
        {
            return math::triangleNormal(pos[0], pos[1], pos[2]);
        }

        //
        // TODO: this belongs in a utility class, certainly not here nor in TriMesh<T>::Triangle
        //
        bool intersect(const Ray<FloatType> &r, FloatType& _t, FloatType& _u, FloatType& _v, FloatType tmin = (FloatType)0, FloatType tmax = std::numeric_limits<FloatType>::max()) const;
    };

    void initInternal(const std::vector< std::pair<const TriMesh<FloatType> *, mat4f> > &meshes, bool storeLocalCopy);
    bool intersect(const Ray<FloatType> &ray, TriMeshRayAccelerator<FloatType>::Intersection &result) const;

private:
    //
    // exactly one of m_tris or m_meshes contains data, depending on storeLocalCopy
    //
    std::vector< Triangle > m_tris;
    std::vector< std::pair<const TriMesh<FloatType> *, mat4f> > m_meshes;
    BoundingBox3d<FloatType> m_bbox;
};

} // ml

#include "TriMeshRayAccelerator.inl"
*/

#endif
