
#ifndef _TRIMESH_ACCELERATOR_BRUTEFORCE_H_
#define _TRIMESH_ACCELERATOR_BRUTEFORCE_H_

namespace ml {

template <class FloatType>
class TriMeshAcceleratorBruteForce : public TriMeshRayAccelerator<FloatType> , public TriMeshCollisionAccelerator<FloatType,TriMeshAcceleratorBruteForce<FloatType>>
{
public:

	TriMeshAcceleratorBruteForce() {

	}

	TriMeshAcceleratorBruteForce(const TriMesh<FloatType>& triMesh, bool storeLocalCopy = false) {
		build(triMesh, storeLocalCopy);
	}


private:

	//! interface definition
	bool collisionInternal(const TriMeshAcceleratorBruteForce<FloatType>& accel) const {
		if (triangleCount() > 0) {
			for (const auto* triA : m_TrianglePointers)	{
				for (const auto* triB : accel.m_TrianglePointers) {
					if (intersection::intersectTriangleTriangle(
						triA->getV0().position, triA->getV1().position, triA->getV2().position,
						triB->getV0().position, triB->getV1().position, triB->getV2().position))
							return true;
				}
			}
		}
		return false;
	}

    //! interface definition
    bool collisionTransformInternal(const TriMeshAcceleratorBruteForce<FloatType>& accel, const Matrix4x4<FloatType>& transform) const {
        if (triangleCount() > 0) {
            for (const auto* triA : m_TrianglePointers)	{
                for (const auto* triB : accel.m_TrianglePointers) {
                    if (intersection::intersectTriangleTriangle(
                        triA->getV0().position, triA->getV1().position, triA->getV2().position,
                        transform * triB->getV0().position, 
                        transform * triB->getV1().position, 
                        transform * triB->getV2().position))
                        return true;
                }
            }
        }
        return false;
    }

    //! interface definition
    bool collisionTransformBBoxOnlyInternal(const TriMeshAcceleratorBruteForce<FloatType>& accel, const Matrix4x4<FloatType>& transform) const {
        //TODO: have Matthias do this
        return false;
    }


	//! interface definition
	typename const TriMesh<FloatType>::Triangle<FloatType>* intersectInternal(const Ray<FloatType>& r, FloatType& t, FloatType& u, FloatType& v, FloatType tmin = (FloatType)0, FloatType tmax = std::numeric_limits<FloatType>::max(), bool onlyFrontFaces = false) const {

		typename TriMesh<FloatType>::Triangle<FloatType>* tri = nullptr;
		for (size_t i = 0; i < m_TrianglePointers.size(); i++) {
			if (m_TrianglePointers[i]->intersect(r, t, u, v, tmin, tmax, onlyFrontFaces)) {
				tmax = t;
				tri = m_TrianglePointers[i];
			}
		}
		return tri;
	}

	void buildInternal() {
		//nothing to do here
	}

};

typedef TriMeshAcceleratorBruteForce<float>		TriMeshAcceleratorBruteForcef;
typedef TriMeshAcceleratorBruteForce<double>	TriMeshAcceleratorBruteForced;

} //namespace ml

#endif