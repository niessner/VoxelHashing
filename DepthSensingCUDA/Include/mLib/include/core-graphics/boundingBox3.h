
#ifndef CORE_GRAPHICS_BOUNDINGBOX3_H_
#define CORE_GRAPHICS_BOUNDINGBOX3_H_

#include "Ray.h"

#include <cfloat>
#include <vector>
#include <cmath>
#include <iostream>

namespace ml {

template<class T>
class OrientedBoundingBox3;

template<class FloatType>
class BoundingBox3
{
public:

	BoundingBox3() {
		reset();
	}

	explicit BoundingBox3(const std::vector< point3d<FloatType> >& verts) {
		reset();
        for (const auto &v : verts)
			include(v);
	}

	explicit BoundingBox3(typename std::vector<point3d<FloatType>>::const_iterator pBegin, typename std::vector<point3d<FloatType>>::const_iterator pEnd) {
		reset();
		for (const auto& iter = pBegin; iter != pEnd; iter++) {
			include(*iter);
		}
	}

	BoundingBox3(const point3d<FloatType>& p0, const point3d<FloatType>& p1, const point3d<FloatType>& p2) {
		reset();
		include(p0);
		include(p1);
		include(p2);
	}

	BoundingBox3(const point3d<FloatType>& minBound, const point3d<FloatType>& maxBound) {
		reset();
		minB = minBound;
		maxB = maxBound;
	}

    explicit BoundingBox3(const OrientedBoundingBox3<FloatType> &obb) {
        reset();
		std::vector< point3d <FloatType > > vertices = obb.getVertices();
        for (const auto &v : vertices)
		//for (const auto &v : obb.getVertices())
            include(v);
    }

	void reset() {
		minX = minY = minZ = std::numeric_limits<FloatType>::max();
		maxX = maxY = maxZ = -std::numeric_limits<FloatType>::max();
	}

	void include(const BoundingBox3 &other) {
		if (other.minX < minX)	minX = other.minX;
		if (other.minY < minY)	minY = other.minY;
		if (other.minZ < minZ)	minZ = other.minZ;

		if (other.maxX > maxX)	maxX = other.maxX;
		if (other.maxY > maxY)	maxY = other.maxY;
		if (other.maxZ > maxZ)	maxZ = other.maxZ;
	}

	void include(const point3d<FloatType> &v) {
		if (v.x < minX)	minX = v.x;
		if (v.y < minY)	minY = v.y;
		if (v.z < minZ)	minZ = v.z;

		if (v.x > maxX)	maxX = v.x;
		if (v.y > maxY)	maxY = v.y;
		if (v.z > maxZ)	maxZ = v.z;
	}

    void include(const std::vector<point3d<FloatType>> &v) {
        for (const auto &p : v)
            include(p);
    }

	bool isInitialized() const {
		return (minX != std::numeric_limits<FloatType>::max());
	}
	bool isValid() const {
		return (minX <= maxX && minY <= maxY && minZ <= maxZ);
	}

	bool intersect(const Ray<FloatType> &r, FloatType tmin, FloatType tmax ) const
	{
		//TODO move to intersection

		//const FloatType t0 = 0.0;
		//const FloatType t1 = r.t;

		const FloatType t0 = tmin;
		const FloatType t1 = tmax;

		FloatType txmin, txmax, tymin, tymax, tzmin, tzmax;

        auto sign = r.sign();
        auto origin = r.origin();
        auto invDir = r.inverseDirection();

        txmin = (parameters[sign.x * 3] - origin.x) * invDir.x;
        txmax = (parameters[3 - sign.x * 3] - origin.x) * invDir.x;
        tymin = (parameters[sign.y * 3 + 1] - origin.y) * invDir.y;
        tymax = (parameters[3 - sign.y * 3 + 1] - origin.y) * invDir.y;

		if ( (txmin > tymax) || (tymin > txmax) ) return false;

		if (tymin > txmin)	txmin = tymin;
		if (tymax < txmax)	txmax = tymax;

        tzmin = (parameters[sign.z * 3 + 2] - origin.z) * invDir.z;
        tzmax = (parameters[3 - sign.z * 3 + 2] - origin.z) * invDir.z;

		if ( (txmin > tzmax) || (tzmin > txmax) ) 
			return false;
		if (tzmin > txmin)
			txmin = tzmin;
		if (tzmax < txmax)
			txmax = tzmax;
		return ( (txmin <= t1) && (txmax >= t0) );

	}


	void getVertices(point3d<FloatType> *result) const {
		result[0] = point3d<FloatType>(minX, minY, minZ);
		result[1] = point3d<FloatType>(maxX, minY, minZ);
		result[2] = point3d<FloatType>(maxX, maxY, minZ);
		result[3] = point3d<FloatType>(minX, maxY, minZ);
		result[4] = point3d<FloatType>(minX, minY, maxZ);
		result[5] = point3d<FloatType>(maxX, minY, maxZ);
		result[6] = point3d<FloatType>(maxX, maxY, maxZ);
		result[7] = point3d<FloatType>(minX, maxY, maxZ);
	}

	std::vector< point3d<FloatType> > getVertices() const {
		std::vector< point3d<FloatType> > result;
		result.resize(8);

		getVertices(result.data());

		return result;
	}





	std::vector< LineSegment3<FloatType> > getEdges() const
	{
		std::vector< LineSegment3<FloatType> > result;

		auto v = getVertices();

		result.push_back(LineSegment3<FloatType>(v[0], v[1]));
		result.push_back(LineSegment3<FloatType>(v[1], v[2]));
		result.push_back(LineSegment3<FloatType>(v[2], v[3]));
		result.push_back(LineSegment3<FloatType>(v[3], v[0]));

		result.push_back(LineSegment3<FloatType>(v[4], v[5]));
		result.push_back(LineSegment3<FloatType>(v[5], v[6]));
		result.push_back(LineSegment3<FloatType>(v[6], v[7]));
		result.push_back(LineSegment3<FloatType>(v[7], v[4]));

		result.push_back(LineSegment3<FloatType>(v[0], v[4]));
		result.push_back(LineSegment3<FloatType>(v[1], v[5]));
		result.push_back(LineSegment3<FloatType>(v[2], v[6]));
		result.push_back(LineSegment3<FloatType>(v[3], v[7]));

		return result;
	}

	//! point collision
	bool intersects(const point3d<FloatType>& p) const {
        if (p.x >= minX && p.x <= maxX &&
            p.y >= minY && p.y <= maxY &&
            p.z >= minZ && p.z <= maxZ)
            return true;
		return false;
	}

	//! triangle collision
	bool intersects(const point3d<FloatType>& p0, const point3d<FloatType>& p1, const point3d<FloatType>& p2) const {
		return intersection::intersectTriangleAABB(minB, maxB, p0, p1, p2);
	}

	//! bounding box collision
	bool intersects(const BoundingBox3<FloatType>& other) const {
		return 
			minX <= other.maxX && other.minX <= maxX &&
			minY <= other.maxY && other.minY <= maxY &&
			minZ <= other.maxZ && other.minZ <= maxZ;
	}

    //! transformed bounding box collision
    //bool collision(const BoundingBox3<FloatType>& other, const matrix4x4<FloatType> &transform) const {
    //    BoundingBox3<FloatType> otherTransformed = other * transform;
    //    return collision(otherTransformed);
    //}

	FloatType getMaxExtent() const {
		FloatType d0 = maxX - minX;
		FloatType d1 = maxY - minY;
		FloatType d2 = maxZ - minZ;
		return math::max(d0, d1, d2);
	}

	FloatType getExtentX() const {
		return maxX - minX;
	}

	FloatType getExtentY() const {
		return maxY - minY;
	}

	FloatType getExtentZ() const {
		return maxZ - minZ;
	}

	point3d<FloatType> getExtent() const {
		return point3d<FloatType>(maxX - minX, maxY - minY, maxZ - minZ);
	}

	point3d<FloatType> getMin() const {
		return point3d<FloatType>(minX, minY, minZ);
	}

	point3d<FloatType> getMax() const {
		return point3d<FloatType>(maxX, maxY, maxZ);
	}

	point3d<FloatType> getCenter() const {
		point3d<FloatType> center = getMin() + getMax();
		center *= (FloatType)0.5;
		return center;
	}

	void setMin(const point3d<FloatType>& minValue) {
		minX = minValue.x;
		minY = minValue.y;
		minZ = minValue.z;
	}

	void setMax(const point3d<FloatType>& maxValue) {
		maxX = maxValue.x;
		maxY = maxValue.y;
		maxZ = maxValue.z;
	}

	void setMinX(FloatType v) { minX = v; }
	void setMinY(FloatType v) { minY = v; }
	void setMinZ(FloatType v) { minZ = v; }
	void setMaxX(FloatType v) { maxX = v; }
	void setMaxY(FloatType v) { maxY = v; }
	void setMaxZ(FloatType v) { maxZ = v; }

	FloatType getMinX() const { return minX; }
	FloatType getMinY() const { return minY; }
	FloatType getMinZ() const { return minZ; }
	FloatType getMaxX() const { return maxX; }
	FloatType getMaxY() const { return maxY; }
	FloatType getMaxZ() const { return maxZ; }

	//! scales the bounding box by the factor t (for t=1 the bb remains unchanged)
	void scale(FloatType x, FloatType y, FloatType z) {

		FloatType scale[] = {x, y, z};
		for (unsigned int i = 0; i < 3; i++) {
			FloatType center = (FloatType)0.5 * (parameters[i] + parameters[i+3]);
			FloatType diff = parameters[i+3] - parameters[i];
			diff *= scale[i];
			diff *= (FloatType)0.5;
			parameters[i] = center - diff;
			parameters[i+3] = center + diff;
		}
	}

	//! scales the bounding box by the factor t (for t=1 the bb remains unchanged)
	void scale(FloatType t) {
		for (unsigned int i = 0; i < 3; i++) {
			FloatType center = (FloatType)0.5 * (parameters[i] + parameters[i+3]);
			FloatType diff = parameters[i+3] - parameters[i];
			diff *= t;
			diff *= (FloatType)0.5;
			parameters[i] = center - diff;
			parameters[i+3] = center + diff;
		}
	}

	//! transforms the bounding box (conservatively)
	void transform(const Matrix4x4<FloatType>& m) {
        point3d<FloatType> verts[8];
		getVertices(verts);
		reset();
		for (const auto& p : verts) {
			include(m.transformAffine(p));
		}
	}

	void translate(const point3d<FloatType>& t) {
		minB += t;
		maxB += t;
	}

	//! scales the bounding box (see scale)
	BoundingBox3<FloatType> operator*(FloatType t) const {
		BoundingBox3<FloatType> res = *this;
		res.scale(t);
		return res;
	}

	//! transforms the bounding box (see transform)
	BoundingBox3<FloatType> operator*(const Matrix4x4<FloatType>& m) const {
		BoundingBox3<FloatType> res = *this;
		res.transform(m);
		return res;
	}


	Plane<FloatType> getBottomPlane() const {
		std::vector<point3d<FloatType>> vertices; vertices.resize(3);
		vertices[0] = point3d<FloatType>(minX, minY, minZ);
		vertices[2] = point3d<FloatType>(maxX, minY, minZ);
		vertices[1] = point3d<FloatType>(maxX, maxY, minZ);
		return Plane<FloatType>(&vertices[0]);
	}

	Plane<FloatType> getTopPlane() const {
		std::vector<point3d<FloatType>> vertices; vertices.resize(3);
		vertices[0] = point3d<FloatType>(minX, minY, maxZ);
		vertices[1] = point3d<FloatType>(maxX, minY, maxZ);
		vertices[2] = point3d<FloatType>(maxX, maxY, maxZ);
		return Plane<FloatType>(&vertices[0]);
	}

	void makeTriMeshBottomPlane(std::vector<point3d<FloatType>>& vertices, std::vector<vec3ui>& indices, std::vector<point3d<FloatType>>& normals) const {
		vertices.resize(4);
		normals.resize(4);
		indices.resize(2);

		vertices[0] = point3d<FloatType>(minX, minY, minZ);
		vertices[1] = point3d<FloatType>(maxX, minY, minZ);
		vertices[2] = point3d<FloatType>(maxX, maxY, minZ);
		vertices[3] = point3d<FloatType>(minX, maxY, minZ);
		indices[0].x = 0;	indices[0].y = 1;	indices[0].z = 2;
		indices[1].x = 2;	indices[1].y = 3;	indices[1].z = 0;
		normals[0] = normals[1] = normals[2] = normals[3] = point3d<FloatType>(0,0,-1);
	}

	//! generates vertices, indices, and normals which can be used to initialize a triMesh
	void makeTriMesh(point3d<FloatType>* vertices, vec3ui* indices, point3d<FloatType>* normals) const {

		//bottom
		vertices[0] = point3d<FloatType>(minX, minY, minZ);
		vertices[1] = point3d<FloatType>(maxX, minY, minZ);
		vertices[2] = point3d<FloatType>(maxX, maxY, minZ);
		vertices[3] = point3d<FloatType>(minX, maxY, minZ);
		indices[0].x = 0;	indices[0].y = 1;	indices[0].z = 2;
		indices[1].x = 2;	indices[1].y = 3;	indices[1].z = 0;
		normals[0] = normals[1] = normals[2] = normals[3] = point3d<FloatType>(0,0,-1);
		//front
		vertices[4] = point3d<FloatType>(minX, minY, minZ);
		vertices[5] = point3d<FloatType>(maxX, minY, minZ);
		vertices[6] = point3d<FloatType>(maxX, minY, maxZ);
		vertices[7] = point3d<FloatType>(minX, minY, maxZ);
		indices[2].x = 4;	indices[2].y = 5;	indices[2].z = 6;
		indices[3].x = 6;	indices[3].y = 7;	indices[3].z = 4;
		normals[4] = normals[5] = normals[6] = normals[7] = point3d<FloatType>(0,-1,0);
		//left
		vertices[8] = point3d<FloatType>(minX, minY, minZ);
		vertices[9] = point3d<FloatType>(minX, minY, maxZ);
		vertices[10] = point3d<FloatType>(minX, maxY, maxZ);
		vertices[11] = point3d<FloatType>(minX, maxY, minZ);
		indices[4].x = 8;	indices[4].y = 9;	indices[4].z = 10;
		indices[5].x = 10;	indices[5].y = 11;	indices[5].z = 8;
		normals[8] = normals[9] = normals[10] = normals[11] = point3d<FloatType>(-1,0,0);
		//right
		vertices[12] = point3d<FloatType>(maxX, minY, minZ);
		vertices[13] = point3d<FloatType>(maxX, minY, maxZ);
		vertices[14] = point3d<FloatType>(maxX, maxY, maxZ);
		vertices[15] = point3d<FloatType>(maxX, maxY, minZ);
		indices[6].x = 12;	indices[6].y = 13;	indices[6].z = 14;
		indices[7].x = 14;	indices[7].y = 15;	indices[7].z = 12;
		normals[12] = normals[13] = normals[14] = normals[15] = point3d<FloatType>(1,0,0);
		//back
		vertices[16] = point3d<FloatType>(minX, maxY, minZ);
		vertices[17] = point3d<FloatType>(maxX, maxY, minZ);
		vertices[18] = point3d<FloatType>(maxX, maxY, maxZ);
		vertices[19] = point3d<FloatType>(minX, maxY, maxZ);
		indices[8].x = 16;	indices[8].y = 17;	indices[8].z = 18;
		indices[9].x = 18;	indices[9].y = 19;	indices[9].z = 16;
		normals[16] = normals[17] = normals[18] = normals[19] = point3d<FloatType>(0,1,0);
		//top
		vertices[20] = point3d<FloatType>(minX, minY, maxZ);
		vertices[21] = point3d<FloatType>(maxX, minY, maxZ);
		vertices[22] = point3d<FloatType>(maxX, maxY, maxZ);
		vertices[23] = point3d<FloatType>(minX, maxY, maxZ);
		indices[10].x = 20;	indices[10].y = 21;	indices[10].z = 22;
		indices[11].x = 22;	indices[11].y = 23;	indices[11].z = 20;
		normals[20] = normals[21] = normals[22] = normals[23] = point3d<FloatType>(0,0,1);
	}

	//! generates vertices, indices, and normals which can be used to initialize a triMesh
	void makeTriMesh(std::vector<point3d<FloatType>>& vertices, std::vector<vec3ui>& indices, std::vector<point3d<FloatType>>& normals) const {
		//TODO check face and normal orientation
		vertices.resize(24);
		normals.resize(24);
		indices.resize(12);

    makeTriMesh(vertices.data(), indices.data(), normals.data());
	}

	void makeTriMesh(point3d<FloatType>* vertices, vec3ui* indices) const {
		vertices[0] = point3d<FloatType>(maxX, maxY, maxZ);
		vertices[1] = point3d<FloatType>(minX, maxY, maxZ);
		vertices[2] = point3d<FloatType>(minX, minY, maxZ);
		vertices[3] = point3d<FloatType>(maxX, minY, maxZ);
		vertices[4] = point3d<FloatType>(maxX, maxY, minZ);
		vertices[5] = point3d<FloatType>(minX, maxY, minZ);
		vertices[6] = point3d<FloatType>(minX, minY, minZ);
		vertices[7] = point3d<FloatType>(maxX, minY, minZ);

		indices[0].x = 1;	indices[0].y = 2;	indices[0].z = 3; 
		indices[1].x = 1;	indices[1].y = 3;	indices[1].z = 0; 
		indices[2].x = 0;	indices[2].y = 3;	indices[2].z = 7; 
		indices[3].x = 0;	indices[3].y = 7;	indices[3].z = 4; 
		indices[4].x = 3;	indices[4].y = 2;	indices[4].z = 6; 
		indices[5].x = 3;	indices[5].y = 6;	indices[5].z = 7; 
		indices[6].x = 1;	indices[6].y = 6;	indices[6].z = 2; 
		indices[7].x = 1;	indices[7].y = 5;	indices[7].z = 6; 
		indices[8].x = 0;	indices[8].y = 5;	indices[8].z = 1; 
		indices[9].x = 0;	indices[9].y = 4;	indices[9].z = 5; 
		indices[10].x = 6;	indices[10].y = 5;	indices[10].z = 4; 
		indices[11].x = 6;	indices[11].y = 4;	indices[11].z = 7; 
	}

	//! generates vertices, indices which can be used to initialize a triMesh
	void makeTriMesh(std::vector<point3d<FloatType>>& vertices, std::vector<vec3ui>& indices) const {
		//TODO check face and normal orientation
		vertices.resize(8);
		indices.resize(12);
		makeTriMesh(vertices.data(), indices.data());
	}

	void setUnitCube() {
		minX = minY = minZ = 0;
		maxX = maxY = maxZ = 1;
	}

	//! transforms a point in [-1;1]^3 to the bounding box space
    Matrix4x4<FloatType> cubeToWorldTransform() const {
        return  Matrix4x4<FloatType>::translation(getCenter()) *  Matrix4x4<FloatType>::scale((maxB - minB) * (FloatType)0.5);
    }
	
	//! transforms a point from bounding box space to [-1;1]^3
	Matrix4x4<FloatType> worldToCubeTransform() const {
		return cubeToWorldTransform().getInverse();	//TODO avoid the inverse
	}

protected:

  // boost archive serialization functions
  friend class boost::serialization::access;
  template <class Archive>
  inline void serialize(Archive& ar, const unsigned int verion) {
    ar & boost::serialization::make_array(parameters, 6);
  }

	union {
		struct {
			point3d<FloatType> minB;
			point3d<FloatType> maxB;
		};
		struct {
			FloatType minX, minY, minZ;
			FloatType maxX, maxY, maxZ;
		};
		FloatType parameters[6];
	};
};

template<class FloatType>
std::ostream& operator<< (std::ostream& s, const BoundingBox3<FloatType>& bb) {
	s << bb.getMin() << std::endl << bb.getMax() << std::endl;
	return s;
}

typedef BoundingBox3<float> BoundingBox3f;
typedef BoundingBox3<double> BoundingBox3d;

typedef BoundingBox3<float> bbox3f;
typedef BoundingBox3<double> bbox3d;

}  // namespace ml



#endif  // CORE_GRAPHICS_BOUNDINGBOX3_H_
