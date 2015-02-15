
#ifndef CORE_GRAPHICS_BOUNDINGBOX2_H_
#define CORE_GRAPHICS_BOUNDINGBOX2_H_

#include "Ray.h"

#include <cfloat>
#include <vector>
#include <cmath>
#include <iostream>

namespace ml {

	template<class T>
	class ObjectOrientedBoundingBox2;

	template<class FloatType>
	class BoundingBox2
	{
	public:

		BoundingBox2(void) {
			reset();
		}

		BoundingBox2(const std::vector< point2d<FloatType> >& verts) {
			reset();
			for (const auto &v : verts)
				include(v);
		}

		BoundingBox2(const point2d<FloatType>& p0, const point2d<FloatType>& p1, const point2d<FloatType>& p2) {
			reset();
			include(p0);
			include(p1);
			include(p2);
		}

		BoundingBox2(const point2d<FloatType>& minBound, const point2d<FloatType>& maxBound) {
			reset();
			minB = minBound;
			maxB = maxBound;
		}

		//TODO
		//explicit BoundingBox2(const ObjectOrientedBoundingBox2<FloatType> &oobb) {
		//	reset();
		//	for (const auto &v : oobb.getVertices())
		//		include(v);
		//}

		void reset() {
			minX = minY = std::numeric_limits<FloatType>::max();
			maxX = maxY = -std::numeric_limits<FloatType>::max();
		}

		void include(const BoundingBox2 &other) {
			if (other.minX < minX)	minX = other.minX;
			if (other.minY < minY)	minY = other.minY;

			if (other.maxX > maxX)	maxX = other.maxX;
			if (other.maxY > maxY)	maxY = other.maxY;
		}

		void include(const point2d<FloatType> &v) {
			if (v.x < minX)	minX = v.x;
			if (v.y < minY)	minY = v.y;

			if (v.x > maxX)	maxX = v.x;
			if (v.y > maxY)	maxY = v.y;
		}

		void include(const std::vector<point2d<FloatType>> &v) {
			for (const auto &p : v)
				include(p);
		}

		bool isInitialized() const {
			return (minX != std::numeric_limits<FloatType>::max());
		}

		bool isValid() const {
			return (minX <= maxX && minY <= maxY);
		}

		void getVertices(point2d<FloatType> *result) const {
			result[0] = point2d<FloatType>(minX, minY);
			result[1] = point2d<FloatType>(maxX, minY);
			result[2] = point2d<FloatType>(maxX, maxY);
			result[3] = point2d<FloatType>(minX, maxY);
		}

		std::vector< point2d<FloatType> > getVertices() const {
			std::vector< point2d<FloatType> > result;
			result.resize(4);

			getVertices(result.data());

			return result;
		}


		std::vector< LineSegment2<FloatType> > getEdges() const
		{
			std::vector< LineSegment2<FloatType> > result;

			auto v = getVertices();

			result.push_back(LineSegment2<FloatType>(v[0], v[1]));
			result.push_back(LineSegment2<FloatType>(v[1], v[2]));
			result.push_back(LineSegment2<FloatType>(v[2], v[3]));
			result.push_back(LineSegment2<FloatType>(v[3], v[0]));

			result.push_back(LineSegment2<FloatType>(v[4], v[5]));
			result.push_back(LineSegment2<FloatType>(v[5], v[6]));
			result.push_back(LineSegment2<FloatType>(v[6], v[7]));
			result.push_back(LineSegment2<FloatType>(v[7], v[4]));

			result.push_back(LineSegment2<FloatType>(v[0], v[4]));
			result.push_back(LineSegment2<FloatType>(v[1], v[5]));
			result.push_back(LineSegment2<FloatType>(v[2], v[6]));
			result.push_back(LineSegment2<FloatType>(v[3], v[7]));

			return result;
		}

		//! point collision
		bool intersects(const point2d<FloatType>& p) const {
			if (p.x >= minX && p.x <= maxX &&
				p.y >= minY && p.y <= maxY) return true;
			else  return false;
		}

		//! triangle collision
		bool intersects(const point2d<FloatType>& p0, const point2d<FloatType>& p1, const point2d<FloatType>& p2) const {
			return intersection::intersectTriangleABBB(minB, maxB, p0, p1, p2);
		}

		//! bounding box collision
		bool intersects(const BoundingBox2<FloatType>& other) const {
			return 
				minX <= other.maxX && other.minX <= maxX &&
				minY <= other.maxY && other.minY <= maxY;
		}

		//! transformed bounding box collision
		//bool collision(const BoundingBox2<FloatType>& other, const matrix4x4<FloatType> &transform) const {
		//    BoundingBox2<FloatType> otherTransformed = other * transform;
		//    return collision(otherTransformed);
		//}

		FloatType getMaxExtent() const {
			FloatType d0 = maxX - minX;
			FloatType d1 = maxY - minY;
			return std::max(d0,d1);
		}

		FloatType getExtentX() const {
			return maxX - minX;
		}

		FloatType getExtentY() const {
			return maxY - minY;
		}

		point2d<FloatType> getExtent() const {
			return point2d<FloatType>(maxX - minX, maxY - minY);
		}

		point2d<FloatType> getMin() const {
			return point2d<FloatType>(minX, minY);
		}

		point2d<FloatType> getMax() const {
			return point2d<FloatType>(maxX, maxY);
		}

		point2d<FloatType> getCenter() const {
			point2d<FloatType> center = getMin() + getMax();
			center *= (FloatType)0.5;
			return center;
		}

		void setMin(const point2d<FloatType>& minValue) {
			minX = minValue.x;
			minY = minValue.y;
		}

		void setMax(const point2d<FloatType>& maxValue) {
			maxX = maxValue.x;
			maxY = maxValue.y;
		}

		void setMinX(FloatType v) { minX = v; }
		void setMinY(FloatType v) { minY = v; }
		void setMaxX(FloatType v) { maxX = v; }
		void setMaxY(FloatType v) { maxY = v; }

		FloatType getMinX() const { return minX; }
		FloatType getMinY() const { return minY; }
		FloatType getMaxX() const { return maxX; }
		FloatType getMaxY() const { return maxY; }

		//! scales the bounding box by the factor t (for t=1 the bb remains unchanged)
		void scale(FloatType x, FloatType y) {

			float scale[] = {x, y};
			for (unsigned int i = 0; i < 2; i++) {
				FloatType center = (FloatType)0.5 * (parameters[i] + parameters[i+2]);
				FloatType diff = parameters[i+2] - parameters[i];
				diff *= scale[i];
				diff *= (FloatType)0.5;
				parameters[i] = center - diff;
				parameters[i+2] = center + diff;
			}
		}

		//! scales the bounding box by the factor t (for t=1 the bb remains unchanged)
		void scale(FloatType t) {
			for (unsigned int i = 0; i < 2; i++) {
				FloatType center = (FloatType)0.5 * (parameters[i] + parameters[i+2]);
				FloatType diff = parameters[i+2] - parameters[i];
				diff *= t;
				diff *= (FloatType)0.5;
				parameters[i] = center - diff;
				parameters[i+2] = center + diff;
			}
		}

		//! transforms the bounding box (conservatively)
		void transform(const Matrix2x2<FloatType>& m) {
			point2d<FloatType> verts[8];
			getVertices(verts);
			reset();
			for (const auto& p : verts) {
				include(m * p);
			}
		}

		void translate(const point2d<FloatType>& t) {
			minB += t;
			maxB += t;
		}

		//! scales the bounding box (see scale)
		BoundingBox2<FloatType> operator*(FloatType t) const {
			BoundingBox2<FloatType> res = *this;
			res.scale(t);
			return res;
		}

		//! transforms the bounding box (see transform)
		BoundingBox2<FloatType> operator*(const Matrix2x2<FloatType>& m) const {
			BoundingBox2<FloatType> res = *this;
			res.transform(m);
			return res;
		}


		void setUnitCube() {
			minX = minY = 0;
			maxX = maxY = 1;
		}
	protected:
		union {
			struct {
				point2d<FloatType> minB;
				point2d<FloatType> maxB;
			};
			struct {
				FloatType minX, minY;
				FloatType maxX, maxY;
			};
			FloatType parameters[4];
		};
	};

	template<class FloatType>
	std::ostream& operator<< (std::ostream& s, const BoundingBox2<FloatType>& bb) {
		s << bb.getMin() << std::endl << bb.getMax() << std::endl;
		return s;
	}

	typedef BoundingBox2<float> BoundingBox2f;
	typedef BoundingBox2<double> BoundingBox2d;
	typedef BoundingBox2<int> BoundingBox2i;

	typedef BoundingBox2<float> bbox2f;
	typedef BoundingBox2<double> bbox2d;
	typedef BoundingBox2<int> bbox2i;

}  // namespace ml

#endif  // CORE_GRAPHICS_BoundingBox2_H_
