#pragma once

#ifndef OBJECT_ORIENTED_BOUNDIG_BOX3_H_
#define OBJECT_ORIENTED_BOUNDIG_BOX3_H_

namespace ml {

template <class FloatType>
class OrientedBoundingBox3 {
public:

	OrientedBoundingBox3() {
		setInvalid();
	}

    OrientedBoundingBox3(const BoundingBox3<FloatType> &box)
    {
        m_Anchor = box.getMin();
        m_AxesScaled[0] = vec3f::eX * box.getExtentX();
        m_AxesScaled[1] = vec3f::eY * box.getExtentY();
        m_AxesScaled[2] = vec3f::eZ * box.getExtentZ();
    }

	//! constructs an oriented bounding box using PCA
	OrientedBoundingBox3(const std::vector<point3d<FloatType>>& points) {

		auto pca = math::pointSetPCA(points);
		m_AxesScaled[0] = pca[0].first.getNormalized();
		m_AxesScaled[1] = pca[1].first.getNormalized();
		m_AxesScaled[2] = pca[2].first.getNormalized();

		computeAnchorAndExtentsForGivenNormalizedAxis(points);
	}

	//! creates an object oriented bounding for a given set of points with the same axes as the other OBB
	OrientedBoundingBox3(const std::vector<point3d<FloatType>>& points, const OrientedBoundingBox3& other) {
		m_AxesScaled[0] = other.m_AxesScaled[0].getNormalized();
		m_AxesScaled[1] = other.m_AxesScaled[1].getNormalized();
		m_AxesScaled[2] = other.m_AxesScaled[2].getNormalized();

		computeAnchorAndExtentsForGivenNormalizedAxis(points);
	}

	//! creates an object oriented bounding box given a set of points and 3 axes
	OrientedBoundingBox3(const std::vector<point3d<FloatType>>& points, const point3d<FloatType>& xAxis, const point3d<FloatType>& yAxis, const point3d<FloatType>& zAxis) {
		m_AxesScaled[0] = xAxis.getNormalized();
		m_AxesScaled[1] = yAxis.getNormalized();
		m_AxesScaled[2] = zAxis.getNormalized();

		computeAnchorAndExtentsForGivenNormalizedAxis(points);
	}

	//! creates an object oriented bounding box around a set of points with a given zAxis
	OrientedBoundingBox3(const std::vector<point3d<FloatType>>& points, const point3d<FloatType>& zAxis) {

		m_AxesScaled[2] = zAxis.getNormalized();

		point3d<FloatType> v, v0, v1;
		if (m_AxesScaled[2].x != (FloatType)0)		v = point3d<FloatType>(m_AxesScaled[2].z, -m_AxesScaled[2].x, m_AxesScaled[2].y);
		else if (m_AxesScaled[2].y != (FloatType)0)	v = point3d<FloatType>(m_AxesScaled[2].z, m_AxesScaled[2].x, -m_AxesScaled[2].y);
		else										v = point3d<FloatType>(-m_AxesScaled[2].z, m_AxesScaled[2].x, m_AxesScaled[2].y);
		v0 = (v ^ m_AxesScaled[2]);
		v1 = (v0 ^ m_AxesScaled[2]);
		v0.normalize();
		v1.normalize();

		std::vector < point2d<FloatType> > pointsProj(points.size());

		point2d<FloatType> pointsProjMean((FloatType)0, (FloatType)0);
		for (size_t i = 0; i < points.size(); i++) {
			pointsProj[i] = point2d<FloatType>(points[i] | v0, points[i] | v1);
		}

		vector< std::pair<point3d<FloatType>, FloatType> > pca = math::pointSetPCA(points);

		const point2d<FloatType>& ev0 = pca[0].first;
		const point2d<FloatType>& ev1 = pca[1].first;
		

		//Eigenvector computation has some numerical issues...
		assert((ev0 | ev1) < 0.001);

		m_AxesScaled[0] = v0 * ev0.x + v1 * ev0.y;
		m_AxesScaled[1] = v0 * ev1.x + v1 * ev1.y;

		m_AxesScaled[0].normalize();
		m_AxesScaled[1].normalize();

		computeAnchorAndExtentsForGivenNormalizedAxis(points);
	}

	//! creates an object oriented bounding box given an anchor and 3 axes
	OrientedBoundingBox3(const point3d<FloatType>& anchor, const point3d<FloatType>& xAxis, const point3d<FloatType>& yAxis, const point3d<FloatType>& zAxis) {
		m_AxesScaled[0] = xAxis;
		m_AxesScaled[1] = yAxis;
		m_AxesScaled[2] = zAxis;

		m_Anchor = anchor;
	}

	bool isValid() const {
		if (m_Anchor.x == -std::numeric_limits<FloatType>::max() || m_Anchor.y == -std::numeric_limits<FloatType>::max() || m_Anchor.z == -std::numeric_limits<FloatType>::max())	
			return false;
		else return true;
	}

	void setInvalid() {
		m_Anchor.x = m_Anchor.y = m_Anchor.z = -std::numeric_limits<FloatType>::max();
	}

	std::vector< point3d<FloatType> > getVertices() const
	{
		std::vector< point3d<FloatType> > result(8);

		result[0] = (m_Anchor + m_AxesScaled[0] * (FloatType)0.0 + m_AxesScaled[1] * (FloatType)0.0 + m_AxesScaled[2] * (FloatType)0.0);
		result[1] = (m_Anchor + m_AxesScaled[0] * (FloatType)1.0 + m_AxesScaled[1] * (FloatType)0.0 + m_AxesScaled[2] * (FloatType)0.0);
		result[2] = (m_Anchor + m_AxesScaled[0] * (FloatType)1.0 + m_AxesScaled[1] * (FloatType)1.0 + m_AxesScaled[2] * (FloatType)0.0);
		result[3] = (m_Anchor + m_AxesScaled[0] * (FloatType)0.0 + m_AxesScaled[1] * (FloatType)1.0 + m_AxesScaled[2] * (FloatType)0.0);

		result[4] = (m_Anchor + m_AxesScaled[0] * (FloatType)0.0 + m_AxesScaled[1] * (FloatType)0.0 + m_AxesScaled[2] * (FloatType)1.0);
		result[5] = (m_Anchor + m_AxesScaled[0] * (FloatType)1.0 + m_AxesScaled[1] * (FloatType)0.0 + m_AxesScaled[2] * (FloatType)1.0);
		result[6] = (m_Anchor + m_AxesScaled[0] * (FloatType)1.0 + m_AxesScaled[1] * (FloatType)1.0 + m_AxesScaled[2] * (FloatType)1.0);
		result[7] = (m_Anchor + m_AxesScaled[0] * (FloatType)0.0 + m_AxesScaled[1] * (FloatType)1.0 + m_AxesScaled[2] * (FloatType)1.0);

		return result;
	}


	//! returns the transformation matrix that transforms points into the space of the OBB
	inline Matrix4x4<FloatType> getOBBToWorld() const 
	{
		return Matrix4x4<FloatType>(
			m_AxesScaled[0].x, m_AxesScaled[1].x, m_AxesScaled[2].x, m_Anchor.x,
			m_AxesScaled[0].y, m_AxesScaled[1].y, m_AxesScaled[2].y, m_Anchor.y,
			m_AxesScaled[0].z, m_AxesScaled[1].z, m_AxesScaled[2].z, m_Anchor.z,
			0, 0, 0, 1);
	}

	//! returns a matrix that transforms to OBB space [0,1]x[0,1]x[0,1]
	inline Matrix4x4<FloatType> getWorldToOBB() const {
		//return getOOBBToWorld().getInverse();

		FloatType scaleValues[3] = { m_AxesScaled[0].length(), m_AxesScaled[1].length(), m_AxesScaled[2].length() };
		Matrix3x3<FloatType> worldToOBB3x3(m_AxesScaled[0] / scaleValues[0], m_AxesScaled[1] / scaleValues[1], m_AxesScaled[2] / scaleValues[2]);

		worldToOBB3x3(0, 0) /= scaleValues[0];	worldToOBB3x3(0, 1) /= scaleValues[0];	worldToOBB3x3(0, 2) /= scaleValues[0];
		worldToOBB3x3(1, 0) /= scaleValues[1];	worldToOBB3x3(1, 1) /= scaleValues[1];	worldToOBB3x3(1, 2) /= scaleValues[1];
		worldToOBB3x3(2, 0) /= scaleValues[2];	worldToOBB3x3(2, 1) /= scaleValues[2];	worldToOBB3x3(2, 2) /= scaleValues[2];

		point3d<FloatType> trans = worldToOBB3x3 * (-m_Anchor);
		Matrix4x4<FloatType> worldToOBB = worldToOBB3x3;
		worldToOBB.at(0, 3) = trans.x;
		worldToOBB.at(1, 3) = trans.y;
		worldToOBB.at(2, 3) = trans.z;

		return worldToOBB;
	}

	//! returns the center of the OBB
	point3d<FloatType> getCenter() const {
		return m_Anchor + (m_AxesScaled[0] + m_AxesScaled[1] + m_AxesScaled[2]) * (FloatType)0.5;
	}

	//! returns the n'th axis of the OBB
	const point3d<FloatType>& getAxis(unsigned int n) const {
		return m_AxesScaled[n];
	}

	//! returns the first axis of the OBB
	const point3d<FloatType>& getAxisX() const {
		return m_AxesScaled[0];
	}

	//! returns the second axis of the OBB
	const point3d<FloatType>& getAxisY() const {
		return m_AxesScaled[1];
	}

	//! returns the third axis of the OBB
	const point3d<FloatType>& getAxisZ() const {
		return m_AxesScaled[2];
	}

	point3d<FloatType> getExtent() const {
		return point3d<FloatType>(m_AxesScaled[0].length(), m_AxesScaled[1].length(), m_AxesScaled[2].length());
	}

	point3d<FloatType> getAnchor() const {
		return m_Anchor;
	}

	FloatType getVolume() const {
		point3d<FloatType> extent = getExtent();
		return extent.x * extent.y * extent.z;
	}

	//! returns the diagonal extent of the OBB
	FloatType getDiagonalLength() const {
		return (m_AxesScaled[0] + m_AxesScaled[1] + m_AxesScaled[2]).length();
	}

	void getCornerPoints(std::vector<point3d<FloatType>>& points) const {
		points.resize(8);
		getCornerPoints(&points[0]);
	} 

	std::vector< LineSegment3<FloatType> > getEdges() const
	{
		std::vector< LineSegment3<FloatType> > result;	result.reserve(12);
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

	//! scales the OBB
	void operator*=(const FloatType& scale) {
		point3d<FloatType> center = getCenter();
		m_AxesScaled[0] *= scale;
		m_AxesScaled[1] *= scale;
		m_AxesScaled[2] *= scale;
		m_Anchor = center - (m_AxesScaled[0] + m_AxesScaled[1] + m_AxesScaled[2]) * (FloatType)0.5;
	}
	//! returns a scaled OBB
	OrientedBoundingBox3<FloatType> operator*(const FloatType& scale) const {
		OrientedBoundingBox3<FloatType> res = *this;
		res *= scale;
		return res;
	}

	//! extends the OBB
	void operator+=(const FloatType& ext) {
		FloatType scaleValues[3] = { m_AxesScaled[0].length(), m_AxesScaled[1].length(), m_AxesScaled[2].length() };
		point3d<FloatType> center = getCenter();
		m_AxesScaled[0] *= (scaleValues[0] + ext) / scaleValues[0];
		m_AxesScaled[1] *= (scaleValues[1] + ext) / scaleValues[1];
		m_AxesScaled[2] *= (scaleValues[2] + ext) / scaleValues[2];
		m_Anchor = center - (m_AxesScaled[0] + m_AxesScaled[1] + m_AxesScaled[2]) * (FloatType)0.5;
	}
	//! returns an extended OBB
	OrientedBoundingBox3<FloatType> operator+(const FloatType& ext) const {
		OrientedBoundingBox3<FloatType> res = *this;
		res += ext;
		return res;
	}

	//! returns a transformed OBB
	void operator*=(const Matrix4x4<FloatType>& mat) {
		assert(mat.isAffine());
		m_Anchor = mat * m_Anchor;
		Matrix3x3<FloatType> rot = mat.getMatrix3x3();
		m_AxesScaled[0] = rot * m_AxesScaled[0];
		m_AxesScaled[1] = rot * m_AxesScaled[1];
		m_AxesScaled[2] = rot * m_AxesScaled[2];
	}

	bool intersects(const OrientedBoundingBox3<FloatType>& other) const {
		return intersection::intersectOBBOBB<FloatType>(m_Anchor, &m_AxesScaled[0], other.m_Anchor, &other.m_AxesScaled[0]);
	}


	//! scales the bounding box by the factor t (for t=1 the bb remains unchanged)
	void scale(FloatType x, FloatType y, FloatType z) {
		point3d<FloatType> extent = getExtent();
		FloatType scale[] = { x, y, z };
		point3d<FloatType> center(0, 0, 0);
		for (unsigned int i = 0; i < 3; i++) {
			center += (FloatType)0.5 * m_AxesScaled[i];
			m_AxesScaled[i] *= scale[i];
		}
		m_Anchor = center - (FloatType)0.5 * (m_AxesScaled[0] + m_AxesScaled[1] + m_AxesScaled[2]);
	}

	//! scales the bounding box by the factor t (for t=1 the bb remains unchanged)
	void scale(FloatType t) {
		scale(t, t, t);
	}
private:

	void computeAnchorAndExtentsForGivenNormalizedAxis(const std::vector<point3d<FloatType>>& points)
	{
		assert((m_AxesScaled[0] | m_AxesScaled[1]) < (FloatType)0.001);
		assert((m_AxesScaled[1] | m_AxesScaled[2]) < (FloatType)0.001);
		assert((m_AxesScaled[2] | m_AxesScaled[0]) < (FloatType)0.001);

		Matrix3x3<FloatType> worldToOBBSpace(m_AxesScaled[0], m_AxesScaled[1], m_AxesScaled[2]);
		Matrix3x3<FloatType> OBBSpaceToWorld = worldToOBBSpace.getTranspose();	//is an orthogonal matrix

		point3d<FloatType> minValues(std::numeric_limits<FloatType>::max(), std::numeric_limits<FloatType>::max(), std::numeric_limits<FloatType>::max());
		point3d<FloatType> maxValues(-std::numeric_limits<FloatType>::max(), -std::numeric_limits<FloatType>::max(), -std::numeric_limits<FloatType>::max());

		for (size_t i = 0; i < points.size(); i++) {
			point3d<FloatType> curr = worldToOBBSpace * points[i];
			if (curr.x < minValues.x)	minValues.x = curr.x;
			if (curr.y < minValues.y)	minValues.y = curr.y;
			if (curr.z < minValues.z)	minValues.z = curr.z;

			if (curr.x > maxValues.x)	maxValues.x = curr.x;
			if (curr.y > maxValues.y)	maxValues.y = curr.y;
			if (curr.z > maxValues.z)	maxValues.z = curr.z;
		}

		m_Anchor = OBBSpaceToWorld * minValues;

		FloatType extent[3];

		extent[0] = maxValues.x - minValues.x;
		extent[1] = maxValues.y - minValues.y;
		extent[2] = maxValues.z - minValues.z;


		//if bounding box has no extent; set invalid and return
		if (extent[0] < (FloatType)0.00001 ||
			extent[1] < (FloatType)0.00001 ||
			extent[2] < (FloatType)0.00001) {
			setInvalid();
			return;
		}

		m_AxesScaled[0] *= extent[0];
		m_AxesScaled[1] *= extent[1];
		m_AxesScaled[2] *= extent[2];
	}

	point3d<FloatType>	m_Anchor;
	point3d<FloatType>	m_AxesScaled[3];	//these axes are not normalized; they are scaled according to the extent

	/*

	OrientedBoundingBox3(const point3d<FloatType>* points, unsigned int numPoints) {
		computeFromPCA(points, numPoints);
	}

	OrientedBoundingBox3(const std::vector<point3d<FloatType>>& points) {
		computeFromPCA(points);
	}



	//! computes the bounding box using a pca
	void computeFromPCA(const std::vector<point3d<FloatType>>& points) {
		if (points.size() < 4) {
			setInvalid();
			return;
		}
		computeFromPCA(&points[0], points.size());
	}

	void computeFromPCA(const point3d<FloatType>* points, size_t numPoints) {

		//at least 4 points are required for a valid bounding box
		if (numPoints < 4)	{
			setInvalid();
			return;
		}

		point3d<FloatType> mean(0.0, 0.0, 0.0);
		for (unsigned int i = 0; i < numPoints; i++) {
			mean += points[i];
		}
		mean /= (FloatType)numPoints;

		Matrix3x3<FloatType> cov;
		cov.setZero();
		for (unsigned int i = 0; i < numPoints; i++) {
			point3d<FloatType> curr = points[i] - mean;
			cov += Matrix3x3<FloatType>::tensorProduct(curr, curr);
		}
		cov /= (FloatType)(numPoints - 1);

		//FloatType lambda[3];
		//bool validEVs = cov.computeEigenvaluesAndEigenvectorsNR(lambda[0], lambda[1], lambda[2], m_AxesScaled[0], m_AxesScaled[1], m_AxesScaled[2]);
		//assert(validEVs);

        //
        // TODO: implement this
        //
		//cov.computeEigenvaluesAndEigenvectorsNR(lambda[0], lambda[1], lambda[2], m_AxesScaled[0], m_AxesScaled[1], m_AxesScaled[2]);

		m_AxesScaled[0].normalize();
		m_AxesScaled[1].normalize();
		m_AxesScaled[2].normalize();

		computeAnchorAndExtentsForGivenNormalizedAxis(points, numPoints);
	}


	void getCornerPoints(point3d<FloatType>* points) const {
		points[0] = m_Anchor;
		points[1] = m_Anchor + m_AxesScaled[0];
		points[2] = m_Anchor + m_AxesScaled[0] + m_AxesScaled[1];
		points[3] = m_Anchor + m_AxesScaled[1];

		points[4] = m_Anchor + m_AxesScaled[2];
		points[5] = m_Anchor + m_AxesScaled[0] + m_AxesScaled[2];
		points[6] = m_Anchor + m_AxesScaled[0] + m_AxesScaled[1] + m_AxesScaled[2];
		points[7] = m_Anchor + m_AxesScaled[1] + m_AxesScaled[2];
	}

	void getEdgeIndices(std::vector<unsigned int>& edgeIndexList) const {
		edgeIndexList.resize(24);
		getEdgeIndices(&edgeIndexList[0]);
	}

	//! returns the edge indices corresponding to getCornerPoints (e.g., line rendering)
	void getEdgeIndices(unsigned int* edgeIndexList) const {
		//floor
		edgeIndexList[0] = 0;
		edgeIndexList[1] = 1;

		edgeIndexList[2] = 1;
		edgeIndexList[3] = 2;

		edgeIndexList[4] = 2;
		edgeIndexList[5] = 3;

		edgeIndexList[6] = 3;
		edgeIndexList[7] = 0;

		//verticals
		edgeIndexList[8] = 0;
		edgeIndexList[9] = 4;

		edgeIndexList[10] = 1;
		edgeIndexList[11] = 5;

		edgeIndexList[12] = 2;
		edgeIndexList[13] = 6;

		edgeIndexList[14] = 3;
		edgeIndexList[15] = 7;

		//ceiling
		edgeIndexList[16] = 4;
		edgeIndexList[17] = 5;

		edgeIndexList[18] = 5;
		edgeIndexList[19] = 6;

		edgeIndexList[20] = 6;
		edgeIndexList[21] = 7;

		edgeIndexList[22] = 7;
		edgeIndexList[23] = 4;
	}

	void getEdgeList(std::vector<point3d<FloatType>> &edges) const {
		edges.resize(24);
		getEdgeList(&edges[0]);
	}

	void getEdgeList(point3d<FloatType>* edges) const {
		unsigned int indices[24];
		getEdgeIndices(indices);
		point3d<FloatType> corners[8];
		getCornerPoints(corners);
		for (unsigned int i = 0; i < 24; i++) {
			edges[i] = corners[indices[i]];
		}
	}





	//! returns a matrix that transforms to OOBB space [0;extentX]x[0;extentY]x[0;extentZ]
	inline Matrix4x4<FloatType>  getWorldToOOBBNormalized() const {
		Matrix4x4<FloatType> worldToOOBB(m_AxesScaled[0].getNormalized(), m_AxesScaled[1].getNormalized(), m_AxesScaled[2].getNormalized());
		point3d<FloatType> trans = worldToOOBB * (-m_Anchor);
		worldToOOBB.at(0,3) = trans.x;
		worldToOOBB.at(1,3) = trans.y;
		worldToOOBB.at(2,3) = trans.z;
		return worldToOOBB;
	}

	//! tests whether a point lies within the bounding box or not
	inline bool contains(const point3d<FloatType>& p ) const {
		return isInUnitCube(getWorldToOOBB() * p);
	}

	//! tests whether a point is outside of the bounding box or not
	inline bool outside(const point3d<FloatType>& p) const {
		return !contains(p);
	}

	//! tests whether a set of points lies within the bounding box or not
	inline bool contains(const point3d<FloatType>* points, unsigned int numPoints, FloatType eps = (FloatType)0.00001) const {
		assert(numPoints);
		Matrix4x4<FloatType> worldToOOBB = getWorldToOOBB();
		for (unsigned int i = 0; i < numPoints; i++) {
			if (!isInUnitCube(worldToOOBB * points[i], eps))	return false;
		}
		return true;
	}

	inline bool contains(const OrientedBoundingBox3<FloatType> &other) {
		point3d<FloatType> cornerPoints[8];
		other.getCornerPoints(cornerPoints);
		return contains(cornerPoints, 8);
	}

	//! tests whether a set of points lies outside of the bounding box or not
	inline bool outside(const point3d<FloatType>* points, unsigned int numPoints, FloatType eps = (FloatType)0.00001) const {
		assert(numPoints);
		Matrix4x4<FloatType> worldToOOBB = getWorldToOOBB();
		for (unsigned int i = 0; i < numPoints; i++) {
			if (isInUnitCube(worldToOOBB * points[i], -eps))	return false;
		}
		return true;
	}

	//! returns the signed distance of the box to a plane (negative if plane is inside; positive if outside)
	inline FloatType planeDistance(const Plane<FloatType> &p) const {
		point3d<FloatType> extent = getExtent();

		FloatType r = (FloatType)0.5 * (
			extent.x * std::abs(p.getNormal() | (m_AxesScaled[0] / extent.x)) +
			extent.y * std::abs(p.getNormal() | (m_AxesScaled[1] / extent.y)) +
			extent.z * std::abs(p.getNormal() | (m_AxesScaled[2] / extent.z))
			);

		FloatType s = p.distanceToPointAbs(getCenter());
		return s - r;
	}

	//! tests a face against the box
	inline bool testFace(const point3d<FloatType>* points, FloatType eps = (FloatType)0.00001) const {
		//point3d<FloatType> planeNormal = ((points[1] - points[0])^(points[2] - points[0])).getNormalized();
		//FloatType planeDistance = (planeNormal | points[0]);
		//if (planeDistance < (FloatType)0) {	//make sure normal points away from origin (i.e., distance is positive)
		//	planeDistance = -planeDistance;
		//	planeNormal = -planeNormal;
		//}

		//assert(floatEqual(planeDistance, planeNormal | points[3], (FloatType)0.0001));

		//point3d<FloatType> center = getCenter();
		//point3d<FloatType> extent = getExtent();
		//
		//FloatType r = (FloatType)0.5 * (
		//	extent.x * std::abs(planeNormal | (m_AxesScaled[0] / extent.x)) +
		//	extent.y * std::abs(planeNormal | (m_AxesScaled[1] / extent.y)) +
		//	extent.z * std::abs(planeNormal | (m_AxesScaled[2] / extent.z))
		//	);

		//FloatType s = (planeNormal | center) - planeDistance;
		//if (std::abs(s) <= r + eps)	{	//box intersects face plane

		Plane<FloatType> p(points);
		FloatType planeDist = planeDistance(p);
		if (planeDist <= -eps) {
			Matrix4x4<FloatType> worldToOOBB = getWorldToOOBB();
			point3d<FloatType> pointsOOBB[4];
			for (unsigned int i = 0; i < 4; i++) {
				pointsOOBB[i] = worldToOOBB * points[i];
			}
			if (pointsOOBB[0].x >= (FloatType)1.0 - eps && pointsOOBB[1].x >= (FloatType)1.0 - eps && pointsOOBB[2].x >= (FloatType)1.0 - eps && pointsOOBB[3].x >= (FloatType)1.0 - eps)	return false;
			if (pointsOOBB[0].y >= (FloatType)1.0 - eps && pointsOOBB[1].y >= (FloatType)1.0 - eps && pointsOOBB[2].y >= (FloatType)1.0 - eps && pointsOOBB[3].y >= (FloatType)1.0 - eps)	return false;
			if (pointsOOBB[0].z >= (FloatType)1.0 - eps && pointsOOBB[1].z >= (FloatType)1.0 - eps && pointsOOBB[2].z >= (FloatType)1.0 - eps && pointsOOBB[3].z >= (FloatType)1.0 - eps)	return false;

			if (pointsOOBB[0].x <= eps && pointsOOBB[1].x <= eps && pointsOOBB[2].x <= eps && pointsOOBB[3].x <= eps)	return false;
			if (pointsOOBB[0].y <= eps && pointsOOBB[1].y <= eps && pointsOOBB[2].y <= eps && pointsOOBB[3].y <= eps)	return false;
			if (pointsOOBB[0].z <= eps && pointsOOBB[1].z <= eps && pointsOOBB[2].z <= eps && pointsOOBB[3].z <= eps)	return false;

			return true;	//face intersects box
		} else {
			return false;	//outside
		}

	}

	FloatType getMinPlaneExtension(const OrientedBoundingBox3& box, point3d<FloatType>* facePoints, const point3d<FloatType>& faceNormal) {
		FloatType minPlaneExtension = FLT_MAX;
		for (unsigned int i = 0; i < 6; i++) {
			OOBB_PLANE which = (OOBB_PLANE)i;
			Plane<FloatType> currPlane = box.getPlane(which);
			assert(floatEqual(currPlane.getNormal().length(), 1.0f));
			assert(floatEqual(faceNormal.length(), 1.0f));
			if ((currPlane.getNormal() | faceNormal) < (FloatType)0.1)	continue;
			FloatType maxNegDist = FLT_MAX; 
			unsigned int whichPoint = INT_MAX;
			for (unsigned int j = 0; j < 4; j++) {
				FloatType dist = currPlane.distanceToPoint(facePoints[j]);
				if (dist >= (FloatType)0)	continue;
				else if (dist < maxNegDist) {
					whichPoint = j;
					maxNegDist = dist;
				}
			}
			if (maxNegDist == FLT_MAX)	continue;
			FloatType planeExt = -maxNegDist / (currPlane.getNormal() | faceNormal);
			assert(planeExt > 0);
			if (planeExt < minPlaneExtension)	minPlaneExtension = planeExt;
		}
		//assert(minPlaneExtension != FLT_MAX);
		if (minPlaneExtension == FLT_MAX)	minPlaneExtension = (FloatType)0;
		return minPlaneExtension;
	}


	void makeZPlaneOutsideBoxes(const OrientedBoundingBox3& box0, const OrientedBoundingBox3& box1) {

		////swap axis until z is min extent
		//point3d<FloatType> extent = getExtent();
		//FloatType minExt = std::min(std::min(extent.x, extent.y), extent.z);
		//if (minExt == extent.x)	{
		//	swapAxes();
		//} else if (minExt == extent.y)  {
		//	swapAxes();
		//	swapAxes();
		//}
		//assert(getExtent().z == minExt);

		{
			unsigned int which = (unsigned int)-1;
			FloatType maxAssess = -FLT_MAX;
			for (unsigned int i = 0; i < 3; i++) {
				Plane<FloatType> front, back;
				front = getPlaneZFront();
				back = getPlaneZBack();

				//distance in box is negative; outside positive - we must make sure that we have at least two positive distances
				FloatType dist0front = box0.planeDistance(front);
				FloatType dist0back = box0.planeDistance(back);

				FloatType dist1front = box1.planeDistance(front);
				FloatType dist1back = box1.planeDistance(back);

				FloatType assess0 = std::min(dist0front, (FloatType)0) + std::min(dist1back, (FloatType)0);
				FloatType assess1 = std::min(dist1front, (FloatType)0) + std::min(dist0back, (FloatType)0);

				FloatType assess = std::max(assess0, assess1);
				if (assess > maxAssess) {
					maxAssess = assess;
					which = i;
				}
				swapAxes();
			}

			for (unsigned int i = 0; i < which; i++) {
				swapAxes();
			}

		}




		Plane<FloatType> front, back;
		front = getPlaneZFront();
		back = getPlaneZBack();

		//distance in box is negative; outside positive - we must make sure that we have at least two positive distances
		FloatType dist0front = box0.planeDistance(front);
		FloatType dist0back = box0.planeDistance(back);
		
		FloatType dist1front = box1.planeDistance(front);
		FloatType dist1back = box1.planeDistance(back);

		FloatType assess0 = std::min(dist0front, (FloatType)0) + std::min(dist1back, (FloatType)0);
		FloatType assess1 = std::min(dist1front, (FloatType)0) + std::min(dist0back, (FloatType)0);

		FloatType eps = (FloatType)0.0001;
		if (assess0 > assess1) {
			////extend front to bb0 and back to bb1
			//FloatType extFront = dist0front;
			//extendInZFront(std::abs(extFront) + eps);
			//FloatType extBack = dist1back;
			//extendInZBack(std::abs(extBack) + eps);

			point3d<FloatType> zFront[4];
			getFaceZFront(zFront);
			FloatType extFront = std::min(getMinPlaneExtension(box0, zFront, front.getNormal()), std::abs(dist0front));
			extendInZFront(std::abs(extFront) + eps);

			point3d<FloatType> zBack[4];
			getFaceZBack(zBack);
			FloatType extBack = std::min(getMinPlaneExtension(box1, zBack, back.getNormal()), std::abs(dist1back));
			extendInZBack(std::abs(extBack) + eps);

		} else {
			////extend back to bb0 and front to bb1
			//FloatType extBack = dist0back;
			//extendInZBack(std::abs(extBack) + eps);
			//FloatType extFront = dist1front;
			//extendInZFront(std::abs(extFront) + eps);

			point3d<FloatType> zBack[4];
			getFaceZBack(zBack);
			FloatType extBack = std::min(getMinPlaneExtension(box0, zBack, back.getNormal()), std::abs(dist0back));
			extendInZBack(std::abs(extBack) + eps);

			point3d<FloatType> zFront[4];
			getFaceZFront(zFront);
			FloatType extFront = std::min(getMinPlaneExtension(box1, zFront, front.getNormal()), std::abs(dist1front));
			extendInZFront(std::abs(extFront) + eps);
		}


		////debug stuff
		//front = getPlaneZFront();
		//back = getPlaneZBack();

		//FloatType resdist0front = box0.planeDistance(front);
		//FloatType resdist0back = box0.planeDistance(back);

		//FloatType resdist1front = box1.planeDistance(front);
		//FloatType resdist1back = box1.planeDistance(back);


		//point3d<FloatType> zFront[4];
		//point3d<FloatType> zBack[4];
		//getFaceZFront(zFront);
		//getFaceZBack(zBack);
		//assert(
		//	(!box0.testFace(zFront) && !box1.testFace(zBack)) ||
		//	(!box0.testFace(zBack) && !box1.testFace(zFront))
		//	);
	}

	//! intersects two object oriented bounding boxes; the return value is another OOBB that convervatively bounds the intersecting volume
	OrientedBoundingBox3 intersect(const OrientedBoundingBox3& other) const {

		if (this == &other) {
			return *this;
		}

		OrientedBoundingBox3 res;
		
		//try early reject (if there is no intersection)
		if ((getCenter() - other.getCenter()).length() > (FloatType)0.5 * (getDiagonalLength() + other.getDiagonalLength()))	return res;
	
		std::vector<point3d<FloatType>> contactPoints;
		computeContactPoints(other, contactPoints);

		std::vector<point3d<FloatType>> contactPointsOther;
		other.computeContactPoints(*this, contactPointsOther);

		contactPoints.insert(contactPoints.end(), contactPointsOther.begin(), contactPointsOther.end());

		res.computeFromPCA(contactPoints);

		//make sure either front and back plane is outside of either input OOBB
		if (res.isValid()) {
			for (unsigned int i = 0; i < 4; i++) {
				point3d<FloatType> zFront[4];
				point3d<FloatType> zBack[4];
				res.getFaceZFront(zFront);
				res.getFaceZBack(zBack);
				if (
					(!testFace(zFront) && !other.testFace(zBack)) ||
					(!testFace(zBack) && !other.testFace(zFront))
					) {
					break; 
				} else {
					res.swapAxes();
					if (i == 3) {
						res.makeZPlaneOutsideBoxes(*this, other);
						break;
					}
				}
			}
		}

		return res;
	}


	//! swaps the axes of the OOBB (x->z; z->y; y->x)
	inline void swapAxes() {
		point3d<FloatType> tmp = m_AxesScaled[0];
		m_AxesScaled[0] = m_AxesScaled[1];
		m_AxesScaled[1] = m_AxesScaled[2];
		m_AxesScaled[2] = tmp;
	}

	//! returns the four corner pointds of the z = 0; front plane
	inline void getFaceZFront(point3d<FloatType>* points) const {
		points[0] = m_Anchor;
		points[1] = m_Anchor + m_AxesScaled[0];
		points[2] = m_Anchor + m_AxesScaled[0] + m_AxesScaled[1];
		points[3] = m_Anchor + m_AxesScaled[1];
	}

	enum OOBB_PLANE {
		X_FRONT,
		X_BACK,
		Y_FRONT,
		Y_BACK,
		Z_FRONT,
		Z_BACK
	};

	inline Plane<FloatType> getPlane(OOBB_PLANE which) const {
		switch (which) {
			case X_FRONT:	return getPlaneXFront();
			case X_BACK:	return getPlaneXBack();
			case Y_FRONT:	return getPlaneYFront();
			case Y_BACK:	return getPlaneYBack();
			case Z_FRONT:	return getPlaneZFront();
			case Z_BACK:	return getPlaneZBack();
			default: assert(false);
		}
		return Plane<FloatType>();
	}


	inline Plane<FloatType> getPlaneXFront() const {
		return Plane<FloatType>(-m_AxesScaled[0].getNormalized(), m_Anchor);
	}

	inline Plane<FloatType> getPlaneXBack() const {
		return Plane<FloatType>(m_AxesScaled[0].getNormalized(), m_Anchor + m_AxesScaled[0]);
	}

	inline Plane<FloatType> getPlaneYFront() const {
		return Plane<FloatType>(-m_AxesScaled[1].getNormalized(), m_Anchor);
	}

	inline Plane<FloatType> getPlaneYBack() const {
		return Plane<FloatType>(m_AxesScaled[1].getNormalized(), m_Anchor + m_AxesScaled[1]);
	}

	inline Plane<FloatType> getPlaneZFront() const {
		return Plane<FloatType>(-m_AxesScaled[2].getNormalized(), m_Anchor);
	}

	inline Plane<FloatType> getPlaneZBack() const {
		return Plane<FloatType>(m_AxesScaled[2].getNormalized(), m_Anchor + m_AxesScaled[2]);
	}


	//! returns the four corner points of the z = 1; back plane
	inline void getFaceZBack(point3d<FloatType>* points) const {
		points[0] = m_Anchor + m_AxesScaled[2];
		points[1] = m_Anchor + m_AxesScaled[0] + m_AxesScaled[2];
		points[2] = m_Anchor + m_AxesScaled[0] + m_AxesScaled[1] + m_AxesScaled[2];
		points[3] = m_Anchor + m_AxesScaled[1] + m_AxesScaled[2];
	}

	inline void extendInZBack(FloatType ext) {
		m_AxesScaled[2] = m_AxesScaled[2] + m_AxesScaled[2].getNormalized() * ext;
	}

	inline void extendInZFront(FloatType ext) {
		m_Anchor = m_Anchor - m_AxesScaled[2].getNormalized() * ext;
		m_AxesScaled[2] = m_AxesScaled[2] + m_AxesScaled[2].getNormalized() * ext;
	}


	static OrientedBoundingBox3 interpolateLinear(const OrientedBoundingBox3& oobb0, const OrientedBoundingBox3& oobb1, float t) {
		OrientedBoundingBox3 ret;
		assert(floatEqual((oobb0.m_AxesScaled[0].getNormalized() - oobb1.m_AxesScaled[0].getNormalized()).length(), (FloatType)0));
		assert(floatEqual((oobb0.m_AxesScaled[1].getNormalized() - oobb1.m_AxesScaled[1].getNormalized()).length(), (FloatType)0));
		assert(floatEqual((oobb0.m_AxesScaled[2].getNormalized() - oobb1.m_AxesScaled[2].getNormalized()).length(), (FloatType)0));

		ret.m_Anchor = lerp(oobb0.m_Anchor, oobb1.m_Anchor, t);
		ret.m_AxesScaled[0] = lerp(oobb0.m_AxesScaled[0], oobb1.m_AxesScaled[0], t);
		ret.m_AxesScaled[1] = lerp(oobb0.m_AxesScaled[1], oobb1.m_AxesScaled[1], t);
		ret.m_AxesScaled[2] = lerp(oobb0.m_AxesScaled[2], oobb1.m_AxesScaled[2], t);

		assert(ret.isValid());

		return ret;
	}


private:
	static inline bool isInUnitInterval(const FloatType &v, FloatType eps = (FloatType)0.00001) {
		if (v >= -eps && v <= 1 + eps) return true;
		return false;
	}
	static inline bool isInUnitCube(const point3d<FloatType>& p, FloatType eps = (FloatType)0.00001) {
		if (p.x >= -eps && p.x <= 1 + eps &&
			p.y >= -eps && p.y <= 1 + eps &&
			p.z >= -eps && p.z <= 1 + eps) return true;
		return false;
	}
	point3d<FloatType>	m_Anchor;
	point3d<FloatType>	m_AxesScaled[3];

	//! computes the contact points of edges
	inline void computeContactPoints( const OrientedBoundingBox3 &other, std::vector<point3d<FloatType>> &contactPoints ) const
	{
		Matrix4x4<FloatType> OOBBToWorld = getOOBBToWorld();
		Matrix4x4<FloatType> worldToOOBB = getWorldToOOBB();


		point3d<FloatType> otherPoints[8];
		other.getCornerPoints(otherPoints);

		for (unsigned int i = 0; i < 8; i++) {
			otherPoints[i] = worldToOOBB * otherPoints[i];
			if (isInUnitCube(otherPoints[i])) contactPoints.push_back(otherPoints[i]);
		}

		unsigned int edgeIndices[24];
		getEdgeIndices(edgeIndices);

		for (unsigned int i = 0; i < 12; i++) {
			point3d<FloatType> o = otherPoints[edgeIndices[2*i+0]];
			point3d<FloatType> d = otherPoints[edgeIndices[2*i+1]] - o;

			//test against all 6 planes
			FloatType tx0 = -o.x/d.x;
			point3d<FloatType> px0 = o + tx0 * d;
			if (isInUnitInterval(tx0) && isInUnitCube(px0)) contactPoints.push_back(px0);

			FloatType tx1 = ((FloatType)1.0 - o.x)/d.x;
			point3d<FloatType> px1 = o + tx1 * d;
			if (isInUnitInterval(tx1) && isInUnitCube(px1)) contactPoints.push_back(px1);

			FloatType ty0 = -o.y/d.y;
			point3d<FloatType> py0 = o + ty0 * d;
			if (isInUnitInterval(ty0) && isInUnitCube(py0)) contactPoints.push_back(py0);

			FloatType ty1 = ((FloatType)1.0 - o.y)/d.y;
			point3d<FloatType> py1 = o + ty1 * d;
			if (isInUnitInterval(ty1) && isInUnitCube(py1)) contactPoints.push_back(py1);

			FloatType tz0 = -o.z/d.z;
			point3d<FloatType> pz0 = o + tz0 * d;
			if (isInUnitInterval(tz0) && isInUnitCube(pz0)) contactPoints.push_back(pz0);

			FloatType tz1 = ((FloatType)1.0 - o.z)/d.z;
			point3d<FloatType> pz1 = o + tz1 * d;
			if (isInUnitInterval(tz1) && isInUnitCube(pz1)) contactPoints.push_back(pz1);
		}

		for (unsigned int i = 0; i < contactPoints.size(); i++) {
			contactPoints[i] = OOBBToWorld * contactPoints[i];
		}
		
	}
	*/
};

template<class FloatType>
OrientedBoundingBox3<FloatType> operator*(const Matrix4x4<FloatType> &mat, const OrientedBoundingBox3<FloatType>& oobb) {
	OrientedBoundingBox3<FloatType> res = oobb;
	res *= mat;
	return res;
}

template<class FloatType>
std::ostream& operator<<(std::ostream& os, const OrientedBoundingBox3<FloatType>& obb)  {
	os << obb.getAxisX() << std::endl << obb.getAxisY() << std::endl << obb.getAxisZ() << std::endl;
	os << "Extent: " << obb.getExtent() << std::endl;
	os << "Anchor: " << obb.getAnchor() << std::endl;
	os << "Volume: " << obb.getVolume() << std::endl;
	return os;
}

typedef OrientedBoundingBox3<float> OrientedBoundingBox3f;
typedef OrientedBoundingBox3<double> OrientedBoundingBox3d;

typedef OrientedBoundingBox3<float> OBBf;
typedef OrientedBoundingBox3<double> OBBd;

} //namespace ml

#endif
