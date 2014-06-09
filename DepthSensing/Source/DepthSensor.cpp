
#include "stdafx.h"

#include "DepthSensor.h"
#include "GlobalAppState.h"

#include <limits>

DepthSensor::DepthSensor()
{
	m_depthWidth  = 0;
	m_depthHeight = 0;

	m_colorWidth  = 0;
	m_colorHeight = 0;

	m_depthD16 = NULL;
	m_colorRGBX = NULL;
}

void DepthSensor::init(unsigned int depthWidth, unsigned int depthHeight, unsigned int colorWidth, unsigned int colorHeight)
{
	m_depthWidth  = static_cast<LONG>(depthWidth);
	m_depthHeight = static_cast<LONG>(depthHeight);

	m_colorWidth  = static_cast<LONG>(colorWidth);
	m_colorHeight = static_cast<LONG>(colorHeight);

	SAFE_DELETE_ARRAY(m_depthD16);
	m_depthD16 = new USHORT[m_depthWidth*m_depthHeight];
	memset(m_depthD16, 0, sizeof(USHORT)*m_depthWidth*m_depthHeight);

	SAFE_DELETE_ARRAY(m_colorRGBX);
	m_colorRGBX = new BYTE[m_colorWidth*m_colorHeight*cBytesPerPixel];
	memset(m_colorRGBX, 0, sizeof(BYTE)*m_colorWidth*m_colorHeight*cBytesPerPixel);
}

DepthSensor::~DepthSensor()
{
	// done with pixel data
	SAFE_DELETE_ARRAY(m_colorRGBX);
	SAFE_DELETE_ARRAY(m_depthD16);

	reset();
}

void DepthSensor::savePointCloud( const std::string& filename, const mat4f& transform) const
{
	std::vector<vec3f> points;
	std::vector<vec3f> colors;
	std::vector<vec3f> normals;

	computePointCurrentPointCloud(points, colors, normals, transform);

	PointCloudIOf::saveToFile(filename, &points, &normals, &colors);
	//if (colors.size() > 0) {
	//	assert(points.size() == colors.size());
	//	PointCloudIOf::saveToFile(filename, &points, NULL, &colors);
	//} else {
	//	PointCloudIOf::saveToFile(filename, &points, NULL, NULL);
	//}
}

void DepthSensor::initializeIntrinsics( float fovX, float fovY, float centerX, float centerY, float k1 /*= 0.0f*/, float k2 /*= 0.0f*/, float k3 /*= 0.0f*/, float p1 /*= 0.0f*/, float p2 /*= 0.0f*/ )
{
	//depth width and height must be already set here
	const float convX = (float)GlobalAppState::getInstance().s_windowWidth / (float)getDepthWidth();
	const float convY = (float)GlobalAppState::getInstance().s_windowHeight / (float)getDepthHeight();



	m_intrinsics.fx = fovX;
	m_intrinsics.fy = fovY;
	m_intrinsics.mx = centerX;
	m_intrinsics.my = centerY;
	m_intrinsics.k1 = k1;
	m_intrinsics.k2 = k2;
	m_intrinsics.k3 = k3;
	m_intrinsics.p1 = p1;
	m_intrinsics.p2 = p2;

	m_intrinsicsOriginal = m_intrinsics;
	m_intrinsics.fx *= convX;
	m_intrinsics.fy *= convY;
	m_intrinsics.mx *= convX;
	m_intrinsics.my *= convY;


	//m_intrinsics = mat4f(	
	//	fovX, 0.0f, 0.0f, centerX,										
	//	0.0f, -fovY, 0.0f, centerY,										
	//	0.0f, 0.0f, 1.0f, 0.0f,
	//	0.0f, 0.0f, 0.0f, 1.0f);

	//m_intrinsicsInv = mat4f(	
	//	1.0f/fovX, 0.0f, 0.0f, -centerX*1.0f/fovX,
	//	0.0f, -1.0f/fovY, 0.0f, centerY*1.0f/fovY,
	//	0.0f, 0.0f, 1.0f, 0.0f,
	//	0.0f, 0.0f, 0.0f, 1.0f);

	m_intrinsics.print();
}
