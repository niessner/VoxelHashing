
#include "stdafx.h"

#include "ChristophSensor.h"

#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <string>


ChristophSensor::ChristophSensor()
{
	unsigned int depthWidth = 1024;
	unsigned int depthHeight = 768;

	unsigned int colorWidth = 1024;
	unsigned int colorHeight = 768;

	DepthSensor::init(depthWidth, depthHeight, colorWidth, colorHeight);

	//mat4f intrinsics(
	//	(float)1.2264958231570761e+003, 0.0f, (float)5.1823698806762695e+002, 0.0f,
	//	0.0f, (float)1.2264958231570761e+003, (float)3.8550175094604492e+002, 0.0f,
	//	0.0f, 0.0f, 1.0f, 0.0f,
	//	0.0f, 0.0f, 0.0f, 1.0f
	//	);
	//DepthSensor::initializeIntrinsics(intrinsics);
	//DepthSensor::initializeIntrinsics((float)1.2264958231570761e+003, (float)1.2264958231570761e+003, (float)5.1823698806762695e+002, (float)3.8550175094604492e+002);

	mat4f intrinsics(
		(float)1.2362915678039572e+003, 0.0f, (float)5.0902931976318359e+002, 0.0f, 
		0.0f, (float)1.2362915678039572e+003, (float)3.8060153961181641e+002, 0.0f, 
		0.0f, 0.0f, 1.0f, 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f);
	initializeIntrinsics(intrinsics(0,0),intrinsics(1,1),intrinsics(0,2),intrinsics(1,2));
	m_StringCounter = NULL;
}

ChristophSensor::~ChristophSensor()
{
	SAFE_DELETE(m_StringCounter);
}

HRESULT ChristophSensor::createFirstConnected()
{
	std::string basePath = "..\\Data\\Christoph\\results_face1\\";
	m_StringCounter = new StringCounter(basePath + "disparity_", "bin", 6, 1);
	return S_OK;
}

HRESULT ChristophSensor::processDepth()
{
	std::string filename = m_StringCounter->getNext();
	std::ifstream input(filename, std::ofstream::binary);

	float* rawDepth = new float[getDepthWidth()*getDepthHeight()];
	input.read((char*)rawDepth, sizeof(float)*getDepthWidth()*getDepthHeight());

	const float fx = (float)1.2362915678039572e+003;
	const float baseline = (float)550.84;

	for (unsigned int i = 0; i < getDepthWidth()*getDepthHeight(); i++) {
		float depth = (fx*baseline) / rawDepth[i];
		if (rawDepth[i] == 0.0f) depth = 0.0f;
		m_depthD16[i] = (USHORT)(depth);
	}

	SAFE_DELETE_ARRAY(rawDepth);
	input.close();
	return S_OK;
}
