#include "stdafx.h"

#include "ImageReaderSensor.h"


ImageReaderSensor::ImageReaderSensor()
{
	init(640, 480, 640, 480);

	//default path should be actually overwritten
	m_BaseFilename = "";
	//m_BaseFilename = "../stanfordData/copyroom_png/";
	m_NumFrames = 0;
}

ImageReaderSensor::~ImageReaderSensor()
{

}

HRESULT ImageReaderSensor::createFirstConnected()
{
	HRESULT hr = S_OK;

	//what Qian-Yi / Vladlen tell us
	float focalLengthX = 525.0f;
	float focalLengthY = 525.0f;
	//float cx = 319.5f;
	//float cy = 239.5f;

	//what the oni framework gives us
	//float focalLengthX = 570.34f;
	//float focalLengthY = 570.34f;
	float cx = 320.0f;
	float cy = 240.0f;
	initializeIntrinsics(focalLengthX, focalLengthY, cx, cy);

	m_CurrentFrameNumberColor = 0;
	m_CurrentFrameNumberDepth = 0;
	return hr;
}

HRESULT ImageReaderSensor::processDepth()
{
	HRESULT hr = S_OK;
	if (m_CurrentFrameNumberDepth >= m_NumFrames) {
		return S_FALSE;
	}
	std::cout << "Processing Depth Frame " << m_CurrentFrameNumberDepth << std::endl;
	char frameNumber_c[10];
	sprintf_s(frameNumber_c,"%06d", m_CurrentFrameNumberDepth+1);
	std::string frameNumber(frameNumber_c);
	std::string currFileName = m_BaseFilename;
	currFileName.append("depth/").append(frameNumber).append(".png");
	DepthImage image;
	FreeImageWrapper::loadImage(currFileName, image);
	image.flipY();
	for (UINT i = 0; i < getDepthWidth() * getDepthHeight(); i++) {
		m_depthD16[i] = (USHORT)(image.getDataPointer()[i] * 1000);
	}
	m_CurrentFrameNumberDepth++;
	return hr;
}

HRESULT ImageReaderSensor::processColor()
{
	HRESULT hr = S_OK;
	if (m_CurrentFrameNumberColor >= m_NumFrames) {
		return S_FALSE;
	}

	bool readColor = false;

	if (readColor) {
		char frameNumber_c[10];
		sprintf_s(frameNumber_c,"%06d", m_CurrentFrameNumberDepth+1);
		std::string frameNumber(frameNumber_c);
		std::string currFileName = m_BaseFilename;
		currFileName.append("color/").append(frameNumber).append(".png");
		ColorImageRGB image;
		FreeImageWrapper::loadImage(currFileName, image);
		image.flipY();
		for (UINT i = 0; i < getDepthWidth() * getDepthHeight(); i++) {
			vec3f c = image.getDataPointer()[i];
			c = 255.0f*c;

			m_colorRGBX[4*i+0] = (BYTE)c.x;
			m_colorRGBX[4*i+1] = (BYTE)c.y; 
			m_colorRGBX[4*i+2] = (BYTE)c.z;
			m_colorRGBX[4*i+3] = 255; 
		}
	}
	m_CurrentFrameNumberColor++;
	return hr;
}
