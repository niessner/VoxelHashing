
#include "stdafx.h"

#include "BinaryDumpReader.h"
#include "GlobalAppState.h"

#include <algorithm>
#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <string>
 
BinaryDumpReader::BinaryDumpReader()
{
	m_NumFrames = 0;
	m_CurrFrame = 0;
	m_bHasColorData = false;
	m_DepthD16Array = NULL;
	m_ColorRGBXArray = NULL;
	//parameters are read from the calibration file
}

BinaryDumpReader::~BinaryDumpReader()
{
	releaseData();
}

HRESULT BinaryDumpReader::createFirstConnected()
{
	std::string filename = GlobalAppState::getInstance().s_BinaryDumpReaderSourceFile;
	std::cout << "Start loading binary dump" << std::endl;
	//BinaryDataStreamZLibFile inputStream(filename, false);
	BinaryDataStreamFile inputStream(filename, false);
	CalibratedSensorData sensorData;
	inputStream >> sensorData;
	std::cout << "Loading finished" << std::endl;
	std::cout << sensorData << std::endl;

	DepthSensor::init(sensorData.m_DepthImageWidth, sensorData.m_DepthImageHeight, std::max(sensorData.m_ColorImageWidth,1u), std::max(sensorData.m_ColorImageHeight,1u));
	mat4f intrinsics(sensorData.m_CalibrationDepth.m_Intrinsic);
	initializeIntrinsics(sensorData.m_CalibrationDepth.m_Intrinsic(0,0), sensorData.m_CalibrationDepth.m_Intrinsic(1,1), sensorData.m_CalibrationDepth.m_Intrinsic(0,2), sensorData.m_CalibrationDepth.m_Intrinsic(1,2));

	m_NumFrames = sensorData.m_DepthNumFrames;
	assert(sensorData.m_ColorNumFrames == sensorData.m_DepthNumFrames || sensorData.m_ColorNumFrames == 0);		
	releaseData();
	m_DepthD16Array = new USHORT*[m_NumFrames];
	for (unsigned int i = 0; i < m_NumFrames; i++) {
		m_DepthD16Array[i] = new USHORT[getDepthWidth()*getDepthHeight()];
		for (unsigned int k = 0; k < getDepthWidth()*getDepthHeight(); k++) {
			m_DepthD16Array[i][k] = (USHORT)(sensorData.m_DepthImages[i][k]*1000.0f + 0.5f);
		}
	}

	std::cout << "loading depth done" << std::endl;
	if (sensorData.m_ColorImages.size() > 0) {
		m_bHasColorData = true;
		m_ColorRGBXArray = new BYTE*[m_NumFrames];
		for (unsigned int i = 0; i < m_NumFrames; i++) {
			m_ColorRGBXArray[i] = new BYTE[getColorWidth()*getColorHeight()*getColorBytesPerPixel()];
			for (unsigned int k = 0; k < getColorWidth()*getColorHeight(); k++) {
				const BYTE* c = (BYTE*)&(sensorData.m_ColorImages[i][k]); 
				m_ColorRGBXArray[i][k*getColorBytesPerPixel()+0] = c[0];
				m_ColorRGBXArray[i][k*getColorBytesPerPixel()+1] = c[1];
				m_ColorRGBXArray[i][k*getColorBytesPerPixel()+2] = c[2];
				m_ColorRGBXArray[i][k*getColorBytesPerPixel()+3] = 255;
				//I don't know really why this has to be swapped...
			}
			//std::string outFile = "colorout//color" + std::to_string(i) + ".png";
			//ColorImageR8G8B8A8 image(getColorHeight(), getColorWidth(), (vec4uc*)m_ColorRGBXArray[i]);
			//FreeImageWrapper::saveImage(outFile, image);
		}
	} else {
		m_bHasColorData = false;
	}
	sensorData.deleteData();

	std::cout << "loading color done" << std::endl;


	return S_OK;
}

HRESULT BinaryDumpReader::processDepth()
{
	if (m_CurrFrame < m_NumFrames) {
		std::cout << "curr Frame " << m_CurrFrame << std::endl;
		memcpy(m_depthD16, m_DepthD16Array[m_CurrFrame], sizeof(USHORT)*getDepthWidth()*getDepthHeight());
		
		if (m_bHasColorData) {
			memcpy(m_colorRGBX, m_ColorRGBXArray[m_CurrFrame], getColorBytesPerPixel()*getColorWidth()*getColorHeight());
		}

		m_CurrFrame++;
		return S_OK;
	} else {
		return S_FALSE;
	}
}

void BinaryDumpReader::releaseData()
{
	for (unsigned int i = 0; i < m_NumFrames; i++) {
		if (m_DepthD16Array)	SAFE_DELETE(m_DepthD16Array[i]);
		if (m_ColorRGBXArray)	SAFE_DELETE(m_ColorRGBXArray[i]);
	}
	SAFE_DELETE_ARRAY(m_DepthD16Array);
	SAFE_DELETE_ARRAY(m_ColorRGBXArray);
	m_CurrFrame = 0;
	m_bHasColorData = false;
}
