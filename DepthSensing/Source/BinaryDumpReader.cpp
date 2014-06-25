
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
}

BinaryDumpReader::~BinaryDumpReader()
{
	releaseData();
}

HRESULT BinaryDumpReader::createFirstConnected()
{
	releaseData();

	std::string filename = GlobalAppState::getInstance().s_BinaryDumpReaderSourceFile;
	std::cout << "Start loading binary dump" << std::endl;
	//BinaryDataStreamZLibFile inputStream(filename, false);
	BinaryDataStreamFile inputStream(filename, false);
	inputStream >> m_data;
	std::cout << "Loading finished" << std::endl;
	std::cout << m_data << std::endl;

	DepthSensor::init(m_data.m_DepthImageWidth, m_data.m_DepthImageHeight, std::max(m_data.m_ColorImageWidth,1u), std::max(m_data.m_ColorImageHeight,1u));
	mat4f intrinsics(m_data.m_CalibrationDepth.m_Intrinsic);
	initializeIntrinsics(m_data.m_CalibrationDepth.m_Intrinsic(0,0), m_data.m_CalibrationDepth.m_Intrinsic(1,1), m_data.m_CalibrationDepth.m_Intrinsic(0,2), m_data.m_CalibrationDepth.m_Intrinsic(1,2));

	m_NumFrames = m_data.m_DepthNumFrames;
	assert(m_data.m_ColorNumFrames == m_data.m_DepthNumFrames || m_data.m_ColorNumFrames == 0);

	return S_OK;
}

HRESULT BinaryDumpReader::processDepth()
{
	if (m_CurrFrame < m_NumFrames) {
		std::cout << "curr Frame " << m_CurrFrame << std::endl;

		for (unsigned int k = 0; k < getDepthWidth()*getDepthHeight(); k++) {
			m_depthD16[k] = (USHORT)(m_data.m_DepthImages[m_CurrFrame][k]*1000.0f + 0.5f);
		}
		if (m_data.m_ColorImages.size() > 0) {
			for (unsigned int k = 0; k < getColorWidth()*getColorHeight(); k++) {
			const BYTE* c = (BYTE*)&(m_data.m_ColorImages[m_CurrFrame][k]); 
				m_colorRGBX[k*getColorBytesPerPixel()+0] = c[0];
				m_colorRGBX[k*getColorBytesPerPixel()+1] = c[1];
				m_colorRGBX[k*getColorBytesPerPixel()+2] = c[2];
				m_colorRGBX[k*getColorBytesPerPixel()+3] = 255;
			}
		}

		m_CurrFrame++;
		return S_OK;
	} else {
		return S_FALSE;
	}
}

void BinaryDumpReader::releaseData()
{
	m_data.deleteData();
	m_CurrFrame = 0;
}
