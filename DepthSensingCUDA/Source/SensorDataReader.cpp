
#include "stdafx.h"

#include "SensorDataReader.h"
#include "GlobalAppState.h"
#include "MatrixConversion.h"

#ifdef SENSOR_DATA_READER

#include "sensorData/sensorData.h"

#include <algorithm>
#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <string>

#include <conio.h>

SensorDataReader::SensorDataReader()
{
	m_numFrames = 0;
	m_currFrame = 0;
	m_bHasColorData = false;
	//parameters are read from the calibration file

	m_sensorData = NULL;
	m_sensorDataCache = NULL;

	m_currSensFileIdx = 0;
}

SensorDataReader::~SensorDataReader()
{
	releaseData();
}


HRESULT SensorDataReader::createFirstConnected()
{
	releaseData();

	if (GlobalAppState::get().s_binaryDumpSensorFile.size() <= m_currSensFileIdx) throw MLIB_EXCEPTION("need to specify s_binaryDumpSensorFile[" + std::to_string(m_currSensFileIdx) + "]");
	std::string filename = GlobalAppState::get().s_binaryDumpSensorFile[m_currSensFileIdx];

	std::cout << "Start loading binary dump... ";
	m_sensorData = new SensorData;
	m_sensorData->loadFromFile(filename);
	std::cout << "DONE!" << std::endl;
	std::cout << *m_sensorData << std::endl;

	//std::cout << "intrinsics:" << std::endl;
//	std::cout << m_sensorData->m_calibrationDepth.m_intrinsic << std::endl;

	RGBDSensor::init(m_sensorData->m_depthWidth, m_sensorData->m_depthHeight, std::max(m_sensorData->m_colorWidth, 1u), std::max(m_sensorData->m_colorHeight, 1u), 1);
	initializeDepthIntrinsics(m_sensorData->m_calibrationDepth.m_intrinsic(0, 0), m_sensorData->m_calibrationDepth.m_intrinsic(1, 1), m_sensorData->m_calibrationDepth.m_intrinsic(0, 2), m_sensorData->m_calibrationDepth.m_intrinsic(1, 2));
	initializeColorIntrinsics(m_sensorData->m_calibrationColor.m_intrinsic(0, 0), m_sensorData->m_calibrationColor.m_intrinsic(1, 1), m_sensorData->m_calibrationColor.m_intrinsic(0, 2), m_sensorData->m_calibrationColor.m_intrinsic(1, 2));

	initializeDepthExtrinsics(m_sensorData->m_calibrationDepth.m_extrinsic);
	initializeColorExtrinsics(m_sensorData->m_calibrationColor.m_extrinsic);


	m_numFrames = (unsigned int)m_sensorData->m_frames.size();

	if (m_numFrames > 0 && m_sensorData->m_frames[0].getColorCompressed()) {
		m_bHasColorData = true;
	}
	else {
		m_bHasColorData = false;
	}

	const unsigned int cacheSize = 10;
	m_sensorDataCache = new RGBDFrameCacheRead(m_sensorData, cacheSize);

	return S_OK;
}

void SensorDataReader::loadNextSensFile() {
	if (!GlobalAppState::get().s_playData) return;

	if (m_currFrame >= m_numFrames)	{
		std::cout << "Loading new sens file!" << std::endl;

		if (m_currSensFileIdx + 1 < GlobalAppState::get().s_binaryDumpSensorFile.size()) {
			m_currSensFileIdx++;
			createFirstConnected();
			//in case we have more .sens files specified
		}
		else {
			GlobalAppState::get().s_playData = false;
			std::cout << "binary dump sequence complete - press space to run again" << std::endl;
			m_currFrame = 0;
		}
	}

}

HRESULT SensorDataReader::processDepth()
{
	//if (m_currFrame >= m_numFrames)
	//{
	//	if (m_currSensFileIdx + 1 < GlobalAppState::get().s_binaryDumpSensorFile.size()) {
	//		m_currSensFileIdx++;
	//		createFirstConnected();
	//		//in case we have more .sens files specified
	//	}
	//	else {
	//		GlobalAppState::get().s_playData = false;
	//		std::cout << "binary dump sequence complete - press space to run again" << std::endl;
	//		m_currFrame = 0;
	//	}
	//}

	if (GlobalAppState::get().s_playData) {

		float* depth = getDepthFloat();
		//memcpy(depth, m_data.m_DepthImages[m_currFrame], sizeof(float)*getDepthWidth()*getDepthHeight());

		ml::RGBDFrameCacheRead::FrameState frameState = m_sensorDataCache->getNext();
		//ml::RGBDFrameCacheRead::FrameState frameState;
		//frameState.m_colorFrame = m_sensorData->m_frames[m_currFrame].decompressColorAlloc();
		//frameState.m_depthFrame = m_sensorData->m_frames[m_currFrame].decompressDepthAlloc();


		for (unsigned int i = 0; i < getDepthWidth()*getDepthHeight(); i++) {
			depth[i] = (float)frameState.m_depthFrame[i] / m_sensorData->m_depthShift;
		}

		//{
		//	//debug
		//	ColorImageR8G8B8 c(getColorHeight(), getColorWidth(), frameState.m_colorFrame);
		//	FreeImageWrapper::saveImage("test_color.png", c);
		//}
		//{
		//	//debug
		//	DepthImage d(getDepthHeight(), getDepthWidth(), depth);
		//	ColorImageRGBA c(d);
		//	FreeImageWrapper::saveImage("test_depth.png", d);
		//}

		incrementRingbufIdx();

		if (m_bHasColorData) {
			//memcpy(m_colorRGBX, m_data.m_ColorImages[m_currFrame], sizeof(vec4uc)*getColorWidth()*getColorHeight());
			for (unsigned int i = 0; i < getColorWidth()*getColorHeight(); i++) {
				m_colorRGBX[i] = vec4uc(frameState.m_colorFrame[i]);
			}
		}
		frameState.free();

		//if (m_currFrame == 50) {
		//	m_sensorDataCache->endDecompression();
		//	std::cout << "END END" << std::endl;
		//	getchar();
		//}
		m_currFrame++;
		return S_OK;
	}
	else {
		return S_FALSE;
	} 
}

std::string SensorDataReader::getSensorName() const
{
	return m_sensorData->m_sensorName;
}

ml::mat4f SensorDataReader::getRigidTransform(int offset) const
{
	unsigned int idx = m_currFrame - 1 + offset;
	if (idx >= m_sensorData->m_frames.size()) throw MLIB_EXCEPTION("invalid trajectory index " + std::to_string(idx));
	const mat4f& transform = m_sensorData->m_frames[idx].getCameraToWorld();
	return transform;
	//return m_data.m_trajectory[idx];
}

void SensorDataReader::releaseData()
{
	m_currFrame = 0;
	m_bHasColorData = false;

	 
	SAFE_DELETE(m_sensorDataCache);
	if (m_sensorData) {
		m_sensorData->free();
		SAFE_DELETE(m_sensorData);
	}
}



#endif
