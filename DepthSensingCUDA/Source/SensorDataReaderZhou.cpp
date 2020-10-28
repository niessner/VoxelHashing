
#include "stdafx.h"

#include "SensorDataReaderZhou.h"
#include "GlobalAppState.h"
#include "MatrixConversion.h"

#ifdef SENSOR_DATA_FILE_READER

#include "sensorData/sensorData.h"

#include <algorithm>
#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <string>

#include <conio.h>

SensorDataReaderZhou::SensorDataReaderZhou()
{
	m_numFrames = 0;
	m_currFrame = 0;
	m_bHasColorData = false;

	m_sensorData = NULL;
	m_sensorDataCache = NULL;
}

SensorDataReaderZhou::SensorDataReaderZhou(int x, int nFrames, int startFrame)
{
	m_numFrames = nFrames;

	m_currFrame = 0;
	m_startFrame = startFrame;
	m_bHasColorData = false;

	m_sensorData = NULL;
	m_sensorDataCache = NULL;
	m_type = x;
}

SensorDataReaderZhou::~SensorDataReaderZhou()
{
	releaseData();
}


HRESULT SensorDataReaderZhou::createFirstConnected()
{
	releaseData();

	std::string filename = GlobalAppState::get().s_binaryDumpSensorFile[0];

	std::cout << "Start loading binary dump... " << filename;
	m_sensorData = new SensorDataZhou;

	if (m_type == 0) {
		m_sensorData->loadFromFile(filename, m_numFrames);
		std::cout << "DONE!" << std::endl;
	}
	if (m_type == 1)
		m_sensorData->loadFromFileObject(filename, m_numFrames, m_startFrame);
	std::cout << "DONE!" << std::endl;
	std::cout << *m_sensorData << std::endl;

	std::cout << m_sensorData->m_depthWidth << " " << m_sensorData->m_depthHeight << " " << m_sensorData->m_colorWidth <<" "<< m_sensorData->m_colorHeight << std::endl;

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

ml::mat4f SensorDataReaderZhou::getRigidTransform(int offset) const
{
	unsigned int idx = m_currFrame - 1 + offset;
	if (idx >= m_sensorData->m_frames.size()) throw MLIB_EXCEPTION("invalid trajectory index " + std::to_string(idx));
	const mat4f& transform = m_sensorData->m_frames[idx].getCameraToWorld();
	return transform;
	//return m_data.m_trajectory[idx];
}

HRESULT SensorDataReaderZhou::processDepth()
{
	//if (m_currFrame >= m_numFrames)
	//{
	//	GlobalAppState::get().s_playData = false;
	//	//std::cout << "binary dump sequence complete - press space to run again" << std::endl;
	//	stopReceivingFrames();
	//	std::cout << "binary dump sequence complete - stopped receiving frames" << std::endl;
	//	m_currFrame = 0;
	//}
	//if (m_currFrame >= m_numFrames)
	//{
	//	
	//	int da = m_currFrame / m_numFrames;
	//	//std::cout << "binary dump sequence complete - press space to run again" << std::endl;

	//}
	if (m_sensorDataCache->getCheck()) GlobalAppState::get().s_integrationEnabled = false;

	if (GlobalAppState::get().s_playData) {

		float* depth = getDepthFloat();

		//TODO check why the frame cache is not used?
	    RGBDFrameCacheRead::FrameState frameState = m_sensorDataCache->getNextOurs();

		int depthsize = getDepthWidth()*getDepthHeight();

		for (unsigned int i = 0; i <depthsize; i++) {
			if (frameState.m_depthFrame[i] == 0) depth[i] = -std::numeric_limits<float>::infinity();
			else depth[i] = (float)frameState.m_depthFrame[i] / m_sensorData->m_depthShift;
		}

		incrementRingbufIdx();

		if (m_bHasColorData) {
			int colorsize= getColorWidth()*getColorHeight();
			for (unsigned int i = 0; i < colorsize; i++) {
				m_colorRGBX[i] = vec4uc(frameState.m_colorFrame[i]);
			}
		}
		//frameState.free();

		m_currFrame++;
		return S_OK;
	}
	else {
		return S_FALSE;
	}
}

std::string SensorDataReaderZhou::getSensorName() const
{
	return m_sensorData->m_sensorName;
}

void SensorDataReaderZhou::releaseData()
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
