#pragma once


/************************************************************************/
/* Reads sensor data files from .sens files                            */
/************************************************************************/

#include "GlobalAppState.h"
#include "RGBDSensor.h"
#include "stdafx.h"

#ifdef SENSOR_DATA_READER

namespace ml {
	class SensorData;
	class RGBDFrameCacheRead;
}

class SensorDataReader : public RGBDSensor
{
public:
	//! Constructor
	SensorDataReader();

	//! Destructor; releases allocated ressources
	~SensorDataReader();

	//! initializes the sensor
	HRESULT createFirstConnected();

	//! reads the next depth frame
	HRESULT processDepth();


	HRESULT processColor()	{
		//everything done in process depth since order is relevant (color must be read first)
		return S_OK;
	}

	std::string getSensorName() const;

	mat4f getRigidTransform(int offset) const;

	const SensorData* getSensorData() const {
		return m_sensorData;
	}

	unsigned int getNumFrames() const {
		return m_numFrames;
	}
	unsigned int getCurrFrame() const {
		return m_currFrame;
	}
	unsigned int getCurrSensFileIdx() const {
		return m_currSensFileIdx;
	}

	void loadNextSensFile();
private:
	//! deletes all allocated data
	void releaseData();

	ml::SensorData* m_sensorData;
	ml::RGBDFrameCacheRead* m_sensorDataCache;

	unsigned int	m_numFrames;
	unsigned int	m_currFrame;
	bool			m_bHasColorData;

	unsigned int	m_currSensFileIdx;

};


#endif	//sensor data reader
