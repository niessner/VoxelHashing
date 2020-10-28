#pragma once


/************************************************************************/
/* Reads sensor data files from .sens files                            */
/************************************************************************/

#include "GlobalAppState.h"
#include "RGBDSensor.h"
#include "sensordatazhou.h"
#include "stdafx.h"

#ifdef SENSOR_DATA_READER

namespace ml {
	class SensorDataZhou;
	class RGBDFrameCacheRead;
}

class SensorDataReaderZhou : public RGBDSensor
{
public:

	//load Kun's dataset.
	//1234567 - 123456789ABC
	//0004443 - 000148226209 : dept
	//0000001 - 000000000000 : image
	//frame num - timestamp
	void loadFromFileZhou(const std::string& filename);

	//! Constructor
	SensorDataReaderZhou();

	SensorDataReaderZhou(int x, int nFrames = 0, int startFrame=0);

	//! Destructor; releases allocated ressources
	~SensorDataReaderZhou();

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
	
private:
	//! deletes all allocated data
	void releaseData();

	//ml::SensorData* m_sensorData;
	//ml::SensorData::RGBDFrameCacheRead* m_sensorDataCache;
	ml::SensorDataZhou* m_sensorData;
	ml::RGBDFrameCacheRead* m_sensorDataCache;

	unsigned int	m_numFrames;
	unsigned int	m_currFrame;
	unsigned int	m_startFrame;
	bool			m_bHasColorData;
	int m_nFrames;
	int m_type;

};


#endif	//sensor data reader
