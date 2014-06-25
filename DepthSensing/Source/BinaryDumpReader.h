#pragma once

/************************************************************************/
/* Reads binary dump data from .sensor files                            */
/************************************************************************/

#include "DepthSensor.h"
#include "calibratedSensorData.h"

class BinaryDumpReader : public DepthSensor
{
public:

	//! Constructor
	BinaryDumpReader();

	//! Destructor; releases allocated ressources
	~BinaryDumpReader();

	//! initializes the sensor
	HRESULT createFirstConnected();

	//! reads the next depth frame
	HRESULT processDepth();

	HRESULT processColor()	{
		//everything done in process depth since order is relevant (color must be read first)
		return S_OK;
	}

	HRESULT BinaryDumpReader::toggleNearMode()	{
		return S_OK;
	}

	//! Toggle enable auto white balance
	HRESULT toggleAutoWhiteBalance() {
		return S_OK;
	}

	bool isKinect4Windows()	{
		return true;
	}

	mat4f getRigidTransform() const {
		if (m_CurrFrame-1 >= m_data.m_trajectory.size()) throw MLIB_EXCEPTION("invalid trajectory index " + std::to_string(m_CurrFrame-1));
		return m_data.m_trajectory[m_CurrFrame-1];
	}
private:
	//! deletes all allocated data
	void releaseData();

	CalibratedSensorData m_data;
	
	unsigned int	m_NumFrames;
	unsigned int	m_CurrFrame;
};
