#pragma once

/************************************************************************/
/* Reads binary dump data from .sensor files                            */
/************************************************************************/

#include "DepthSensor.h"

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

private:
	//! deletes all allocated data
	void releaseData();

	BYTE**			m_ColorRGBXArray;
	USHORT**		m_DepthD16Array;
	unsigned int	m_NumFrames;
	unsigned int	m_CurrFrame;
	bool			m_bHasColorData;
};
