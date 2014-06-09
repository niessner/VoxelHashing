#pragma once

/************************************************************************/
/* Reads data from a set of image sequence                              */
/************************************************************************/

#include "DepthSensor.h"

#include <string>

class ImageReaderSensor : public DepthSensor
{
public:
	ImageReaderSensor();
	~ImageReaderSensor();

	HRESULT createFirstConnected();

	HRESULT processDepth();

	HRESULT processColor();

	HRESULT toggleAutoWhiteBalance()
	{
		HRESULT hr = S_OK;

		return hr;
	}

	void setBaseFilePath(const std::string& basePath) {
		m_BaseFilename = basePath;
	}

	void setNumFrames(unsigned int i) {
		m_NumFrames = i;
	}

	unsigned int getNumFrames() {
		return m_NumFrames;
	}

private:
	std::string m_BaseFilename;
	unsigned int m_CurrentFrameNumberDepth;
	unsigned int m_CurrentFrameNumberColor;
	unsigned int m_NumFrames;


};

