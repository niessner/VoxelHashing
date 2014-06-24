#pragma once

/************************************************************************/
/* Prime sense depth camera: Warning this is highly untested atm        */
/************************************************************************/

#include "GlobalAppState.h"

//Only working with OpenNI 2 SDK
#ifdef OPEN_NI

#pragma comment( lib, "OpenNI2" )

#include "DepthSensor.h"
#include <OpenNI.h>

#include <vector>
#include <list>

class PrimeSenseSensor : public DepthSensor
{
public:

	//! Constructor; allocates CPU memory and creates handles
	PrimeSenseSensor();

	//! Destructor; releases allocated ressources
	~PrimeSenseSensor();

	//! Initializes the sensor
	HRESULT createFirstConnected();

	//! Processes the depth data (and color)
	HRESULT processDepth();
	

	//! Processes the Kinect color data
	HRESULT processColor()
	{
		HRESULT hr = S_OK;
		return hr;
	}

	//! Toggles the Kinect to near-mode; default is far mode
	HRESULT toggleNearMode()
	{
		// PrimeSense is always in near mode
		return S_OK;
	}
	
	//! Toggle enable auto white balance
	HRESULT toggleAutoWhiteBalance()
	{
		HRESULT hr = S_OK;

		// TODO

		return hr;
	}

protected:
	//! reads depth and color from the sensor
	HRESULT readDepthAndColor(USHORT* depthD16, BYTE* colorRGBX);


	// to prevent drawing until we have data for both streams
	bool			m_bDepthReceived;
	bool			m_bColorReceived;

	bool			m_bDepthImageIsUpdated;
	bool			m_bDepthImageCameraIsUpdated;
	bool			m_bNormalImageCameraIsUpdated;

	bool			m_kinect4Windows;

	openni::VideoMode			m_depthVideoMode;
	openni::VideoMode			m_colorVideoMode;


	openni::VideoFrameRef		m_depthFrame;
	openni::VideoFrameRef		m_colorFrame;

	openni::Device				m_device;
	openni::VideoStream			m_depthStream;
	openni::VideoStream			m_colorStream;
	openni::VideoStream**		m_streams;
	
	float *uvmap;

	bool					m_bUseCache;
	std::list<USHORT*>		m_CachedDataDepth;
	std::list<BYTE*>		m_CachedDataColor;
};

#endif
