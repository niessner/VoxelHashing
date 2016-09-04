#pragma once


#include "GlobalAppState.h"

//win8 only
#ifdef REAL_SENSE

#include "RGBDSensor.h"
#include "RealSense/pxcsensemanager.h"

#include <vector>
#include <list>

class RealSenseSensor : public RGBDSensor
{
public:

	//! Constructor; allocates CPU memory and creates handles
	RealSenseSensor();

	//! Destructor; releases allocated ressources
	~RealSenseSensor();

	//! Initializes the sensor
	HRESULT createFirstConnected();

	//! Processes the depth & color data
	HRESULT processDepth();
	

	//! processing happends in processdepth()
	HRESULT processColor() {
		return S_OK;
	}

	std::string getSensorName() const {
		return "RealSense";
	}
	

protected:

	PXCSession *m_session;
	PXCCapture *m_capture;
	PXCCapture::Device *m_device;
	PXCSenseManager *m_senseManager;

	float m_depthFps;
	float m_colorFps;
};

#endif
