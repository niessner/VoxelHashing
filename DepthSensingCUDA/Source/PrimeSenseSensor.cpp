
#include "stdafx.h"

#include "PrimeSenseSensor.h"

//Only working with OpenNI 2 SDK (which wants to run on Win8)
#ifdef OPEN_NI

PrimeSenseSensor::PrimeSenseSensor()
{
	m_bDepthReceived = false;
	m_bColorReceived = false;

	m_bDepthImageIsUpdated = false;
	m_bDepthImageCameraIsUpdated = false;
	m_bNormalImageCameraIsUpdated = false;
}

PrimeSenseSensor::~PrimeSenseSensor()
{
	if (m_streams != NULL)
	{
		delete [] m_streams;
	}

	m_depthStream.stop();
	m_colorStream.stop();
	m_depthStream.destroy();
	m_colorStream.destroy();
	m_device.close();
	openni::OpenNI::shutdown();

	/*openni::Recorder recorder;
	recorder.create("test.oni");

	recorder.attach(m_depthStream, false);
	recorder.attach(m_colorStream, false);

	recorder.start();
	recorder.stop();

	recorder.destroy();*/
}

HRESULT PrimeSenseSensor::createFirstConnected()
{
	HRESULT hr = S_OK;

	openni::Status rc = openni::STATUS_OK;
	const char* deviceURI = openni::ANY_DEVICE;

	rc = openni::OpenNI::initialize();

	std::cout << "After initialization: " << openni::OpenNI::getExtendedError() << std::endl;

	// Create Device
	rc = m_device.open(deviceURI);
	if (rc != openni::STATUS_OK)
	{
		std::cout << "Device open failed: " << openni::OpenNI::getExtendedError() << std::endl;
		openni::OpenNI::shutdown();
		return S_FALSE;
	}

	openni::PlaybackControl* pc = m_device.getPlaybackControl();

	// Create Depth Stream
	rc = m_depthStream.create(m_device, openni::SENSOR_DEPTH);
	if (rc == openni::STATUS_OK)
	{
		rc = m_depthStream.start();
		if (rc != openni::STATUS_OK)
		{
			std::cout << "Couldn't start depth stream: " << openni::OpenNI::getExtendedError() << std::endl;
			m_depthStream.destroy();
		}
	}
	else
	{
		std::cout << "Couldn't find depth stream: " << openni::OpenNI::getExtendedError() << std::endl;
	}

	// Create Color Stream
	rc = m_colorStream.create(m_device, openni::SENSOR_COLOR);
	if (rc == openni::STATUS_OK)
	{
		rc = m_colorStream.start();
		if (rc != openni::STATUS_OK)
		{
			std::cout << "Couldn't start color stream: " << openni::OpenNI::getExtendedError() << " Return code: " << rc << std::endl;
			m_colorStream.destroy();
		}
	}
	else
	{
		std::cout << "Couldn't find color stream: " << openni::OpenNI::getExtendedError() << std::endl;
	}
	if (GlobalAppState::get().s_enableColorCropping) 
	{
		if (m_colorStream.isCroppingSupported()) {
			m_colorStream.setCropping(GlobalAppState::get().s_colorCropX, GlobalAppState::get().s_colorCropY, GlobalAppState::get().s_colorCropWidth, GlobalAppState::get().s_colorCropHeight);
		}
		else {
			std::cout << "cropping is not supported for this color stream" << std::endl;
		}
	}

	// Check Streams
	if (!m_depthStream.isValid() || !m_colorStream.isValid())
	{
		std::cout << "No valid streams. Exiting" << std::endl;
		openni::OpenNI::shutdown();
		return S_FALSE;
	}

	openni::CameraSettings* cs = m_colorStream.getCameraSettings();

	//cs->setAutoWhiteBalanceEnabled(false);
	//cs->setAutoExposureEnabled(false);
	//cs->setGain(1);
	//cs->setExposure(10);

	std::cout << "getGain: " << cs->getGain() << std::endl;
	std::cout << "getExposure: " << cs->getExposure() << std::endl;
	std::cout << "getAutoExposureEnabled: " << cs->getAutoExposureEnabled() << std::endl;
	std::cout << "getAutoWhiteBalanceEnabled: " << cs->getAutoWhiteBalanceEnabled() << std::endl;

	// Get Dimensions
	m_depthVideoMode = m_depthStream.getVideoMode();
	m_colorVideoMode = m_colorStream.getVideoMode();

	int depthWidth = m_depthVideoMode.getResolutionX();
	int depthHeight = m_depthVideoMode.getResolutionY();
	int colorWidth = m_colorVideoMode.getResolutionX();
	int colorHeight = m_colorVideoMode.getResolutionY();

	RGBDSensor::init(depthWidth, depthHeight, colorWidth, colorHeight, 1);

	m_streams = new openni::VideoStream*[2];
	m_streams[0] = &m_depthStream;
	m_streams[1] = &m_colorStream;

	if (rc != openni::STATUS_OK)
	{
		openni::OpenNI::shutdown();
		return 3;
	}

	m_device.setImageRegistrationMode(openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR);

	float focalLengthX = (depthWidth/2.0f)/tan(m_depthStream.getHorizontalFieldOfView()/2.0f);
	float focalLengthY = (depthHeight/2.0f)/tan(m_depthStream.getVerticalFieldOfView()/2.0f);
	initializeDepthIntrinsics(focalLengthX, focalLengthY, depthWidth/2.0f, depthHeight/2.0f);

	focalLengthX =  (colorWidth/2.0f)/tan(m_colorStream.getHorizontalFieldOfView()/2.0f);
	focalLengthY =  (colorHeight/2.0f)/tan(m_colorStream.getVerticalFieldOfView()/2.0f);
	initializeColorIntrinsics(focalLengthX, focalLengthY, colorWidth/2.0f, colorHeight/2.0f);

	Matrix3f R; R.setIdentity(); Vector3f t; t.setZero();
	initializeColorExtrinsics(R, t);

	R(0, 0) =  9.9991741106823473e-001; R(0, 1) =  3.0752530258331304e-003; R(0, 2) = -1.2478536028949385e-002;
	R(1, 0) = -3.0607678272497924e-003; R(1, 1) =  9.9999461994140826e-001; R(1, 2) =  1.1797408808971066e-003;
	R(2, 0) =  1.2482096895408091e-002; R(2, 1) = -1.1414495457493831e-003; R(2, 2) =  9.9992144408949846e-001;

	//t[0] = -2.5331974929667012e+001;  t[1] = 6.1798287248283634e-001; t[2] = 3.8510108109251804e+000;
	//t[0] /= 1000.0f; t[1] /= 1000.0f; t[2] /= 1000.0f;
	//
	//initializeDepthExtrinsics(R, t);

	return hr;
}

HRESULT PrimeSenseSensor::processDepth()
{

	HRESULT hr = S_OK;

	m_bDepthImageIsUpdated = false;
	m_bDepthImageCameraIsUpdated = false;
	m_bNormalImageCameraIsUpdated = false;

	hr = readDepthAndColor(getDepthFloat(), m_colorRGBX);

	m_bDepthImageIsUpdated = true;
	m_bDepthImageCameraIsUpdated = true;
	m_bNormalImageCameraIsUpdated = true;

	m_bDepthReceived = true;
	m_bColorReceived = true;

	return hr;
}

HRESULT PrimeSenseSensor::readDepthAndColor(float* depthFloat, vec4uc* colorRGBX )
{
	HRESULT hr = S_OK;

	int changedIndex;
	openni::Status rc = openni::OpenNI::waitForAnyStream(&m_streams[0], 1, &changedIndex, 0);
	if (rc != openni::STATUS_OK) {
		return S_FALSE;	//no frame available
	}

	rc = openni::OpenNI::waitForAnyStream(&m_streams[1], 1, &changedIndex, 0);
	if (rc != openni::STATUS_OK) {
		return S_FALSE;	//no frame available
	}

	openni::Status sd = m_depthStream.readFrame(&m_depthFrame);
	openni::Status sc = m_colorStream.readFrame(&m_colorFrame);

	assert(m_colorFrame.getWidth() == m_depthFrame.getWidth());
	assert(m_colorFrame.getHeight() == m_depthFrame.getHeight());

	const openni::DepthPixel* pDepth = (const openni::DepthPixel*)m_depthFrame.getData();
	const openni::RGB888Pixel* pImage = (const openni::RGB888Pixel*)m_colorFrame.getData();

	// check if we need to draw depth frame to texture
	if (m_depthFrame.isValid() && m_colorFrame.isValid())
	{	
		unsigned int width   = m_depthFrame.getWidth();
		unsigned int nPixels = m_depthFrame.getWidth()*m_depthFrame.getHeight();

		for (unsigned int i = 0; i < nPixels; i++)	{
			const int x = i%width;
			const int y = i/width;
			const int src = y*width + (width-1-x);
			const openni::DepthPixel& p = pDepth[src];

			float dF = (float)p*0.001f;
			if(dF >= GlobalAppState::get().s_sensorDepthMin && dF <= GlobalAppState::get().s_sensorDepthMax) depthFloat[i] = dF;
			else																	 depthFloat[i] = -std::numeric_limits<float>::infinity();
		}
		incrementRingbufIdx();
	}

	// check if we need to draw depth frame to texture
	if (m_depthFrame.isValid() && m_colorFrame.isValid())
	{
		int cropX, cropY, cropW, cropH;

		if(m_colorStream.getCropping(&cropX, &cropY, &cropW, &cropH))
		{
			const openni::RGB888Pixel* pImage = (const openni::RGB888Pixel*)m_colorFrame.getData();
			ZeroMemory(m_colorRGBX, sizeof(LONG)*m_colorWidth*m_colorHeight);
			#pragma omp parallel for
			for (int y = 0; y < cropH; ++y)
			{
				for (int x = 0; x < cropW; ++x)
				{
					unsigned int width   = m_colorFrame.getWidth();
					unsigned int height  = m_colorFrame.getHeight();
					int y2 = 0;
					if(m_colorWidth == 1280)	y2 = (y+cropY)-64/2/*-(unsigned int)(((float)(y+cropY)/((float)(height-1)))*64+0.5f)*/-cropY;
					else						y2 = y;
				
					if(y2 >= 0 && y2 < (int)height)
					{
						const openni::RGB888Pixel& pixel = pImage[(y2)*cropW+(cropW - x - 1)];

						unsigned int c = 0;
						c |= pixel.b;
						c <<= 8;
						c |= pixel.g;
						c <<= 8;
						c |= pixel.r;
						c |= 0xFF000000;

						((LONG*)m_colorRGBX)[(cropY+y)*m_colorWidth+(cropX+x)] = c;
					}
				}
			}
		}
		else
		{
			unsigned int width   = m_colorFrame.getWidth();
			unsigned int height  = m_colorFrame.getHeight();
			unsigned int nPixels = m_colorFrame.getWidth()*m_colorFrame.getHeight();

			for(unsigned int i = 0; i<nPixels; i++)
			{
				const int x = i%width;
				const int y = i/width;

				int y2 = 0;
				if(m_colorWidth == 1280)	y2 = y+64/2-10-(unsigned int)(((float)y/((float)(height-1)))*64+0.5f);
				else						y2 = y;

				if(y2 >= 0 && y2 < (int)height)
				{
					//unsigned int Index1D = y2*width+x;
					unsigned int Index1D = y2*width+ (width-1-x);	//x-flip here

					const openni::RGB888Pixel& pixel = pImage[Index1D];

					unsigned int c = 0;
					c |= pixel.b;
					c <<= 8;
					c |= pixel.g;
					c <<= 8;
					c |= pixel.r;
					c |= 0xFF000000;

					((LONG*)colorRGBX)[y*width+x] = c;
				}
			}
		}
	}


	return hr;
}

#endif
