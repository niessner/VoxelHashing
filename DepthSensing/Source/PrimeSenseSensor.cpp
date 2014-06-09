
#include "stdafx.h"

#include "PrimeSenseSensor.h"

//Only working with OpenNI 2 SDK (which wants to run on Win8)
#ifdef OPEN_NI

PrimeSenseSensor::PrimeSenseSensor()
{
	unsigned int depthWidth = 640;
	unsigned int depthHeight = 480;

	unsigned int colorWidth = 640;
	unsigned int colorHeight = 480;

	DepthSensor::init(depthWidth, depthHeight, colorWidth, colorHeight);

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

	while(!m_CachedDataColor.empty()) {
		delete[] m_CachedDataColor.front();
		m_CachedDataColor.pop_front();
	}
	while(!m_CachedDataDepth.empty()) {
		delete[] m_CachedDataDepth.front();
		m_CachedDataDepth.pop_front();
	}
}

HRESULT PrimeSenseSensor::createFirstConnected()
{
	HRESULT hr = S_OK;

	openni::Status rc = openni::STATUS_OK;
	const char* deviceURI = openni::ANY_DEVICE;
	//const char* deviceURI = "../stanfordData/burghers.oni";
	//const char* deviceURI = "../stanfordData/cactusgarden.oni";	//4941 frames
	//const char* deviceURI = "../stanfordData/stonewall.oni";	//2250 frames
	//const char* deviceURI = "../stanfordData/lounge.oni";		//2501 frames
	//const char* deviceURI = "../stanfordData/copyroom.oni";	//4370 frames

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
			std::cout << "Couldn't start color stream: " << openni::OpenNI::getExtendedError() << std::endl;
			m_colorStream.destroy();
		}
	}
	else
	{
		std::cout << "Couldn't find color stream: " << openni::OpenNI::getExtendedError() << std::endl;
	}

	// Check Streams
	if (!m_depthStream.isValid() || !m_colorStream.isValid())
	{
		std::cout << "No valid streams. Exiting" << std::endl;
		openni::OpenNI::shutdown();
		return S_FALSE;
	}

	//pc->setSpeed(1.0);
	//pc->seek(m_depthStream, 2700);
	//pc->seek(m_colorStream, 2700);

	//pc->setRepeatEnabled(false);

	//std::cout << "Sequence Num Frames: \t" << pc->getNumberOfFrames(m_depthStream) << std::endl;
	//std::cout << "Sequence Play Speed: \t" << pc->getSpeed() << std::endl;
	//std::cout << "Sequence Repeat Enabl.: \t" << pc->getRepeatEnabled() << std::endl;


	// Get Dimensions
	m_depthVideoMode = m_depthStream.getVideoMode();
	m_colorVideoMode = m_colorStream.getVideoMode();

	int depthWidth = m_depthVideoMode.getResolutionX();
	int depthHeight = m_depthVideoMode.getResolutionY();
	int colorWidth = m_colorVideoMode.getResolutionX();
	int colorHeight = m_colorVideoMode.getResolutionY();

	if (depthWidth != getDepthWidth() || depthHeight != getDepthHeight() || colorWidth != getColorWidth() || colorHeight != getColorHeight())
	{
		std::cout << depthWidth << " " << depthHeight << " " << colorWidth << " " << colorHeight << std::endl;
		std::cout << "Error - expect color and depth to have different resolutions: " << std::endl;
		openni::OpenNI::shutdown();
		return S_FALSE;
	}

	m_streams = new openni::VideoStream*[2];
	m_streams[0] = &m_depthStream;
	m_streams[1] = &m_colorStream;

	if (rc != openni::STATUS_OK)
	{
		openni::OpenNI::shutdown();
		return 3;
	}

	//m_device.setImageRegistrationMode(openni::IMAGE_REGISTRATION_OFF);
	m_device.setImageRegistrationMode(openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR);

	//float focalLengthX = 320.0f/tan(m_depthStream.getHorizontalFieldOfView()/2.0f);
	//float focalLengthY = 240.0f/tan(m_depthStream.getVerticalFieldOfView()/2.0f);
	//initializeIntrinsics(focalLengthX, focalLengthY, 320.0f, 240.0f);

	float focalLengthX = 525.0f;
	float focalLengthY = 525.0f;
	float cx = 319.5f;
	float cy = 239.5f;
	initializeIntrinsics(focalLengthX, focalLengthY, cx, cy);
	//
	//
	//std::cout << "Focal LengthX: " << focalLengthX << std::endl;
	//std::cout << "Focal LengthY: " << focalLengthY << std::endl;


	//m_bUseCache = true;
	m_bUseCache = false;
	UINT numFrames = pc->getNumberOfFrames(m_depthStream);
	//numFrames = 500;
	//TODO MADDI fix that
	/*
	if (m_bUseCache) {
		for (UINT i = 0; i < numFrames; i++) {
			std::cout << "pre-caching frame " << i << std::endl;
			USHORT* depthD16 = new USHORT[getDepthWidth() * getDepthHeight()];
			BYTE* colorRGBX = new BYTE[getColorWidth() * getColorHeight() * getColorBytesPerPixel()];
			readDepthAndColor(depthD16, colorRGBX);
			m_CachedDataDepth.push_back(depthD16);
			m_CachedDataColor.push_back(colorRGBX);
		}

		std::string filename = "mydump.bdump";
		mat4f extr(mat4f::Identity);
		BinaryDumpSensorOld::writeBinaryDump(filename, getDepthWidth(), getDepthHeight(), 1, &getIntrinsics(), &extr, m_CachedDataDepth, m_CachedDataColor);
		std::cout << "dump done!" << std::endl;;
	}
	*/
	return hr;
}

HRESULT PrimeSenseSensor::processDepth()
{

	HRESULT hr = S_OK;

	m_bDepthImageIsUpdated = false;
	m_bDepthImageCameraIsUpdated = false;
	m_bNormalImageCameraIsUpdated = false;

	if (!m_bUseCache) {
		hr = readDepthAndColor(m_depthD16, m_colorRGBX);
	} else {
		if (m_CachedDataColor.empty() || m_CachedDataDepth.empty())	return S_FALSE;
		m_depthD16 = m_CachedDataDepth.front();
		m_colorRGBX = m_CachedDataColor.front();
		m_CachedDataDepth.push_back(m_CachedDataDepth.front());
		m_CachedDataDepth.pop_front();
		m_CachedDataColor.push_back(m_CachedDataColor.front());
		m_CachedDataColor.pop_front();
	}

	m_bDepthImageIsUpdated = true;
	m_bDepthImageCameraIsUpdated = true;
	m_bNormalImageCameraIsUpdated = true;

	m_bDepthReceived = true;
	m_bColorReceived = true;

	return hr;
}

HRESULT PrimeSenseSensor::readDepthAndColor( USHORT* depthD16, BYTE* colorRGBX )
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


	//// check if we need to draw image frame to texture
	//if (m_colorFrame.isValid())
	//{
	//	const openni::RGB888Pixel* pImage = (const openni::RGB888Pixel*)m_colorFrame.getData();
	//	
	//	for (int y = 0; y < m_colorFrame.getHeight(); ++y)
	//	{
	//		for (int x = 0; x < m_colorFrame.getWidth(); ++x)
	//		{
	//			const openni::RGB888Pixel& pixel = pImage[y*m_colorFrame.getWidth()+x];
	//			
	//			unsigned int c = 0;
	//			c |= pixel.r;
	//			c <<= 8;
	//			c |= pixel.g;
	//			c <<= 8;
	//			c |= pixel.b;
	//			c |= 0xFF000000;

	//			((LONG*)colorRGBX)[y*m_colorFrame.getWidth()+x] = c;
	//		}
	//	}
	//}

	//// check if we need to draw depth frame to texture
	//if (m_depthFrame.isValid())
	//{
	//	const openni::DepthPixel* pDepth = (const openni::DepthPixel*)m_depthFrame.getData();

	//	for (int y = 0; y < m_depthFrame.getHeight(); ++y)
	//	{
	//		for (int x = 0; x < m_depthFrame.getWidth(); ++x)
	//		{
	//			const openni::DepthPixel& p = pDepth[y*m_depthFrame.getWidth()+x];
	//			if (p != 0)
	//			{
	//				depthD16[y*m_depthFrame.getWidth()+x] = p;//*8;
	//			}
	//			else
	//			{
	//				depthD16[y*m_depthFrame.getWidth()+x] = 0;
	//			}
	//		}
	//	}
	//}



	assert(m_colorFrame.getWidth() == m_depthFrame.getWidth());
	assert(m_colorFrame.getHeight() == m_depthFrame.getHeight());
	int width = m_depthFrame.getWidth();
	int height = m_depthFrame.getHeight();
	// check if we need to draw depth frame to texture
	if (m_depthFrame.isValid() && m_colorFrame.isValid())
	{
		const openni::DepthPixel* pDepth = (const openni::DepthPixel*)m_depthFrame.getData();
		const openni::RGB888Pixel* pImage = (const openni::RGB888Pixel*)m_colorFrame.getData();

		for (int y = 0; y < height; ++y)
		{
			for (int x = 0; x < width; ++x)
			{
				const openni::RGB888Pixel& pixel = pImage[y*width+x];

				unsigned int c = 0;
				c |= pixel.r;
				c <<= 8;
				c |= pixel.g;
				c <<= 8;
				c |= pixel.b;
				c |= 0xFF000000;

				((LONG*)colorRGBX)[y*width+x] = c;


				const openni::DepthPixel& p = pDepth[y*width+x];

				if (p != 0)	{
					depthD16[y*width+x] = p;//*8;
				} else {
					depthD16[y*width+x] = 0;
				}
			}
		}
	}

	return hr;
}

#endif
