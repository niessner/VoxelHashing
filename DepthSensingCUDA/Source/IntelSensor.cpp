
#include "stdafx.h"
#include "IntelSensor.h"

#ifdef INTEL_SENSOR

#include "DSCommon.h"


IntelSensor::IntelSensor()
{
	unsigned int depthWidth = 480; // Valid resolutions: 628x468, 480x360
	unsigned int depthHeight = 360;
	unsigned int colorWidth = 640; // Valid resolutions: 1920x1080, 640x480
	unsigned int colorHeight = 480;

	m_depthFPS = 60;
	m_colorFPS = 30;
	m_exposure = 16.3f;
	m_gain = 2.0f;

	//DepthSensor::init(depthWidth, depthHeight, colorWidth, colorHeight);
	RGBDSensor::init(depthWidth, depthHeight, depthWidth, depthHeight);
}

IntelSensor::~IntelSensor()
{
	if (NULL != m_sensor)
	{
		DSDestroy(m_sensor);
		m_sensor = 0;
	}
}

HRESULT IntelSensor::createFirstConnected()
{
	m_sensor = DSCreate(DS_DS4_PLATFORM);
	m_colorSensor = m_sensor->accessThird();
	m_hardware = m_sensor->accessHardware();
	if (!m_sensor->probeConfiguration()) return S_FALSE;

	// Check calibration data
	if (!m_sensor->isCalibrationValid())
	{
		std::cout << "Calibration data on camera is invalid" << std::endl;
		return S_FALSE;
	}

	// Configure core Z-from-stereo capabilities
	if (!m_sensor->enableZ(true)) return S_FALSE;
	if (!m_sensor->enableLeft(false)) return S_FALSE;
	if (!m_sensor->enableRight(false)) return S_FALSE;
	if (!m_sensor->setLRZResolutionMode(true, getDepthWidth(), getDepthHeight(), m_depthFPS, DS_LUMINANCE8)) return S_FALSE; // Valid resolutions: 628x468, 480x360
	m_sensor->enableLRCrop(true);
	m_sensor->setZUnits(1000);

	DSCalibIntrinsicsRectified intrinsicsDepth;
	m_sensor->getCalibIntrinsicsZ(intrinsicsDepth);
	initializeDepthIntrinsics(intrinsicsDepth.rfx, intrinsicsDepth.rfy, intrinsicsDepth.rpx, intrinsicsDepth.rpy);
	initializeDepthExtrinsics(mat4f::identity());

	//MLIB_WARNING("TODO initialize color intrs/extr");
	initializeColorIntrinsics(intrinsicsDepth.rfx, intrinsicsDepth.rfy, intrinsicsDepth.rpx, intrinsicsDepth.rpy);
	initializeColorExtrinsics(mat4f::identity());
	m_intrinsicsDepth = mat4f(
		intrinsicsDepth.rfx, 0.0f, intrinsicsDepth.rpx, 0.0f,
		0.0f, -intrinsicsDepth.rfy, intrinsicsDepth.rpy, 0.0f,
		0.0f, 0.0f, 1.0f, 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f
		);

	// Configure third camera
	if (m_colorSensor)
	{
		if (!m_colorSensor->enableThird(true)) return S_FALSE;
		if (!m_colorSensor->setThirdResolutionMode(true, 640, 480, m_colorFPS, DS_RGB8)) return S_FALSE; // Valid resolutions: 1920x1080, 640x480

		// intrinsics / extrinsics
		DSCalibIntrinsicsRectified intrinsicsColor;
		m_colorSensor->getCalibIntrinsicsRectThird(intrinsicsColor);
		m_intrinsicsColor = mat4f(
			intrinsicsColor.rfx, 0.0f, intrinsicsColor.rpx, 0.0f,
			0.0f, -intrinsicsColor.rfy, intrinsicsColor.rpy, 0.0f,
			0.0f, 0.0f, 1.0f, 0.0f,
			0.0f, 0.0f, 0.0f, 1.0f
			);
		m_intrinsicsColorInv = m_intrinsicsColor.getInverse();
		double translation[3];
		m_colorSensor->getCalibExtrinsicsZToRectThird(translation);
		m_extrinsicsColorToDepth.setTranslation((float)-translation[0]*0.001f, (float)-translation[1]*0.001f, (float)-translation[2]*0.001f);
	}

	// Set initial exposure and gain values
	if (!m_hardware->setImagerExposure(m_exposure, DS_BOTH_IMAGERS)) return S_FALSE;
	if (!m_hardware->setImagerGain(m_gain, DS_BOTH_IMAGERS)) return S_FALSE;

	// Begin capturing images
	if (!m_sensor->startCapture()) return S_FALSE;

	m_depthFrameNumber = 0;
	m_colorFrameNumber = 0;

	return S_OK;
}

HRESULT IntelSensor::processDepth()
{
	HRESULT hr = S_OK;

	if (!m_sensor->grab()) return S_FALSE;

	if (m_sensor->isZEnabled())
	{

		if (m_sensor->getFrameNumber() != m_depthFrameNumber) {
			float* depth = getDepthFloat();
			USHORT* receivedDepth = m_sensor->getZImage();
			//#pragma omp parallel for
			for (int j = 0; j < (int)getDepthWidth()*(int)getDepthHeight(); j++)	{
				depth[j] = receivedDepth[j];
			}

		}
		else
			return S_FALSE; // same frame
		m_depthFrameNumber = m_sensor->getFrameNumber();

		m_bDepthReceived = true;
	}
	else return S_FALSE;

	return hr;
}

HRESULT IntelSensor::processColor()
{
	//return S_FALSE;

	HRESULT hr = S_OK;
	if (m_colorSensor && m_colorSensor->isThirdEnabled()) // && m_colorSensor->getThirdFrameNumber() != m_colorFrameNumber)
	{
		assert(m_colorSensor->getThirdPixelFormat() == DS_RGB8);
		m_colorFrameNumber = m_colorSensor->getThirdFrameNumber();

		BYTE* color = (BYTE*)m_colorSensor->getThirdImage();
		unsigned int width = m_colorSensor->thirdWidth();
		unsigned int height = m_colorSensor->thirdHeight();

		for (unsigned int i = 0; i < getDepthWidth()*getDepthHeight(); i++)
			m_colorRGBX[i] = vec4uc(0, 0, 0, 255);
		// map to depth space
		for (unsigned int y = 0; y < height; y++) {
			for (unsigned int x = 0; x < width; x++) {
				vec3f camPos = m_intrinsicsColorInv * vec3f((float)x, (float)y, 1.f);
				vec3f worldPos = m_extrinsicsColorToDepth * camPos;
				vec3i pixel = vec3i(m_intrinsicsDepth * worldPos);

				if (pixel.y >= 0 && pixel.y < (int)getDepthHeight() &&
					pixel.x >= 0 && pixel.x < (int)getDepthWidth()) {
						unsigned int idx = y * width + x;
						unsigned int nidx = pixel.y*getDepthWidth()+pixel.x;
						m_colorRGBX[nidx] = vec4uc(color[idx*3+2], color[idx*3+1], color[idx*3], 255); //! later wants gbr
				}
			}
		}

		m_bColorReceived = true;
	}
	else return S_FALSE;

	return hr;
}



#endif