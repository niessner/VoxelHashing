
#include "stdafx.h"

#include "RealSenseSensor.h"

//win8 only
#ifdef REAL_SENSE

RealSenseSensor::RealSenseSensor()
{
	unsigned int depthWidth = 640; //! todo
	unsigned int depthHeight = 480;
	m_depthFps = 30;
	unsigned int colorWidth = 1920;
	unsigned int colorHeight = 1080;
	m_colorFps = 30;

	RGBDSensor::init(depthWidth, depthHeight, colorWidth, colorHeight);

	m_session = NULL;
	m_capture = NULL;
	m_device = NULL;
	m_senseManager = NULL;
}

RealSenseSensor::~RealSenseSensor()
{
	if (m_senseManager) {
		m_senseManager->Close();
		m_senseManager->Release();
	}
	if (m_device) m_device->Release();
	if (m_capture) m_capture->Release();
	m_device = NULL;
	m_capture = NULL;
	if (m_session) m_session->Release();
}

HRESULT RealSenseSensor::createFirstConnected()
{
	HRESULT hr = S_OK;
	m_session = PXCSession::CreateInstance();
	if (!m_session) return S_FALSE;

	PXCSession::ImplDesc mdesc={};
	mdesc.group=PXCSession::IMPL_GROUP_SENSOR;
	mdesc.subgroup=PXCSession::IMPL_SUBGROUP_VIDEO_CAPTURE;
	PXCSession::ImplDesc desc;
	if (m_session->QueryImpl(&mdesc,0,&desc) < PXC_STATUS_NO_ERROR)
		return S_FALSE;
	PXCCapture* capture = NULL;
	if (m_session->CreateImpl<PXCCapture>(&desc, &capture) < PXC_STATUS_NO_ERROR) 
		return S_FALSE;

	PXCCapture::DeviceInfo dinfo;
	if (capture->QueryDeviceInfo(0,&dinfo)<PXC_STATUS_NO_ERROR) 
		return S_FALSE;

	std::cout << "found device " << dinfo.name << std::endl;

	capture->Release();
	if (m_session->CreateImpl<PXCCapture>(desc.iuid, &m_capture) < PXC_STATUS_NO_ERROR)
		return S_FALSE;
	m_device = m_capture->CreateDevice(0);
	if (!m_device) return S_FALSE;

	m_senseManager = m_session->CreateSenseManager();
	if (!m_senseManager) return S_FALSE;

	PXCVideoModule::StreamDesc depthDesc;
	memset(&depthDesc, 0, sizeof(PXCVideoModule::StreamDesc));
	depthDesc.sizeMin.width = getDepthWidth();
	depthDesc.sizeMin.height = getDepthHeight();
	depthDesc.sizeMax.width = getDepthWidth();
	depthDesc.sizeMax.height = getDepthHeight();
	depthDesc.frameRate.min = m_depthFps;
	depthDesc.frameRate.max = m_depthFps;

	PXCVideoModule::StreamDesc colorDesc;
	memset(&depthDesc, 0, sizeof(PXCVideoModule::StreamDesc));
	colorDesc.sizeMin.width = getColorWidth();
	colorDesc.sizeMin.height = getColorHeight();
	colorDesc.sizeMax.width = getColorWidth();
	colorDesc.sizeMax.height = getColorHeight();
	colorDesc.frameRate.min = m_colorFps;
	colorDesc.frameRate.max = m_colorFps;

	PXCVideoModule::DataDesc ddesc = {};
	memset(&ddesc, 0, sizeof(PXCVideoModule::DataDesc));
	ddesc.streams.depth = depthDesc;
	ddesc.streams.color = colorDesc;
	m_capture->QueryDeviceInfo(0, &ddesc.deviceInfo);

	m_senseManager->EnableStreams(&ddesc);

	//PXCPointF32 depthFov = m_device->QueryDepthFieldOfView();
	PXCPointF32 depthFocalLength = m_device->QueryDepthFocalLength();
	initializeDepthIntrinsics(depthFocalLength.x, depthFocalLength.y, getDepthWidth()/2.f, getDepthHeight()/2.f);

	//PXCPointF32 colorFov = m_device->QueryColorFieldOfView();
	PXCPointF32 colorFocalLength = m_device->QueryColorFocalLength();
	initializeColorIntrinsics(colorFocalLength.x, colorFocalLength.y, getColorWidth()/2.f, getColorHeight()/2.f);
	
	initializeColorExtrinsics(mat4f(
		0.9999409077110664, -0.004026476916329122, 0.01009794878235079, -.02572834519558893,
		0.004128179014034619, 0.9999407645726361, -0.01007102198984382, 0.0000846951962910934,
		-0.01005679988847323, 0.01011211301035062, 0.999898297801566, 0.004717521358323926,
		0.0, 0.0, 0.0, 1.0));
	initializeDepthExtrinsics(mat4f::identity());

	pxcStatus sts = m_senseManager->Init();
	if (sts < PXC_STATUS_NO_ERROR) return S_FALSE;

	return hr;
}

HRESULT RealSenseSensor::processDepth()
{
	HRESULT hr = S_OK;

	// waits until new frame is available and locks it for application processing
	pxcStatus sts = m_senseManager->AcquireFrame(false);
	if (sts < PXC_STATUS_NO_ERROR) return S_FALSE;

	//float unit = m_device->QueryDepthUnit();

	const PXCCapture::Sample *sample = m_senseManager->QuerySample();
	if (sample) {
		if (sample->depth) {
			PXCImage::ImageInfo info = sample->depth->QueryInfo();
			PXCImage::ImageData data;
			if (sample->depth->AcquireAccess(PXCImage::ACCESS_READ,PXCImage::PIXEL_FORMAT_DEPTH, &data) >= PXC_STATUS_NO_ERROR) {
				float* depth = getDepthFloat();
				USHORT* sensorDepth = (USHORT*) data.planes[0];
				for (int j = 0; j < (int)getDepthWidth()*(int)getDepthHeight(); j++) {
					const USHORT& d = sensorDepth[j];
					if (d == 0)
						depth[j] = -std::numeric_limits<float>::infinity();
					else
						depth[j] = (float) d * 0.001f;
				}
				sample->depth->ReleaseAccess(&data);
			}
			else 
				return S_FALSE;
		}
		if (sample->color) {
			PXCImage::ImageInfo info = sample->color->QueryInfo();
			PXCImage::ImageData data;
			PXCImage* image = sample->color;
			if (image->AcquireAccess(PXCImage::ACCESS_READ,PXCImage::PIXEL_FORMAT_RGB24, &data) >= PXC_STATUS_NO_ERROR) {
				for (int j = 0; j < (int)getColorWidth()*(int)getColorHeight(); j++) {
					m_colorRGBX[j] = vec4uc(data.planes[0][j*3], data.planes[0][j*3+1], data.planes[0][j*3+2], 255); // both gbr
				}
				image->ReleaseAccess(&data);
			}
		}
	}

	// Releases lock so pipeline can process next frame
	m_senseManager->ReleaseFrame();

	return hr;
}

#endif
