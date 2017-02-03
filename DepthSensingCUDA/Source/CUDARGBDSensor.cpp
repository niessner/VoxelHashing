#include "stdafx.h"

#include "CUDARGBDSensor.h"
#include "TimingLog.h"
#include "DepthCameraUtil.h"
#include <algorithm>


extern "C" void convertDepthRawToFloat(float* d_output, unsigned short* d_input, unsigned int width, unsigned int height, float minDepth, float maxDepth);
extern "C" void convertColorRawToFloat4(float4* d_output, BYTE* d_input, unsigned int width, unsigned int height);
extern "C" void convertDepthFloatToCameraSpaceFloat4(float4* d_output, float* d_input, float4x4 intrinsicsInv, unsigned int width, unsigned int height, const DepthCameraData& depthCameraData);

extern "C" void convertDepthToColorSpace(float* d_output, float* d_input, float4x4 depthIntrinsicsInv, float4x4 colorIntrinsics, float4x4 depthExtrinsicsInvs, float4x4 colorExtrinsics, unsigned int depthWidth, unsigned int depthHeight, unsigned int colorWidth, unsigned int colorHeight);

extern "C" void convertColorToIntensityFloat(float* d_output, float4* d_input, unsigned int width, unsigned int height);

extern "C" void bilateralFilterFloatMap(float* d_output, float* d_input, float sigmaD, float sigmaR, unsigned int width, unsigned int height);
extern "C" void bilateralFilterFloat4Map(float4* d_output, float4* d_input, float sigmaD, float sigmaR, unsigned int width, unsigned int height);

extern "C" void gaussFilterFloatMap(float* d_output, float* d_input, float sigmaD, float sigmaR, unsigned int width, unsigned int height);
extern "C" void gaussFilterFloat4Map(float4* d_output, float4* d_input, float sigmaD, float sigmaR, unsigned int width, unsigned int height);

extern "C" void computeNormals(float4* d_output, float4* d_input, unsigned int width, unsigned int height);
extern "C" void computeNormals2(float4* d_output, float4* d_input, unsigned int width, unsigned int height);

extern "C" void copyFloatMap(float* d_output, float* d_input, unsigned int width, unsigned int height);
extern "C" void copyDepthFloatMap(float* d_output, float* d_input, unsigned int width, unsigned int height, float minDepth, float maxDepth);
extern "C" void copyFloat4Map(float4* d_output, float4* d_input, unsigned int width, unsigned int height);

extern "C" void setInvalidFloatMap(float* d_output, unsigned int width, unsigned int height);
extern "C" void setInvalidFloat4Map(float4* d_output, unsigned int width, unsigned int height);

extern "C" void upsampleDepthMap(float* d_output, float* d_input, unsigned int depthWidth, unsigned int depthHeight, unsigned int colorWidth, unsigned int colorHeight);

extern "C" void convertColorFloat4ToUCHAR4(uchar4* d_output, float4* d_input, unsigned int width, unsigned int height);
extern "C" void convertColorToIntensityFloat(float* d_output, float4* d_input, unsigned int width, unsigned int height);

extern "C" void erodeDepthMap(float* d_output, float* d_input, int structureSize, unsigned int width, unsigned int height, float dThresh, float fracReq);

extern "C" void depthToHSV(float4* d_output, float* d_input, unsigned int width, unsigned int height, float minDepth, float maxDepth);

CUDARGBDSensor::CUDARGBDSensor()
{
	m_bFilterDepthValues = false;
	m_fBilateralFilterSigmaD = 1.0f;
	m_fBilateralFilterSigmaR = 1.0f;

	m_bFilterIntensityValues = false;
	m_fBilateralFilterSigmaDIntensity = 1.0f;
	m_fBilateralFilterSigmaRIntensity = 1.0f;

	// depth
	d_depthMapFilteredFloat = NULL;
	d_cameraSpaceFloat4 = NULL;

	// intensity
	d_intensityMapFilteredFloat = NULL;

	// normals
	d_normalMapFloat4 = NULL;

	d_depthErodeHelper = NULL;
	d_colorErodeHelper = NULL;

	d_depthHSV = NULL;
}

CUDARGBDSensor::~CUDARGBDSensor()
{
	
}

void CUDARGBDSensor::OnD3D11DestroyDevice()
{
	// depth
	cutilSafeCall(cudaFree(d_depthMapFilteredFloat));
	cutilSafeCall(cudaFree(d_cameraSpaceFloat4));

	// intensity
	cutilSafeCall(cudaFree(d_intensityMapFilteredFloat));

	// normals
	cutilSafeCall(cudaFree(d_normalMapFloat4));

	cutilSafeCall(cudaFree(d_depthErodeHelper));
	cutilSafeCall(cudaFree(d_colorErodeHelper));

	cutilSafeCall(cudaFree(d_depthHSV));

	g_RGBDRenderer.OnD3D11DestroyDevice();
	g_CustomRenderTarget.OnD3D11DestroyDevice();

	m_depthCameraData.free();
}

HRESULT CUDARGBDSensor::OnD3D11CreateDevice(ID3D11Device* device, CUDARGBDAdapter* CUDARGBDAdapter)
{
	HRESULT hr = S_OK;

	m_RGBDAdapter = CUDARGBDAdapter;

	Matrix4f M(m_RGBDAdapter->getColorIntrinsics().ptr()); M.transposeInPlace();

	const unsigned int bufferDimDepth = m_RGBDAdapter->getWidth()*m_RGBDAdapter->getHeight();
	const unsigned int bufferDimColor = m_RGBDAdapter->getWidth()*m_RGBDAdapter->getHeight();

	cutilSafeCall(cudaMalloc(&d_depthMapFilteredFloat, sizeof(float)*bufferDimDepth));
	cutilSafeCall(cudaMalloc(&d_cameraSpaceFloat4, 4 * sizeof(float)*bufferDimDepth));

	// normal
	cutilSafeCall(cudaMalloc(&d_normalMapFloat4, 4 * sizeof(float)*bufferDimColor));

	// intensity
	cutilSafeCall(cudaMalloc(&d_intensityMapFilteredFloat, sizeof(float)*bufferDimColor));

	cutilSafeCall(cudaMalloc(&d_depthErodeHelper, sizeof(float)*bufferDimDepth));
	cutilSafeCall(cudaMalloc(&d_colorErodeHelper, sizeof(float4)*bufferDimColor));

	cutilSafeCall(cudaMalloc(&d_depthHSV, sizeof(float4)*bufferDimDepth));

	std::vector<DXGI_FORMAT> formats;
	formats.push_back(DXGI_FORMAT_R32_FLOAT);
	formats.push_back(DXGI_FORMAT_R32G32B32A32_FLOAT);
	formats.push_back(DXGI_FORMAT_R32G32B32A32_FLOAT);
	formats.push_back(DXGI_FORMAT_R32G32B32A32_FLOAT);

	V_RETURN(g_RGBDRenderer.OnD3D11CreateDevice(device, GlobalAppState::get().s_adapterWidth, GlobalAppState::get().s_adapterHeight));
	V_RETURN(g_CustomRenderTarget.OnD3D11CreateDevice(device, GlobalAppState::get().s_adapterWidth, GlobalAppState::get().s_adapterHeight, formats));

	m_depthCameraParams.fx = m_RGBDAdapter->getDepthIntrinsics()(0, 0);
	m_depthCameraParams.fy = m_RGBDAdapter->getDepthIntrinsics()(1, 1);
	m_depthCameraParams.mx = m_RGBDAdapter->getDepthIntrinsics()(0, 2);
	m_depthCameraParams.my = m_RGBDAdapter->getDepthIntrinsics()(1, 2);
	m_depthCameraParams.m_sensorDepthWorldMin = GlobalAppState::get().s_sensorDepthMin;
	m_depthCameraParams.m_sensorDepthWorldMax = GlobalAppState::get().s_sensorDepthMax;
	m_depthCameraParams.m_imageHeight = m_RGBDAdapter->getHeight();
	m_depthCameraParams.m_imageWidth = m_RGBDAdapter->getWidth();

	m_depthCameraData.alloc(m_depthCameraParams);

	return hr;
}

HRESULT CUDARGBDSensor::process(ID3D11DeviceContext* context)
{
	HRESULT hr = S_OK;

	if (m_RGBDAdapter->process(context) == S_FALSE)	return S_FALSE;

	////////////////////////////////////////////////////////////////////////////////////
	// Process Color
	////////////////////////////////////////////////////////////////////////////////////

	//Start Timing
	if (GlobalAppState::get().s_timingsDetailledEnabled) { cutilSafeCall(cudaDeviceSynchronize()); m_timer.start(); }

	if (m_bFilterIntensityValues)	gaussFilterFloat4Map(m_depthCameraData.d_colorData, m_RGBDAdapter->getColorMapResampledFloat4(), m_fBilateralFilterSigmaDIntensity, m_fBilateralFilterSigmaRIntensity, m_RGBDAdapter->getWidth(), m_RGBDAdapter->getHeight());
	else							copyFloat4Map(m_depthCameraData.d_colorData, m_RGBDAdapter->getColorMapResampledFloat4(), m_RGBDAdapter->getWidth(), m_RGBDAdapter->getHeight());

	// Stop Timing
	if (GlobalAppState::get().s_timingsDetailledEnabled) { cutilSafeCall(cudaDeviceSynchronize()); m_timer.stop(); TimingLog::totalTimeFilterColor += m_timer.getElapsedTimeMS(); TimingLog::countTimeFilterColor++; }

	////////////////////////////////////////////////////////////////////////////////////
	// Process Depth
	////////////////////////////////////////////////////////////////////////////////////

	//Start Timing
	if (GlobalAppState::get().s_timingsDetailledEnabled) { cutilSafeCall(cudaDeviceSynchronize()); m_timer.start(); }

	if (m_bFilterDepthValues) gaussFilterFloatMap(d_depthMapFilteredFloat, m_RGBDAdapter->getDepthMapResampledFloat(), m_fBilateralFilterSigmaD, m_fBilateralFilterSigmaR, m_RGBDAdapter->getWidth(), m_RGBDAdapter->getHeight());
	else					 copyFloatMap(d_depthMapFilteredFloat, m_RGBDAdapter->getDepthMapResampledFloat(), m_RGBDAdapter->getWidth(), m_RGBDAdapter->getHeight());

	//TODO this call seems not needed as the depth map is overwriten later anyway later anyway...
	setInvalidFloatMap(m_depthCameraData.d_depthData, m_RGBDAdapter->getWidth(), m_RGBDAdapter->getHeight());

	// Stop Timing
	if (GlobalAppState::get().s_timingsDetailledEnabled) { cutilSafeCall(cudaDeviceSynchronize()); m_timer.stop(); TimingLog::totalTimeFilterDepth += m_timer.getElapsedTimeMS(); TimingLog::countTimeFilterDepth++; }

	////////////////////////////////////////////////////////////////////////////////////
	// Render to Color Space
	////////////////////////////////////////////////////////////////////////////////////

	//Start Timing
	if (GlobalAppState::get().s_timingsDetailledEnabled) { cutilSafeCall(cudaDeviceSynchronize()); m_timer.start(); }

	if (GlobalAppState::get().s_bUseCameraCalibration)
	{
		mat4f depthExt = m_RGBDAdapter->getDepthExtrinsics();

		g_CustomRenderTarget.Clear(context);
		g_CustomRenderTarget.Bind(context);
		g_RGBDRenderer.RenderDepthMap(context,
			d_depthMapFilteredFloat, m_depthCameraData.d_colorData, m_RGBDAdapter->getWidth(), m_RGBDAdapter->getHeight(),
			m_RGBDAdapter->getDepthIntrinsicsInv(), depthExt, m_RGBDAdapter->getColorIntrinsics(),
			g_CustomRenderTarget.getWidth(), g_CustomRenderTarget.getHeight(),
			GlobalAppState::get().s_remappingDepthDiscontinuityThresOffset, GlobalAppState::get().s_remappingDepthDiscontinuityThresLin);
		g_CustomRenderTarget.Unbind(context);
		g_CustomRenderTarget.copyToCuda(m_depthCameraData.d_depthData, 0);


		//Util::writeToImage(m_depthCameraData.d_depthData, getDepthWidth(), getDepthHeight(), "depth.png");
		//Util::writeToImage(m_depthCameraData.d_colorData, getDepthWidth(), getDepthHeight(), "color.png");
	}
	else
	{
		copyFloatMap(m_depthCameraData.d_depthData, d_depthMapFilteredFloat, m_RGBDAdapter->getWidth(), m_RGBDAdapter->getHeight());
	}


	bool bErode = false;
	if (bErode) {
		unsigned int numIter = 20;

		numIter = 2 * ((numIter + 1) / 2);
		for (unsigned int i = 0; i < numIter; i++) {
			if (i % 2 == 0) {
				erodeDepthMap(d_depthErodeHelper, m_depthCameraData.d_depthData, 5, getDepthWidth(), getDepthHeight(), 0.05f, 0.3f);
			}
			else {
				erodeDepthMap(m_depthCameraData.d_depthData, d_depthErodeHelper, 5, getDepthWidth(), getDepthHeight(), 0.05f, 0.3f);
			}
		}
	}

	//TODO check whether the intensity is actually used
	convertColorToIntensityFloat(d_intensityMapFilteredFloat, m_depthCameraData.d_colorData, m_RGBDAdapter->getWidth(), m_RGBDAdapter->getHeight());


	float4x4 M((m_RGBDAdapter->getColorIntrinsicsInv()).ptr());
	m_depthCameraData.updateParams(getDepthCameraParams());
	convertDepthFloatToCameraSpaceFloat4(d_cameraSpaceFloat4, m_depthCameraData.d_depthData, M, m_RGBDAdapter->getWidth(), m_RGBDAdapter->getHeight(), m_depthCameraData); // !!! todo
	computeNormals(d_normalMapFloat4, d_cameraSpaceFloat4, m_RGBDAdapter->getWidth(), m_RGBDAdapter->getHeight());

	float4x4 Mintrinsics((m_RGBDAdapter->getColorIntrinsics()).ptr());
	cudaMemcpyToArray(m_depthCameraData.d_depthArray, 0, 0, m_depthCameraData.d_depthData, sizeof(float)*m_depthCameraParams.m_imageHeight*m_depthCameraParams.m_imageWidth, cudaMemcpyDeviceToDevice);
	cudaMemcpyToArray(m_depthCameraData.d_colorArray, 0, 0, m_depthCameraData.d_colorData, sizeof(float4)*m_depthCameraParams.m_imageHeight*m_depthCameraParams.m_imageWidth, cudaMemcpyDeviceToDevice);

	// Stop Timing
	if (GlobalAppState::get().s_timingsDetailledEnabled) { cutilSafeCall(cudaDeviceSynchronize()); m_timer.stop(); TimingLog::totalTimeRemapDepth += m_timer.getElapsedTimeMS(); TimingLog::countTimeRemapDepth++; }

	return hr;
}

//! enables bilateral filtering of the depth value
void CUDARGBDSensor::setFiterDepthValues(bool b, float sigmaD, float sigmaR)
{
	m_bFilterDepthValues = b;
	m_fBilateralFilterSigmaD = sigmaD;
	m_fBilateralFilterSigmaR = sigmaR;
}

void CUDARGBDSensor::setFiterIntensityValues(bool b, float sigmaD, float sigmaR)
{
	m_bFilterIntensityValues = b;
	m_fBilateralFilterSigmaDIntensity = sigmaD;
	m_fBilateralFilterSigmaRIntensity = sigmaR;
}

bool CUDARGBDSensor::checkValidT1(unsigned int x, unsigned int y, unsigned int W, float4* h_cameraSpaceFloat4)
{
	return		(h_cameraSpaceFloat4[y*W + x].z != 0.0 && h_cameraSpaceFloat4[(y + 1)*W + x].z != 0.0 && h_cameraSpaceFloat4[y*W + x + 1].z != 0.0)
		&& (h_cameraSpaceFloat4[y*W + x].z != -std::numeric_limits<float>::infinity() && h_cameraSpaceFloat4[(y + 1)*W + x].z != -std::numeric_limits<float>::infinity() && h_cameraSpaceFloat4[y*W + x + 1].z != -std::numeric_limits<float>::infinity())
		&& (h_cameraSpaceFloat4[y*W + x].z > GlobalAppState::get().s_sensorDepthMin && h_cameraSpaceFloat4[(y + 1)*W + x].z > GlobalAppState::get().s_sensorDepthMin && h_cameraSpaceFloat4[y*W + x + 1].z > GlobalAppState::get().s_sensorDepthMin)
		&& (h_cameraSpaceFloat4[y*W + x].z < GlobalAppState::get().s_sensorDepthMax && h_cameraSpaceFloat4[(y + 1)*W + x].z < GlobalAppState::get().s_sensorDepthMax && h_cameraSpaceFloat4[y*W + x + 1].z < GlobalAppState::get().s_sensorDepthMax)
		&& (fabs(h_cameraSpaceFloat4[y*W + x].x) < GlobalAppState::get().s_sensorDepthMax && fabs(h_cameraSpaceFloat4[(y + 1)*W + x].x) < GlobalAppState::get().s_sensorDepthMax && fabs(h_cameraSpaceFloat4[y*W + x + 1].x) < GlobalAppState::get().s_sensorDepthMax)
		&& (fabs(h_cameraSpaceFloat4[y*W + x].y) < GlobalAppState::get().s_sensorDepthMax && fabs(h_cameraSpaceFloat4[(y + 1)*W + x].y) < GlobalAppState::get().s_sensorDepthMax && fabs(h_cameraSpaceFloat4[y*W + x + 1].y) < GlobalAppState::get().s_sensorDepthMax);
}

bool CUDARGBDSensor::checkValidT2(unsigned int x, unsigned int y, unsigned int W, float4* h_cameraSpaceFloat4)
{
	return		(h_cameraSpaceFloat4[(y + 1)*W + x].z != 0.0 && h_cameraSpaceFloat4[(y + 1)*W + (x + 1)].z != 0.0 && h_cameraSpaceFloat4[y*W + x + 1].z != 0.0)
		&& (h_cameraSpaceFloat4[(y + 1)*W + x].z != -std::numeric_limits<float>::infinity() && h_cameraSpaceFloat4[(y + 1)*W + (x + 1)].z != -std::numeric_limits<float>::infinity() && h_cameraSpaceFloat4[y*W + x + 1].z != -std::numeric_limits<float>::infinity())
		&& (h_cameraSpaceFloat4[(y + 1)*W + x].z > GlobalAppState::get().s_sensorDepthMin && h_cameraSpaceFloat4[(y + 1)*W + (x + 1)].z > GlobalAppState::get().s_sensorDepthMin && h_cameraSpaceFloat4[y*W + x + 1].z > GlobalAppState::get().s_sensorDepthMin)
		&& (h_cameraSpaceFloat4[(y + 1)*W + x].z < GlobalAppState::get().s_sensorDepthMax && h_cameraSpaceFloat4[(y + 1)*W + (x + 1)].z < GlobalAppState::get().s_sensorDepthMax && h_cameraSpaceFloat4[y*W + x + 1].z < GlobalAppState::get().s_sensorDepthMax)
		&& (fabs(h_cameraSpaceFloat4[(y + 1)*W + x].x) < GlobalAppState::get().s_sensorDepthMax && fabs(h_cameraSpaceFloat4[(y + 1)*W + (x + 1)].x) < GlobalAppState::get().s_sensorDepthMax && fabs(h_cameraSpaceFloat4[y*W + x + 1].x) < GlobalAppState::get().s_sensorDepthMax)
		&& (fabs(h_cameraSpaceFloat4[(y + 1)*W + x].y) < GlobalAppState::get().s_sensorDepthMax && fabs(h_cameraSpaceFloat4[(y + 1)*W + (x + 1)].y) < GlobalAppState::get().s_sensorDepthMax && fabs(h_cameraSpaceFloat4[y*W + x + 1].y) < GlobalAppState::get().s_sensorDepthMax);
}

float4* CUDARGBDSensor::getCameraSpacePositionsFloat4()
{
	return d_cameraSpaceFloat4;
}

float* CUDARGBDSensor::getDepthMapColorSpaceFloat()
{
	return m_depthCameraData.d_depthData;
}

float4* CUDARGBDSensor::getColorMapFilteredFloat4()
{
	return m_depthCameraData.d_colorData;
}

float4* CUDARGBDSensor::getNormalMapFloat4()
{
	return d_normalMapFloat4;
}

unsigned int CUDARGBDSensor::getDepthWidth() const
{
	return m_RGBDAdapter->getWidth();
}

unsigned int CUDARGBDSensor::getDepthHeight() const
{
	return m_RGBDAdapter->getHeight();
}

unsigned int CUDARGBDSensor::getColorWidth() const
{
	return m_RGBDAdapter->getWidth();
}

unsigned int CUDARGBDSensor::getColorHeight() const
{
	return m_RGBDAdapter->getHeight();
}


float4* CUDARGBDSensor::getAndComputeDepthHSV() const
{
	depthToHSV(d_depthHSV, m_depthCameraData.d_depthData, getDepthWidth(), getDepthHeight(), GlobalAppState::get().s_sensorDepthMin, GlobalAppState::get().s_sensorDepthMax);
	return d_depthHSV;
}