#include "stdafx.h"

#include "CUDARGBDAdapter.h"
#include "TimingLog.h"

extern "C" void copyFloat4Map(float4* d_output, float4* d_input, unsigned int width, unsigned int height);

extern "C" void convertDepthRawToFloat(float* d_output, unsigned short* d_input, unsigned int width, unsigned int height, float minDepth, float maxDepth);
extern "C" void convertColorRawToFloat4(float4* d_output, BYTE* d_input, unsigned int width, unsigned int height);

extern "C" void resampleFloatMap(float* d_colorMapResampledFloat, unsigned int outputWidth, unsigned int outputHeight, float* d_colorMapFloat, unsigned int inputWidth, unsigned int inputHeight, float* d_depthMaskMap);
extern "C" void resampleFloat4Map(float4* d_colorMapResampledFloat4, unsigned int outputWidth, unsigned int outputHeight, float4* d_colorMapFloat4, unsigned int inputWidth, unsigned int inputHeight);

CUDARGBDAdapter::CUDARGBDAdapter()
{			
	// depth
	d_depthMapFloat = NULL;
	d_depthMapResampledFloat = NULL;

	// color
	d_colorMapRaw = NULL;
	d_colorMapFloat4 = NULL;
	d_colorMapResampledFloat4 = NULL;

	m_frameNumber = 0;
}

CUDARGBDAdapter::~CUDARGBDAdapter()
{
}

void CUDARGBDAdapter::OnD3D11DestroyDevice() 
{		
	// depth
	cutilSafeCall(cudaFree(d_depthMapFloat));
	cutilSafeCall(cudaFree(d_depthMapResampledFloat));
	
	// color
	cutilSafeCall(cudaFree(d_colorMapRaw));
	cutilSafeCall(cudaFree(d_colorMapFloat4));
	cutilSafeCall(cudaFree(d_colorMapResampledFloat4));
}

HRESULT CUDARGBDAdapter::OnD3D11CreateDevice(ID3D11Device* device, RGBDSensor* RGBDSensor, unsigned int width, unsigned int height)
{
	HRESULT hr = S_OK;

	m_RGBDSensor = RGBDSensor;

	m_width = width;
	m_height = height;

	// adapt intrinsics
	m_depthIntrinsics = m_RGBDSensor->getDepthIntrinsics();
	m_depthIntrinsics._m00 *= (float)m_width / (float)m_RGBDSensor->getDepthWidth();
	m_depthIntrinsics._m11 *= (float)m_height / (float)m_RGBDSensor->getDepthHeight();
	m_depthIntrinsics._m02 *= (float)(m_width - 1) / (float)(m_RGBDSensor->getDepthWidth() - 1);
	m_depthIntrinsics._m12 *= (float)(m_height - 1) / (float)(m_RGBDSensor->getDepthHeight() - 1);
	m_depthIntrinsicsInv = m_depthIntrinsics.getInverse();

	m_colorIntrinsics = m_RGBDSensor->getColorIntrinsics();
	m_colorIntrinsics._m00 *= (float)m_width / (float)m_RGBDSensor->getColorWidth();
	m_colorIntrinsics._m11 *= (float)m_height / (float)m_RGBDSensor->getColorHeight();
	m_colorIntrinsics._m02 *= (float)(m_width - 1) / (float)(m_RGBDSensor->getColorWidth() - 1);
	m_colorIntrinsics._m12 *= (float)(m_height - 1) / (float)(m_RGBDSensor->getColorHeight() - 1);
	m_colorIntrinsicsInv = m_colorIntrinsics.getInverse();

	// adapt extrinsics
	m_depthExtrinsics	 =  m_RGBDSensor->getDepthExtrinsics();
	m_depthExtrinsicsInv =  m_RGBDSensor->getDepthExtrinsicsInv();

	m_colorExtrinsics	 =  m_RGBDSensor->getColorExtrinsics();
	m_colorExtrinsicsInv =  m_RGBDSensor->getColorExtrinsicsInv();

	// allocate memory
	const unsigned int bufferDimDepthInput = m_RGBDSensor->getDepthWidth()*m_RGBDSensor->getDepthHeight();
	const unsigned int bufferDimColorInput = m_RGBDSensor->getColorWidth()*m_RGBDSensor->getColorHeight();

	const unsigned int bufferDimOutput = width*height;
	
	// depth
	cutilSafeCall(cudaMalloc(&d_depthMapFloat,			sizeof(float)			* bufferDimDepthInput));
	cutilSafeCall(cudaMalloc(&d_depthMapResampledFloat,	sizeof(float)			* bufferDimOutput));

	// color
	cutilSafeCall(cudaMalloc(&d_colorMapRaw,			 sizeof(unsigned int)*bufferDimColorInput));
	cutilSafeCall(cudaMalloc(&d_colorMapFloat4,			 4*sizeof(float)*bufferDimColorInput));
	cutilSafeCall(cudaMalloc(&d_colorMapResampledFloat4, 4*sizeof(float)*bufferDimOutput));
	
	return hr;
}

HRESULT CUDARGBDAdapter::process(ID3D11DeviceContext* context)
{
	HRESULT hr = S_OK;

	if (m_RGBDSensor->processDepth() != S_OK)	return S_FALSE; // Order is important!
	if (m_RGBDSensor->processColor() != S_OK)	return S_FALSE;

	//Start Timing
	if (GlobalAppState::get().s_timingsDetailledEnabled) { cutilSafeCall(cudaDeviceSynchronize()); m_timer.start(); }

	////////////////////////////////////////////////////////////////////////////////////
	// Process Color
	////////////////////////////////////////////////////////////////////////////////////

	const unsigned int bufferDimColorInput = m_RGBDSensor->getColorWidth()*m_RGBDSensor->getColorHeight();

	cutilSafeCall(cudaMemcpy(d_colorMapRaw, m_RGBDSensor->getColorRGBX(), sizeof(unsigned int)*bufferDimColorInput, cudaMemcpyHostToDevice));
	convertColorRawToFloat4(d_colorMapFloat4, d_colorMapRaw, m_RGBDSensor->getColorWidth(), m_RGBDSensor->getColorHeight());
	
	if ((m_RGBDSensor->getColorWidth() == m_width) && (m_RGBDSensor->getColorHeight() == m_height)) {
		copyFloat4Map(d_colorMapResampledFloat4, d_colorMapFloat4, m_width, m_height);
	} else {
		resampleFloat4Map(d_colorMapResampledFloat4, m_width, m_height, d_colorMapFloat4, m_RGBDSensor->getColorWidth(), m_RGBDSensor->getColorHeight());
	}

	////////////////////////////////////////////////////////////////////////////////////
	// Process Depth
	////////////////////////////////////////////////////////////////////////////////////
	
	const unsigned int bufferDimDepthInput = m_RGBDSensor->getDepthWidth()*m_RGBDSensor->getDepthHeight();
	cutilSafeCall(cudaMemcpy(d_depthMapFloat, m_RGBDSensor->getDepthFloat(), sizeof(float)*m_RGBDSensor->getDepthWidth()* m_RGBDSensor->getDepthHeight(), cudaMemcpyHostToDevice));
	resampleFloatMap(d_depthMapResampledFloat, m_width, m_height, d_depthMapFloat, m_RGBDSensor->getDepthWidth(), m_RGBDSensor->getDepthHeight(), NULL);

	// Stop Timing
	if (GlobalAppState::get().s_timingsDetailledEnabled) { cutilSafeCall(cudaDeviceSynchronize()); m_timer.stop(); TimingLog::totalTimeRGBDAdapter += m_timer.getElapsedTimeMS(); TimingLog::countTimeRGBDAdapter++; }

	m_frameNumber++;

	return hr;
}

