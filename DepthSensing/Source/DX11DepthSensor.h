#pragma once

/*************************************************************************************/
/* Manages all DX11 resources on the GPU of a Depth Sensor (which is a member)       */
/*************************************************************************************/

#include "DX11Utils.h"
#include "DepthSensor.h"
#include "DX11ImageHelper.h"

class DX11Sensor
{
public:

	//! Constructor
	DX11Sensor();

	//! Destructor
	~DX11Sensor();

	//! Releases all resources
	void OnD3D11DestroyDevice();

	//! Allocates all resources (must get a valid device and depthSensor)
	HRESULT OnD3D11CreateDevice(ID3D11Device* device, DepthSensor* depthSensor);

	//! Gets the next depth frame and loads it to the GPU (calls 'processDepth') of the 
	HRESULT processDepth(ID3D11DeviceContext* context);

	//! maps the color to depth data and copies depth and color data to the GPU
	HRESULT processColor(ID3D11DeviceContext* context);

	//! enables bilateral filtering of the depth value
	void setFiterDepthValues(bool b = true, float sigmaD = 1.0f, float sigmaR = 1.0f) {
		m_bFilterDepthValues = b;
		m_fBilateralFilterSigmaD = sigmaD;
		m_fBilateralFilterSigmaR = sigmaR;
	}

	ID3D11ShaderResourceView* GetDepthUSSRV() {
		return m_pDepthTextureUSSRV;
	}

	ID3D11ShaderResourceView* GetDepthFSRV() {
		return m_pDepthTextureFSRV;
	}

	ID3D11Texture2D* GetDepthFFiltered() {
		return m_pDepthTextureFFiltered2D;
	}

	ID3D11ShaderResourceView* GetDepthFFilteredSRV() {
		return m_pDepthTextureFFilteredSRV;
	}

	ID3D11Texture2D* GetDepthFloat4() {
		return m_pDepthTextureFloat42D;
	}

	ID3D11ShaderResourceView* GetDepthFErodedSRV() {
		return m_pDepthTextureFErodedSRV;
	}

	ID3D11ShaderResourceView* GetDepthFloat4SRV() {
		return m_pDepthTextureFloat4SRV;
	}

	ID3D11ShaderResourceView* GetHSVDepthFloat4SRV() {
		return m_pHSVDepthTextureFloat4SRV;
	}

	ID3D11Texture2D* GetHSVDepthFloatTexture() {
		return m_pHSVDepthTextureFloat42D;
	}

	//ID3D11ShaderResourceView* GetDepthFloat4NoSmoothingSRV() {
	//	return m_pDepthTextureFloat4NoSmoothingSRV;
	//}

	ID3D11ShaderResourceView* GetNormalFloat4SRV() {
		return m_pNormalTextureFloat4SRV;
	}

	ID3D11ShaderResourceView* GetColorSRV() {
		return m_pColorTextureSRV;
	}

	ID3D11Texture2D* GetColorTexture() {
		return m_pColorTexture2D;
	}

	unsigned int GetFrameNumberDepth() const {
		return m_FrameNumberDepth;
	}

	void writeDepthDataToFile(const std::string& filename) const {
		m_depthSensor->writeDepthDataToFile(filename);
	}

	void writeColorDataToFile(const std::string& filename) const {
		m_depthSensor->writeColorDataToFile(filename);
	}

	void savePointCloud(const std::string& filename, const mat4f& transform = mat4f::identity()) const {
		std::cout << "Saving depth frame as point cloud: " << filename << std::endl;
		m_depthSensor->savePointCloud(filename, transform);
	}

	//! resets the frame counter and all recorded date (if available)
	void reset() {
		m_FrameNumberDepth = 0;
		m_depthSensor->reset();
	}
	void recordFrame() {
		m_depthSensor->recordFrame();
	}
	void recordTrajectory(const mat4f& transformation) {
		m_depthSensor->recordTrajectory(transformation);
	}
	void saveRecordedFramesToFile(const std::string& filename) {
		m_depthSensor->saveRecordedFramesToFile(filename);
	}

	void recordPointCloud(const mat4f& transform = mat4f::identity()) {
		m_depthSensor->recordPointCloud(transform);
	}

	void saveRecordedPointCloud(const std::string& filename) {
		m_depthSensor->saveRecordedPointCloud(filename);
	}

	mat4f getRigidTransform() const {
		return m_depthSensor->getRigidTransform();
	}
private:
	unsigned int m_FrameNumberDepth;

	DepthSensor* m_depthSensor;

	//! only interesting for CPU usage
	bool			m_bFilterApprox;
	bool			m_bFilterDepthValues;
	float			m_fBilateralFilterSigmaD;
	float			m_fBilateralFilterSigmaR;

	// for passing depth data as a texture
	ID3D11Texture2D*                    m_pDepthTextureUS2D;
	ID3D11ShaderResourceView*           m_pDepthTextureUSSRV;

	//! depth texture float
	ID3D11Texture2D*					m_pDepthTextureF2D;
	ID3D11ShaderResourceView*           m_pDepthTextureFSRV;
	ID3D11UnorderedAccessView*          m_pDepthTextureFUAV;

	//! depth texture float filtered
	ID3D11Texture2D*					m_pDepthTextureFEroded2D;
	ID3D11ShaderResourceView*           m_pDepthTextureFErodedSRV;
	ID3D11UnorderedAccessView*          m_pDepthTextureFErodedUAV;

	//! depth texture float filtered
	ID3D11Texture2D*					m_pDepthTextureFFiltered2D;
	ID3D11ShaderResourceView*           m_pDepthTextureFFilteredSRV;
	ID3D11UnorderedAccessView*          m_pDepthTextureFFilteredUAV;

	//! HSV depth texture
	ID3D11Texture2D*					m_pHSVDepthTextureFloat42D;
	ID3D11ShaderResourceView*           m_pHSVDepthTextureFloat4SRV;
	ID3D11UnorderedAccessView*          m_pHSVDepthTextureFloat4UAV;

	//! camera space points
	ID3D11Texture2D*					m_pDepthTextureFloat42D;
	ID3D11ShaderResourceView*           m_pDepthTextureFloat4SRV;
	ID3D11UnorderedAccessView*          m_pDepthTextureFloat4UAV;

	////! camera space points
	//ID3D11Texture2D*					m_pDepthTextureFloat4NoSmoothing2D;
	//ID3D11ShaderResourceView*           m_pDepthTextureFloat4NoSmoothingSRV;
	//ID3D11UnorderedAccessView*          m_pDepthTextureFloat4NoSmoothingUAV;

	//! camera space normals
	ID3D11Texture2D*					m_pNormalTextureFloat42D;
	ID3D11ShaderResourceView*           m_pNormalTextureFloat4SRV;
	ID3D11UnorderedAccessView*          m_pNormalTextureFloat4UAV;

	// for passing color data as a texture
	ID3D11Texture2D*                    m_pColorTexture2D;
	ID3D11ShaderResourceView*           m_pColorTextureSRV;

	Timer m_timer;


};
