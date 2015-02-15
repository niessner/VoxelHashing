#pragma once

/************************************************************************/
/* Main ICP class used for tracking                                     */
/************************************************************************/

#include "DepthCameraUtil.h"
#include "MatrixConversion.h"
#include "DX11QuadDrawer.h"
#include "CUDABuildLinearSystemRGBD.h"
#include "ICPErrorLog.h"
#include "TimingLog.h"
#include "Eigen.h"

#include <D3D11.h>
#include "DX11Utils.h"

#include "CameraTrackingInput.h"


using namespace MatrixConversion;

class CUDACameraTrackingMultiResRGBD
{
public:

	CUDACameraTrackingMultiResRGBD(unsigned int imageWidth, unsigned int imageHeight, unsigned int levels);
	~CUDACameraTrackingMultiResRGBD();

	mat4f applyCT(
		float4* d_inputPos, float4* d_inputNormal, float4* d_inputColor,
		float4* d_targetPos, float4* d_targetNormal, float4* d_targetColor,
		const mat4f& lastTransform, const std::vector<unsigned int>& maxInnerIter, const std::vector<unsigned int>& maxOuterIter,
		const std::vector<float>& distThres, const std::vector<float>& normalThres,
		const std::vector<float>& colorGradiantMin,
		const std::vector<float>& colorThres,
		float condThres, float angleThres, 
		const mat4f& deltaTransformEstimate,
		const std::vector<float>& weightsDepth,
		const std::vector<float>& weightsColor,
		const std::vector<float>& earlyOutResidual, 
		const mat4f& intrinsic, const DepthCameraData& depthCameraData,
		ICPErrorLog* errorLog);

private:

	// angleThres in radians, distThres in meter
	bool checkRigidTransformation(Eigen::Matrix3f& R, Eigen::Vector3f& t, float angleThres, float distThres);

	Eigen::Matrix4f delinearizeTransformation(Vector6f& x, Eigen::Vector3f& mean, float meanStDev, unsigned int level);
	Eigen::Matrix4f computeBestRigidAlignment(CameraTrackingInput cameraTrackingInput, Eigen::Matrix3f& intrinsics, Eigen::Matrix4f& globalDeltaTransform, unsigned int level, CameraTrackingParameters cameraTrackingParameters, unsigned int maxInnerIter, float condThres, float angleThres, LinearSystemConfidence& conf);
	Eigen::Matrix4f align(CameraTrackingInput cameraTrackingInput, Eigen::Matrix4f& deltaTransform, unsigned int level, CameraTrackingParameters cameraTrackingParameters, unsigned int maxInnerIter, unsigned maxOuterIter, float condThres, float angleThres, float earlyOut, const mat4f& intrinsic, const DepthCameraData& depthCameraData, ICPErrorLog* errorLog);
	
	float4** d_input;
	float4** d_inputNormal;
	float**  d_inputIntensity;
	float**  d_inputIntensityFiltered;

	float4** d_model;
	float4** d_modelNormal;
	float**  d_modelIntensity;
	float**  d_modelIntensityFiltered;
	float4** d_modelIntensityAndDerivatives;
	
	// Image Pyramid Dimensions
	unsigned int* m_imageWidth;
	unsigned int* m_imageHeight;
	unsigned int m_levels;

	Eigen::Matrix4f m_matrixTrackingLost;

	CUDABuildLinearSystemRGBD* m_CUDABuildLinearSystem;

	static Timer m_timer;
};
