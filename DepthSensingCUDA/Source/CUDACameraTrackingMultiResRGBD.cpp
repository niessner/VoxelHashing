#include "stdafx.h"

#include "CUDACameraTrackingMultiResRGBD.h"
#include "CUDAImageHelper.h"

#include "GlobalAppState.h"

#include <cutil_inline.h>
#include <cutil_math.h>

#include <iostream>
#include <limits>

/////////////////////////////////////////////////////
// Camera Tracking Multi Res
/////////////////////////////////////////////////////

Timer CUDACameraTrackingMultiResRGBD::m_timer;

extern "C" void resampleFloat4Map(float4* d_colorMapResampledFloat4, unsigned int outputWidth, unsigned int outputHeight, float4* d_colorMapFloat4, unsigned int inputWidth, unsigned int inputHeight);
extern "C" void resampleFloatMap(float* d_colorMapResampledFloat, unsigned int outputWidth, unsigned int outputHeight, float* d_colorMapFloat, unsigned int inputWidth, unsigned int inputHeight);

extern "C" void computeNormals(float4* d_output, float4* d_input, unsigned int width, unsigned int height);
extern "C" void computeDerivativesCameraSpace(float4* d_positions, unsigned int imageWidth, unsigned int imageHeight, float4* d_positionsDU, float4* d_positionsDV);

extern "C" void convertColorToIntensityFloat(float* d_output, float4* d_input, unsigned int width, unsigned int height);
extern "C" void convertColorToIntensityFloat4(float4* d_output, float4* d_input, unsigned int width, unsigned int height);

extern "C" void computeGradientIntensityMagnitude(float4* d_inputDU, float4* d_inputDV, unsigned int imageWidth, unsigned int imageHeight, float4* d_ouput);

extern "C" void computeIntensityAndDerivatives(float* d_intensity, unsigned int imageWidth, unsigned int imageHeight, float4* d_intensityAndDerivatives);


extern "C" void gaussFilterFloat4Map(float4* d_output, float4* d_input, float sigmaD, float sigmaR, unsigned int width, unsigned int height);
extern "C" void gaussFilterFloatMap(float* d_output, float* d_input, float sigmaD, float sigmaR, unsigned int width, unsigned int height);

extern "C" void copyFloatMap(float* d_output, float* d_input, unsigned int width, unsigned int height);

CUDACameraTrackingMultiResRGBD::CUDACameraTrackingMultiResRGBD(unsigned int imageWidth, unsigned int imageHeight, unsigned int levels) {
	m_levels = levels;

	d_input = new float4*[m_levels];
	d_inputNormal = new float4*[m_levels];
	
	d_inputIntensity = new float*[m_levels];
	d_inputIntensityFiltered = new float*[m_levels];

	d_model = new float4*[m_levels];
	d_modelNormal = new float4*[m_levels];
	
	d_modelIntensity = new float*[m_levels];
	d_modelIntensityFiltered = new float*[m_levels];
	d_modelIntensityAndDerivatives = new float4*[m_levels];

	m_imageWidth = new unsigned int[m_levels];
	m_imageHeight = new unsigned int[m_levels];

	unsigned int fac = 1;
	for (unsigned int i = 0; i<m_levels; i++) {
		m_imageWidth[i] = imageWidth/fac;
		m_imageHeight[i] = imageHeight/fac;

		// input
		if (i != 0) {  // Not finest level
			cutilSafeCall(cudaMalloc(&d_input[i], sizeof(float4)*m_imageWidth[i]*m_imageHeight[i]));
			cutilSafeCall(cudaMalloc(&d_inputNormal[i], sizeof(float4)*m_imageWidth[i]*m_imageHeight[i]));
		} else {
			d_input[i] = NULL;
			d_inputNormal[i] = NULL;
		}

		cutilSafeCall(cudaMalloc(&d_inputIntensity[i], sizeof(float)*m_imageWidth[i]*m_imageHeight[i]));
		cutilSafeCall(cudaMalloc(&d_inputIntensityFiltered[i], sizeof(float)*m_imageWidth[i]*m_imageHeight[i]));

		// model
		if (i != 0) { // Not finest level
			cutilSafeCall(cudaMalloc(&d_model[i], sizeof(float4)*m_imageWidth[i]*m_imageHeight[i]));
			cutilSafeCall(cudaMalloc(&d_modelNormal[i], sizeof(float4)*m_imageWidth[i]*m_imageHeight[i]));
		} else {
			d_model[i] = NULL;
			d_modelNormal[i] = NULL;
		}

		cutilSafeCall(cudaMalloc(&d_modelIntensity[i], sizeof(float)*m_imageWidth[i]*m_imageHeight[i]));
		cutilSafeCall(cudaMalloc(&d_modelIntensityFiltered[i], sizeof(float)*m_imageWidth[i]*m_imageHeight[i]));
		cutilSafeCall(cudaMalloc(&d_modelIntensityAndDerivatives[i], sizeof(float4)*m_imageWidth[i]*m_imageHeight[i]));

		fac*=2;
	}

	m_matrixTrackingLost.fill(-std::numeric_limits<float>::infinity());

	m_CUDABuildLinearSystem = new CUDABuildLinearSystemRGBD(m_imageWidth[0], m_imageHeight[0]);
}

CUDACameraTrackingMultiResRGBD::~CUDACameraTrackingMultiResRGBD() {

	d_input[0] = NULL;
	d_inputNormal[0] = NULL;
	
	d_model[0] = NULL;
	d_modelNormal[0] = NULL;
	
	// input
	if (d_input) {
		for (unsigned int i = 0; i < m_levels; i++) 
			if (d_input[i])	cutilSafeCall(cudaFree(d_input[i]));
		SAFE_DELETE_ARRAY(d_input)
	}

	if (d_inputNormal) {
		for (unsigned int i = 0; i < m_levels; i++) 
			if (d_inputNormal[i])	cutilSafeCall(cudaFree(d_inputNormal[i]));
		SAFE_DELETE_ARRAY(d_inputNormal)
	}
	
	if (d_inputIntensity) {
		for (unsigned int i = 0; i < m_levels; i++) 
			if (d_inputIntensity[i])	cutilSafeCall(cudaFree(d_inputIntensity[i]));
		SAFE_DELETE_ARRAY(d_inputIntensity)
	}

	if (d_inputIntensityFiltered) {
		for (unsigned int i = 0; i < m_levels; i++) 
			if (d_inputIntensityFiltered[i])	cutilSafeCall(cudaFree(d_inputIntensityFiltered[i]));
		SAFE_DELETE_ARRAY(d_inputIntensityFiltered)
	}

	// model
	if (d_model) {
		for (unsigned int i = 0; i < m_levels; i++) 
			if (d_model[i])	cutilSafeCall(cudaFree(d_model[i]));
		SAFE_DELETE_ARRAY(d_model)
	}

	if (d_modelIntensity) {
		for (unsigned int i = 0; i < m_levels; i++) 
			if (d_modelIntensity[i]) cutilSafeCall(cudaFree(d_modelIntensity[i]));
		SAFE_DELETE_ARRAY(d_modelIntensity)
	}

	if (d_modelIntensityFiltered) {
		for (unsigned int i = 0; i < m_levels; i++) 
			if (d_modelIntensityFiltered[i]) cutilSafeCall(cudaFree(d_modelIntensityFiltered[i]));
		SAFE_DELETE_ARRAY(d_modelIntensityFiltered)
	}

	if (d_modelIntensityAndDerivatives) {
		for (unsigned int i = 0; i < m_levels; i++) 
			if (d_modelIntensityAndDerivatives[i]) cutilSafeCall(cudaFree(d_modelIntensityAndDerivatives[i]));
		SAFE_DELETE_ARRAY(d_modelIntensityAndDerivatives)
	}

	if (d_modelNormal) {
		for (unsigned int i = 0; i < m_levels; i++) 
			if (d_modelNormal[i])	cutilSafeCall(cudaFree(d_modelNormal[i]));
		SAFE_DELETE_ARRAY(d_modelNormal)
	}
	
	if (m_imageWidth)	SAFE_DELETE_ARRAY(m_imageWidth);
	if (m_imageHeight)	SAFE_DELETE_ARRAY(m_imageHeight);

	SAFE_DELETE(m_CUDABuildLinearSystem);
}

bool CUDACameraTrackingMultiResRGBD::checkRigidTransformation(Eigen::Matrix3f& R, Eigen::Vector3f& t, float angleThres, float distThres) {
	Eigen::AngleAxisf aa(R);

	if (aa.angle() > angleThres || t.norm() > distThres) {
		std::cout << "Tracking lost: angle " << (aa.angle()/M_PI)*180.0f << " translation " << t.norm() << std::endl;
		return false;
	}

	return true;
}

Eigen::Matrix4f CUDACameraTrackingMultiResRGBD::delinearizeTransformation(Vector6f& x, Eigen::Vector3f& mean, float meanStDev, unsigned int level) 
{
	Eigen::Matrix3f R =	 Eigen::AngleAxisf(x[0], Eigen::Vector3f::UnitZ()).toRotationMatrix()  // Rot Z
						*Eigen::AngleAxisf(x[1], Eigen::Vector3f::UnitY()).toRotationMatrix()  // Rot Y
						*Eigen::AngleAxisf(x[2], Eigen::Vector3f::UnitX()).toRotationMatrix(); // Rot X

	Eigen::Vector3f t = x.segment(3, 3);

	if(!checkRigidTransformation(R, t, GlobalCameraTrackingState::getInstance().s_angleTransThres[level], GlobalCameraTrackingState::getInstance().s_distTransThres[level])) {
		return m_matrixTrackingLost;
	}

	Eigen::Matrix4f res; res.setIdentity();
	res.block(0, 0, 3, 3) = R;
	res.block(0, 3, 3, 1) = meanStDev*t+mean-R*mean;

	return res;
}

Eigen::Matrix4f CUDACameraTrackingMultiResRGBD::computeBestRigidAlignment(CameraTrackingInput cameraTrackingInput, Eigen::Matrix3f& intrinsics, Eigen::Matrix4f& globalDeltaTransform, unsigned int level, CameraTrackingParameters cameraTrackingParameters, unsigned int maxInnerIter, float condThres, float angleThres, LinearSystemConfidence& conf) 
{
	Eigen::Matrix4f deltaTransform = globalDeltaTransform;

	conf.reset();

	Matrix6x7f system;

	Eigen::Matrix3f ROld = deltaTransform.block(0, 0, 3, 3);
	Eigen::Vector3f eulerAngles = ROld.eulerAngles(2, 1, 0);

	float3 anglesOld;	   anglesOld.x = eulerAngles.x(); anglesOld.y = eulerAngles.y(); anglesOld.z = eulerAngles.z();
	float3 translationOld; translationOld.x = deltaTransform(0, 3); translationOld.y = deltaTransform(1, 3); translationOld.z = deltaTransform(2, 3);

	m_CUDABuildLinearSystem->applyBL(cameraTrackingInput, intrinsics, cameraTrackingParameters, anglesOld, translationOld, m_imageWidth[level], m_imageHeight[level], level, system, conf);

	Matrix6x6f ATA = system.block(0, 0, 6, 6);
	Vector6f ATb = system.block(0, 6, 6, 1);

	if (ATA.isZero()) {
		return m_matrixTrackingLost;
	}

	Eigen::JacobiSVD<Matrix6x6f> SVD(ATA, Eigen::ComputeFullU | Eigen::ComputeFullV);
	Vector6f x = SVD.solve(ATb);

	//computing the matrix condition
	Vector6f evs = SVD.singularValues();
	conf.matrixCondition = evs[0]/evs[5];

	Vector6f xNew; xNew.block(0, 0, 3, 1) = eulerAngles; xNew.block(3, 0, 3, 1) = deltaTransform.block(0, 3, 3, 1);
	xNew += x;

	deltaTransform = delinearizeTransformation(xNew, Eigen::Vector3f(0.0f, 0.0f, 0.0f), 1.0f, level);
	if(deltaTransform(0, 0) == -std::numeric_limits<float>::infinity())
	{
		conf.trackingLostTresh = true;
		return m_matrixTrackingLost;
	}

	return deltaTransform;
}

mat4f CUDACameraTrackingMultiResRGBD::applyCT(
	float4* dInputPos, float4* dInputNormal, float4* dInputColor,
	float4* dTargetPos,float4* dTargetNormal, float4* dTargetColor,
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
	ICPErrorLog* errorLog)
{		
	// Input
	d_input[0] = dInputPos;
	d_inputNormal[0] = dInputNormal;
	
	d_model[0] = dTargetPos;
	d_modelNormal[0] = dTargetNormal;
	
	//Start Timing
	if(GlobalAppState::get().s_timingsDetailledEnabled) { cutilSafeCall(cudaDeviceSynchronize()); m_timer.start(); }

	convertColorToIntensityFloat(d_inputIntensity[0], dInputColor, m_imageWidth[0], m_imageHeight[0]);
	convertColorToIntensityFloat(d_modelIntensity[0], dTargetColor, m_imageWidth[0], m_imageHeight[0]);
	computeIntensityAndDerivatives(d_modelIntensity[0], m_imageWidth[0], m_imageHeight[0], d_modelIntensityAndDerivatives[0]);
	copyFloatMap(d_inputIntensityFiltered[0], d_inputIntensity[0], m_imageWidth[0], m_imageHeight[0]);

	for (unsigned int i = 0; i < m_levels-1; i++)
	{
		float sigmaD = 3.0f; float sigmaR = 1.0f;

		resampleFloat4Map(d_input[i+1], m_imageWidth[i+1], m_imageHeight[i+1], d_input[i], m_imageWidth[i], m_imageHeight[i]);
		computeNormals(d_inputNormal[i+1], d_input[i+1], m_imageWidth[i+1], m_imageHeight[i+1]);
		resampleFloatMap(d_inputIntensity[i+1], m_imageWidth[i+1], m_imageHeight[i+1], d_inputIntensity[i], m_imageWidth[i], m_imageHeight[i]);
		gaussFilterFloatMap(d_inputIntensityFiltered[i+1], d_inputIntensity[i+1], sigmaD, sigmaR, m_imageWidth[i+1], m_imageHeight[i+1]);

		resampleFloat4Map(d_model[i+1], m_imageWidth[i+1], m_imageHeight[i+1], d_model[i], m_imageWidth[i], m_imageHeight[i]);
		computeNormals(d_modelNormal[i+1], d_model[i+1], m_imageWidth[i+1], m_imageHeight[i+1]);
		resampleFloatMap(d_modelIntensity[i+1], m_imageWidth[i+1], m_imageHeight[i+1], d_modelIntensity[i], m_imageWidth[i], m_imageHeight[i]);
		gaussFilterFloatMap(d_modelIntensityFiltered[i+1], d_modelIntensity[i+1], sigmaD, sigmaR, m_imageWidth[i+1], m_imageHeight[i+1]);

		computeIntensityAndDerivatives(d_modelIntensityFiltered[i+1], m_imageWidth[i+1], m_imageHeight[i+1], d_modelIntensityAndDerivatives[i+1]);
	}

	//DX11QuadDrawer::RenderQuadDynamic(DXUTGetD3D11Device(), DXUTGetD3D11DeviceContext(), (float*)d_modelIntensityAndDerivatives[0], 4, m_imageWidth[0], m_imageHeight[0], 100.0f);

	Eigen::Matrix4f deltaTransform; deltaTransform = MatToEig(deltaTransformEstimate);
	for (int level = m_levels-1; level>=0; level--)	
	{	
		if (errorLog) {
			errorLog->newICPFrame(level);
		}

		float levelFactor = pow(2.0f, (float)level);
		mat4f intrinsicNew = intrinsic;
		intrinsicNew(0, 0) /= levelFactor; intrinsicNew(1, 1) /= levelFactor; intrinsicNew(0, 2) /= levelFactor; intrinsicNew(1, 2) /= levelFactor;

		CameraTrackingInput input;
		input.d_inputPos = d_input[level];
		input.d_inputNormal = d_inputNormal[level];
		input.d_inputIntensity = d_inputIntensityFiltered[level];
		input.d_targetPos = d_model[level];
		input.d_targetNormal = d_modelNormal[level];
		input.d_targetIntensityAndDerivatives = d_modelIntensityAndDerivatives[level];

		CameraTrackingParameters parameters;
		parameters.weightColor = weightsColor[level];
		parameters.weightDepth =  weightsDepth[level];
		parameters.distThres = distThres[level];
		parameters.normalThres = normalThres[level];
		parameters.sensorMaxDepth = GlobalAppState::get().s_sensorDepthMax;
		parameters.colorGradiantMin = colorGradiantMin[level];
		parameters.colorThres = colorThres[level];

		deltaTransform = align(input, deltaTransform, level, parameters,  maxInnerIter[level], maxOuterIter[level], condThres, angleThres, earlyOutResidual[level], intrinsicNew, depthCameraData, errorLog);

		if(deltaTransform(0, 0) == -std::numeric_limits<float>::infinity()) {
			return EigToMat(m_matrixTrackingLost);
		}
	}

	//End Timing
	if(GlobalAppState::get().s_timingsDetailledEnabled) { cutilSafeCall(cudaDeviceSynchronize()); m_timer.stop(); TimingLog::totalTimeTracking += m_timer.getElapsedTimeMS(); TimingLog::countTimeTracking++; }
	
	return lastTransform*EigToMat(deltaTransform);
}

Eigen::Matrix4f CUDACameraTrackingMultiResRGBD::align(CameraTrackingInput cameraTrackingInput, Eigen::Matrix4f& deltaTransform, unsigned int level, CameraTrackingParameters cameraTrackingParameters, unsigned int maxInnerIter, unsigned maxOuterIter, float condThres, float angleThres, float earlyOut, const mat4f& intrinsic, const DepthCameraData& depthCameraData, ICPErrorLog* errorLog)
{
	float lastICPError = -1.0f;
	for(unsigned int i = 0; i<maxOuterIter; i++)
	{
		LinearSystemConfidence currConf;
	
		Eigen::Matrix4f intrinsics4x4 = MatrixConversion::MatToEig(intrinsic);
		Eigen::Matrix3f intrinsics = intrinsics4x4.block(0, 0, 3, 3);

		deltaTransform = computeBestRigidAlignment(cameraTrackingInput, intrinsics, deltaTransform, level, cameraTrackingParameters, maxInnerIter, condThres, angleThres, currConf);

		if (errorLog) {
			errorLog->addCurrentICPIteration(currConf, level);
		}
		 
		if (std::abs(lastICPError - currConf.sumRegError) < earlyOut) {
		//	std::cout << lastICPError << " " <<  currConf.sumRegError << " ICP aboarted because no further convergence... " << i << std::endl;
			break;
		}

		lastICPError = currConf.sumRegError;
	}

	return deltaTransform;
}
