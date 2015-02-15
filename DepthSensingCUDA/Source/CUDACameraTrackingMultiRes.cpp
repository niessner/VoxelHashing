#include "stdafx.h"

#include "CUDACameraTrackingMultiRes.h"
#include "CUDAImageHelper.h"

#include "GlobalAppState.h"

#include <cutil_inline.h>
#include <cutil_math.h>

#include <iostream>
#include <limits>

/////////////////////////////////////////////////////
// Camera Tracking Multi Res
/////////////////////////////////////////////////////

Timer CUDACameraTrackingMultiRes::m_timer;

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

CUDACameraTrackingMultiRes::CUDACameraTrackingMultiRes(unsigned int imageWidth, unsigned int imageHeight, unsigned int levels) {
	m_levels = levels;

	d_correspondence = new float4*[m_levels];
	d_correspondenceNormal  = new float4*[m_levels];

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

		// correspondences
		cutilSafeCall(cudaMalloc(&d_correspondence[i], sizeof(float4)*m_imageWidth[i]*m_imageHeight[i]));
		cutilSafeCall(cudaMalloc(&d_correspondenceNormal[i], sizeof(float4)*m_imageWidth[i]*m_imageHeight[i]));

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

	m_CUDABuildLinearSystem = new CUDABuildLinearSystem(m_imageWidth[0], m_imageHeight[0]);
}

CUDACameraTrackingMultiRes::~CUDACameraTrackingMultiRes() {

	d_input[0] = NULL;
	d_inputNormal[0] = NULL;
	
	d_model[0] = NULL;
	d_modelNormal[0] = NULL;
	
	// correspondence
	if (d_correspondence) {
		for (unsigned int i = 0; i < m_levels; i++) 
			if (d_correspondence[i])	cutilSafeCall(cudaFree(d_correspondence[i]));
		SAFE_DELETE_ARRAY(d_correspondence)
	}
	if (d_correspondenceNormal) {
		for (unsigned int i = 0; i < m_levels; i++) 
			if (d_correspondenceNormal[i])	cutilSafeCall(cudaFree(d_correspondenceNormal[i]));
		SAFE_DELETE_ARRAY(d_correspondenceNormal)
	}

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

bool CUDACameraTrackingMultiRes::checkRigidTransformation(Eigen::Matrix3f& R, Eigen::Vector3f& t, float angleThres, float distThres) {
	Eigen::AngleAxisf aa(R);

	if (aa.angle() > angleThres || t.norm() > distThres) {
		std::cout << "Tracking lost: angle " << (aa.angle()/M_PI)*180.0f << " translation " << t.norm() << std::endl;
		return false;
	}

	return true;
}

Eigen::Matrix4f CUDACameraTrackingMultiRes::delinearizeTransformation(Vector6f& x, Eigen::Vector3f& mean, float meanStDev, unsigned int level) 
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

Eigen::Matrix4f CUDACameraTrackingMultiRes::computeBestRigidAlignment(float4* dInput, float4* dInputNormals, float3& mean, float meanStDev, float nValidCorres, const Eigen::Matrix4f& globalDeltaTransform, unsigned int level, unsigned int maxInnerIter, float condThres, float angleThres, LinearSystemConfidence& conf)
{
	Eigen::Matrix4f deltaTransform = globalDeltaTransform;

	for (unsigned int i = 0; i < maxInnerIter; i++)
	{
		conf.reset();

		Matrix6x7f system;

		m_CUDABuildLinearSystem->applyBL(dInput, d_correspondence[level], d_correspondenceNormal[level], mean, meanStDev, deltaTransform, m_imageWidth[level], m_imageHeight[level], level, system, conf);

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

		Eigen::Matrix4f t = delinearizeTransformation(x, Eigen::Vector3f(mean.x, mean.y, mean.z), meanStDev, level);
		if(t(0, 0) == -std::numeric_limits<float>::infinity())
		{
			conf.trackingLostTresh = true;
			return m_matrixTrackingLost;
		}

		deltaTransform = t*deltaTransform;
	}

	return deltaTransform;
}

mat4f CUDACameraTrackingMultiRes::applyCT(
	float4* dInput, float4* dInputNormals, float4* dInputColors, 
	float4* dModel, float4* dModelNormals, float4* dModelColors, 
	const mat4f& lastTransform, const std::vector<unsigned int>& maxInnerIter, const std::vector<unsigned int>& maxOuterIter, 
	const std::vector<float>& distThres, const std::vector<float>& normalThres, float condThres, float angleThres, 
	const mat4f& deltaTransformEstimate, const std::vector<float>& earlyOutResidual, 
	const mat4f& intrinsic, const DepthCameraData& depthCameraData,
	ICPErrorLog* errorLog)
{		
	// Input
	d_input[0] = dInput;
	d_inputNormal[0] = dInputNormals;

	d_model[0] = dModel;
	d_modelNormal[0] = dModelNormals;

	//Start Timing
	if(GlobalAppState::get().s_timingsDetailledEnabled) { cutilSafeCall(cudaDeviceSynchronize()); m_timer.start(); }

	for (unsigned int i = 0; i < m_levels-1; i++)
	{
		resampleFloat4Map(d_input[i+1], m_imageWidth[i+1], m_imageHeight[i+1], d_input[i], m_imageWidth[i], m_imageHeight[i]);
		computeNormals(d_inputNormal[i+1], d_input[i+1], m_imageWidth[i+1], m_imageHeight[i+1]);
	
		resampleFloat4Map(d_model[i+1], m_imageWidth[i+1], m_imageHeight[i+1], d_model[i], m_imageWidth[i], m_imageHeight[i]);
		computeNormals(d_modelNormal[i+1], d_model[i+1], m_imageWidth[i+1], m_imageHeight[i+1]);
	}

	Eigen::Matrix4f deltaTransform; deltaTransform = MatToEig(deltaTransformEstimate);
	for (int level = m_levels-1; level>=0; level--)
	{	
		if (errorLog) {
			errorLog->newICPFrame(level);
		}

		deltaTransform = align(d_input[level], d_inputNormal[level], d_model[level], d_modelNormal[level], deltaTransform, level, maxInnerIter[level], maxOuterIter[level], distThres[level], normalThres[level], condThres, angleThres, earlyOutResidual[level], intrinsic, depthCameraData, errorLog);

		if(deltaTransform(0, 0) == -std::numeric_limits<float>::infinity()) {
			return EigToMat(m_matrixTrackingLost);
		}
	}

	//End Timing
	if(GlobalAppState::get().s_timingsDetailledEnabled) { cutilSafeCall(cudaDeviceSynchronize()); m_timer.stop(); TimingLog::totalTimeTracking += m_timer.getElapsedTimeMS(); TimingLog::countTimeTracking++; }
	
	return lastTransform*EigToMat(deltaTransform);
}

Eigen::Matrix4f CUDACameraTrackingMultiRes::align(float4* dInput, float4* dInputNormals, float4* dModel, float4* dModelNormals, Eigen::Matrix4f& deltaTransform, unsigned int level, unsigned int maxInnerIter, unsigned maxOuterIter, float distThres, float normalThres, float condThres, float angleThres, float earlyOut, const mat4f& intrinsic, const DepthCameraData& depthCameraData, ICPErrorLog* errorLog)
{
	float lastICPError = -1.0f;
	for(unsigned int i = 0; i<maxOuterIter; i++)
	{
		float3 mean;
		float meanStDev;
		float nValidCorres;

		LinearSystemConfidence currConfWiReject;
		LinearSystemConfidence currConfNoReject;

		if (errorLog) {
			//run ICP without correspondence rejection (must be run before because it needs the old delta transform)
			float dThresh = 1000.0f;	float nThresh = 0.0f;
			computeCorrespondences(dInput, dInputNormals, dModel, dModelNormals, mean, meanStDev, nValidCorres, deltaTransform, level, dThresh, nThresh, intrinsic, depthCameraData);
			
			computeBestRigidAlignment(dInput, dInputNormals, mean, meanStDev, nValidCorres, deltaTransform, level, maxInnerIter, condThres, angleThres, currConfNoReject);
			errorLog->addCurrentICPIteration(currConfNoReject, level);
		}

		//standard correspondence search and alignment
		computeCorrespondences(dInput, dInputNormals, dModel, dModelNormals, mean, meanStDev, nValidCorres, deltaTransform, level, distThres, normalThres, intrinsic, depthCameraData);
		

		deltaTransform = computeBestRigidAlignment(dInput, dInputNormals, mean, meanStDev, nValidCorres, deltaTransform, level, maxInnerIter, condThres, angleThres, currConfWiReject);
		
		if (std::abs(lastICPError - currConfWiReject.sumRegError) < earlyOut) {
			//std::cout << lastICPError << " " <<  currConfWiReject.sumRegError << " ICP aboarted because no further convergence... " << i << std::endl;
			break;
		}
		lastICPError = currConfWiReject.sumRegError;
	}

	return deltaTransform;
}

void CUDACameraTrackingMultiRes::computeCorrespondences(float4* dInput, float4* dInputNormals, float4* dModel, float4* dModelNormals, float3& mean, float& meanStDev, float& nValidCorres, const Eigen::Matrix4f& deltaTransform, unsigned int level, float distThres, float normalThres, const mat4f& intrinsic, const DepthCameraData& depthCameraData)
{
	float levelFactor = pow(2.0f, (float)level);
	float4x4 deltaTransformCUDA = MatrixConversion::toCUDA(deltaTransform);
	mean = make_float3(0.0f, 0.0f, 0.0f);
	meanStDev = 1.0f;
	CUDAImageHelper::applyProjectiveCorrespondences(
		dInput, dInputNormals, NULL, 
		dModel, dModelNormals, NULL, 
		d_correspondence[level], d_correspondenceNormal[level], deltaTransformCUDA, m_imageWidth[level], m_imageHeight[level], distThres, normalThres, levelFactor, intrinsic, depthCameraData
		);
}
