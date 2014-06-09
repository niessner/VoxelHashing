#include "stdafx.h"

#include "DX11CameraTrackingMultiRes.h"


#include "GlobalAppState.h"

#include <iostream>
#include <limits>

/////////////////////////////////////////////////////
// Camera Tracking Multi Res
/////////////////////////////////////////////////////

Eigen::Matrix4f DX11CameraTrackingMultiRes::m_matrixTrackingLost;

// Correspondences
ID3D11Texture2D** DX11CameraTrackingMultiRes::m_pCorrespondenceTextureFloat42D = NULL;
ID3D11ShaderResourceView** DX11CameraTrackingMultiRes::m_pCorrespondenceTextureFloat4SRV = NULL;
ID3D11UnorderedAccessView** DX11CameraTrackingMultiRes::m_pCorrespondenceTextureFloat4UAV = NULL;

ID3D11Texture2D** DX11CameraTrackingMultiRes::m_pCorrespondenceNormalTextureFloat42D = NULL;
ID3D11ShaderResourceView** DX11CameraTrackingMultiRes::m_pCorrespondenceNormalTextureFloat4SRV = NULL;
ID3D11UnorderedAccessView** DX11CameraTrackingMultiRes::m_pCorrespondenceNormalTextureFloat4UAV = NULL;

// Input
ID3D11Texture2D** DX11CameraTrackingMultiRes::m_inputTextureFloat42D = NULL;
ID3D11ShaderResourceView** DX11CameraTrackingMultiRes::m_inputTextureFloat4SRV = NULL;
ID3D11UnorderedAccessView** DX11CameraTrackingMultiRes::m_inputTextureFloat4UAV = NULL;

ID3D11Texture2D** DX11CameraTrackingMultiRes::m_inputNormalTextureFloat42D = NULL;
ID3D11ShaderResourceView** DX11CameraTrackingMultiRes::m_inputNormalTextureFloat4SRV = NULL;
ID3D11UnorderedAccessView** DX11CameraTrackingMultiRes::m_inputNormalTextureFloat4UAV = NULL;

ID3D11Texture2D** DX11CameraTrackingMultiRes::m_inputColorTextureFloat42D = NULL;
ID3D11ShaderResourceView** DX11CameraTrackingMultiRes::m_inputColorTextureFloat4SRV = NULL;
ID3D11UnorderedAccessView** DX11CameraTrackingMultiRes::m_inputColorTextureFloat4UAV = NULL;

// Model
ID3D11Texture2D** DX11CameraTrackingMultiRes::m_modelTextureFloat42D = NULL;
ID3D11ShaderResourceView** DX11CameraTrackingMultiRes::m_modelTextureFloat4SRV = NULL;
ID3D11UnorderedAccessView** DX11CameraTrackingMultiRes::m_modelTextureFloat4UAV = NULL;

ID3D11Texture2D** DX11CameraTrackingMultiRes::m_modelNormalTextureFloat42D = NULL;
ID3D11ShaderResourceView** DX11CameraTrackingMultiRes::m_modelNormalTextureFloat4SRV = NULL;
ID3D11UnorderedAccessView** DX11CameraTrackingMultiRes::m_modelNormalTextureFloat4UAV = NULL;

ID3D11Texture2D** DX11CameraTrackingMultiRes::m_modelColorTextureFloat42D = NULL;
ID3D11ShaderResourceView** DX11CameraTrackingMultiRes::m_modelColorTextureFloat4SRV = NULL;
ID3D11UnorderedAccessView** DX11CameraTrackingMultiRes::m_modelColorTextureFloat4UAV = NULL;

// Image Pyramid Dimensions
unsigned int* DX11CameraTrackingMultiRes::m_imageWidth = NULL;
unsigned int* DX11CameraTrackingMultiRes::m_imageHeight = NULL;

/////////////////////////////////////////////////////
// Timer
/////////////////////////////////////////////////////

Timer DX11CameraTrackingMultiRes::m_timer;


HRESULT DX11CameraTrackingMultiRes::OnD3D11CreateDevice(ID3D11Device* pd3dDevice)
{
	HRESULT hr = S_OK;

	// Correspondences
	m_pCorrespondenceTextureFloat42D = new ID3D11Texture2D*[GlobalCameraTrackingState::getInstance().s_maxLevels];
	m_pCorrespondenceTextureFloat4SRV = new ID3D11ShaderResourceView*[GlobalCameraTrackingState::getInstance().s_maxLevels];
	m_pCorrespondenceTextureFloat4UAV = new ID3D11UnorderedAccessView*[GlobalCameraTrackingState::getInstance().s_maxLevels];

	m_pCorrespondenceNormalTextureFloat42D = new ID3D11Texture2D*[GlobalCameraTrackingState::getInstance().s_maxLevels];
	m_pCorrespondenceNormalTextureFloat4SRV = new ID3D11ShaderResourceView*[GlobalCameraTrackingState::getInstance().s_maxLevels];
	m_pCorrespondenceNormalTextureFloat4UAV = new ID3D11UnorderedAccessView*[GlobalCameraTrackingState::getInstance().s_maxLevels];

	// Input
	m_inputTextureFloat42D = new ID3D11Texture2D*[GlobalCameraTrackingState::getInstance().s_maxLevels];
	m_inputTextureFloat4SRV = new ID3D11ShaderResourceView*[GlobalCameraTrackingState::getInstance().s_maxLevels];
	m_inputTextureFloat4UAV = new ID3D11UnorderedAccessView*[GlobalCameraTrackingState::getInstance().s_maxLevels];

	m_inputNormalTextureFloat42D = new ID3D11Texture2D*[GlobalCameraTrackingState::getInstance().s_maxLevels];
	m_inputNormalTextureFloat4SRV = new ID3D11ShaderResourceView*[GlobalCameraTrackingState::getInstance().s_maxLevels];
	m_inputNormalTextureFloat4UAV = new ID3D11UnorderedAccessView*[GlobalCameraTrackingState::getInstance().s_maxLevels];

	m_inputColorTextureFloat42D = new ID3D11Texture2D*[GlobalCameraTrackingState::getInstance().s_maxLevels];
	m_inputColorTextureFloat4SRV = new ID3D11ShaderResourceView*[GlobalCameraTrackingState::getInstance().s_maxLevels];
	m_inputColorTextureFloat4UAV = new ID3D11UnorderedAccessView*[GlobalCameraTrackingState::getInstance().s_maxLevels];

	// Model
	m_modelTextureFloat42D = new ID3D11Texture2D*[GlobalCameraTrackingState::getInstance().s_maxLevels];
	m_modelTextureFloat4SRV = new ID3D11ShaderResourceView*[GlobalCameraTrackingState::getInstance().s_maxLevels];
	m_modelTextureFloat4UAV = new ID3D11UnorderedAccessView*[GlobalCameraTrackingState::getInstance().s_maxLevels];

	m_modelNormalTextureFloat42D = new ID3D11Texture2D*[GlobalCameraTrackingState::getInstance().s_maxLevels];
	m_modelNormalTextureFloat4SRV = new ID3D11ShaderResourceView*[GlobalCameraTrackingState::getInstance().s_maxLevels];
	m_modelNormalTextureFloat4UAV = new ID3D11UnorderedAccessView*[GlobalCameraTrackingState::getInstance().s_maxLevels];

	m_modelColorTextureFloat42D = new ID3D11Texture2D*[GlobalCameraTrackingState::getInstance().s_maxLevels];
	m_modelColorTextureFloat4SRV = new ID3D11ShaderResourceView*[GlobalCameraTrackingState::getInstance().s_maxLevels];
	m_modelColorTextureFloat4UAV = new ID3D11UnorderedAccessView*[GlobalCameraTrackingState::getInstance().s_maxLevels];

	// Image Pyramid Dimensions
	m_imageWidth = new unsigned int[GlobalCameraTrackingState::getInstance().s_maxLevels];
	m_imageHeight = new unsigned int[GlobalCameraTrackingState::getInstance().s_maxLevels];
	

	unsigned int fac = 1;
	for (unsigned int i = 0; i<GlobalCameraTrackingState::getInstance().s_maxLevels; i++) {
		m_imageWidth[i] = GlobalAppState::getInstance().s_windowWidth/fac;
		m_imageHeight[i] = GlobalAppState::getInstance().s_windowHeight/fac;

		// Create depth texture
		D3D11_TEXTURE2D_DESC depthTexDesc = {0};
		depthTexDesc.Width = m_imageWidth[i];
		depthTexDesc.Height = m_imageHeight[i];
		depthTexDesc.MipLevels = 1;
		depthTexDesc.ArraySize = 1;
		depthTexDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
		depthTexDesc.SampleDesc.Count = 1;
		depthTexDesc.SampleDesc.Quality = 0;
		depthTexDesc.Usage = D3D11_USAGE_DEFAULT;
		depthTexDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
		depthTexDesc.CPUAccessFlags = 0;
		depthTexDesc.MiscFlags = 0;

		////////////////////////////////////////////////
		// Correspondences
		////////////////////////////////////////////////

		V_RETURN(pd3dDevice->CreateTexture2D(&depthTexDesc, NULL, &m_pCorrespondenceTextureFloat42D[i]));
		V_RETURN(pd3dDevice->CreateTexture2D(&depthTexDesc, NULL, &m_pCorrespondenceNormalTextureFloat42D[i]));

		//create shader resource views
		V_RETURN(pd3dDevice->CreateShaderResourceView(m_pCorrespondenceTextureFloat42D[i], NULL, &m_pCorrespondenceTextureFloat4SRV[i]));
		V_RETURN(pd3dDevice->CreateShaderResourceView(m_pCorrespondenceNormalTextureFloat42D[i], NULL, &m_pCorrespondenceNormalTextureFloat4SRV[i]));

		//create unordered access views
		V_RETURN(pd3dDevice->CreateUnorderedAccessView(m_pCorrespondenceTextureFloat42D[i], NULL, &m_pCorrespondenceTextureFloat4UAV[i]));
		V_RETURN(pd3dDevice->CreateUnorderedAccessView(m_pCorrespondenceNormalTextureFloat42D[i], NULL, &m_pCorrespondenceNormalTextureFloat4UAV[i]));

		////////////////////////////////////////////////
		// Input
		////////////////////////////////////////////////
		if (i != 0) {  // Not finest level
			V_RETURN(pd3dDevice->CreateTexture2D(&depthTexDesc, NULL, &m_inputTextureFloat42D[i]));
			V_RETURN(pd3dDevice->CreateTexture2D(&depthTexDesc, NULL, &m_inputNormalTextureFloat42D[i]));
			V_RETURN(pd3dDevice->CreateTexture2D(&depthTexDesc, NULL, &m_inputColorTextureFloat42D[i]));

			//create shader resource views
			V_RETURN(pd3dDevice->CreateShaderResourceView(m_inputTextureFloat42D[i], NULL, &m_inputTextureFloat4SRV[i]));
			V_RETURN(pd3dDevice->CreateShaderResourceView(m_inputNormalTextureFloat42D[i], NULL, &m_inputNormalTextureFloat4SRV[i]));
			V_RETURN(pd3dDevice->CreateShaderResourceView(m_inputColorTextureFloat42D[i], NULL, &m_inputColorTextureFloat4SRV[i]));

			//create unordered access views
			V_RETURN(pd3dDevice->CreateUnorderedAccessView(m_inputTextureFloat42D[i], NULL, &m_inputTextureFloat4UAV[i]));
			V_RETURN(pd3dDevice->CreateUnorderedAccessView(m_inputNormalTextureFloat42D[i], NULL, &m_inputNormalTextureFloat4UAV[i]));
			V_RETURN(pd3dDevice->CreateUnorderedAccessView(m_inputColorTextureFloat42D[i], NULL, &m_inputColorTextureFloat4UAV[i]));
		} else {
			m_inputTextureFloat42D[i] = NULL;
			m_inputNormalTextureFloat42D[i] = NULL;
			m_inputColorTextureFloat42D[i] = NULL;

			m_inputTextureFloat4SRV[i] = NULL;
			m_inputNormalTextureFloat4SRV[i] = NULL;
			m_inputColorTextureFloat4SRV[i] = NULL;

			m_inputTextureFloat4UAV[i] = NULL;
			m_inputNormalTextureFloat4UAV[i] = NULL;
			m_inputColorTextureFloat4UAV[i] = NULL;
		}

		////////////////////////////////////////////////
		// Model
		////////////////////////////////////////////////
		if (i != 0) { // Not fines level
			V_RETURN(pd3dDevice->CreateTexture2D(&depthTexDesc, NULL, &m_modelTextureFloat42D[i]));
			V_RETURN(pd3dDevice->CreateTexture2D(&depthTexDesc, NULL, &m_modelNormalTextureFloat42D[i]));
			V_RETURN(pd3dDevice->CreateTexture2D(&depthTexDesc, NULL, &m_modelColorTextureFloat42D[i]));

			//create shader resource views
			V_RETURN(pd3dDevice->CreateShaderResourceView(m_modelTextureFloat42D[i], NULL, &m_modelTextureFloat4SRV[i]));
			V_RETURN(pd3dDevice->CreateShaderResourceView(m_modelNormalTextureFloat42D[i], NULL, &m_modelNormalTextureFloat4SRV[i]));
			V_RETURN(pd3dDevice->CreateShaderResourceView(m_modelColorTextureFloat42D[i], NULL, &m_modelColorTextureFloat4SRV[i]));

			//create unordered access views
			V_RETURN(pd3dDevice->CreateUnorderedAccessView(m_modelTextureFloat42D[i], NULL, &m_modelTextureFloat4UAV[i]));
			V_RETURN(pd3dDevice->CreateUnorderedAccessView(m_modelNormalTextureFloat42D[i], NULL, &m_modelNormalTextureFloat4UAV[i]));
			V_RETURN(pd3dDevice->CreateUnorderedAccessView(m_modelColorTextureFloat42D[i], NULL, &m_modelColorTextureFloat4UAV[i]));
		} else {
			m_modelTextureFloat42D[i] = NULL;
			m_modelNormalTextureFloat42D[i] = NULL;
			m_modelColorTextureFloat42D[i] = NULL;

			m_modelTextureFloat4SRV[i] = NULL;
			m_modelNormalTextureFloat4SRV[i] = NULL;
			m_modelColorTextureFloat4SRV[i] = NULL;

			m_modelTextureFloat4UAV[i] = NULL;
			m_modelNormalTextureFloat4UAV[i] = NULL;
			m_modelColorTextureFloat4UAV[i] = NULL;

		}

		fac*=2;
	}

	m_matrixTrackingLost.fill(-std::numeric_limits<float>::infinity());


	return  hr;
}

bool DX11CameraTrackingMultiRes::checkRigidTransformation(Eigen::Matrix3f& R, Eigen::Vector3f& t, float angleThres, float distThres)
{
	Eigen::AngleAxisf aa(R);

	if(aa.angle() > angleThres || t.norm() > distThres)
	{
		std::cout << "Tracking lost: angle " << (aa.angle()/M_PI)*180.0f << " translation " << t.norm() << std::endl;
		return false;
	}

	//std::cout << "Tracking successful: anlge " << (aa.angle()/M_PI)*180.0f << " translation " << t.norm() << std::endl;
	return true;
}

Eigen::Matrix4f DX11CameraTrackingMultiRes::delinearizeTransformation(Vector6f& x, Eigen::Vector3f& mean, float meanStDev, unsigned int level)
{
	Eigen::Matrix3f R =	 Eigen::AngleAxisf(x[0], Eigen::Vector3f::UnitZ()).toRotationMatrix()  // Rot Z
		*Eigen::AngleAxisf(x[1], Eigen::Vector3f::UnitY()).toRotationMatrix()  // Rot Y
		*Eigen::AngleAxisf(x[2], Eigen::Vector3f::UnitX()).toRotationMatrix(); // Rot X

	Eigen::Vector3f t = x.segment(3, 3);

	if(!checkRigidTransformation(R, t, GlobalCameraTrackingState::getInstance().s_angleTransThres[level], GlobalCameraTrackingState::getInstance().s_distTransThres[level]))
	{
		return m_matrixTrackingLost;
	}

	Eigen::Matrix4f res; res.setIdentity();
	res.block(0, 0, 3, 3) = R;
	res.block(0, 3, 3, 1) = meanStDev*t+mean-R*mean;

	return res;
}

Eigen::Matrix4f DX11CameraTrackingMultiRes::computeBestRigidAlignment(ID3D11DeviceContext* context, ID3D11ShaderResourceView* inputSRV, ID3D11ShaderResourceView* inputNormalsSRV, D3DXVECTOR3& mean, float meanStDev, float nValidCorres, const Eigen::Matrix4f& globalDeltaTransform, unsigned int level, unsigned int maxInnerIter, float condThres, float angleThres, LinearSystemConfidence& conf)
{
	Eigen::Matrix4f deltaTransform = globalDeltaTransform;

	for(unsigned int i = 0; i<maxInnerIter; i++)
	{
		conf.reset();

		Matrix6x7f system;

		DX11BuildLinearSystem::applyBL(context, inputSRV, m_pCorrespondenceTextureFloat4SRV[level], m_pCorrespondenceNormalTextureFloat4SRV[level], mean, meanStDev, deltaTransform, m_imageWidth[level], m_imageHeight[level], level, system, conf);

		Matrix6x6f ATA = system.block(0, 0, 6, 6);
		Vector6f ATb = system.block(0, 6, 6, 1);

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

		//Eigen::Matrix3f R = deltaTransform.block(0, 0, 3, 3);
		//Eigen::Vector3f v = deltaTransform.block(0, 3, 3, 1);
	}
	return deltaTransform;
}


mat4f DX11CameraTrackingMultiRes::applyCT(ID3D11DeviceContext* context, ID3D11ShaderResourceView* inputSRV, ID3D11ShaderResourceView* inputNormalsSRV, ID3D11ShaderResourceView* inputColorsSRV, ID3D11ShaderResourceView* modelSRV, ID3D11ShaderResourceView* modelNormalsSRV, ID3D11ShaderResourceView* modelColorsSRV, const mat4f& lastTransform, const std::vector<unsigned int>& maxInnerIter, const std::vector<unsigned int>& maxOuterIter, const std::vector<float>& distThres, const std::vector<float>& normalThres, float condThres, float angleThres, const mat4f& deltaTransformEstimate, const std::vector<float>& earlyOutResidual, ICPErrorLog* errorLog)
{		
	// Input
	m_inputTextureFloat4SRV[0] = inputSRV;
	m_inputNormalTextureFloat4SRV[0] = inputNormalsSRV;
	m_inputColorTextureFloat4SRV[0] = inputColorsSRV;

	m_modelTextureFloat4SRV[0] = modelSRV;
	m_modelNormalTextureFloat4SRV[0] = modelNormalsSRV;
	m_modelColorTextureFloat4SRV[0] = modelColorsSRV;

	// Start query for timing
	if(GlobalAppState::getInstance().s_timingsDetailledEnabled)
	{
		GlobalAppState::getInstance().WaitForGPU();
		m_timer.start();
	}

	for(unsigned int i = 0; i<GlobalCameraTrackingState::getInstance().s_maxLevels-1; i++)
	{
		// Downsample Depth Maps directly ? -> better ?
		DX11ImageHelper::applyDownsampling(context, m_inputTextureFloat4SRV[i], m_inputTextureFloat4UAV[i+1], m_imageWidth[i+1], m_imageHeight[i+1]);
		DX11ImageHelper::applyDownsampling(context, m_modelTextureFloat4SRV[i], m_modelTextureFloat4UAV[i+1], m_imageWidth[i+1], m_imageHeight[i+1]);

		DX11ImageHelper::applyNormalComputation(context, m_inputTextureFloat4SRV[i+1], m_inputNormalTextureFloat4UAV[i+1], m_imageWidth[i+1], m_imageHeight[i+1]);
		DX11ImageHelper::applyNormalComputation(context, m_modelTextureFloat4SRV[i+1], m_modelNormalTextureFloat4UAV[i+1], m_imageWidth[i+1], m_imageHeight[i+1]);

		DX11ImageHelper::applyDownsampling(context, m_inputColorTextureFloat4SRV[i], m_inputColorTextureFloat4UAV[i+1], m_imageWidth[i+1], m_imageHeight[i+1]);
		DX11ImageHelper::applyDownsampling(context, m_modelColorTextureFloat4SRV[i], m_modelColorTextureFloat4UAV[i+1], m_imageWidth[i+1], m_imageHeight[i+1]);
	}

	Eigen::Matrix4f deltaTransform; 
	//deltaTransform.setIdentity();
	deltaTransform = MatToEig(deltaTransformEstimate);
	for(int level = GlobalCameraTrackingState::getInstance().s_maxLevels-1; level>=0; level--)
	{	
		if (errorLog) {
			errorLog->newICPFrame(level);
		}

		deltaTransform = align(context, m_inputTextureFloat4SRV[level], m_inputNormalTextureFloat4SRV[level], m_inputColorTextureFloat4SRV[level], m_modelTextureFloat4SRV[level], m_modelNormalTextureFloat4SRV[level], m_modelColorTextureFloat4SRV[level], deltaTransform, level, maxInnerIter[level], maxOuterIter[level], distThres[level], normalThres[level], condThres, angleThres, earlyOutResidual[level], errorLog);

		if(deltaTransform(0, 0) == -std::numeric_limits<float>::infinity()) {
			return EigToMat(m_matrixTrackingLost);
		}
	}

	// Wait for query
	if(GlobalAppState::getInstance().s_timingsDetailledEnabled)
	{
		GlobalAppState::getInstance().WaitForGPU();
		TimingLog::totalTimeTrackCamera += m_timer.getElapsedTimeMS(); TimingLog::countTrackCamera++;
	}

	return lastTransform*EigToMat(deltaTransform);
}


Eigen::Matrix4f DX11CameraTrackingMultiRes::align(ID3D11DeviceContext* context, ID3D11ShaderResourceView* inputSRV, ID3D11ShaderResourceView* inputNormalsSRV, ID3D11ShaderResourceView* inputColorsSRV, 
												  ID3D11ShaderResourceView* modelSRV, ID3D11ShaderResourceView* modelNormalsSRV, ID3D11ShaderResourceView* modelColorsSRV, Eigen::Matrix4f& deltaTransform, 
												  unsigned int level, unsigned int maxInnerIter, unsigned maxOuterIter, float distThres, float normalThres, float condThres, float angleThres, float earlyOut, ICPErrorLog* errorLog)
{
	float lastICPError = -1.0f;
	for(unsigned int i = 0; i<maxOuterIter; i++)
	{
		D3DXVECTOR3 mean;
		float meanStDev;
		float nValidCorres;

		LinearSystemConfidence currConfWiReject;
		LinearSystemConfidence currConfNoReject;

		if (errorLog) {
			//run ICP without correspondence rejection (must be run before because it needs the old delta transform)
			float dThresh = 1000.0f;	float nThresh = 0.0f;
			computeCorrespondences(context, inputSRV, inputNormalsSRV, inputColorsSRV, modelSRV, modelNormalsSRV, modelColorsSRV, mean, meanStDev, nValidCorres, deltaTransform, level, dThresh, nThresh);
			computeBestRigidAlignment(context, inputSRV, inputNormalsSRV, mean, meanStDev, nValidCorres, deltaTransform, level, maxInnerIter, condThres, angleThres, currConfNoReject);
			errorLog->addCurrentICPIteration(currConfNoReject, level);
		}

		//standard correspondence search and alignment
		computeCorrespondences(context, inputSRV, inputNormalsSRV, inputColorsSRV, modelSRV, modelNormalsSRV, modelColorsSRV, mean, meanStDev, nValidCorres, deltaTransform, level, distThres, normalThres);
		deltaTransform = computeBestRigidAlignment(context, inputSRV, inputNormalsSRV, mean, meanStDev, nValidCorres, deltaTransform, level, maxInnerIter, condThres, angleThres, currConfWiReject);
		if (std::abs(lastICPError - currConfWiReject.sumRegError) < earlyOut) {
			//std::cout << "ICP aboarted because no further convergence... " << i << std::endl;
			break;
		}
		lastICPError = currConfWiReject.sumRegError;

		//std::cout << currConfWiReject.numCorr << std::endl;
		//std::cout << "i " << i << std::endl;
		//currConf.print();

		//if(level == 0) std::cout << deltaTransform << std::endl;
		//deltaTransform.setIdentity();

		/*if(deltaTransform(0, 0) == -std::numeric_limits<float>::infinity())
		{
		return m_matrixTrackingLost;
		}*/
	}

	return deltaTransform;
}

HRESULT DX11CameraTrackingMultiRes::computeCorrespondences(ID3D11DeviceContext* context, ID3D11ShaderResourceView* inputSRV, ID3D11ShaderResourceView* inputNormalsSRV, ID3D11ShaderResourceView* inputColorsSRV, 
	ID3D11ShaderResourceView* modelSRV, ID3D11ShaderResourceView* modelNormalsSRV, ID3D11ShaderResourceView* modelColorsSRV, 
	D3DXVECTOR3& mean, float& meanStDev, float& nValidCorres, const Eigen::Matrix4f& deltaTransform, unsigned int level, float distThres, float normalThres)
{
	float levelFactor = pow(2.0f, (float)level);

	HRESULT hr;
	hr = DX11ImageHelper::applyProjectiveCorrespondences(context, inputSRV, inputNormalsSRV, inputColorsSRV, modelSRV, modelNormalsSRV, modelColorsSRV, m_pCorrespondenceTextureFloat4UAV[level], m_pCorrespondenceNormalTextureFloat4UAV[level], deltaTransform, m_imageWidth[level], m_imageHeight[level], distThres, normalThres, levelFactor);
	hr = DX11NormalizeReduction::applyNorm(context, m_pCorrespondenceTextureFloat4SRV[level], level,  m_imageWidth[level], m_imageHeight[level], mean, meanStDev, nValidCorres);

	return hr;
}

void DX11CameraTrackingMultiRes::OnD3D11DestroyDevice()
{
	// Do not free input buffers of last applyCT call
	m_inputTextureFloat4SRV[0] = NULL;
	m_inputNormalTextureFloat4SRV[0] = NULL;
	m_inputColorTextureFloat4SRV[0] = NULL;

	m_modelTextureFloat4SRV[0] = NULL;
	m_modelNormalTextureFloat4SRV[0] = NULL;
	m_modelColorTextureFloat4SRV[0] = NULL;

	/////////////////////////////////////////////////////
	// Camera Tracking Multi Res
	/////////////////////////////////////////////////////

	for(unsigned int i = 0; i<GlobalCameraTrackingState::getInstance().s_maxLevels; i++)
	{
		SAFE_RELEASE(m_pCorrespondenceTextureFloat42D[i]);
		SAFE_RELEASE(m_pCorrespondenceTextureFloat4SRV[i]);
		SAFE_RELEASE(m_pCorrespondenceTextureFloat4UAV[i]);

		SAFE_RELEASE(m_pCorrespondenceNormalTextureFloat42D[i]);
		SAFE_RELEASE(m_pCorrespondenceNormalTextureFloat4SRV[i]);
		SAFE_RELEASE(m_pCorrespondenceNormalTextureFloat4UAV[i]);

		SAFE_RELEASE(m_inputTextureFloat42D[i]);
		SAFE_RELEASE(m_inputTextureFloat4SRV[i]);
		SAFE_RELEASE(m_inputTextureFloat4UAV[i]);

		SAFE_RELEASE(m_inputNormalTextureFloat42D[i]);
		SAFE_RELEASE(m_inputNormalTextureFloat4SRV[i]);
		SAFE_RELEASE(m_inputNormalTextureFloat4UAV[i]);

		SAFE_RELEASE(m_inputColorTextureFloat42D[i]);
		SAFE_RELEASE(m_inputColorTextureFloat4SRV[i]);
		SAFE_RELEASE(m_inputColorTextureFloat4UAV[i]);

		SAFE_RELEASE(m_modelColorTextureFloat42D[i]);
		SAFE_RELEASE(m_modelColorTextureFloat4SRV[i]);
		SAFE_RELEASE(m_modelColorTextureFloat4UAV[i]);

		SAFE_RELEASE(m_modelTextureFloat42D[i]);
		SAFE_RELEASE(m_modelTextureFloat4SRV[i]);
		SAFE_RELEASE(m_modelTextureFloat4UAV[i]);

		SAFE_RELEASE(m_modelNormalTextureFloat42D[i]);
		SAFE_RELEASE(m_modelNormalTextureFloat4SRV[i]);
		SAFE_RELEASE(m_modelNormalTextureFloat4UAV[i]);
	}


	SAFE_DELETE_ARRAY(m_pCorrespondenceTextureFloat42D);
	SAFE_DELETE_ARRAY(m_pCorrespondenceTextureFloat4SRV);
	SAFE_DELETE_ARRAY(m_pCorrespondenceTextureFloat4UAV);

	SAFE_DELETE_ARRAY(m_pCorrespondenceNormalTextureFloat42D);
	SAFE_DELETE_ARRAY(m_pCorrespondenceNormalTextureFloat4SRV);
	SAFE_DELETE_ARRAY(m_pCorrespondenceNormalTextureFloat4UAV);

	SAFE_DELETE_ARRAY(m_inputTextureFloat42D);
	SAFE_DELETE_ARRAY(m_inputTextureFloat4SRV);
	SAFE_DELETE_ARRAY(m_inputTextureFloat4UAV);

	SAFE_DELETE_ARRAY(m_inputNormalTextureFloat42D);
	SAFE_DELETE_ARRAY(m_inputNormalTextureFloat4SRV);
	SAFE_DELETE_ARRAY(m_inputNormalTextureFloat4UAV);

	SAFE_DELETE_ARRAY(m_inputColorTextureFloat42D);
	SAFE_DELETE_ARRAY(m_inputColorTextureFloat4SRV);
	SAFE_DELETE_ARRAY(m_inputColorTextureFloat4UAV);

	SAFE_DELETE_ARRAY(m_modelColorTextureFloat42D);
	SAFE_DELETE_ARRAY(m_modelColorTextureFloat4SRV);
	SAFE_DELETE_ARRAY(m_modelColorTextureFloat4UAV);

	SAFE_DELETE_ARRAY(m_modelTextureFloat42D);
	SAFE_DELETE_ARRAY(m_modelTextureFloat4SRV);
	SAFE_DELETE_ARRAY(m_modelTextureFloat4UAV);

	SAFE_DELETE_ARRAY(m_modelNormalTextureFloat42D);
	SAFE_DELETE_ARRAY(m_modelNormalTextureFloat4SRV);
	SAFE_DELETE_ARRAY(m_modelNormalTextureFloat4UAV);


	SAFE_DELETE_ARRAY(m_imageHeight);
	SAFE_DELETE_ARRAY(m_imageWidth);
}