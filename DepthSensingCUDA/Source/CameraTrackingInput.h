#include "cudaUtil.h"

#pragma once

struct CameraTrackingParameters
{
	 float weightDepth;
	 float weightColor;
	 float distThres;
	 float normalThres;

	 float sensorMaxDepth;
	 float colorGradiantMin;
	 float colorThres;
};

struct CameraTrackingInput
{
	float4* d_inputPos;
	float4* d_inputNormal;
	float*  d_inputIntensity;
	
	float4* d_targetPos;
	float4* d_targetNormal;
	float4* d_targetIntensityAndDerivatives;
};