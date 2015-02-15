#pragma once

/************************************************************************/
/* Linear System Build on the GPU for ICP                               */
/************************************************************************/

#include "stdafx.h"

#include "Eigen.h"
#include "ICPErrorLog.h"

#include <cutil_inline.h>
#include <cutil_math.h>

#include "CameraTrackingInput.h"


class CUDABuildLinearSystem
{
	public:
		CUDABuildLinearSystem(unsigned int imageWidth, unsigned int imageHeight);
		~CUDABuildLinearSystem();
				
		void applyBL(float4* input, float4* target, float4* targetNormals, float3& mean, float meanStDev, Eigen::Matrix4f& deltaTransform, unsigned int imageWidth, unsigned int imageHeight, unsigned int level, Matrix6x7f& res, LinearSystemConfidence& conf);
	
		//! builds AtA, AtB, and confidences
		Matrix6x7f reductionSystemCPU(const float* data, unsigned int nElems, LinearSystemConfidence& conf);
				
	private:

		float* d_output;
		float* h_output;
};
