#pragma once

/************************************************************************/
/* Linear System Build on the GPU for ICP                               */
/************************************************************************/

#include "stdafx.h"

#include "Eigen.h"
#include "ICPErrorLog.h"

#include <D3D11.h>
#include "DX11Utils.h"


class DX11BuildLinearSystem
{
	public:

		static HRESULT OnD3D11CreateDevice(ID3D11Device* pd3dDevice);
		
		static HRESULT applyBL(ID3D11DeviceContext* context, ID3D11ShaderResourceView* inputSRV, ID3D11ShaderResourceView* correspondenceSRV, ID3D11ShaderResourceView* correspondenceNormalsSRV, D3DXVECTOR3& mean, float meanStDev, Eigen::Matrix4f& deltaTransform, unsigned int imageWidth, unsigned int imageHeight, unsigned int level, Matrix6x7f& res, LinearSystemConfidence& conf);

		//! builds AtA, AtB, and confidences
		static Matrix6x7f reductionSystemCPU(const float* data, unsigned int nElems, LinearSystemConfidence& conf);
				
		static void OnD3D11DestroyDevice();
		
	private:

		/////////////////////////////////////////////////////
		// Build Linear System
		/////////////////////////////////////////////////////
		static unsigned int m_blockSizeBL;
	
		struct CBufferBL
		{
			int imageWidth;
			int imageHeigth;
			int2 dummy;
			float deltaTransform[16];
			D3DXVECTOR3 mean;
			float meanStDevInv;
		};
		
		static ID3D11Buffer* m_constantBufferBL;
		static ID3D11ComputeShader** m_pComputeShaderBL;

		static ID3D11Buffer* m_pOutputFloat;
		static ID3D11UnorderedAccessView* m_pOutputFloatUAV;
		static ID3D11Buffer* m_pOutputFloatCPU;

		/////////////////////////////////////////////////////
		// Query
		/////////////////////////////////////////////////////

		static Timer m_timer;
};
