#pragma once

/************************************************************************/
/* Reduction on the GPU (required for ICP system matrix                 */
/************************************************************************/

#include "stdafx.h"

#include <D3D11.h>
#include "DX11Utils.h"

#include "Eigen.h"

class DX11NormalizeReduction
{
	public:

		static HRESULT OnD3D11CreateDevice(ID3D11Device* pd3dDevice);
		
		static HRESULT applyNorm(ID3D11DeviceContext* context, ID3D11ShaderResourceView* inputSRV, unsigned int level, unsigned int imageWidth, unsigned int imageHeight, D3DXVECTOR3& mean, float& meanStDev, float& nValidCorres);

		static void reductionCPU(const float* data, unsigned int nElems, D3DXVECTOR3& mean, float& meanStDev, float& nValidCorres);
				
		static void OnD3D11DestroyDevice();

	private:

		struct CBufferNorm
		{
			unsigned int imageWidth;
			unsigned int numElements;
			unsigned int pad0;
			unsigned int pad1;
		};

		static ID3D11Buffer* m_constantBufferNorm;

		static ID3D11ComputeShader** m_pComputeShaderNorm;
		static ID3D11ComputeShader** m_pComputeShader2Norm;

		static ID3D11Buffer** m_pAuxBufNorm;
		static ID3D11Buffer** m_pAuxBufNormCPU;

		static ID3D11ShaderResourceView** m_pAuxBufNormSRV;
		static ID3D11UnorderedAccessView** m_pAuxBufNormUAV;

		/////////////////////////////////////////////////////
		// Query
		/////////////////////////////////////////////////////

		static Timer m_timer;
};
