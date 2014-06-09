#pragma once

/**************************************************************************/
/* Histogram class for the voxel data structure (including linked lists)  */
/**************************************************************************/

#include <D3D11.h>

#include <string>

#define NUM_GROUPS_X 1024 // to be in-sync with the define in the shader

class DX11HistogramHashSDF
{
	public:
		
		static HRESULT OnD3D11CreateDevice(ID3D11Device* pd3dDevice);

		static void OnD3D11DestroyDevice();

		static HRESULT computeHistrogram(ID3D11DeviceContext* context, ID3D11ShaderResourceView* hash, ID3D11Buffer* CBsceneRepSDF, unsigned int hashNumBuckets, unsigned int hashBucketSize, std::string hashName);

	private:

		static HRESULT initialize(ID3D11Device* pd3dDevice);
		
		static void printHistogram(const unsigned int* data, unsigned int hashNumBuckets, unsigned int hashBucketSize, std::string hashName);
						
		static void destroy();
		
		
		static unsigned int s_blockSize;

		static ID3D11ComputeShader* m_pComputeShader;
		static ID3D11ComputeShader* m_pComputeShaderReset;
	
		static ID3D11Buffer* s_pHistogram;
		static ID3D11UnorderedAccessView* s_pHistogramUAV;
		static ID3D11Buffer* s_pHistogramCPU;
};
