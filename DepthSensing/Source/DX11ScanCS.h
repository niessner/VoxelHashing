
#pragma once

#include "DXUT.h"
#include "DX11Utils.h"

class DX11ScanCS
{
public:
    DX11ScanCS();
    HRESULT OnD3D11CreateDevice( ID3D11Device* pd3dDevice );
    void OnD3D11DestroyDevice();

	//if output_Buf != NULL -> obtain return last element of scan
	UINT ScanCS( ID3D11DeviceContext* pd3dImmediateContext,
		UINT numElements,
		ID3D11ShaderResourceView* input_SRV,
		ID3D11UnorderedAccessView* input_UAV,
		ID3D11ShaderResourceView* output_SRV,
		ID3D11UnorderedAccessView* output_UAV,
		ID3D11Buffer* output_Buf);

	UINT GetMaxScanSize();
private:
	bool m_bIsCreated;

	UINT m_BucketSize;
	UINT m_BucketBlockSize;

	UINT m_NumBuckets;
	UINT m_NumBucketBlocks;

	ID3D11Buffer*               m_pcbCS;

	ID3D11Buffer*               m_pBucketResults;
	ID3D11ShaderResourceView*   m_pBucketResultsSRV;
	ID3D11UnorderedAccessView*  m_pBucketResultsUAV;


	ID3D11Buffer*               m_pBucketBlockResults;
	ID3D11ShaderResourceView*   m_pBucketBlockResultsSRV;
	ID3D11UnorderedAccessView*  m_pBucketBlockResultsUAV;

	ID3D11Buffer*				m_PrefixSumLastElemStagging;
	ID3D11Buffer*				m_PrefixSumLastElem;
	ID3D11UnorderedAccessView*	m_PrefixSumLastElemUAV;
	ID3D11ShaderResourceView*	m_PrefixSumLastElemSRV;


	ID3D11ComputeShader*     m_pScanBucketsCS;
	ID3D11ComputeShader*     m_pScanBucketResultsCS;
	ID3D11ComputeShader*     m_pScanBucketBlockResultsCS;
	ID3D11ComputeShader*     m_pScanApplyBucketBlockResultsCS;
	ID3D11ComputeShader*     m_pScanApplyBucketResultsCS;

	struct CBScanCS
	{
		UINT numElements;
		UINT numBlocks;
		UINT pad1;
		UINT pad2;
	};
};


