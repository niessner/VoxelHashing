#pragma once

/****************************************************************************/
/* Helper to perform GPU image operations such as filtering, copying etc.   */
/****************************************************************************/

#include "stdafx.h"

#include <D3D11.h>
#include "DX11Utils.h"
#include "TimingLog.h"
#include "Eigen.h"

class DX11ImageHelper
{
	public:

		static HRESULT OnD3D11CreateDevice(ID3D11Device* pd3dDevice);
		
		static HRESULT applyBilateralFilter(ID3D11DeviceContext* context, ID3D11ShaderResourceView* inputSRV, ID3D11UnorderedAccessView* outputUAV, unsigned int imageWidth, unsigned int imageHeight, float sigmaD = 5.0f, float sigmaR = 0.1f);

		static HRESULT applyBilateralFilterForSSAO(ID3D11DeviceContext* context, ID3D11ShaderResourceView* inputDepthSRV, ID3D11ShaderResourceView* inputSRV, ID3D11UnorderedAccessView* outputUAV, unsigned int imageWidth, unsigned int imageHeight, float sigmaD = 5.0f, float sigmaR = 0.1f);
		
		static HRESULT applyBilateralFilterApprox(ID3D11DeviceContext* context, ID3D11ShaderResourceView* inputSRV, ID3D11UnorderedAccessView* outputUAV, unsigned int imageWidth, unsigned int imageHeight, unsigned int kernelRadius, float distThres);

		static HRESULT applyBilateralCompletion(ID3D11DeviceContext* context, ID3D11ShaderResourceView* inputSRV, ID3D11UnorderedAccessView* outputUAV, unsigned int imageWidth, unsigned int imageHeight, float sigmaD = 5.0f, float sigmaR = 0.1f);

		static HRESULT applyBilateralCompletionWithColor(
			ID3D11DeviceContext* context, 
			ID3D11ShaderResourceView* inputDepthSRV, ID3D11UnorderedAccessView* outputDepthUAV, 
			ID3D11ShaderResourceView* inputColorSRV, ID3D11UnorderedAccessView* outputColorUAV,
			unsigned int imageWidth, unsigned int imageHeight, float sigmaD = 5.0f, float sigmaR = 0.1f);

		static HRESULT applyBilateralFilter4F(ID3D11DeviceContext* context, ID3D11ShaderResourceView* inputSRV, ID3D11UnorderedAccessView* outputUAV, unsigned int imageWidth, unsigned int imageHeight, float sigmaD = 5.0f, float sigmaR = 0.1f);

		static HRESULT applyNormalComputation(ID3D11DeviceContext* context, ID3D11ShaderResourceView* inputSRV, ID3D11UnorderedAccessView* outputUAV, unsigned int imageWidth, unsigned int imageHeight);

		static HRESULT applyCameraSpace(ID3D11DeviceContext* context, ID3D11ShaderResourceView* inputSRV, ID3D11UnorderedAccessView* outputUAV, unsigned int imageWidth, unsigned int imageHeight);

		static HRESULT applyCameraSpaceProjection(ID3D11DeviceContext* context, ID3D11ShaderResourceView* inputSRV, ID3D11UnorderedAccessView* outputUAV, unsigned int imageWidth, unsigned int imageHeight);

		static HRESULT StereoCameraSpaceProjection(ID3D11DeviceContext* context, ID3D11ShaderResourceView* inputSRV, ID3D11UnorderedAccessView* outputUAV, unsigned int imageWidth, unsigned int imageHeight);

		static HRESULT applyDepthMap(ID3D11DeviceContext* context, ID3D11ShaderResourceView* inputSRV, ID3D11UnorderedAccessView* outputUAV, unsigned int imageWidth, unsigned int imageHeight);

		static HRESULT applySSAOMap(ID3D11DeviceContext* context, ID3D11ShaderResourceView* inputSRV, ID3D11UnorderedAccessView* outputUAV, unsigned int imageWidth, unsigned int imageHeight);

		static HRESULT applyHSVDepth(ID3D11DeviceContext* context, ID3D11ShaderResourceView* inputSRV, ID3D11UnorderedAccessView* outputUAV, unsigned int imageWidth, unsigned int imageHeight);

		static HRESULT applyProjectiveCorrespondences(ID3D11DeviceContext* context, ID3D11ShaderResourceView* inputSRV, ID3D11ShaderResourceView* inputNormalsSRV, ID3D11ShaderResourceView* inputColorsSRV, ID3D11ShaderResourceView* targetSRV, ID3D11ShaderResourceView* targetNormalsSRV, ID3D11ShaderResourceView* targetColorsSRV, ID3D11UnorderedAccessView* outputUAV, ID3D11UnorderedAccessView* outputNormalUAV, const Eigen::Matrix4f& deltaTransform, unsigned int imageWidth, unsigned int imageHeight, float distThres, float normalThres, float levelFactor);

		// Input Image (2*targetImageWidth, 2*targetImageHeight) -> Target Image (targetImageWidth, targetImageHeight)
		static HRESULT applySubSamp(ID3D11DeviceContext* context, ID3D11ShaderResourceView* inputSRV, ID3D11UnorderedAccessView* outputUAV, unsigned int targetImageWidth, unsigned int targetImageHeight);

		// Input Image (2*targetImageWidth, 2*targetImageHeight) -> Target Image (targetImageWidth, targetImageHeight)
		static HRESULT applyDownsampling(ID3D11DeviceContext* context, ID3D11ShaderResourceView* inputSRV, ID3D11UnorderedAccessView* outputUAV, unsigned int targetImageWidth, unsigned int targetImageHeight);

		static HRESULT applyBlockAveraging(ID3D11DeviceContext* context, ID3D11ShaderResourceView* inputSRV, ID3D11UnorderedAccessView* outputUAV, unsigned int imageWidth, unsigned int imageHeight, float sigmaD, float sigmaR);

		static HRESULT applyCopy(ID3D11DeviceContext* context, ID3D11ShaderResourceView* inputSRV, ID3D11UnorderedAccessView* outputUAV, unsigned int imageWidth, unsigned int imageHeight);

		static HRESULT applyErode(ID3D11DeviceContext* context, ID3D11ShaderResourceView* inputSRV, ID3D11ShaderResourceView* inputColorSRV, ID3D11UnorderedAccessView* outputUAV, float distThres, int stencilSize, unsigned int imageWidth, unsigned int imageHeight);

		static void OnD3D11DestroyDevice();
		
	private:
		
		/////////////////////////////////////////////////////
		// Bilateral Completion
		/////////////////////////////////////////////////////
		static unsigned int m_blockSizeBilateralCompletion;

		struct CBufferBilateralCompletion
		{
			int imageWidth;
			int imageHeigth;

			float sigmaD;
			float sigmaR;
		};

		static ID3D11Buffer* m_constantBufferBilateralCompletion;
		static ID3D11ComputeShader* m_pComputeShaderBilateralCompletion;
		static ID3D11ComputeShader* m_pComputeShaderBilateralCompletionWithColor;

		/////////////////////////////////////////////////////
		// Bilateral Filter
		/////////////////////////////////////////////////////
		static unsigned int m_blockSizeBilateralFilter;
	
		struct CBufferBilateralFilter
		{
			int imageWidth;
			int imageHeigth;

			float sigmaD;
			float sigmaR;
		};

		static ID3D11Buffer* m_constantBufferBilateralFilter;
		static ID3D11ComputeShader* m_pComputeShaderBilateralFilter;

		/////////////////////////////////////////////////////
		// Bilateral Filter SSAO
		/////////////////////////////////////////////////////
		static unsigned int m_blockSizeBilateralFilterForSSAO;
	
		struct CBufferBilateralFilterSSAO
		{
			int imageWidth;
			int imageHeigth;

			float sigmaD;
			float sigmaR;
		};

		static ID3D11Buffer* m_constantBufferBilateralFilterForSSAO;
		static ID3D11ComputeShader* m_pComputeShaderBilateralFilterForSSAO;

		/////////////////////////////////////////////////////
		// Bilateral Filter Approximation
		/////////////////////////////////////////////////////
		static unsigned int m_blockSizeBilateralFilterApprox;
	
		struct CBufferBilateralFilterApprox
		{
			int imageWidth;
			int imageHeigth;

			int kernelRadius;
			float thres;
		};

		static ID3D11Buffer* m_constantBufferBilateralFilterApprox;
		static ID3D11ComputeShader* m_pComputeShaderBilateralFilterApprox;

		/////////////////////////////////////////////////////
		// Bilateral Filter 4F
		/////////////////////////////////////////////////////
		static unsigned int m_blockSizeBilateralFilter4F;
	
		struct CBufferBilateralFilter4F
		{
			int imageWidth;
			int imageHeigth;

			float sigmaD;
			float sigmaR;
		};

		static ID3D11Buffer* m_constantBufferBilateralFilter4F;
		static ID3D11ComputeShader* m_pComputeShaderBilateralFilter4F;

		/////////////////////////////////////////////////////
		// Normal Computation
		/////////////////////////////////////////////////////

		static unsigned int m_blockSizeNormalComputation;
	
		struct CBufferNormalComputation
		{
			int imageWidth;
			int imageHeigth;
			unsigned int dummy0;
			unsigned int dummy1;
		};

		static ID3D11Buffer* m_constantBufferNormalComputation;
		static ID3D11ComputeShader* m_pComputeShaderNormalComputation;

		/////////////////////////////////////////////////////
		// Camera Space To Depth Map
		/////////////////////////////////////////////////////

		static unsigned int m_blockSizeCameraSpace;
	
		struct CBufferCameraSpace
		{
			int imageWidth;
			int imageHeigth;
			unsigned int dummy0;
			unsigned int dummy1;
		};

		static ID3D11Buffer* m_constantBufferCameraSpace;
		static ID3D11ComputeShader* m_pComputeShaderCameraSpace;

		/////////////////////////////////////////////////////
		// Compute Depth Maps
		/////////////////////////////////////////////////////

		static unsigned int m_blockSizeDepthMap;
	
		struct CBufferDepthMap
		{
			int imageWidth;
			int imageHeigth;
			unsigned int dummy0;
			unsigned int dummy1;
		};

		static ID3D11Buffer* m_constantBufferDepthMap;
		static ID3D11ComputeShader* m_pComputeShaderDepthMap;

		/////////////////////////////////////////////////////
		// Compute SSAO Map
		/////////////////////////////////////////////////////

		static unsigned int m_blockSizeSSAOMap;
	
		struct CBufferSSAOMap
		{
			int imageWidth;
			int imageHeigth;
			unsigned int dummy0;
			unsigned int dummy1;

			float4 rotationVectors[16];
		};

		static std::vector<vec4f> m_randomRotations;
		
		static ID3D11Buffer* m_constantBufferSSAOMap;
		static ID3D11ComputeShader* m_pComputeShaderSSAOMap;

		/////////////////////////////////////////////////////
		// Compute HSV Depth Maps
		/////////////////////////////////////////////////////

		static unsigned int m_blockSizeHSVDepth;
	
		struct CBufferHSVDepth
		{
			int imageWidth;
			int imageHeigth;
			unsigned int dummy0;
			unsigned int dummy1;
		};

		static ID3D11Buffer* m_constantBufferHSVDepth;
		static ID3D11ComputeShader* m_pComputeShaderHSVDepth;

		/////////////////////////////////////////////////////
		// Camera Space Projection
		/////////////////////////////////////////////////////

		static unsigned int m_blockSizeCameraSpaceProjection;
	
		struct CBufferCameraSpaceProjection
		{
			int imageWidth;
			int imageHeigth;
			unsigned int dummy0;
			unsigned int dummy1;
		};

		static ID3D11Buffer* m_constantBufferCameraSpaceProjection;
		static ID3D11ComputeShader* m_pComputeShaderCameraSpaceProjection;

		/////////////////////////////////////////////////////
		// Stereo Mask Camera Space Projection
		/////////////////////////////////////////////////////

		static unsigned int m_blockSizeStereoCameraSpaceProjection;

		struct CBufferStereoCameraSpaceProjection
		{
			int imageWidth;
			int imageHeigth;
			unsigned int dummy0;
			unsigned int dummy1;
		};

		static ID3D11Buffer* m_constantBufferStereoCameraSpaceProjection;
		static ID3D11ComputeShader* m_pComputeShaderStereoCameraSpaceProjection;

		/////////////////////////////////////////////////////
		// Projective Correspondences
		/////////////////////////////////////////////////////

		static unsigned int m_blockSizeProjectiveCorrespondences;
	
		struct CBufferProjectiveCorrespondences
		{
			int imageWidth;
			int imageHeigth;
			float distThres;
			float normalThres;
			float transform[16];
			float levelFactor;
			float dummy00;
			float dummy01;
			float dummy02;
			//float transformNormal[9];
		};

		static ID3D11Buffer* m_constantBufferProjectiveCorrespondences;
		static ID3D11ComputeShader* m_pComputeShaderProjectiveCorrespondences;

		/////////////////////////////////////////////////////
		// Depth aware block averaging
		/////////////////////////////////////////////////////

		static unsigned int m_blockSizeBlockAverage;
	
		struct CBufferBlockAverage
		{
			int imageWidth;
			int imageHeigth;
			float sigmaD;
			float sigmaR;
		};

		static ID3D11Buffer* m_constantBufferBlockAverage;
		static ID3D11ComputeShader* m_pComputeShaderBlockAverage;

		/////////////////////////////////////////////////////
		// Subsample
		/////////////////////////////////////////////////////

		static unsigned int m_blockSizeSubSamp;
	
		struct CBufferSubSamp
		{
			int imageWidth;
			int imageHeigth;
			unsigned int dummy0;
			unsigned int dummy1;
		};

		static ID3D11Buffer* m_constantBufferSubSamp;
		static ID3D11ComputeShader* m_pComputeShaderSubSamp;

		/////////////////////////////////////////////////////
		// Downsampling
		/////////////////////////////////////////////////////

		static unsigned int m_blockSizeDownsampling;
	
		struct CBufferDS
		{
			int imageWidth;
			int imageHeigth;
			unsigned int dummy0;
			unsigned int dummy1;
		};

		static ID3D11Buffer* m_constantBufferDownsampling;
		static ID3D11ComputeShader* m_pComputeShaderDownsampling;

		/////////////////////////////////////////////////////
		// Copy
		/////////////////////////////////////////////////////

		static unsigned int m_blockSizeCopy;
	
		struct CBufferCopy
		{
			int imageWidth;
			int imageHeigth;
			unsigned int dummy0;
			unsigned int dummy1;
		};

		static ID3D11Buffer* m_constantBufferCopy;
		static ID3D11ComputeShader* m_pComputeShaderCopy;

		/////////////////////////////////////////////////////
		// Erode
		/////////////////////////////////////////////////////

		static unsigned int m_blockSizeErode;
	
		struct CBufferErode
		{
			int imageWidth;
			int imageHeigth;
			float distThres;
			int stencilSize;
		};

		static ID3D11Buffer* m_constantBufferErode;
		static ID3D11ComputeShader* m_pComputeShaderErode;
		
		/////////////////////////////////////////////////////
		// Timer
		/////////////////////////////////////////////////////

		static Timer m_timer;
};
