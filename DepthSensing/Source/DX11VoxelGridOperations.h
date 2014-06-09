#pragma once

#include <D3D11.h>
#include "DXUT.h"
#include "DX11VoxelGrid.h"
#include "TimingLog.h"
#include "GlobalAppState.h"
#include "matrix4x4.h"

class DX11VoxelGridOperations
{
	public:

		//---------------------------------------------------
		// Voxel Grid Operations
		//---------------------------------------------------

		static HRESULT OnD3D11CreateDevice(ID3D11Device* pd3dDevice)
		{
			HRESULT hr = S_OK;

			V_RETURN(initializeReset(pd3dDevice));
			V_RETURN(initializeSetDistanceFunctionEllipsoid(pd3dDevice));
			V_RETURN(initializeIntegrateDepthFrame(pd3dDevice));
			V_RETURN(initializeExtractIsoSurface(pd3dDevice));

			return  hr;
		}

		static void OnD3D11DestroyDevice()
		{
			destroyReset();
			destroySetDistanceFunctionEllipsoid();
			destroyIntegrateDepthFrame();
			destroyExtractIsoSurface();
		}
		
		//---------------------------------------------------------------------------------------------------------------
		// Reset
		//---------------------------------------------------------------------------------------------------------------

	public:
		
		static HRESULT reset(ID3D11DeviceContext* context, ID3D11UnorderedAccessView* voxelBuffer, D3DXVECTOR3* gridPosition, int3* gridDimensions, D3DXVECTOR3* voxelExtends) 
		{
			m_LastRigidTransform.setIdentity();
			m_NumIntegratedImages = 0;

			HRESULT hr = S_OK;

			// Initialize constant buffer
			D3D11_MAPPED_SUBRESOURCE mappedResource;
			V_RETURN(context->Map(m_constantBufferReset, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource));

				CBufferReset *cbuffer = (CBufferReset*)mappedResource.pData;
				memcpy(&cbuffer->gridPosition, gridPosition, sizeof(D3DXVECTOR3));
				memcpy(&cbuffer->voxelExtends, voxelExtends, sizeof(D3DXVECTOR3));
				memcpy(&cbuffer->gridDimensions, gridDimensions, sizeof(int3));

			context->Unmap(m_constantBufferReset, 0);

			// Setup pipeline
			context->CSSetUnorderedAccessViews(0, 1, &voxelBuffer, 0);
			context->CSSetConstantBuffers(0, 1, &m_constantBufferReset);
			context->CSSetShader(m_pComputeShaderReset, 0, 0);

			// Run compute shader
			unsigned int dimX = (unsigned int)ceil(((float)gridDimensions->x)/m_blockSizeReset);
			unsigned int dimY = (unsigned int)ceil(((float)gridDimensions->y)/m_blockSizeReset);
			unsigned int dimZ = (unsigned int)ceil(((float)gridDimensions->z)/m_blockSizeReset);
			
			assert(dimX <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);
			assert(dimY <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);
			assert(dimZ <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);

			context->Dispatch(dimX, dimY, dimZ);

			// Cleanup
			ID3D11UnorderedAccessView* nullUAV[1] = {NULL};
			ID3D11Buffer* nullB[1] = {NULL};

			context->CSSetUnorderedAccessViews(0, 1, nullUAV, 0);
			context->CSSetConstantBuffers(0, 1, nullB);
			context->CSSetShader(0, 0, 0);

			return hr;
		}
		
	private:

		static HRESULT initializeReset(ID3D11Device* pd3dDevice)
		{
			HRESULT hr = S_OK;

			char BLOCK_SIZE_Reset[5];
			sprintf_s(BLOCK_SIZE_Reset, "%d", m_blockSizeReset);

			D3D_SHADER_MACRO shaderDefinesReset[] = {{"groupthreads", BLOCK_SIZE_Reset}, {0}};

			ID3DBlob* pBlob = NULL;
			V_RETURN(CompileShaderFromFile(L"Shaders\\REset.hlsl", "resetCS", "cs_5_0", &pBlob, shaderDefinesReset));
			V_RETURN(pd3dDevice->CreateComputeShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &m_pComputeShaderReset))
			SAFE_RELEASE(pBlob);

			D3D11_BUFFER_DESC bDesc;
			bDesc.BindFlags	= D3D11_BIND_CONSTANT_BUFFER;
			bDesc.Usage	= D3D11_USAGE_DYNAMIC;
			bDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
			bDesc.MiscFlags	= 0;

			bDesc.ByteWidth	= sizeof(CBufferReset) ;
			V_RETURN(pd3dDevice->CreateBuffer(&bDesc, NULL, &m_constantBufferReset));

			return hr;
		}

		static void destroyReset()
		{
			SAFE_RELEASE(m_pComputeShaderReset);
			SAFE_RELEASE(m_constantBufferReset);
		}

		// State
		struct CBufferReset
		{
			// Grid
			D3DXVECTOR3 gridPosition;
			float align0;

			int3 gridDimensions;
			int align1;

			D3DXVECTOR3 voxelExtends;
			float align2;
		};
				
		static unsigned int m_blockSizeReset;

		static ID3D11ComputeShader* m_pComputeShaderReset;
		static ID3D11Buffer* m_constantBufferReset;

	//---------------------------------------------------------------------------------------------------------------
	// Set distance function ellipsoid
	//---------------------------------------------------------------------------------------------------------------

	public:
		
		static HRESULT setDistanceFunctionEllipsoid(ID3D11DeviceContext* context, ID3D11UnorderedAccessView* voxelBuffer, D3DXVECTOR3* gridPosition, int3* gridDimensions, D3DXVECTOR3* voxelExtends, D3DXVECTOR3& center, float a, float b, float c) 
		{
			HRESULT hr = S_OK;

			// Initialize constant buffer
			D3D11_MAPPED_SUBRESOURCE mappedResource;
			V_RETURN(context->Map(m_constantBufferSetDistanceFunctionEllipsoid, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource));

				CBufferSetDistanceFunctionEllipsoid *cbuffer = (CBufferSetDistanceFunctionEllipsoid*)mappedResource.pData;
				memcpy(&cbuffer->gridPosition, gridPosition, sizeof(D3DXVECTOR3));
				memcpy(&cbuffer->voxelExtends, voxelExtends, sizeof(D3DXVECTOR3));
				memcpy(&cbuffer->gridDimensions, gridDimensions, sizeof(int3));
				memcpy(&cbuffer->center, &center, sizeof(D3DXVECTOR3));
				cbuffer->a = a;
				cbuffer->b = b;
				cbuffer->c = c;
			
			context->Unmap(m_constantBufferSetDistanceFunctionEllipsoid, 0);

			// Setup pipeline
			context->CSSetUnorderedAccessViews(0, 1, &voxelBuffer, 0);
			context->CSSetConstantBuffers(0, 1, &m_constantBufferSetDistanceFunctionEllipsoid);
			context->CSSetShader(m_pComputeShaderSetDistanceFunctionEllipsoid, 0, 0);

			// Run compute shader
			unsigned int dimX = (unsigned int)ceil(((float)gridDimensions->x)/m_blockSizeSetDistanceFunctionEllipsoid);
			unsigned int dimY = (unsigned int)ceil(((float)gridDimensions->y)/m_blockSizeSetDistanceFunctionEllipsoid);
			unsigned int dimZ = (unsigned int)ceil(((float)gridDimensions->z)/m_blockSizeSetDistanceFunctionEllipsoid);
		
			// Start query for timing
			if(GlobalAppState::getInstance().s_timingsDetailledEnabled)
			{
				GlobalAppState::getInstance().WaitForGPU();
				m_timer.start();
			}

			assert(dimX <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);
			assert(dimY <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);
			assert(dimZ <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);

			context->Dispatch(dimX, dimY, dimZ);

			// Wait for query
			if(GlobalAppState::getInstance().s_timingsDetailledEnabled)
			{
				GlobalAppState::getInstance().WaitForGPU();
				TimingLog::totalTimeFillVoxelGrid+=m_timer.getElapsedTimeMS();
				TimingLog::countFillVoxelGrid++;
			}

			// Cleanup
			ID3D11UnorderedAccessView* nullUAV[1] = {NULL};
			ID3D11Buffer* nullB[1] = {NULL};

			context->CSSetUnorderedAccessViews(0, 1, nullUAV, 0);
			context->CSSetConstantBuffers(0, 1, nullB);
			context->CSSetShader(0, 0, 0);

			return hr;
		}
		
	private:

		static HRESULT initializeSetDistanceFunctionEllipsoid(ID3D11Device* pd3dDevice)
		{
			HRESULT hr = S_OK;

			char BLOCK_SIZE_SetDistanceFunctionEllipsoid[5];
			sprintf_s(BLOCK_SIZE_SetDistanceFunctionEllipsoid, "%d", m_blockSizeSetDistanceFunctionEllipsoid);

			D3D_SHADER_MACRO shaderDefinesSetDistanceFunctionEllipsoid[] = {{"groupthreads", BLOCK_SIZE_SetDistanceFunctionEllipsoid}, {0}};

			ID3DBlob* pBlob = NULL;
			V_RETURN(CompileShaderFromFile(L"Shaders\\SetDistanceFunctionEllipsoid.hlsl", "setDistanceFunctionEllipsoidCS", "cs_5_0", &pBlob, shaderDefinesSetDistanceFunctionEllipsoid));
			V_RETURN(pd3dDevice->CreateComputeShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &m_pComputeShaderSetDistanceFunctionEllipsoid))
			SAFE_RELEASE(pBlob);

			D3D11_BUFFER_DESC bDesc;
			bDesc.BindFlags	= D3D11_BIND_CONSTANT_BUFFER;
			bDesc.Usage	= D3D11_USAGE_DYNAMIC;
			bDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
			bDesc.MiscFlags	= 0;

			bDesc.ByteWidth	= sizeof(CBufferSetDistanceFunctionEllipsoid) ;
			V_RETURN(pd3dDevice->CreateBuffer(&bDesc, NULL, &m_constantBufferSetDistanceFunctionEllipsoid));

			return hr;
		}

		static void destroySetDistanceFunctionEllipsoid()
		{
			SAFE_RELEASE(m_pComputeShaderSetDistanceFunctionEllipsoid);
			SAFE_RELEASE(m_constantBufferSetDistanceFunctionEllipsoid);
		}

		// State
		struct CBufferSetDistanceFunctionEllipsoid
		{
			// Grid
			D3DXVECTOR3 gridPosition;
			float align0;

			int3 gridDimensions;
			int align1;

			D3DXVECTOR3 voxelExtends;
			float align2;

			// Sphere
			D3DXVECTOR3 center;
			float align3;

			float a;
			float b;
			float c;
			float align4;
		};
		
		static unsigned int m_blockSizeSetDistanceFunctionEllipsoid;

		static ID3D11ComputeShader* m_pComputeShaderSetDistanceFunctionEllipsoid;
		static ID3D11Buffer* m_constantBufferSetDistanceFunctionEllipsoid;

	//---------------------------------------------------------------------------------------------------------------
	// Integrate depth frame
	//---------------------------------------------------------------------------------------------------------------

	public:
		
		static HRESULT integrateDepthFrame(ID3D11DeviceContext* context, ID3D11UnorderedAccessView* voxelBuffer, D3DXVECTOR3* gridPosition, int3* gridDimensions, D3DXVECTOR3* voxelExtends, ID3D11ShaderResourceView* depth, ID3D11ShaderResourceView* color, mat4f* lastRigidTransform, unsigned int imageWidth, unsigned int imageHeight)
		{
			HRESULT hr = S_OK;

			m_LastRigidTransform = *lastRigidTransform;
			m_NumIntegratedImages++;

			// Initialize constant buffer
			D3D11_MAPPED_SUBRESOURCE mappedResource;
			V_RETURN(context->Map(m_constantBufferIntegrateDepthFrame, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource));

				CBufferIntegrateDepthFrame *cbuffer = (CBufferIntegrateDepthFrame*)mappedResource.pData;
				memcpy(&cbuffer->gridPosition, gridPosition, sizeof(D3DXVECTOR3));
				memcpy(&cbuffer->voxelExtends, voxelExtends, sizeof(D3DXVECTOR3));
				memcpy(&cbuffer->gridDimensions, gridDimensions, sizeof(int3));

				mat4f worldToLastKinectSpace = lastRigidTransform->getInverse();
				memcpy(&cbuffer->viewMat, &worldToLastKinectSpace, sizeof(mat4f));

				cbuffer->imageWidth = imageWidth;
				cbuffer->imageHeight = imageHeight;
			
			context->Unmap(m_constantBufferIntegrateDepthFrame, 0);

			// Setup pipeline
			context->CSSetUnorderedAccessViews(0, 1, &voxelBuffer, 0);
			context->CSSetShaderResources(0, 1, &depth);
			context->CSSetShaderResources(1, 1, &color);
			context->CSSetConstantBuffers(0, 1, &m_constantBufferIntegrateDepthFrame);
			context->CSSetShader(m_pComputeShaderIntegrateDepthFrame, 0, 0);

			// Run compute shader
			unsigned int dimX = (unsigned int)ceil(((float)gridDimensions->x)/m_blockSizeIntegrateDepthFrame);
			unsigned int dimY = (unsigned int)ceil(((float)gridDimensions->y)/m_blockSizeIntegrateDepthFrame);
			unsigned int dimZ = (unsigned int)ceil(((float)gridDimensions->z)/m_blockSizeIntegrateDepthFrame);
					
			// Start query for timing
			if(GlobalAppState::getInstance().s_timingsDetailledEnabled)
			{
				GlobalAppState::getInstance().WaitForGPU();
				m_timer.start();
			}

			assert(dimX <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);
			assert(dimY <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);
			assert(dimZ <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);
			context->Dispatch(dimX, dimY, dimZ);

			// Wait for query
			if(GlobalAppState::getInstance().s_timingsDetailledEnabled)
			{
				GlobalAppState::getInstance().WaitForGPU();
				TimingLog::totalTimeFillVoxelGrid+=m_timer.getElapsedTimeMS();
				TimingLog::countFillVoxelGrid++;
			}

			// Cleanup
			ID3D11UnorderedAccessView* nullUAV[1] = {NULL};
			ID3D11ShaderResourceView* nullSRV[1] = {NULL};
			ID3D11Buffer* nullB[1] = {NULL};

			context->CSSetUnorderedAccessViews(0, 1, nullUAV, 0);
			context->CSSetShaderResources(0, 1, nullSRV);
			context->CSSetShaderResources(1, 1, nullSRV);
			context->CSSetConstantBuffers(0, 1, nullB);
			context->CSSetShader(0, 0, 0);

			return hr;
		}

		static const mat4f& GetLastRigidTransform()
		{	
			return m_LastRigidTransform;
		}

		static unsigned int GetNumIntegratedImages()
		{
			return m_NumIntegratedImages;
		}
				
	private:

		static HRESULT initializeIntegrateDepthFrame(ID3D11Device* pd3dDevice)
		{
			HRESULT hr = S_OK;

			char BLOCK_SIZE_IntegrateDepthFrame[5];
			sprintf_s(BLOCK_SIZE_IntegrateDepthFrame, "%d", m_blockSizeIntegrateDepthFrame);

			D3D_SHADER_MACRO shaderDefinesIntegrateDepthFrame[] = {{"groupthreads", BLOCK_SIZE_IntegrateDepthFrame}, {0}};

			ID3DBlob* pBlob = NULL;
			V_RETURN(CompileShaderFromFile(L"Shaders\\IntegrateDepthFrame.hlsl", "integrateDepthFrameCS", "cs_5_0", &pBlob, shaderDefinesIntegrateDepthFrame));
			V_RETURN(pd3dDevice->CreateComputeShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &m_pComputeShaderIntegrateDepthFrame))
			SAFE_RELEASE(pBlob);

			D3D11_BUFFER_DESC bDesc;
			bDesc.BindFlags	= D3D11_BIND_CONSTANT_BUFFER;
			bDesc.Usage	= D3D11_USAGE_DYNAMIC;
			bDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
			bDesc.MiscFlags	= 0;

			bDesc.ByteWidth	= sizeof(CBufferIntegrateDepthFrame) ;
			V_RETURN(pd3dDevice->CreateBuffer(&bDesc, NULL, &m_constantBufferIntegrateDepthFrame));

			m_LastRigidTransform.setIdentity();
			m_NumIntegratedImages = 0;

			return S_OK;
		}
		
		static void destroyIntegrateDepthFrame()
		{
			SAFE_RELEASE(m_pComputeShaderIntegrateDepthFrame);
			SAFE_RELEASE(m_constantBufferIntegrateDepthFrame);
		}

		// State
		struct CBufferIntegrateDepthFrame
		{
			// Grid
			D3DXVECTOR3 gridPosition;
			float align0;

			int3 gridDimensions;
			int align1;

			D3DXVECTOR3 voxelExtends;
			float align2;

			int imageWidth;
			int imageHeight;
			int align3;
			int align4;

			float viewMat[16];
		};
		
		static mat4f m_LastRigidTransform;
		static unsigned int m_NumIntegratedImages;

		static unsigned int m_blockSizeIntegrateDepthFrame;

		static ID3D11ComputeShader* m_pComputeShaderIntegrateDepthFrame;
		static ID3D11Buffer* m_constantBufferIntegrateDepthFrame;

	//---------------------------------------------------------------------------------------------------------------
	// Extract Iso-Surface (Marching Cubes)
	//---------------------------------------------------------------------------------------------------------------

	public:
		
		static HRESULT extractIsoSurface(ID3D11DeviceContext* context, ID3D11ShaderResourceView* voxelBuffer, D3DXVECTOR3* gridPosition, int3* gridDimensions, D3DXVECTOR3* voxelExtends)
		{
			HRESULT hr = S_OK;

			// Initialize constant buffer
			D3D11_MAPPED_SUBRESOURCE mappedResource;
			V_RETURN(context->Map(m_constantBufferExtractIsoSurface, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource));

				CBufferExtractIsoSurface *cbuffer = (CBufferExtractIsoSurface*)mappedResource.pData;
				memcpy(&cbuffer->gridPosition, gridPosition, sizeof(D3DXVECTOR3));
				memcpy(&cbuffer->voxelExtends, voxelExtends, sizeof(D3DXVECTOR3));
				memcpy(&cbuffer->gridDimensions, gridDimensions, sizeof(int3));
			
			context->Unmap(m_constantBufferExtractIsoSurface, 0);

			// Setup pipeline
			unsigned int initialCount = 0;
			context->CSSetUnorderedAccessViews(0, 1, &s_pTrianglesUAV, &initialCount);
			context->CSSetShaderResources(0, 1, &voxelBuffer);
			context->CSSetConstantBuffers(0, 1, &m_constantBufferExtractIsoSurface);
			context->CSSetShader(m_pComputeShaderExtractIsoSurface, 0, 0);

			// Run compute shader
			unsigned int dimX = (unsigned int)ceil(((float)gridDimensions->x)/m_blockSizeExtractIsoSurface);
			unsigned int dimY = (unsigned int)ceil(((float)gridDimensions->y)/m_blockSizeExtractIsoSurface);
			unsigned int dimZ = (unsigned int)ceil(((float)gridDimensions->z)/m_blockSizeExtractIsoSurface);
					
			// Start query for timing
			/*if(GlobalAppState::getInstance().s_timingsEnabled)
			{
				GlobalAppState::getInstance().WaitForGPU();
				m_timer.start();
			}*/

			context->Dispatch(dimX, dimY, dimZ);

			// Wait for query
			/*if(GlobalAppState::getInstance().s_timingsEnabled)
			{
				GlobalAppState::getInstance().WaitForGPU();
				TimingLog::totalTimeFillVoxelGrid+=m_timer.getElapsedTimeMS();
				TimingLog::countFillVoxelGrid++;
			}*/

			// Cleanup
			ID3D11UnorderedAccessView* nullUAV[1] = {NULL};
			ID3D11ShaderResourceView* nullSRV[1] = {NULL};
			ID3D11Buffer* nullB[1] = {NULL};

			context->CSSetUnorderedAccessViews(0, 1, nullUAV, 0);
			context->CSSetShaderResources(0, 1, nullSRV);
			context->CSSetConstantBuffers(0, 1, nullB);
			context->CSSetShader(0, 0, 0);

			// Copy to CPU
			context->CopyStructureCount(s_BuffCountTriangles, 0, s_pTrianglesUAV);
			V_RETURN(context->Map(s_BuffCountTriangles, 0, D3D11_MAP_READ, 0, &mappedResource));
			unsigned int nTriangles = ((unsigned int*)mappedResource.pData)[0];
			context->Unmap(s_BuffCountTriangles, 0);


			context->CopyResource(s_pOutputFloatCPU, s_pTriangles);
			V_RETURN(context->Map(s_pOutputFloatCPU, 0, D3D11_MAP_READ, 0, &mappedResource));
			saveAsMeshOFF((float*)mappedResource.pData, nTriangles);
			context->Unmap(s_pOutputFloatCPU, 0);

			return hr;
		}

		static void saveAsMeshOFF(const float* data, unsigned int nTriangles)
		{
			if(nTriangles <= s_maxNumberOfTriangles)
			{
				std::ofstream out("dump.off");

				if(out.fail()) { std::cerr << "Error -> File: " << __FILE__ << " Line: " << __LINE__ << " Function: " __FUNCTION__ << std::endl; return; }
			
				out << "COFF" << std::endl;
				out << 3*nTriangles << " ";
				out << nTriangles << " ";
				out << "0" << "\n";
			
				for(unsigned int i = 0; i < 3*nTriangles; i++)
				{
					out << data[6*i+0] << " " << data[6*i+1] << " " << data[6*i+2] << " " << (int)data[6*i+3] << " " << (int)data[6*i+4] << " " << (int)data[6*i+5] << " " << (int)255 << std::endl;
				}

				for(unsigned int i = 0; i < nTriangles; i++)
				{
					out << 3 << " " << 3*i+0 << " " << 3*i+1 << " " << 3*i+2 << std::endl;
				}

				out.close();

				std::cout << "Dumping finished" << std::endl;
			}
			else
			{
				std::cout << "Dumping FAILED: increase append buffer size (" << nTriangles << ", " << s_maxNumberOfTriangles << ")" << std::endl;
			}
		}

	private:

		static HRESULT initializeExtractIsoSurface(ID3D11Device* pd3dDevice)
		{
			HRESULT hr = S_OK;

			char BLOCK_SIZE_ExtractIsoSurface[5];
			sprintf_s(BLOCK_SIZE_ExtractIsoSurface, "%d", m_blockSizeExtractIsoSurface);

			D3D_SHADER_MACRO shaderDefinesExtractIsoSurface[] = {{"groupthreads", BLOCK_SIZE_ExtractIsoSurface}, {0}};

			ID3DBlob* pBlob = NULL;
			V_RETURN(CompileShaderFromFile(L"Shaders\\ExtractIsoSurface.hlsl", "extractIsoSurfaceCS", "cs_5_0", &pBlob, shaderDefinesExtractIsoSurface));
			V_RETURN(pd3dDevice->CreateComputeShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &m_pComputeShaderExtractIsoSurface))
			SAFE_RELEASE(pBlob);

			D3D11_BUFFER_DESC bDesc;
			bDesc.BindFlags	= D3D11_BIND_CONSTANT_BUFFER;
			bDesc.Usage	= D3D11_USAGE_DYNAMIC;
			bDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
			bDesc.MiscFlags	= 0;

			bDesc.ByteWidth	= sizeof(CBufferExtractIsoSurface) ;
			V_RETURN(pd3dDevice->CreateBuffer(&bDesc, NULL, &m_constantBufferExtractIsoSurface));

			// Create Append Buffer
			ZeroMemory(&bDesc, sizeof(D3D11_BUFFER_DESC));
			bDesc.BindFlags	= D3D11_BIND_UNORDERED_ACCESS;
			bDesc.Usage	= D3D11_USAGE_DEFAULT;
			bDesc.CPUAccessFlags = 0;
			bDesc.MiscFlags	= D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
			bDesc.ByteWidth	= 2*3*sizeof(float3)*s_maxNumberOfTriangles;
			bDesc.StructureByteStride = 2*3*sizeof(float3);
			V_RETURN(pd3dDevice->CreateBuffer(&bDesc, NULL, &s_pTriangles));
		
			D3D11_UNORDERED_ACCESS_VIEW_DESC UAVDesc;
			ZeroMemory(&UAVDesc, sizeof(D3D11_UNORDERED_ACCESS_VIEW_DESC));
			UAVDesc.Format = DXGI_FORMAT_UNKNOWN;
			UAVDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
			UAVDesc.Buffer.FirstElement = 0;
			UAVDesc.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_APPEND;
			UAVDesc.Buffer.NumElements = s_maxNumberOfTriangles;
			V_RETURN(pd3dDevice->CreateUnorderedAccessView(s_pTriangles, &UAVDesc, &s_pTrianglesUAV)); 

			// Create Output Buffer CPU
			bDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
			bDesc.BindFlags = 0;
			bDesc.Usage = D3D11_USAGE_STAGING;
			V_RETURN(pd3dDevice->CreateBuffer(&bDesc, NULL, &s_pOutputFloatCPU));

			// Create Output Buffer 
			ZeroMemory(&bDesc, sizeof(D3D11_BUFFER_DESC));
			bDesc.BindFlags	= 0;
			bDesc.Usage	= D3D11_USAGE_STAGING;
			bDesc.CPUAccessFlags =  D3D11_CPU_ACCESS_READ;
			bDesc.MiscFlags	= 0;
			bDesc.ByteWidth	= sizeof(int);
			V_RETURN(pd3dDevice->CreateBuffer(&bDesc, NULL, &s_BuffCountTriangles));

			return S_OK;
		}
		
		static void destroyExtractIsoSurface()
		{
			SAFE_RELEASE(m_pComputeShaderExtractIsoSurface);
			SAFE_RELEASE(m_constantBufferExtractIsoSurface);

			SAFE_RELEASE(s_pTriangles);
			SAFE_RELEASE(s_pTrianglesUAV);
			SAFE_RELEASE(s_BuffCountTriangles);
			SAFE_RELEASE(s_pOutputFloatCPU);
		}

		// State
		struct CBufferExtractIsoSurface
		{
			// Grid
			D3DXVECTOR3 gridPosition;
			float align0;

			int3 gridDimensions;
			int align1;

			D3DXVECTOR3 voxelExtends;
			float align2;
		};

		static unsigned int m_blockSizeExtractIsoSurface;
		static unsigned int s_maxNumberOfTriangles;

		static ID3D11ComputeShader* m_pComputeShaderExtractIsoSurface;
		static ID3D11Buffer* m_constantBufferExtractIsoSurface;

		static ID3D11Buffer* s_pTriangles;
		static ID3D11UnorderedAccessView* s_pTrianglesUAV;
		static ID3D11Buffer* s_BuffCountTriangles;
		static ID3D11Buffer* s_pOutputFloatCPU;

		//---------------------------------------------------------------------------------------------------------------
		// Other operation
		//---------------------------------------------------------------------------------------------------------------

	public:

		//---------------------------------------------------------------------------------------------------------------
		// Timer
		//---------------------------------------------------------------------------------------------------------------

		static Timer m_timer;

	private:
};
