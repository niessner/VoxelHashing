#pragma once

#include "stdafx.h"

#include "DX11Utils.h"
//#include "KinectSensor2.h"

#include <iostream>
#include <cassert>
#include <vector>

#include "GlobalAppState.h"
#include "VoxelUtilHashSDF.h"
#include "DX11SceneRepHashSDF.h"
#include "BitArray.h"

#define NUM_GROUPS_X 1024 // to be in-sync with the define in the shader

LONG WINAPI StreamingFunc(LONG lParam);

class DX11SceneRepChunkGrid
{
	public:

		DX11SceneRepChunkGrid()
		{
			m_pSDFBlockDescOutput = NULL;
			m_pSDFBlockDescOutputUAV = NULL;
			m_pSDFBlockDescOutputSRV = NULL;

			m_pSDFBlockDescInput = NULL;
			m_pSDFBlockDescInputUAV = NULL;
			m_pSDFBlockDescInputSRV = NULL;

			m_currentPart = 0;
			//m_streamOutParts = GlobalAppState::getInstance().s_hashStreamOutParts;
			m_streamOutParts = 0;	//must be set later when global app state is actually initialized

			m_blockSizeIntegrateFromGlobalHash = 256;
			m_blockSizeChunkToGlobalHash = 256;
			m_maxNumberOfSDFBlocksIntegrateFromGlobalHash = 100000;

			m_pSDFBlockInputCPU = NULL;

			m_pSDFBlockDescOutputCPU = NULL;
			m_pSDFBlockDescInputCPU = NULL;
			
			m_constantBufferChunkToGlobalHash = NULL;
			m_constantBufferIntegrateFromGlobalHash = NULL;

			m_pComputeShaderIntegrateFromGlobalHashPass1 = NULL;
			m_pComputeShaderIntegrateFromGlobalHashPass2 = NULL;

			m_pComputeShaderChunkToGlobalHashPass1 = NULL;
			m_pComputeShaderChunkToGlobalHashPass2 = NULL;

			m_pSDFBlockOutput = NULL;
			m_pSDFBlockOutputUAV = NULL;
			m_pSDFBlockOutputSRV = NULL;

			m_pSDFBlockInput = NULL;
			m_pSDFBlockInputUAV = NULL;
			m_pSDFBlockInputSRV = NULL;

			m_pSDFBlockOutputCPU = NULL;
			m_BuffCountSDFBlockOutput = NULL;

			m_mappedResourceDescStreamOut.pData = NULL;
			m_mappedResourceBlockStreamOut.pData = NULL;

			m_mappedResourceDescStreamIn.pData = NULL;
			m_mappedResourceBlockStreamIn.pData = NULL;

			m_pBitMask = NULL;
			m_pBitMaskCPU = NULL;
			m_pBitMaskSRV = NULL;
		}
		
		HRESULT Init() {
			m_streamOutParts = GlobalAppState::getInstance().s_hashStreamOutParts;
			return S_OK;
		}


		// Stream Out
		HRESULT StreamOutToCPUAll(ID3D11DeviceContext* context, DX11SceneRepHashSDF& sceneRepHashSDF);

		HRESULT StreamOutToCPU(ID3D11DeviceContext* context, DX11SceneRepHashSDF& sceneRepHashSDF, const vec3f& posCamera, float radius, bool useParts, unsigned int& nStreamedBlocks);
		HRESULT StreamOutToCPUPass0GPU(ID3D11DeviceContext* context, DX11SceneRepHashSDF& sceneRepHashSDF, const vec3f& posCamera, float radius, bool useParts, bool multiThreaded = true);
		HRESULT StreamOutToCPUPass1CPU(ID3D11DeviceContext* context, bool multiThreaded = true);
		void IntegrateInChunkGrid(const int* desc, const int* block, unsigned int nSDFBlocks);

		// Stream In
		HRESULT StreamInToGPUAll(ID3D11DeviceContext* context, DX11SceneRepHashSDF& sceneRepHashSDF);
		HRESULT StreamInToGPUAll(ID3D11DeviceContext* context, DX11SceneRepHashSDF& sceneRepHashSDF, const vec3f& posCamera, float radius, bool useParts, unsigned int& nStreamedBlocks);
		HRESULT StreamInToGPUChunk(ID3D11DeviceContext* context, DX11SceneRepHashSDF& sceneRepHashSDF, const vec3i& chunkPos);
		HRESULT StreamInToGPUChunkNeighborhood(ID3D11DeviceContext* context, DX11SceneRepHashSDF& sceneRepHashSDF, const vec3i& chunkPos, int kernelRadius);

		HRESULT StreamInToGPU(ID3D11DeviceContext* context,  DX11SceneRepHashSDF& sceneRepHashSDF, const vec3f& posCamera, float radius, bool useParts, unsigned int& nStreamedBlocks);
		HRESULT StreamInToGPUPass0CPU(ID3D11DeviceContext* context, const vec3f& posCamera, float radius, bool useParts, bool multiThreaded = true);
		HRESULT StreamInToGPUPass1GPU(ID3D11DeviceContext* context, DX11SceneRepHashSDF& sceneRepHashSDF, bool multiThreaded = true);
		unsigned int IntegrateInHash(int* desc, int* blockOutput, const vec3f& posCamera, float radius, bool useParts);
		
		void checkForDuplicates();

		~DX11SceneRepChunkGrid()
		{
		}

		HRESULT startMultiThreading(ID3D11DeviceContext* context)
		{
			HRESULT hr = S_OK;

			// Mapped State
			V_RETURN(initializeMappedState(context));

			initializeCriticalSection();

			s_terminateThread = false;

			hStreamingThread = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)StreamingFunc, NULL, 0, &dwStreamingThreadID);

			if(hStreamingThread == NULL)
			{
				std::cout << "Thread GPU CPU could not be created" << std::endl;
			}

			return hr;
		}

		void stopMultiThreading()
		{
			s_terminateThread = true;

			SetEvent(hEventOutProduce);
			SetEvent(hEventOutConsume);

			SetEvent(hEventInProduce);
			SetEvent(hEventInConsume);

			WaitForSingleObject(hStreamingThread, INFINITE);

			if(CloseHandle(hStreamingThread) == 0)
			{
				std::cout << "Thread Handle GPU CPU could not be closed" << std::endl;
			}

			// Mutex
			deleteCritialSection();
		}

		void clearGrid()
		{
			for(unsigned int i = 0; i<m_grid.size(); i++)
			{
				if(m_grid[i] != NULL)
				{
					delete m_grid[i];
					m_grid[i] = NULL;
				}
			}
		}

		HRESULT Reset(ID3D11DeviceContext* context)
		{
			HRESULT hr = S_OK;

			stopMultiThreading();
		
			clearGrid();

			m_bitMask.reset();

			V_RETURN(startMultiThreading(context));

			return hr;
		}

		// Caution maps the buffer and performs the copy
		ID3D11ShaderResourceView* getBitMask(ID3D11DeviceContext* context)
		{
			D3D11_MAPPED_SUBRESOURCE mappedResource;
			context->Map(m_pBitMaskCPU, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource); // Unsafe
			unsigned int* buffer = (unsigned int*)mappedResource.pData;
			memcpy(&buffer[0], &(m_bitMask.getRawData()[0]), m_bitMask.getByteWidth());
			context->Unmap(m_pBitMaskCPU, 0);

			context->CopyResource(m_pBitMask, m_pBitMaskCPU);

			return m_pBitMaskSRV;
		}

		bool containsSDFBlocksChunk(vec3i chunk)
		{
			unsigned int index = linearizeChunkPos(chunk);
			return ((m_grid[index] != 0) && m_grid[index]->isStreamedOut());
		}

		bool isChunkInSphere(vec3i chunk, vec3f center, float radius)
		{
			vec3f posWorld = chunkToWorld(chunk);
			vec3f offset = m_voxelExtends/2.0f;

			for(int x = -1; x<=1; x+=2)
			{
				for(int y = -1; y<=1; y+=2)
				{
					for(int z = -1; z<=1; z+=2)
					{
						vec3f p = vec3f(posWorld.x+x*offset.x, posWorld.y+y*offset.y, posWorld.z+z*offset.z);
						float d = (p-center).length();

						if(d > radius)
						{
							return false;
						}
					}
				}
			}

			return true;
		}

		bool containsSDFBlocksChunkInRadius(vec3i chunk, int chunkRadius)
		{
			vec3i startChunk = vec3i(std::max(chunk.x-chunkRadius, m_minGridPos.x), std::max(chunk.y-chunkRadius, m_minGridPos.y), std::max(chunk.z-chunkRadius, m_minGridPos.z));
			vec3i endChunk = vec3i(std::min(chunk.x+chunkRadius, m_maxGridPos.x), std::min(chunk.y+chunkRadius, m_maxGridPos.y), std::min(chunk.z+chunkRadius, m_maxGridPos.z));

			for(int x = startChunk.x; x<=endChunk.x; x++)
			{
				for(int y = startChunk.y; y<=endChunk.y; y++)
				{
					for(int z = startChunk.z; z<=endChunk.z; z++)
					{
						unsigned int index = linearizeChunkPos(vec3i(x, y, z));
						if((m_grid[index] != 0) && m_grid[index]->isStreamedOut())
						{
							return true;
						}
					}
				}
			}

			return false;
		}

		HRESULT Init(ID3D11Device* pd3dDevice, ID3D11DeviceContext* context, vec3f& voxelExtends, vec3i gridDimensions, vec3i minGridPos, float virtualVoxelSize, unsigned int initialChunkListSize)
		{
			// CPU stuff
			m_voxelExtends = voxelExtends;
			m_gridDimensions = gridDimensions;
			m_VirtualVoxelSize = virtualVoxelSize;
			m_initialChunkDescListSize = initialChunkListSize;

			m_minGridPos = minGridPos;
			m_maxGridPos = -m_minGridPos;

			m_grid.resize(m_gridDimensions.x*m_gridDimensions.y*m_gridDimensions.z, NULL);

			m_bitMask = BitArray<unsigned int>(m_gridDimensions.x*m_gridDimensions.y*m_gridDimensions.z);

			// Shader
			HRESULT hr = S_OK;

			ID3DBlob* pBlob = NULL;

			char SDFBLOCKSIZE[5];
			sprintf_s(SDFBLOCKSIZE, "%d", SDF_BLOCK_SIZE);

			char HANDLECOLLISIONS[5];
			sprintf_s(HANDLECOLLISIONS, "%d", GlobalAppState::getInstance().s_HANDLE_COLLISIONS);
	
			char BLOCK_SIZE_IntegrateFromGlobalHash[5];
			sprintf_s(BLOCK_SIZE_IntegrateFromGlobalHash, "%d", m_blockSizeIntegrateFromGlobalHash);
			D3D_SHADER_MACRO shaderDefine_IntegrateFromGlobalHash[] = { { "groupthreads", BLOCK_SIZE_IntegrateFromGlobalHash }, { "SDF_BLOCK_SIZE", SDFBLOCKSIZE }, {"HANDLE_COLLISIONS", HANDLECOLLISIONS } , { 0 } };
			D3D_SHADER_MACRO shaderDefine_IntegrateFromGlobalHashWithOut[] = { { "groupthreads", BLOCK_SIZE_IntegrateFromGlobalHash }, { "SDF_BLOCK_SIZE", SDFBLOCKSIZE }, { 0 } };

			D3D_SHADER_MACRO* validDefines = shaderDefine_IntegrateFromGlobalHash;
			if(GlobalAppState::getInstance().s_HANDLE_COLLISIONS == 0)
			{
				validDefines = shaderDefine_IntegrateFromGlobalHashWithOut;
			}


			V_RETURN(CompileShaderFromFile(L"Shaders\\IntegrateFromGlobalHash.hlsl", "integrateFromGlobalHashPass1CS", "cs_5_0", &pBlob, validDefines));
			V_RETURN(pd3dDevice->CreateComputeShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &m_pComputeShaderIntegrateFromGlobalHashPass1))
			SAFE_RELEASE(pBlob);

			V_RETURN(CompileShaderFromFile(L"Shaders\\IntegrateFromGlobalHash.hlsl", "integrateFromGlobalHashPass2CS", "cs_5_0", &pBlob, validDefines));
			V_RETURN(pd3dDevice->CreateComputeShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &m_pComputeShaderIntegrateFromGlobalHashPass2))
			SAFE_RELEASE(pBlob);

			char BLOCK_SIZE_ChunkToGlobalHash[5];
			sprintf_s(BLOCK_SIZE_ChunkToGlobalHash, "%d", m_blockSizeChunkToGlobalHash);
			D3D_SHADER_MACRO shaderDefine_ChunkToGlobalHash[] = { { "groupthreads", BLOCK_SIZE_ChunkToGlobalHash }, { "SDF_BLOCK_SIZE", SDFBLOCKSIZE }, {"HANDLE_COLLISIONS", HANDLECOLLISIONS }, { 0 } };
			D3D_SHADER_MACRO shaderDefine_ChunkToGlobalHashWithout[] = { { "groupthreads", BLOCK_SIZE_ChunkToGlobalHash }, { "SDF_BLOCK_SIZE", SDFBLOCKSIZE }, { 0 } };

			D3D_SHADER_MACRO* validDefines2 = shaderDefine_ChunkToGlobalHash;
			if(GlobalAppState::getInstance().s_HANDLE_COLLISIONS == 0)
			{
				validDefines2 = shaderDefine_ChunkToGlobalHashWithout;
			}

			V_RETURN(CompileShaderFromFile(L"Shaders\\ChunkToGlobalHash.hlsl", "chunkToGlobalHashPass1CS", "cs_5_0", &pBlob, validDefines2));
			V_RETURN(pd3dDevice->CreateComputeShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &m_pComputeShaderChunkToGlobalHashPass1))
			SAFE_RELEASE(pBlob);

			V_RETURN(CompileShaderFromFile(L"Shaders\\ChunkToGlobalHash.hlsl", "chunkToGlobalHashPass2CS", "cs_5_0", &pBlob, validDefines2));
			V_RETURN(pd3dDevice->CreateComputeShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &m_pComputeShaderChunkToGlobalHashPass2))
			SAFE_RELEASE(pBlob);

			// Buffer
			D3D11_BUFFER_DESC bDesc;
			bDesc.BindFlags	= D3D11_BIND_CONSTANT_BUFFER;
			bDesc.Usage	= D3D11_USAGE_DYNAMIC;
			bDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
			bDesc.MiscFlags	= 0;

			bDesc.ByteWidth	= sizeof(CBufferIntegrateFromGlobalHash);
			V_RETURN(pd3dDevice->CreateBuffer(&bDesc, NULL, &m_constantBufferIntegrateFromGlobalHash));

			bDesc.ByteWidth	= sizeof(CBufferIntegrateFromGlobalHash);
			V_RETURN(pd3dDevice->CreateBuffer(&bDesc, NULL, &m_constantBufferChunkToGlobalHash));

			// Create Append Buffer Output
			ZeroMemory(&bDesc, sizeof(D3D11_BUFFER_DESC));
			bDesc.BindFlags	= D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
			bDesc.Usage	= D3D11_USAGE_DEFAULT;
			bDesc.CPUAccessFlags = 0;
			bDesc.MiscFlags	= D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
			bDesc.ByteWidth	= 4*sizeof(int)*m_maxNumberOfSDFBlocksIntegrateFromGlobalHash;
			bDesc.StructureByteStride = 4*sizeof(int);
			V_RETURN(pd3dDevice->CreateBuffer(&bDesc, NULL, &m_pSDFBlockDescOutput));

			D3D11_SHADER_RESOURCE_VIEW_DESC descSRV;
			ZeroMemory (&descSRV, sizeof(D3D11_SHADER_RESOURCE_VIEW_DESC));
			descSRV.Format = DXGI_FORMAT_UNKNOWN;
			descSRV.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
			descSRV.Buffer.FirstElement = 0;
			descSRV.Buffer.NumElements = m_maxNumberOfSDFBlocksIntegrateFromGlobalHash;
			V_RETURN(pd3dDevice->CreateShaderResourceView(m_pSDFBlockDescOutput, &descSRV, &m_pSDFBlockDescOutputSRV)); 

			D3D11_UNORDERED_ACCESS_VIEW_DESC descUAV;
			ZeroMemory(&descUAV, sizeof(D3D11_UNORDERED_ACCESS_VIEW_DESC));
			descUAV.Format = DXGI_FORMAT_UNKNOWN;
			descUAV.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
			descUAV.Buffer.FirstElement = 0;
			descUAV.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_APPEND;
			descUAV.Buffer.NumElements =  m_maxNumberOfSDFBlocksIntegrateFromGlobalHash;
			V_RETURN(pd3dDevice->CreateUnorderedAccessView(m_pSDFBlockDescOutput, &descUAV, &m_pSDFBlockDescOutputUAV)); 

			// Create Append Buffer Input
			ZeroMemory(&bDesc, sizeof(D3D11_BUFFER_DESC));
			bDesc.BindFlags	= D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
			bDesc.Usage	= D3D11_USAGE_DEFAULT;
			bDesc.CPUAccessFlags = 0;
			bDesc.MiscFlags	= D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
			bDesc.ByteWidth	= 4*sizeof(int)*m_maxNumberOfSDFBlocksIntegrateFromGlobalHash;
			bDesc.StructureByteStride = 4*sizeof(int);
			V_RETURN(pd3dDevice->CreateBuffer(&bDesc, NULL, &m_pSDFBlockDescInput));

			//D3D11_SHADER_RESOURCE_VIEW_DESC descSRV;
			ZeroMemory (&descSRV, sizeof(D3D11_SHADER_RESOURCE_VIEW_DESC));
			descSRV.Format = DXGI_FORMAT_UNKNOWN;
			descSRV.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
			descSRV.Buffer.FirstElement = 0;
			descSRV.Buffer.NumElements = m_maxNumberOfSDFBlocksIntegrateFromGlobalHash;
			V_RETURN(pd3dDevice->CreateShaderResourceView(m_pSDFBlockDescInput, &descSRV, &m_pSDFBlockDescInputSRV)); 

			//D3D11_UNORDERED_ACCESS_VIEW_DESC descUAV;
			ZeroMemory(&descUAV, sizeof(D3D11_UNORDERED_ACCESS_VIEW_DESC));
			descUAV.Format = DXGI_FORMAT_UNKNOWN;
			descUAV.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
			descUAV.Buffer.FirstElement = 0;
			descUAV.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_APPEND;
			descUAV.Buffer.NumElements =  m_maxNumberOfSDFBlocksIntegrateFromGlobalHash;
			V_RETURN(pd3dDevice->CreateUnorderedAccessView(m_pSDFBlockDescInput, &descUAV, &m_pSDFBlockDescInputUAV)); 
			
			// Create Append Buffer CPU Ouput
			bDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
			bDesc.BindFlags = 0;
			bDesc.Usage = D3D11_USAGE_STAGING;
			V_RETURN(pd3dDevice->CreateBuffer(&bDesc, NULL, &m_pSDFBlockDescOutputCPU));

			// Create Append Buffer GPU Input
			bDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
			bDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
			bDesc.Usage = D3D11_USAGE_DYNAMIC;
			bDesc.MiscFlags	= 0;
			V_RETURN(pd3dDevice->CreateBuffer(&bDesc, NULL, &m_pSDFBlockDescInputCPU));

			unsigned int linBlockSize = 2*SDF_BLOCK_SIZE*SDF_BLOCK_SIZE*SDF_BLOCK_SIZE;

			// Create SDFBlock output buffer
			ZeroMemory(&bDesc, sizeof(D3D11_BUFFER_DESC));
			bDesc.BindFlags	= D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
			bDesc.Usage	= D3D11_USAGE_DEFAULT;
			bDesc.CPUAccessFlags = 0;
			bDesc.MiscFlags	= 0;
			bDesc.ByteWidth	= sizeof(int)*linBlockSize*m_maxNumberOfSDFBlocksIntegrateFromGlobalHash;
			V_RETURN(pd3dDevice->CreateBuffer(&bDesc, NULL, &m_pSDFBlockOutput));
		
			ZeroMemory (&descSRV, sizeof(D3D11_SHADER_RESOURCE_VIEW_DESC));
			descSRV.Format = DXGI_FORMAT_R32_SINT;
			descSRV.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
			descSRV.Buffer.FirstElement = 0;
			descSRV.Buffer.NumElements = linBlockSize*m_maxNumberOfSDFBlocksIntegrateFromGlobalHash;
			V_RETURN(pd3dDevice->CreateShaderResourceView(m_pSDFBlockOutput, &descSRV, &m_pSDFBlockOutputSRV));

			D3D11_UNORDERED_ACCESS_VIEW_DESC UAVDesc;
			ZeroMemory(&UAVDesc, sizeof(D3D11_UNORDERED_ACCESS_VIEW_DESC));
			UAVDesc.Format = DXGI_FORMAT_R32_SINT;
			UAVDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
			UAVDesc.Buffer.FirstElement = 0;
			UAVDesc.Buffer.Flags = 0;
			UAVDesc.Buffer.NumElements = linBlockSize*m_maxNumberOfSDFBlocksIntegrateFromGlobalHash;
			V_RETURN(pd3dDevice->CreateUnorderedAccessView(m_pSDFBlockOutput, &UAVDesc, &m_pSDFBlockOutputUAV));

			// Create SDFBlock input buffer
			ZeroMemory(&bDesc, sizeof(D3D11_BUFFER_DESC));
			bDesc.BindFlags	= D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
			bDesc.Usage	= D3D11_USAGE_DEFAULT;
			bDesc.CPUAccessFlags = 0;
			bDesc.MiscFlags	= 0;
			bDesc.ByteWidth	= sizeof(int)*linBlockSize*m_maxNumberOfSDFBlocksIntegrateFromGlobalHash;
			V_RETURN(pd3dDevice->CreateBuffer(&bDesc, NULL, &m_pSDFBlockInput));
		
			ZeroMemory (&descSRV, sizeof(D3D11_SHADER_RESOURCE_VIEW_DESC));
			descSRV.Format = DXGI_FORMAT_R32_SINT;
			descSRV.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
			descSRV.Buffer.FirstElement = 0;
			descSRV.Buffer.NumElements = linBlockSize*m_maxNumberOfSDFBlocksIntegrateFromGlobalHash;
			V_RETURN(pd3dDevice->CreateShaderResourceView(m_pSDFBlockInput, &descSRV, &m_pSDFBlockInputSRV));

			//D3D11_UNORDERED_ACCESS_VIEW_DESC UAVDesc;
			ZeroMemory(&UAVDesc, sizeof(D3D11_UNORDERED_ACCESS_VIEW_DESC));
			UAVDesc.Format = DXGI_FORMAT_R32_SINT;
			UAVDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
			UAVDesc.Buffer.FirstElement = 0;
			UAVDesc.Buffer.Flags = 0;
			UAVDesc.Buffer.NumElements = linBlockSize*m_maxNumberOfSDFBlocksIntegrateFromGlobalHash;
			V_RETURN(pd3dDevice->CreateUnorderedAccessView(m_pSDFBlockInput, &UAVDesc, &m_pSDFBlockInputUAV));

			// Create Output Buffer CPU
			bDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
			bDesc.BindFlags = 0;
			bDesc.Usage = D3D11_USAGE_STAGING;
			V_RETURN(pd3dDevice->CreateBuffer(&bDesc, NULL, &m_pSDFBlockOutputCPU));

			// Create Input Buffer GPU
			bDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
			bDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
			bDesc.Usage = D3D11_USAGE_DYNAMIC;
			bDesc.MiscFlags	= 0;
			V_RETURN(pd3dDevice->CreateBuffer(&bDesc, NULL, &m_pSDFBlockInputCPU));

			// Create Output Count Buffer 
			ZeroMemory(&bDesc, sizeof(D3D11_BUFFER_DESC));
			bDesc.BindFlags	= 0;
			bDesc.Usage	= D3D11_USAGE_STAGING;
			bDesc.CPUAccessFlags =  D3D11_CPU_ACCESS_READ;
			bDesc.MiscFlags	= 0;
			bDesc.ByteWidth	= sizeof(int);
			V_RETURN(pd3dDevice->CreateBuffer(&bDesc, NULL, &m_BuffCountSDFBlockOutput));

			// Create BitMask buffer
			ZeroMemory(&bDesc, sizeof(D3D11_BUFFER_DESC));
			bDesc.BindFlags	= D3D11_BIND_SHADER_RESOURCE;
			bDesc.Usage	= D3D11_USAGE_DEFAULT;
			bDesc.CPUAccessFlags = 0;
			bDesc.MiscFlags	= 0;
			bDesc.ByteWidth	= m_bitMask.getByteWidth();
			V_RETURN(pd3dDevice->CreateBuffer(&bDesc, NULL, &m_pBitMask));

			bDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
			bDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
			bDesc.Usage = D3D11_USAGE_DYNAMIC;
			bDesc.MiscFlags	= 0;
			V_RETURN(pd3dDevice->CreateBuffer(&bDesc, NULL, &m_pBitMaskCPU));

			ZeroMemory(&descSRV, sizeof(D3D11_SHADER_RESOURCE_VIEW_DESC));
			descSRV.Format = DXGI_FORMAT_R32_UINT;
			descSRV.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
			descSRV.Buffer.FirstElement = 0;
			descSRV.Buffer.NumElements = m_bitMask.getByteWidth()/sizeof(unsigned int);
			V_RETURN(pd3dDevice->CreateShaderResourceView(m_pBitMask, &descSRV, &m_pBitMaskSRV));

			V_RETURN(startMultiThreading(context));

			return  hr;
		}

		void Destroy(ID3D11DeviceContext* context)
		{
			stopMultiThreading();

			clearGrid();
		
			UnmapOutput(context);
			UnmapInput(context);
			
			SAFE_RELEASE(m_pSDFBlockDescOutput);
			SAFE_RELEASE(m_pSDFBlockDescOutputUAV);
			SAFE_RELEASE(m_pSDFBlockDescOutputSRV);

			SAFE_RELEASE(m_pSDFBlockDescInput);
			SAFE_RELEASE(m_pSDFBlockDescInputUAV);
			SAFE_RELEASE(m_pSDFBlockDescInputSRV);

			SAFE_RELEASE(m_pSDFBlockInputCPU);

			SAFE_RELEASE(m_pSDFBlockDescOutputCPU);
			SAFE_RELEASE(m_pSDFBlockDescInputCPU);
			
			SAFE_RELEASE(m_constantBufferIntegrateFromGlobalHash);
			SAFE_RELEASE(m_constantBufferChunkToGlobalHash);

			SAFE_RELEASE(m_pComputeShaderIntegrateFromGlobalHashPass1);
			SAFE_RELEASE(m_pComputeShaderIntegrateFromGlobalHashPass2);

			SAFE_RELEASE(m_pComputeShaderChunkToGlobalHashPass1);
			SAFE_RELEASE(m_pComputeShaderChunkToGlobalHashPass2);

			SAFE_RELEASE(m_pSDFBlockOutput);
			SAFE_RELEASE(m_pSDFBlockOutputUAV);
			SAFE_RELEASE(m_pSDFBlockOutputSRV);

			SAFE_RELEASE(m_pSDFBlockInput);
			SAFE_RELEASE(m_pSDFBlockInputUAV);
			SAFE_RELEASE(m_pSDFBlockInputSRV);

			SAFE_RELEASE(m_pSDFBlockOutputCPU);
			SAFE_RELEASE(m_BuffCountSDFBlockOutput);

			SAFE_RELEASE(m_pBitMask);
			SAFE_RELEASE(m_pBitMaskCPU);
			SAFE_RELEASE(m_pBitMaskSRV);
		}

		vec3i& getMinGridPos()
		{
			return m_minGridPos;
		}

		vec3i& getMaxGridPos()
		{
			return m_maxGridPos;
		}

		vec3f& getVoxelExtends()
		{
			return m_voxelExtends;
		}

		float getVirtualVoxelSize()
		{
			return m_VirtualVoxelSize;
		}

		vec3f getWorldPosChunk(vec3i& chunk)
		{
			return chunkToWorld(chunk);
		}		

		void printStatistics()
		{
			unsigned int nChunks = (unsigned int)m_grid.size();
			
			unsigned int nSDFBlocks = 0;
			for(unsigned int i = 0; i<nChunks; i++)
			{
				if(m_grid[i] != NULL)
				{
					nSDFBlocks+=m_grid[i]->getNElements();
				}
			}

			std::cout << "Total number of Blocks on the CPU: " << nSDFBlocks << std::endl;
		}

		void setPositionAndRadius(vec3f position, float radius, bool multiThreaded)
		{
			if(multiThreaded)
			{
				WaitForSingleObject(hEventSetTransformProduce, INFINITE);
				WaitForSingleObject(hMutexSetTransform, INFINITE);
			}

			s_posCamera = position;
			s_radius = radius;

			if(multiThreaded)
			{
				SetEvent(hEventSetTransformConsume);
				ReleaseMutex(hMutexSetTransform);
			}
		}



		HRESULT DumpVoxelHash(ID3D11DeviceContext* context, const std::string &filename, DX11SceneRepHashSDF& hash, const vec3f& camPos, float radius, float dumpRadius = 0.0f, vec3f dumpCenter = vec3f(0.0f,0.0f,0.0f)) {
			HRESULT hr = S_OK;

			if (dumpRadius < 0.0f) dumpRadius = 0.0f;

			struct HashEntry
			{
				point3d<short> pos;		//hash position (lower left corner of SDFBlock))
				unsigned short offset;	//offset for collisions
				int ptr;				//pointer into heap to SDFBlock
			};
			struct Voxel
			{
				float sdf;
				vec3uc color;
				unsigned char weight;
			};
			struct VoxelBlock 
			{
				Voxel voxels[SDF_BLOCK_SIZE*SDF_BLOCK_SIZE*SDF_BLOCK_SIZE];	//typically of size 512
			};

			unsigned int occupiedBlocks = 0;
			SparseGrid3D<VoxelBlock> grid;

			vec3i minGridPos = getMinGridPos();
			vec3i maxGridPos = getMaxGridPos();
			
			V_RETURN(StreamOutToCPUAll(context, hash));

			for (int x = minGridPos.x; x<maxGridPos.x; x+=1)	{
				for (int y = minGridPos.y; y<maxGridPos.y; y+=1)	{
					for (int z = minGridPos.z; z<maxGridPos.z; z+=1)	{

						vec3i chunk(x, y, z);
						if (containsSDFBlocksChunk(chunk))	{
							std::cout << "Dump Hash on chunk (" << x << ", " << y << ", " << z << ") " << std::endl;

							vec3f& chunkCenter = getWorldPosChunk(chunk);
							vec3f& voxelExtends = getVoxelExtends();
							float virtualVoxelSize = getVirtualVoxelSize();

							vec3f minCorner = chunkCenter-voxelExtends/2.0f-vec3f(virtualVoxelSize, virtualVoxelSize, virtualVoxelSize)*SDF_BLOCK_SIZE;
							vec3f maxCorner = chunkCenter+voxelExtends/2.0f+vec3f(virtualVoxelSize, virtualVoxelSize, virtualVoxelSize)*SDF_BLOCK_SIZE;


							unsigned int index = linearizeChunkPos(vec3i(x, y, z));
							if(m_grid[index] != NULL && m_grid[index]->isStreamedOut()) // As been allocated and has streamed out blocks
							{
								ChunkDesc* chunkDesc = m_grid[index];
								for (size_t i = 0; i < chunkDesc->m_ChunkDesc.size(); i++) {
									SDFDesc& sdfDesc = chunkDesc->m_ChunkDesc[i];
									const unsigned int ptr = sdfDesc.ptr;

									//if (ptr != -2) {
										VoxelBlock vBlock;
										//memcpy(vBlock.voxels, &voxels[ptr], sizeof(VoxelBlock));
										for (unsigned int j = 0; j < SDF_BLOCK_SIZE*SDF_BLOCK_SIZE*SDF_BLOCK_SIZE; j++) {
											int first = chunkDesc->m_SDFBlocks[i].data[2*j+0];
											vBlock.voxels[j].sdf = *(float*)(&first);
											int last = chunkDesc->m_SDFBlocks[i].data[2*j+1];
											vBlock.voxels[j].weight = last & 0x000000ff;
											last >>= 0x8;
											vBlock.voxels[j].color.x = last & 0x000000ff;
											last >>= 0x8;
											vBlock.voxels[j].color.y = last & 0x000000ff;
											last >>= 0x8;
											vBlock.voxels[j].color.z = last & 0x000000ff;

											//std::cout << vBlock.voxels[j].sdf << std::endl;
										}

										vec3i coord(sdfDesc.pos.x, sdfDesc.pos.y, sdfDesc.pos.z);

										//std::cout << coord << std::endl;

										if (dumpRadius == 0.0f) {
											grid(coord) = vBlock;
										} else {
											//vec3f center = GetLastRigidTransform()*dumpCenter;
											vec3f center = dumpCenter;
											vec3f coordf = vec3f(coord*SDF_BLOCK_SIZE)*m_VirtualVoxelSize;
											if (vec3f::dist(center, coordf) <= dumpRadius) {
												grid(coord) = vBlock;
											}					
										}

										occupiedBlocks++;
									//}
								}
							}


							//V_RETURN(chunkGrid.StreamInToGPUChunkNeighborhood(context, hash, chunk, 1));
							//V_RETURN(DX11MarchingCubesHashSDF::extractIsoSurface(context, hash.GetHashSRV(), hash.GetSDFBlocksSDFSRV(), hash.GetSDFBlocksRGBWSRV(), hash.MapAndGetConstantBuffer(context), hash.GetHashNumBuckets(), hash.GetHashBucketSize(), minCorner, maxCorner, true));
							//V_RETURN(chunkGrid.StreamOutToCPUAll(context, hash));
						}
					}
				}
			}		
			
			std::cout << "found " << occupiedBlocks << " voxel blocks   "; 
			grid.writeBinaryDump(filename);

			unsigned int nStreamedBlock;
			V_RETURN(StreamInToGPUAll(context, hash, camPos, radius, true, nStreamedBlock));

			return hr;
		}
	private:

		//-------------------------------------------------------
		// Helper
		//-------------------------------------------------------

		bool isValidChunk(vec3i chunk)
		{
			if(chunk.x < m_minGridPos.x || chunk.y < m_minGridPos.y || chunk.z < m_minGridPos.z) return false;
			if(chunk.x > m_maxGridPos.x || chunk.y > m_maxGridPos.y || chunk.z > m_maxGridPos.z) return false;

			return true;
		}
		
		float getChunkRadiusInMeter()
		{
			return m_voxelExtends.length()/2.0f;
		}

		float getGridRadiusInMeter()
		{
			vec3f minPos = chunkToWorld(m_minGridPos)-m_voxelExtends/2.0f;
			vec3f maxPos = chunkToWorld(m_maxGridPos)+m_voxelExtends/2.0f;

			return (minPos-maxPos).length()/2.0f;
		}

		vec3f numberOfChunksToMeter(vec3i c)
		{
			return vec3f(c.x*m_voxelExtends.x, c.y*m_voxelExtends.y, c.z*m_voxelExtends.z);
		}

		vec3f meterToNumberOfChunks(float f)
		{
			return vec3f(f/m_voxelExtends.x, f/m_voxelExtends.y, f/m_voxelExtends.z);
		}

		vec3i meterToNumberOfChunksCeil(float f)
		{
			return vec3i((int)ceil(f/m_voxelExtends.x), (int)ceil(f/m_voxelExtends.y), (int)ceil(f/m_voxelExtends.z));
		}

		vec3i worldToChunks(vec3f posWorld)
		{
			vec3f p;
			p.x = posWorld.x/m_voxelExtends.x;
			p.y = posWorld.y/m_voxelExtends.y;
			p.z = posWorld.z/m_voxelExtends.z;

			vec3f s;
			s.x = (float)math::sign(p.x);
			s.y = (float)math::sign(p.y);
			s.z = (float)math::sign(p.z);

			return vec3i(p+s*0.5f);
		}

		vec3f getChunkCenter(vec3i chunk)
		{
			return chunkToWorld(chunk);
		}

		vec3f chunkToWorld(vec3i posChunk)
		{
			vec3f res;
			res.x = posChunk.x*m_voxelExtends.x;
			res.y = posChunk.y*m_voxelExtends.y;
			res.z = posChunk.z*m_voxelExtends.z;

			return res;
		}

		vec3i delinearizeChunkIndex(unsigned int idx)
		{
			unsigned int x = idx % m_gridDimensions.x;
			unsigned int y = (idx % (m_gridDimensions.x * m_gridDimensions.y)) / m_gridDimensions.x;
			unsigned int z = idx / (m_gridDimensions.x * m_gridDimensions.y);
						
			return m_minGridPos+vec3i(x,y,z);
		}

		unsigned int linearizeChunkPos(vec3i chunkPos)
		{
			vec3ui p = chunkPos-m_minGridPos;

			return  p.z * m_gridDimensions.x * m_gridDimensions.y +
					p.y * m_gridDimensions.x +
					p.x;
		}

		// Mutex
		void deleteCritialSection()
		{
			CloseHandle(hMutexOut);
			CloseHandle(hEventOutProduce);
			CloseHandle(hEventOutConsume);

			CloseHandle(hMutexIn);
			CloseHandle(hEventInProduce);
			CloseHandle(hEventInConsume);

			CloseHandle(hMutexSetTransform);
			CloseHandle(hEventSetTransformProduce);
			CloseHandle(hEventSetTransformConsume);
		}

		void initializeCriticalSection()
		{
			hMutexOut = CreateMutex(NULL, FALSE, NULL);
			hEventOutProduce = CreateEvent(NULL, FALSE, TRUE, NULL);
			hEventOutConsume = CreateEvent(NULL, FALSE, FALSE, NULL);

			hMutexIn = CreateMutex(NULL, FALSE, NULL);
			hEventInProduce = CreateEvent(NULL, FALSE, TRUE, NULL);
			hEventInConsume = CreateEvent(NULL, FALSE, FALSE, NULL); //! has to be TRUE if stream out and in calls are splitted !!!

			hMutexSetTransform = CreateMutex(NULL, FALSE, NULL);
			hEventSetTransformProduce = CreateEvent(NULL, FALSE, TRUE, NULL);
			hEventSetTransformConsume = CreateEvent(NULL, FALSE, FALSE, NULL);
		}

		HRESULT initializeMappedState(ID3D11DeviceContext* context)
		{
			HRESULT hr = S_OK;

			UnmapInput(context);
			UnmapOutput(context);

			// Map for correct start state;
			V_RETURN(context->Map(m_pSDFBlockInputCPU, 0, D3D11_MAP_WRITE_DISCARD, 0, &m_mappedResourceBlockStreamIn));
			V_RETURN(context->Map(m_pSDFBlockDescInputCPU, 0, D3D11_MAP_WRITE_DISCARD, 0, &m_mappedResourceDescStreamIn));

			V_RETURN(context->Map(m_pSDFBlockOutputCPU, 0, D3D11_MAP_READ, 0, &m_mappedResourceBlockStreamOut));
			V_RETURN(context->Map(m_pSDFBlockDescOutputCPU, 0, D3D11_MAP_READ, 0, &m_mappedResourceDescStreamOut));

			return hr;
		}

		// Map Unmap wrapper
		void UnmapOutput(ID3D11DeviceContext* context)
		{
			if(m_mappedResourceDescStreamOut.pData != NULL)
			{
				context->Unmap(m_pSDFBlockDescOutputCPU, 0);
				m_mappedResourceDescStreamOut.pData = NULL;
			}

			if(m_mappedResourceBlockStreamOut.pData != NULL)
			{
				context->Unmap(m_pSDFBlockOutputCPU, 0);
				m_mappedResourceBlockStreamOut.pData = NULL;
			}
		}

		void UnmapInput(ID3D11DeviceContext* context)
		{
			if(m_mappedResourceDescStreamIn.pData != NULL)
			{
				context->Unmap(m_pSDFBlockDescInputCPU, 0);
				m_mappedResourceDescStreamIn.pData = NULL;
			}

			if(m_mappedResourceBlockStreamIn.pData != NULL)
			{
				context->Unmap(m_pSDFBlockInputCPU, 0);
				m_mappedResourceBlockStreamIn.pData = NULL;
			}
		}

		//-------------------------------------------------------
		// Integrate from Global Hash
		//-------------------------------------------------------
		
		struct CBufferIntegrateFromGlobalHash
		{
			unsigned int nSDFBlockDescs;
			float radius;
			unsigned int start;
			unsigned int aling1;

			D3DXVECTOR3 cameraPosition;
			unsigned int aling2;
		};

		struct CBufferChunkToGlobalHash
		{
			unsigned int nSDFBlockDescs;
			unsigned int heapFreeCountPrev;
			unsigned int heapFreeCountNow;
			unsigned int aling1;
		};

		unsigned int m_blockSizeIntegrateFromGlobalHash;
		unsigned int m_blockSizeChunkToGlobalHash;

		unsigned int m_maxNumberOfSDFBlocksIntegrateFromGlobalHash;

		ID3D11Buffer* m_constantBufferIntegrateFromGlobalHash;
		ID3D11Buffer* m_constantBufferChunkToGlobalHash;

		ID3D11Buffer* m_pSDFBlockInputCPU;

		ID3D11Buffer* m_pSDFBlockDescOutputCPU;
		ID3D11Buffer* m_pSDFBlockDescInputCPU;

		ID3D11ComputeShader* m_pComputeShaderIntegrateFromGlobalHashPass1;
		ID3D11ComputeShader* m_pComputeShaderIntegrateFromGlobalHashPass2;

		ID3D11ComputeShader* m_pComputeShaderChunkToGlobalHashPass1;
		ID3D11ComputeShader* m_pComputeShaderChunkToGlobalHashPass2;

		ID3D11Buffer* m_pSDFBlockDescOutput;
		ID3D11UnorderedAccessView* m_pSDFBlockDescOutputUAV;
		ID3D11ShaderResourceView* m_pSDFBlockDescOutputSRV;

		ID3D11Buffer* m_pSDFBlockDescInput;
		ID3D11UnorderedAccessView* m_pSDFBlockDescInputUAV;
		ID3D11ShaderResourceView* m_pSDFBlockDescInputSRV;

		ID3D11Buffer* m_pSDFBlockOutput;
		ID3D11UnorderedAccessView* m_pSDFBlockOutputUAV;
		ID3D11ShaderResourceView* m_pSDFBlockOutputSRV;

		ID3D11Buffer* m_pSDFBlockInput;
		ID3D11UnorderedAccessView* m_pSDFBlockInputUAV;
		ID3D11ShaderResourceView* m_pSDFBlockInputSRV;


		ID3D11Buffer* m_pSDFBlockOutputCPU;
		ID3D11Buffer* m_BuffCountSDFBlockOutput;

		ID3D11Buffer* m_pBitMask;
		ID3D11Buffer* m_pBitMaskCPU;
		ID3D11ShaderResourceView* m_pBitMaskSRV;
		
		//-------------------------------------------------------
		// Chunk Grid
		//-------------------------------------------------------
		
		vec3f m_voxelExtends;		// extend of the voxels in meters
		vec3i m_gridDimensions;	    // number of voxels in each dimension

		vec3i m_minGridPos;
		vec3i m_maxGridPos;

		float m_VirtualVoxelSize;   // Resolution of the virtual voxel grid on the GPU

		unsigned int m_initialChunkDescListSize;	 // Inital size for vectors in the ChunkDesc
	
		std::vector<ChunkDesc*> m_grid; // Grid data
		BitArray<unsigned int> m_bitMask;

		unsigned int m_currentPart;
		unsigned int m_streamOutParts;

		// Multi-threading
		HANDLE hStreamingThread;
		DWORD dwStreamingThreadID;

		// Mutex
		HANDLE hMutexOut;
		HANDLE hEventOutProduce;
		HANDLE hEventOutConsume;

		HANDLE hMutexIn;
		HANDLE hEventInProduce;
		HANDLE hEventInConsume;

		HANDLE hMutexSetTransform;
		HANDLE hEventSetTransformProduce;
		HANDLE hEventSetTransformConsume;
	
		// Mapped Data StreamOut
		D3D11_MAPPED_SUBRESOURCE m_mappedResourceDescStreamOut;
		D3D11_MAPPED_SUBRESOURCE m_mappedResourceBlockStreamOut;

		// Mapped Data StreamIn
		D3D11_MAPPED_SUBRESOURCE m_mappedResourceDescStreamIn;
		D3D11_MAPPED_SUBRESOURCE m_mappedResourceBlockStreamIn;

		Timer m_timer;

	public:

		static vec3f s_posCamera;
		static float s_radius;
		static unsigned int s_nStreamdInBlocks;
		static unsigned int s_nStreamdOutBlocks;
		static bool s_terminateThread;
};
