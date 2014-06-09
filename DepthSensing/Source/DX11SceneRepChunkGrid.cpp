#include "stdafx.h"

#include "DX11SceneRepChunkGrid.h"
#include <list>

vec3f DX11SceneRepChunkGrid::s_posCamera;
float DX11SceneRepChunkGrid::s_radius;
unsigned int DX11SceneRepChunkGrid::s_nStreamdInBlocks = 0;
unsigned int DX11SceneRepChunkGrid::s_nStreamdOutBlocks = 0;
bool DX11SceneRepChunkGrid::s_terminateThread = false;

// Worker Thread
extern DX11SceneRepChunkGrid g_SceneRepChunkGrid;

LONG WINAPI StreamingFunc(LONG lParam)
{
	while(true)
	{
		//std::cout <<" Shouldnt run" << std::endl;
		HRESULT hr = S_OK;

		hr = g_SceneRepChunkGrid.StreamOutToCPUPass1CPU(DXUTGetD3D11DeviceContext(), true);
		hr = g_SceneRepChunkGrid.StreamInToGPUPass0CPU(DXUTGetD3D11DeviceContext(), DX11SceneRepChunkGrid::s_posCamera, DX11SceneRepChunkGrid::s_radius, true);

		if(DX11SceneRepChunkGrid::s_terminateThread)
		{
			return 0;
		}
	}

	return 0;
}

HRESULT DX11SceneRepChunkGrid::StreamOutToCPUAll(ID3D11DeviceContext* context, DX11SceneRepHashSDF& sceneRepHashSDF)
{
	HRESULT hr = S_OK;

	unsigned int nStreamedBlocksSum = 1;
	while(nStreamedBlocksSum != 0)
	{
		nStreamedBlocksSum = 0;
		for(unsigned int i = 0; i<m_streamOutParts; i++)
		{
			unsigned int nStreamedBlocks;
			V_RETURN(StreamOutToCPU(context, sceneRepHashSDF, worldToChunks(m_minGridPos-vec3i(1, 1, 1)), 0.0f, true, nStreamedBlocks));

			nStreamedBlocksSum += nStreamedBlocks;
		}
	}

	return hr;
}

HRESULT DX11SceneRepChunkGrid::StreamOutToCPUPass0GPU(ID3D11DeviceContext* context, DX11SceneRepHashSDF& sceneRepHashSDF, const vec3f& posCamera, float radius, bool useParts, bool multiThreaded)
{
	HRESULT hr = S_OK;

	if(multiThreaded)
	{
		WaitForSingleObject(hEventOutProduce, INFINITE);
		WaitForSingleObject(hMutexOut, INFINITE);
	}
    
	s_posCamera = posCamera;
	s_radius = radius;

	UnmapOutput(context);

	ID3D11UnorderedAccessView* hashUAV = sceneRepHashSDF.GetHashUAV();
	ID3D11UnorderedAccessView* sdfBlocksSDFUAV = sceneRepHashSDF.GetSDFBlocksSDFUAV();
	ID3D11UnorderedAccessView* sdfBlocksRGBWUAV = sceneRepHashSDF.GetSDFBlocksRGBWUAV();
	ID3D11UnorderedAccessView* heapAppendUAV = sceneRepHashSDF.GetHeapUAV();
	ID3D11UnorderedAccessView* hashBucketMutex = sceneRepHashSDF.GetAndClearHashBucketMutex(context);
	ID3D11Buffer* CBsceneRepSDF = sceneRepHashSDF.MapAndGetConstantBuffer(context);
	unsigned int hashNumBuckets = sceneRepHashSDF.GetHashNumBuckets();
	unsigned int hashBucketSize = sceneRepHashSDF.GetHashBucketSize();

	//-------------------------------------------------------
	// Pass 1: Find all SDFBlocks that have to be transfered
	//-------------------------------------------------------

	unsigned int threadsPerPart = (hashNumBuckets*hashBucketSize + m_streamOutParts - 1) / m_streamOutParts;
	if(!useParts) threadsPerPart = hashNumBuckets*hashBucketSize;

	// Initialize constant buffer
	D3D11_MAPPED_SUBRESOURCE mappedResource;
	V_RETURN(context->Map(m_constantBufferIntegrateFromGlobalHash, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource));
	CBufferIntegrateFromGlobalHash *cbuffer = (CBufferIntegrateFromGlobalHash*)mappedResource.pData;

		cbuffer->nSDFBlockDescs = 0;
		cbuffer->radius = radius;
		cbuffer->start = m_currentPart*threadsPerPart;
		if(!useParts) cbuffer->start = 0;

		memcpy(cbuffer->cameraPosition, posCamera.array, 3*sizeof(float));
	
	context->Unmap(m_constantBufferIntegrateFromGlobalHash, 0);

	// Setup pipeline
	unsigned int initialCountReset = 0;
	unsigned int initialCountKeep = -1;
	context->CSSetUnorderedAccessViews(0, 1, &m_pSDFBlockDescOutputUAV, &initialCountReset);
	context->CSSetUnorderedAccessViews(1, 1, &hashUAV, 0);
	context->CSSetUnorderedAccessViews(2, 1, &heapAppendUAV, &initialCountKeep);
	context->CSSetUnorderedAccessViews(3, 1, &hashBucketMutex, NULL);
	context->CSSetConstantBuffers(0, 1, &CBsceneRepSDF);
	context->CSSetConstantBuffers(1, 1, &m_constantBufferIntegrateFromGlobalHash);
	ID3D11Buffer* CBGlobalAppState = GlobalAppState::getInstance().MapAndGetConstantBuffer(context);
	context->CSSetConstantBuffers(8, 1, &CBGlobalAppState);
	context->CSSetShader(m_pComputeShaderIntegrateFromGlobalHashPass1, 0, 0);
	
	// Run compute shader
	unsigned int dimX = NUM_GROUPS_X;
	unsigned int dimY = ((threadsPerPart + m_blockSizeIntegrateFromGlobalHash - 1)/m_blockSizeIntegrateFromGlobalHash + NUM_GROUPS_X - 1) / NUM_GROUPS_X;
	assert(dimX <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);
	assert(dimY <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);

	context->Dispatch(dimX, dimY, 1);

	// Cleanup
	ID3D11UnorderedAccessView* nullUAV[] = {NULL, NULL, NULL, NULL};
	ID3D11Buffer* nullB[2] = {NULL, NULL};

	context->CSSetUnorderedAccessViews(0, 4, nullUAV, 0);
	context->CSSetConstantBuffers(0, 2, nullB);
	context->CSSetConstantBuffers(8, 1, nullB);
	context->CSSetShader(0, 0, 0);

	context->CopyStructureCount(m_BuffCountSDFBlockOutput, 0, m_pSDFBlockDescOutputUAV);
	hr = (context->Map(m_BuffCountSDFBlockOutput, 0, D3D11_MAP_READ, 0, &mappedResource));
	V_RETURN(hr);
	unsigned int nSDFBlockDescs = ((unsigned int*)mappedResource.pData)[0];
	context->Unmap(m_BuffCountSDFBlockOutput, 0);

	if(useParts) m_currentPart = (m_currentPart+1) % m_streamOutParts;

	if(nSDFBlockDescs != 0)
	{
		//std::cout << "SDFBlocks streamed out: " << nSDFBlockDescs << std::endl;

		//-------------------------------------------------------
		// Pass 2: Copy SDFBlocks to output buffer
		//-------------------------------------------------------

		// Initialize constant buffer
		V_RETURN(context->Map(m_constantBufferIntegrateFromGlobalHash, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource));
		cbuffer = (CBufferIntegrateFromGlobalHash*)mappedResource.pData;
	
			cbuffer->nSDFBlockDescs = nSDFBlockDescs;

		context->Unmap(m_constantBufferIntegrateFromGlobalHash, 0);

		// Setup pipeline
		context->CSSetUnorderedAccessViews(0, 1, &sdfBlocksSDFUAV, 0);
		context->CSSetUnorderedAccessViews(1, 1, &m_pSDFBlockOutputUAV, 0);
		context->CSSetUnorderedAccessViews(2, 1, &sdfBlocksRGBWUAV, 0);
		context->CSSetShaderResources(0, 1, &m_pSDFBlockDescOutputSRV);
		context->CSSetConstantBuffers(0, 1, &CBsceneRepSDF);
		context->CSSetConstantBuffers(1, 1, &m_constantBufferIntegrateFromGlobalHash);
		context->CSSetConstantBuffers(8, 1, &CBGlobalAppState);
		context->CSSetShader(m_pComputeShaderIntegrateFromGlobalHashPass2, 0, 0);

		// Run compute shader
		dimX = NUM_GROUPS_X;
		dimY = (nSDFBlockDescs + NUM_GROUPS_X - 1) / NUM_GROUPS_X;
		assert(dimX <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);
		assert(dimY <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);

		context->Dispatch(dimX, dimY, 1);

		// Cleanup
		ID3D11UnorderedAccessView* nullUAV2[] = {NULL, NULL, NULL};
		ID3D11ShaderResourceView* nullSRV[1] = {NULL};
		ID3D11Buffer* nullB2[2] = {NULL, NULL};

		context->CSSetUnorderedAccessViews(0, 3, nullUAV2, 0);
		context->CSSetShaderResources(0, 1, nullSRV);
		context->CSSetConstantBuffers(0, 2, nullB2);
		context->CSSetConstantBuffers(8, 1, nullB2);
		context->CSSetShader(0, 0, 0);
			
		// Copy to CPU and Integrate
		const unsigned int linBlockSize = 2*SDF_BLOCK_SIZE*SDF_BLOCK_SIZE*SDF_BLOCK_SIZE;
		D3D11_BOX sourceRegionDesc = {0, 0, 0, 4*sizeof(int)*nSDFBlockDescs, 1, 1};
		context->CopySubresourceRegion(m_pSDFBlockDescOutputCPU, 0, 0, 0, 0, m_pSDFBlockDescOutput, 0, &sourceRegionDesc);

		D3D11_BOX sourceRegionBlock = {0, 0, 0, sizeof(int)*linBlockSize*nSDFBlockDescs, 1, 1};
		context->CopySubresourceRegion(m_pSDFBlockOutputCPU, 0, 0, 0, 0, m_pSDFBlockOutput, 0, &sourceRegionBlock);
	}

	s_nStreamdOutBlocks = nSDFBlockDescs;
	
	hr = context->Map(m_pSDFBlockOutputCPU, 0, D3D11_MAP_READ, 0, &m_mappedResourceBlockStreamOut);
	V_RETURN(context->Map(m_pSDFBlockDescOutputCPU, 0, D3D11_MAP_READ, 0, &m_mappedResourceDescStreamOut));

	if(multiThreaded)
	{
		SetEvent(hEventOutConsume);
		ReleaseMutex(hMutexOut);
	}

	return hr;
}

HRESULT DX11SceneRepChunkGrid::StreamOutToCPUPass1CPU(ID3D11DeviceContext* context, bool multiThreaded)
{
	HRESULT hr = S_OK;

	if(multiThreaded)
	{
		WaitForSingleObject(hEventOutConsume, INFINITE);
		WaitForSingleObject(hMutexOut, INFINITE);
	}

	if(s_nStreamdOutBlocks != 0)
	{
		IntegrateInChunkGrid((int*)m_mappedResourceDescStreamOut.pData, (int*)m_mappedResourceBlockStreamOut.pData, s_nStreamdOutBlocks);
	}

	if(multiThreaded)
	{
		SetEvent(hEventOutProduce);
		ReleaseMutex(hMutexOut);
	}

	return hr;
}

HRESULT DX11SceneRepChunkGrid::StreamOutToCPU(ID3D11DeviceContext* context, DX11SceneRepHashSDF& sceneRepHashSDF, const vec3f& posCamera, float radius, bool useParts, unsigned int& nStreamedBlocks)
{
	HRESULT hr = S_OK;

	s_posCamera = posCamera;
	s_radius = radius;

	V_RETURN(StreamOutToCPUPass0GPU(context, sceneRepHashSDF, posCamera, radius, useParts, false));
	V_RETURN(StreamOutToCPUPass1CPU(context,  false));

	nStreamedBlocks = s_nStreamdOutBlocks;

	return hr;
}

void DX11SceneRepChunkGrid::IntegrateInChunkGrid(const int* desc, const int* block, unsigned int nSDFBlocks)
{	
	const unsigned int descSize = 4;

	for(unsigned int i = 0; i < nSDFBlocks; i++)
	{
		vec3i pos(&desc[i*descSize]);
		vec3f posWorld = VoxelUtilHelper::SDFBlockToWorld(pos);
		vec3i chunk = worldToChunks(posWorld);

		if(!isValidChunk(chunk))
		{
			std::cout << "Chunk out of bounds" << std::endl;
			continue;
		}

		unsigned int index = linearizeChunkPos(chunk);

		if(m_grid[index] == NULL) // Allocate memory for chunk
		{
			m_grid[index] = new ChunkDesc(m_initialChunkDescListSize);
		}
		
		// Add element
		m_grid[index]->addSDFBlock(((const SDFDesc*)desc)[i], ((const SDFBlock*)block)[i]);
		m_bitMask.setBit(index);
	}
}

void DX11SceneRepChunkGrid::checkForDuplicates()
{	
	const unsigned int descSize = 4;

	for(unsigned int i = 0; i < m_grid.size(); i++)
	{
		if(m_grid[i] != NULL)
		{
			std::vector<SDFDesc>& descsCopy = m_grid[i]->getSDFBlockDescs();
		
			std::list<SDFDesc> l;
			for (unsigned int k = 0; k < descsCopy.size(); k++)
			{			
				l.push_back(descsCopy[k]);
			}

			l.sort();
		
			unsigned int sizeBefore = (unsigned int)l.size();
			l.unique();
			unsigned int sizeAfter = (unsigned int)l.size();

			unsigned int diff = sizeBefore - sizeAfter;
			if(diff != 0)
			{
				std::cout << "Chunk: " << i << " diff: " << sizeBefore - sizeAfter << std::endl;
			}
		}
	}
}

HRESULT DX11SceneRepChunkGrid::StreamInToGPUAll(ID3D11DeviceContext* context, DX11SceneRepHashSDF& sceneRepHashSDF)
{
	HRESULT hr = S_OK;

	unsigned int nStreamedBlocks = 1;
	while(nStreamedBlocks != 0)
	{
		V_RETURN(StreamInToGPU(context, sceneRepHashSDF, getChunkCenter(vec3i(0, 0, 0)), 1.1f*getGridRadiusInMeter(), true, nStreamedBlocks));
	}

	return hr;
}

HRESULT DX11SceneRepChunkGrid::StreamInToGPUChunk(ID3D11DeviceContext* context, DX11SceneRepHashSDF& sceneRepHashSDF, const vec3i& chunkPos)
{
	HRESULT hr = S_OK;

	unsigned int nStreamedBlocks = 1;
	while(nStreamedBlocks != 0) // Should not be necessary
	{
		V_RETURN(StreamInToGPU(context, sceneRepHashSDF, getChunkCenter(chunkPos), 1.1f*getChunkRadiusInMeter(), true, nStreamedBlocks));
	}

	return hr;
}

HRESULT DX11SceneRepChunkGrid::StreamInToGPUChunkNeighborhood(ID3D11DeviceContext* context, DX11SceneRepHashSDF& sceneRepHashSDF, const vec3i& chunkPos, int kernelRadius)
{
	HRESULT hr = S_OK;

	vec3i startChunk = vec3i(std::max(chunkPos.x-kernelRadius, m_minGridPos.x), std::max(chunkPos.y-kernelRadius, m_minGridPos.y), std::max(chunkPos.z-kernelRadius, m_minGridPos.z));
	vec3i endChunk = vec3i(std::min(chunkPos.x+kernelRadius, m_maxGridPos.x), std::min(chunkPos.y+kernelRadius, m_maxGridPos.y), std::min(chunkPos.z+kernelRadius, m_maxGridPos.z));

	for(int x = startChunk.x; x<endChunk.x; x++)
	{
		for(int y = startChunk.y; y<endChunk.y; y++)
		{
			for(int z = startChunk.z; z<endChunk.z; z++)
			{
				V_RETURN(StreamInToGPUChunk(context, sceneRepHashSDF, vec3i(x, y, z)));
			}
		}
	}

	return hr;
}

HRESULT DX11SceneRepChunkGrid::StreamInToGPUPass0CPU(ID3D11DeviceContext* context, const vec3f& posCamera, float radius, bool useParts, bool multiThreaded)
{
	HRESULT hr = S_OK;

	//WaitForSingleObject(hEventSetTransformConsume, INFINITE);
	//WaitForSingleObject(hMutexSetTransform, INFINITE);

	WaitForSingleObject(hEventInProduce, INFINITE);
	WaitForSingleObject(hMutexIn, INFINITE);

	unsigned int nSDFBlockDescs = IntegrateInHash((int*)m_mappedResourceDescStreamIn.pData, (int*)m_mappedResourceBlockStreamIn.pData, posCamera, radius, useParts);
	
	s_nStreamdInBlocks = nSDFBlockDescs;

	SetEvent(hEventInConsume);
	ReleaseMutex(hMutexIn);

	//SetEvent(hEventSetTransformProduce);
	//ReleaseMutex(hMutexSetTransform);

	return hr;
}

HRESULT DX11SceneRepChunkGrid::StreamInToGPUPass1GPU(ID3D11DeviceContext* context, DX11SceneRepHashSDF& sceneRepHashSDF, bool multiThreaded)
{
	HRESULT hr = S_OK;

	if(multiThreaded)
	{
		WaitForSingleObject(hEventInConsume, INFINITE);
		WaitForSingleObject(hMutexIn, INFINITE);
	}

	UnmapInput(context);

	ID3D11UnorderedAccessView* hashUAV = sceneRepHashSDF.GetHashUAV();
	ID3D11UnorderedAccessView* sdfBlocksSDFUAV = sceneRepHashSDF.GetSDFBlocksSDFUAV();
	ID3D11UnorderedAccessView* sdfBlocksRGBWUAV = sceneRepHashSDF.GetSDFBlocksRGBWUAV();
	ID3D11UnorderedAccessView* heapConsumeUAV = sceneRepHashSDF.GetHeapUAV();
	ID3D11UnorderedAccessView* heapStaticUAV = sceneRepHashSDF.GetHeapStaticUAV();
	ID3D11Buffer* CBsceneRepSDF = sceneRepHashSDF.MapAndGetConstantBuffer(context);

	if(s_nStreamdInBlocks != 0)
	{
		//std::cout << "SDFBlocks streamed in: " << s_nStreamdInBlocks << std::endl;

		// Copy to GPU
		const unsigned int linBlockSize = 2*SDF_BLOCK_SIZE*SDF_BLOCK_SIZE*SDF_BLOCK_SIZE;
		D3D11_BOX sourceRegionDesc = {0, 0, 0, 4*sizeof(int)*s_nStreamdInBlocks, 1, 1};
		context->CopySubresourceRegion(m_pSDFBlockDescInput, 0, 0, 0, 0, m_pSDFBlockDescInputCPU, 0, &sourceRegionDesc);

		D3D11_BOX sourceRegionBlock = {0, 0, 0, sizeof(int)*linBlockSize*s_nStreamdInBlocks, 1, 1};
		context->CopySubresourceRegion(m_pSDFBlockInput, 0, 0, 0, 0, m_pSDFBlockInputCPU, 0, &sourceRegionBlock);

		//-------------------------------------------------------
		// Pass 1: Alloc memory for chunks
		//-------------------------------------------------------

		unsigned heapFreeCountPrev = sceneRepHashSDF.GetHeapFreeCount(context);

		// Initialize constant buffer
		D3D11_MAPPED_SUBRESOURCE mappedResource;
		V_RETURN(context->Map(m_constantBufferChunkToGlobalHash, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource));
		CBufferChunkToGlobalHash *cbuffer = (CBufferChunkToGlobalHash*)mappedResource.pData;
	
			cbuffer->nSDFBlockDescs = s_nStreamdInBlocks;
			cbuffer->heapFreeCountPrev = heapFreeCountPrev;

		context->Unmap(m_constantBufferChunkToGlobalHash, 0);

		// Setup pipeline
		unsigned int initialCountReset = 0;
		unsigned int initialCountKeep = -1;
		context->CSSetUnorderedAccessViews(0, 1, &hashUAV, 0);
		context->CSSetUnorderedAccessViews(1, 1, &heapStaticUAV, 0);
		context->CSSetShaderResources(0, 1, &m_pSDFBlockDescInputSRV);
		context->CSSetConstantBuffers(0, 1, &CBsceneRepSDF);
		context->CSSetConstantBuffers(1, 1, &m_constantBufferChunkToGlobalHash);
		ID3D11Buffer* CBGlobalAppState = GlobalAppState::getInstance().MapAndGetConstantBuffer(context);
		context->CSSetConstantBuffers(8, 1, &CBGlobalAppState);
		context->CSSetShader(m_pComputeShaderChunkToGlobalHashPass1, 0, 0);
	
		// Run compute shader
		unsigned int dimX = NUM_GROUPS_X;
		unsigned int dimY = ((s_nStreamdInBlocks + m_blockSizeChunkToGlobalHash - 1)/m_blockSizeChunkToGlobalHash + NUM_GROUPS_X - 1) / NUM_GROUPS_X;
		assert(dimX <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);
		assert(dimY <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);

		context->Dispatch(dimX, dimY, 1);

		//Modify Counter
		unsigned int initialCountNew = heapFreeCountPrev-s_nStreamdInBlocks;
		context->CSSetUnorderedAccessViews(1, 1, &heapConsumeUAV, &initialCountNew);

		ID3D11UnorderedAccessView* nullUAV[2] = {NULL, NULL};
		context->CSSetUnorderedAccessViews(1, 1, nullUAV, 0);
	
		// Cleanup
		ID3D11ShaderResourceView* nullSRV[2] = {NULL, NULL};
		ID3D11Buffer* nullB[2] = {NULL, NULL};

		context->CSSetUnorderedAccessViews(0, 2, nullUAV, 0);
		context->CSSetShaderResources(0, 1, nullSRV);
		context->CSSetConstantBuffers(0, 2, nullB);
		context->CSSetConstantBuffers(8, 1, nullB);
		context->CSSetShader(0, 0, 0);

		unsigned heapFreeCountNow = sceneRepHashSDF.GetHeapFreeCount(context);

		//-------------------------------------------------------
		// Pass 2: Initialize corresponding SDFBlocks
		//-------------------------------------------------------

		// Initialize constant buffer
		V_RETURN(context->Map(m_constantBufferChunkToGlobalHash, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource));
		cbuffer = (CBufferChunkToGlobalHash*)mappedResource.pData;
	
			cbuffer->nSDFBlockDescs = s_nStreamdInBlocks;
			cbuffer->heapFreeCountPrev = heapFreeCountPrev;
			cbuffer->heapFreeCountNow = heapFreeCountNow;

		context->Unmap(m_constantBufferChunkToGlobalHash, 0);

		// Setup pipeline
		context->CSSetUnorderedAccessViews(0, 1, &sdfBlocksSDFUAV, 0);
		context->CSSetUnorderedAccessViews(1, 1, &heapStaticUAV, 0);
		context->CSSetUnorderedAccessViews(2, 1, &sdfBlocksRGBWUAV, 0);
		context->CSSetShaderResources(0, 1, &m_pSDFBlockInputSRV);
		context->CSSetConstantBuffers(0, 1, &CBsceneRepSDF);
		context->CSSetConstantBuffers(1, 1, &m_constantBufferChunkToGlobalHash);
		context->CSSetConstantBuffers(8, 1, &CBGlobalAppState);
		context->CSSetShader(m_pComputeShaderChunkToGlobalHashPass2, 0, 0);

		// Run compute shader
		dimX = NUM_GROUPS_X;
		dimY = (s_nStreamdInBlocks + NUM_GROUPS_X - 1) / NUM_GROUPS_X;
		assert(dimX <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);
		assert(dimY <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);

		context->Dispatch(dimX, dimY, 1);

		// Cleanup
		ID3D11UnorderedAccessView* nullUAV2[] = {NULL, NULL, NULL};

		context->CSSetUnorderedAccessViews(0, 3, nullUAV2, 0);
		context->CSSetShaderResources(0, 1, nullSRV);
		context->CSSetConstantBuffers(0, 2, nullB);
		context->CSSetConstantBuffers(8, 1, nullB);
		context->CSSetShader(0, 0, 0);
	}

	V_RETURN(context->Map(m_pSDFBlockInputCPU, 0, D3D11_MAP_WRITE_DISCARD, 0, &m_mappedResourceBlockStreamIn));
	V_RETURN(context->Map(m_pSDFBlockDescInputCPU, 0, D3D11_MAP_WRITE_DISCARD, 0, &m_mappedResourceDescStreamIn));

	if(multiThreaded)
	{
		SetEvent(hEventInProduce);
		ReleaseMutex(hMutexIn);
	}

	return hr;
}

HRESULT DX11SceneRepChunkGrid::StreamInToGPU(ID3D11DeviceContext* context, DX11SceneRepHashSDF& sceneRepHashSDF, const vec3f& posCamera, float radius, bool useParts, unsigned int& nStreamedBlocks)
{
	HRESULT hr = S_OK;

	s_posCamera = posCamera;
	s_radius = radius;

	V_RETURN(StreamInToGPUPass0CPU(context, posCamera, radius, useParts, false));
	V_RETURN(StreamInToGPUPass1GPU(context, sceneRepHashSDF, false));

	nStreamedBlocks = s_nStreamdInBlocks;

	return hr;
}

HRESULT DX11SceneRepChunkGrid::StreamInToGPUAll(ID3D11DeviceContext* context, DX11SceneRepHashSDF& sceneRepHashSDF, const vec3f& posCamera, float radius, bool useParts, unsigned int& nStreamedBlocks)
{
	HRESULT hr = S_OK;

	unsigned int nStreamedBlocksSum = 0;
	unsigned int nBlock = 1;
	while(nBlock != 0) // Should not be necessary
	{
		V_RETURN(StreamInToGPU(context, sceneRepHashSDF, posCamera, radius, useParts, nBlock));
		nStreamedBlocksSum += nBlock;
	}

	nStreamedBlocks = nStreamedBlocksSum;

	return hr;
}

unsigned int DX11SceneRepChunkGrid::IntegrateInHash(int* descOuput, int* blockOutput, const vec3f& posCamera, float radius, bool useParts)
{
	const unsigned int blockSize = sizeof(SDFBlock)/sizeof(int);
	const unsigned int descSize = sizeof(SDFDesc)/sizeof(int);

	vec3i camChunk = worldToChunks(posCamera);
	vec3i chunkRadius = meterToNumberOfChunksCeil(radius);
	vec3i startChunk = vec3i(std::max(camChunk.x-chunkRadius.x, m_minGridPos.x), std::max(camChunk.y-chunkRadius.y, m_minGridPos.y), std::max(camChunk.z-chunkRadius.z, m_minGridPos.z));
	vec3i endChunk = vec3i(std::min(camChunk.x+chunkRadius.x, m_maxGridPos.x), std::min(camChunk.y+chunkRadius.y, m_maxGridPos.y), std::min(camChunk.z+chunkRadius.z, m_maxGridPos.z));
	
	unsigned int nSDFBlocks = 0;
	for(int x = startChunk.x; x<=endChunk.x; x++)
	{
		for(int y = startChunk.y; y<=endChunk.y; y++)
		{
			for(int z = startChunk.z; z<=endChunk.z; z++)
			{
				unsigned int index = linearizeChunkPos(vec3i(x, y, z));
				if(m_grid[index] != NULL && m_grid[index]->isStreamedOut()) // As been allocated and has streamed out blocks
				{
					if(isChunkInSphere(delinearizeChunkIndex(index), posCamera, radius)) // Is in camera range
					{
						unsigned int nBlock = m_grid[index]->getNElements();
						if (nBlock > m_maxNumberOfSDFBlocksIntegrateFromGlobalHash) {
							throw MLIB_EXCEPTION("not enough memory allocated for intermediate GPU buffer");
						}
						// Copy data to GPU
						memcpy(&descOuput[0], &(m_grid[index]->getSDFBlockDescs()[0]), sizeof(SDFDesc)*nBlock);
						memcpy(&blockOutput[0], &(m_grid[index]->getSDFBlocks()[0]), sizeof(SDFBlock)*nBlock);

						// Remove data from CPU
						m_grid[index]->clear();
						m_bitMask.resetBit(index);

						nSDFBlocks += nBlock;

						if(useParts) return nSDFBlocks; // only in one chunk per frame
					}
				}
			}
		}
	}

	return nSDFBlocks;
}
