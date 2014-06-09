#pragma once

#include "DX11Utils.h"
//#include "KinectSensor2.h"
#include "matrix4x4.h"
#include "PointCloudIO.h"
#include "GlobalAppState.h"
#include "TimingLog.h"

#include <cassert>

#define THREAD_GROUP_SIZE_SCENE_REP 8

struct CB_VOXEL_HASH {
	unsigned int	m_HashNumBuckets;
	unsigned int	m_HashBucketSize;
	unsigned int	m_InputImageWidth;
	unsigned int	m_InputImageHeight;
	float			m_VirtualVoxelResolutionScalar;
	float			m_VirtualVoxelSize;
	unsigned int	m_OtherNumVoxels;
	float			m_dummy;
	D3DXMATRIX		m_RigidTransform;
	float4			m_CameraPos;
};

class DX11SceneRep
{
public:
	DX11SceneRep(void);
	~DX11SceneRep(void);

	static HRESULT OnD3D11CreateDevice(ID3D11Device* pd3dDevice);
	static void OnD3D11DestroyDevice();

	HRESULT Init(ID3D11Device* pd3dDevice, unsigned int hashNumBuckets = 1000000, unsigned int hashBucketSize = 10, float voxelSize = 0.002f);
	void Destroy();





	void Integrate(ID3D11DeviceContext* context, ID3D11ShaderResourceView* inputPoints, ID3D11ShaderResourceView* inputColors, const mat4f* rigidTransform) {

		const unsigned int imageWidth = GlobalAppState::getInstance().s_windowWidth;
		const unsigned int imageHeight = GlobalAppState::getInstance().s_windowHeight;

		HRESULT hr = S_OK;

		D3D11_MAPPED_SUBRESOURCE mappedResource;
		V(context->Map(s_VoxelHashCB, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource));

		CB_VOXEL_HASH *cbuffer = (CB_VOXEL_HASH*)mappedResource.pData;
		cbuffer->m_HashBucketSize = m_HashBucketSize;
		cbuffer->m_HashNumBuckets = m_HashNumBuckets;
		cbuffer->m_VirtualVoxelResolutionScalar = m_VirtualVoxelResolutionScalar;
		cbuffer->m_InputImageWidth = GlobalAppState::getInstance().s_windowWidth;
		cbuffer->m_InputImageHeight = GlobalAppState::getInstance().s_windowHeight;
		//D3DXMatrixTranspose(&cbuffer->m_RigidTransform, rigidTransform);
		memcpy(cbuffer->m_RigidTransform, rigidTransform, sizeof(mat4f)); 
		vec3f trans = rigidTransform->getTranslation();
		cbuffer->m_CameraPos = -float4(trans.x, trans.y, trans.z, 1.0f);
		context->Unmap(s_VoxelHashCB, 0);

		// Setup Pipeline
		context->CSSetShaderResources(0, 1, &inputPoints);
		context->CSSetShaderResources(1, 1, &inputColors);
		context->CSSetUnorderedAccessViews( 0, 1, &m_VoxelHashBufferUAV, 0);
		context->CSSetConstantBuffers(0, 1, &s_VoxelHashCB);
		context->CSSetShader(s_VoxelHashIntegrate, 0, 0);


		// Start Compute Shader
		unsigned int dimX = (unsigned int)ceil(((float)imageWidth)/THREAD_GROUP_SIZE_SCENE_REP);
		unsigned int dimY = (unsigned int)ceil(((float)imageHeight)/THREAD_GROUP_SIZE_SCENE_REP);


		// Start query for timing
		if(GlobalAppState::getInstance().s_timingsDetailledEnabled) {
			GlobalAppState::getInstance().WaitForGPU();
			m_timer.start();
		}

		assert(dimX <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);
		assert(dimY <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);

		context->Dispatch(dimX, dimY, 1);

		// Wait for query
		if(GlobalAppState::getInstance().s_timingsDetailledEnabled) {
			GlobalAppState::getInstance().WaitForGPU();
			TimingLog::totalTimeUpdateScene += m_timer.getElapsedTimeMS();
			TimingLog::countUpdateScene++;
		}

		// De-Initialize Pipeline
		ID3D11ShaderResourceView* nullSRV[] = { NULL, NULL };
		ID3D11UnorderedAccessView* nullUAV[] = { NULL };
		ID3D11Buffer* nullCB[] = { NULL };

		context->CSSetShaderResources(0, 2, nullSRV);
		context->CSSetUnorderedAccessViews(0, 1, nullUAV, 0);
		context->CSSetConstantBuffers(0, 1, nullCB);
		context->CSSetShader(0, 0, 0);

		m_LastRigidTransform = *rigidTransform;
		
		m_NumIntegratedImages++;
	}


	void RemoveAndIntegrateToOther(ID3D11DeviceContext* context, DX11SceneRep* other, const mat4f* rigidTransform, bool moveOutsideFrustum) {
		
		HRESULT hr = S_OK;

		D3D11_MAPPED_SUBRESOURCE mappedResource;
		V(context->Map(s_VoxelHashCB, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource));

		CB_VOXEL_HASH *cbuffer = (CB_VOXEL_HASH*)mappedResource.pData;
		//cbuffer->m_HashBucketSize = m_HashBucketSize;
		//cbuffer->m_HashNumBuckets = m_HashNumBuckets;
		cbuffer->m_HashBucketSize = other->m_HashBucketSize;
		cbuffer->m_HashNumBuckets = other->m_HashNumBuckets;
		cbuffer->m_VirtualVoxelResolutionScalar = m_VirtualVoxelResolutionScalar;
		cbuffer->m_OtherNumVoxels = m_HashBucketSize * m_HashNumBuckets;
		cbuffer->m_InputImageWidth = DXUTGetWindowWidth();
		cbuffer->m_InputImageHeight = DXUTGetWindowWidth();
		//D3DXMatrixTranspose(&cbuffer->m_RigidTransform, rigidTransform);
		//memcpy(cbuffer->m_RigidTransform, rigidTransform, sizeof(mat4f)); 
		mat4f worldToLastKinectSpace =  rigidTransform->getInverse();
		memcpy(cbuffer->m_RigidTransform, &worldToLastKinectSpace, sizeof(mat4f)); 
		context->Unmap(s_VoxelHashCB, 0);

		// Setup Pipeline
		context->CSSetUnorderedAccessViews( 0, 1, &m_VoxelHashBufferUAV, 0);
		context->CSSetUnorderedAccessViews( 1, 1, &other->m_VoxelHashBufferUAV, 0);
		context->CSSetConstantBuffers(0, 1, &s_VoxelHashCB);
		if (moveOutsideFrustum)		context->CSSetShader(s_VoxelHashRemoveAndIntegrateOutFrustum, 0, 0);
		else						context->CSSetShader(s_VoxelHashRemoveAndIntegrateInFrustum, 0, 0);


		// Start Compute Shader
		unsigned int groupeThreads = THREAD_GROUP_SIZE_SCENE_REP*THREAD_GROUP_SIZE_SCENE_REP*8;
		unsigned int dimX = (m_HashNumBuckets * m_HashBucketSize + groupeThreads - 1) / groupeThreads;


		//// Start query for timing
		//if(GlobalAppState::getInstance().s_timingsEnabled) {
		//	GlobalAppState::getInstance().WaitForGPU();
		//	m_timer.start();
		//}

		assert(dimX <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);
	
		context->Dispatch(dimX, 1, 1);

		//// Wait for query
		//if(GlobalAppState::getInstance().s_timingsEnabled) {
		//	GlobalAppState::getInstance().WaitForGPU();
		//	TimingLog::totalTimeUpdateScene += m_timer.getElapsedTimeMS();
		//	TimingLog::countUpdateScene++;
		//}

		// De-Initialize Pipeline
		ID3D11ShaderResourceView* nullSRV[] = { NULL, NULL };
		ID3D11UnorderedAccessView* nullUAV[] = { NULL, NULL };
		ID3D11Buffer* nullCB[] = { NULL };

		context->CSSetUnorderedAccessViews(0, 2, nullUAV, 0);
		context->CSSetConstantBuffers(0, 1, nullCB);
		context->CSSetShader(0, 0, 0);

		//m_LastRigidTransform = *rigidTransform;

		//m_NumIntegratedImages++;
	}

	
	void IntegrateFromOther(ID3D11DeviceContext* context, ID3D11ShaderResourceView* otherSRVint4, unsigned int otherNumVoxels, bool remove = false) {

		HRESULT hr = S_OK;

		D3D11_MAPPED_SUBRESOURCE mappedResource;
		V(context->Map(s_VoxelHashCB, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource));

		CB_VOXEL_HASH *cbuffer = (CB_VOXEL_HASH*)mappedResource.pData;
		cbuffer->m_HashBucketSize = m_HashBucketSize;
		cbuffer->m_HashNumBuckets = m_HashNumBuckets;
		cbuffer->m_VirtualVoxelResolutionScalar = m_VirtualVoxelResolutionScalar;
		cbuffer->m_InputImageWidth = DXUTGetWindowWidth();
		cbuffer->m_InputImageHeight = DXUTGetWindowWidth();
		//D3DXMatrixTranspose(&cbuffer->m_RigidTransform, rigidTransform);
		cbuffer->m_OtherNumVoxels = otherNumVoxels;
		context->Unmap(s_VoxelHashCB, 0);

		// Setup Pipeline
		context->CSSetUnorderedAccessViews( 0, 1, &m_VoxelHashBufferUAV, 0);
		context->CSSetShaderResources( 0, 1, &otherSRVint4);
		context->CSSetConstantBuffers(0, 1, &s_VoxelHashCB);
		if (!remove)	context->CSSetShader(s_VoxelHashIntegrateFromOther, 0, 0);
		else			context->CSSetShader(s_VoxelHashRemoveFromOther, 0, 0);


		// Start Compute Shader
		unsigned int groupeThreads = THREAD_GROUP_SIZE_SCENE_REP*THREAD_GROUP_SIZE_SCENE_REP*8;
		unsigned int dimX = (otherNumVoxels * otherNumVoxels + groupeThreads - 1) / groupeThreads;


		//// Start query for timing
		//if(GlobalAppState::getInstance().s_timingsEnabled) {
		//	GlobalAppState::getInstance().WaitForGPU();
		//	m_timer.start();
		//}

		assert(dimX <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);
	
		context->Dispatch(dimX, 1, 1);

		//// Wait for query
		//if(GlobalAppState::getInstance().s_timingsEnabled) {
		//	GlobalAppState::getInstance().WaitForGPU();
		//	TimingLog::totalTimeUpdateScene += m_timer.getElapsedTimeMS();
		//	TimingLog::countUpdateScene++;
		//}

		// De-Initialize Pipeline
		ID3D11ShaderResourceView* nullSRV[] = { NULL, NULL };
		ID3D11UnorderedAccessView* nullUAV[] = { NULL, NULL };
		ID3D11Buffer* nullCB[] = { NULL };

		context->CSSetUnorderedAccessViews(0, 2, nullUAV, 0);
		context->CSSetShaderResources(0, 2, nullSRV);
		context->CSSetConstantBuffers(0, 1, nullCB);
		context->CSSetShader(0, 0, 0);

		//m_LastRigidTransform = *rigidTransform;

		//m_NumIntegratedImages++;
	}

	void StarveVoxelHash(ID3D11DeviceContext* context, bool reset = false) {
		HRESULT hr = S_OK;

		D3D11_MAPPED_SUBRESOURCE mappedResource;
		V(context->Map(s_VoxelHashCB, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource));
		CB_VOXEL_HASH *cbuffer = (CB_VOXEL_HASH*)mappedResource.pData;
		cbuffer->m_HashBucketSize = m_HashBucketSize;
		cbuffer->m_HashNumBuckets = m_HashNumBuckets;
		cbuffer->m_InputImageWidth = GlobalAppState::getInstance().s_windowWidth;;
		cbuffer->m_InputImageHeight = GlobalAppState::getInstance().s_windowHeight;
		context->Unmap(s_VoxelHashCB, 0);

		// Setup Pipeline
		context->CSSetUnorderedAccessViews( 0, 1, &m_VoxelHashBufferUAV, 0);
		context->CSSetConstantBuffers(0, 1, &s_VoxelHashCB);
		if (reset)	{
			context->CSSetShader(s_VoxelHashReset, 0, 0); 
			m_LastRigidTransform.setIdentity();
			m_NumIntegratedImages = 0;
		} else {
			context->CSSetShader(s_VoxelHashStarve, 0, 0);
		}

		// Start Compute Shader
		unsigned int groupeThreads = THREAD_GROUP_SIZE_SCENE_REP*THREAD_GROUP_SIZE_SCENE_REP*8;
		unsigned int dimX = (m_HashNumBuckets * m_HashBucketSize + groupeThreads - 1) / groupeThreads;
		
		assert(dimX <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);
	
		context->Dispatch(dimX, 1, 1);


		// De-Initialize Pipeline
		ID3D11ShaderResourceView* nullSAV[1] = { NULL };
		ID3D11UnorderedAccessView* nullUAV[1] = { NULL };
		ID3D11Buffer* nullCB[1] = { NULL };

		context->CSSetShaderResources(0, 1, nullSAV);
		context->CSSetUnorderedAccessViews(0, 1, nullUAV, 0);
		context->CSSetConstantBuffers(0, 1, nullCB);
		context->CSSetShader(0, 0, 0);
	}


	void StarveVoxelHashVisible(ID3D11DeviceContext* context, ID3D11ShaderResourceView* prevDepth, const mat4f* lastRigidTransform) {
		HRESULT hr = S_OK;

		D3D11_MAPPED_SUBRESOURCE mappedResource;
		V(context->Map(s_VoxelHashCB, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource));
		CB_VOXEL_HASH *cbuffer = (CB_VOXEL_HASH*)mappedResource.pData;
		cbuffer->m_HashBucketSize = m_HashBucketSize;
		cbuffer->m_HashNumBuckets = m_HashNumBuckets;
		cbuffer->m_InputImageWidth = GlobalAppState::getInstance().s_windowWidth;
		cbuffer->m_InputImageHeight = GlobalAppState::getInstance().s_windowHeight;
		cbuffer->m_VirtualVoxelSize = 1.0f/(float)m_VirtualVoxelResolutionScalar;
		mat4f viewTransform = lastRigidTransform->getInverse();
		memcpy(cbuffer->m_RigidTransform, &viewTransform, sizeof(mat4f));
		context->Unmap(s_VoxelHashCB, 0);

		// Setup Pipeline
		context->CSSetUnorderedAccessViews( 0, 1, &m_VoxelHashBufferUAV, 0);
		context->CSSetConstantBuffers(0, 1, &s_VoxelHashCB);
		context->CSSetShaderResources(2, 1, &prevDepth);
		context->CSSetShader(s_VoxelHashStarveVisible, 0, 0);

		// Start Compute Shader
		unsigned int groupeThreads = THREAD_GROUP_SIZE_SCENE_REP*THREAD_GROUP_SIZE_SCENE_REP*8;
		unsigned int dimX = (m_HashNumBuckets * m_HashBucketSize + groupeThreads - 1) / groupeThreads;
		
		assert(dimX <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);
	
		context->Dispatch(dimX, 1, 1);


		// De-Initialize Pipeline
		ID3D11ShaderResourceView* nullSAV[1] = { NULL };
		ID3D11UnorderedAccessView* nullUAV[1] = { NULL };
		ID3D11Buffer* nullCB[1] = { NULL };

		context->CSSetShaderResources(0, 1, nullSAV);
		context->CSSetShaderResources(2, 1, nullSAV);
		context->CSSetUnorderedAccessViews(0, 1, nullUAV, 0);
		context->CSSetConstantBuffers(0, 1, nullCB);
		context->CSSetShader(0, 0, 0);
	}



	void StarveVoxelRegulizeNoise(ID3D11DeviceContext* context, ID3D11ShaderResourceView* prevDepth, ID3D11ShaderResourceView* prevNormals, const mat4f* lastRigidTransform) {
		HRESULT hr = S_OK;

		D3D11_MAPPED_SUBRESOURCE mappedResource;
		V(context->Map(s_VoxelHashCB, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource));
		CB_VOXEL_HASH *cbuffer = (CB_VOXEL_HASH*)mappedResource.pData;
		cbuffer->m_HashBucketSize = m_HashBucketSize;
		cbuffer->m_HashNumBuckets = m_HashNumBuckets;
		cbuffer->m_InputImageWidth = GlobalAppState::getInstance().s_windowWidth;
		cbuffer->m_InputImageHeight = GlobalAppState::getInstance().s_windowHeight;
		cbuffer->m_VirtualVoxelSize = 1.0f/(float)m_VirtualVoxelResolutionScalar;
		mat4f viewTransform = lastRigidTransform->getInverse();
		memcpy(cbuffer->m_RigidTransform, &viewTransform, sizeof(mat4f));
		context->Unmap(s_VoxelHashCB, 0);

		// Setup Pipeline
		context->CSSetUnorderedAccessViews( 0, 1, &m_VoxelHashBufferUAV, 0);
		context->CSSetConstantBuffers(0, 1, &s_VoxelHashCB);
		context->CSSetShaderResources(2, 1, &prevDepth);
		context->CSSetShaderResources(3, 1, &prevNormals);
		context->CSSetShader(s_VoxelHashStarveRegulizeNoise, 0, 0);

		// Start Compute Shader
		unsigned int groupeThreads = THREAD_GROUP_SIZE_SCENE_REP*THREAD_GROUP_SIZE_SCENE_REP*8;
		unsigned int dimX = (m_HashNumBuckets * m_HashBucketSize + groupeThreads - 1) / groupeThreads;
		
		assert(dimX <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);
	
		context->Dispatch(dimX, 1, 1);


		// De-Initialize Pipeline
		ID3D11ShaderResourceView* nullSRV[] = { NULL, NULL, NULL, NULL };
		ID3D11UnorderedAccessView* nullUAV[] = { NULL };
		ID3D11Buffer* nullCB[] = { NULL };

		context->CSSetShaderResources(0, 4, nullSRV);
		context->CSSetUnorderedAccessViews(0, 1, nullUAV, 0);
		context->CSSetConstantBuffers(0, 1, nullCB);
		context->CSSetShader(0, 0, 0);
	}

	const mat4f& GetLastRigidTransform() const {
		return m_LastRigidTransform;
	}

	ID3D11Buffer* GetVoxelHashBuffer() {
		return m_VoxelHashBuffer;
	}

	ID3D11ShaderResourceView* GetVoxelHashBufferSRV() {
		return m_VoxelHashBufferSRV;
	}

	unsigned int GetNumVoxels() {
		return m_HashBucketSize * m_HashNumBuckets;
	}

	float GetVirtualVoxelResolutionScalar() {
		return m_VirtualVoxelResolutionScalar;
	}

	float GetVoxelSize() {
		return 1.0f/m_VirtualVoxelResolutionScalar;
	}

	unsigned int GetNumIntegratedImages() {
		return m_NumIntegratedImages;
	}

	HRESULT DumpPointCloud(const std::string &filename, ID3D11Device* pDevice, ID3D11DeviceContext* pd3dImmediateContext, unsigned int minWeight = 1)
	{
		HRESULT hr = S_OK;

		ID3D11Buffer* pBuffer = m_VoxelHashBuffer;
		ID3D11Buffer* debugbuf = NULL;

		D3D11_BUFFER_DESC desc;
		ZeroMemory( &desc, sizeof(desc) );
		pBuffer->GetDesc( &desc );
		desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
		desc.Usage = D3D11_USAGE_STAGING;
		desc.BindFlags = 0;
		desc.MiscFlags = 0;
		V_RETURN(pDevice->CreateBuffer(&desc, NULL, &debugbuf));

		pd3dImmediateContext->CopyResource( debugbuf, pBuffer );
		unsigned int numElements = desc.ByteWidth/sizeof(INT);
		assert(numElements == m_HashBucketSize * m_HashNumBuckets * 4);
		INT *cpuMemory = new INT[numElements];
		D3D11_MAPPED_SUBRESOURCE mappedResource;
		V(pd3dImmediateContext->Map(debugbuf, D3D11CalcSubresource(0,0,0), D3D11_MAP_READ, 0, &mappedResource));	
		memcpy((void*)cpuMemory, (void*)mappedResource.pData, desc.ByteWidth);
		pd3dImmediateContext->Unmap( debugbuf, 0 );

		std::vector<vec3f> points;
		std::vector<vec3f> colors;
		for (unsigned int i = 0; i < numElements / 4; i++) {
			int last = cpuMemory[4*i+3];			
			int weight = last & 0x000000ff;
			vec3f c;
			last >>= 0x8;
			c.x = (float)(last & 0x000000ff);
			last >>= 0x8;
			c.y = (float)(last & 0x000000ff);
			last >>= 0x8;
			c.z = (float)(last & 0x000000ff);
			c /= 255.0f;

			//if (weight < 100)	continue;
			if (weight < (int)minWeight)	continue;
			vec3f p;
			p.x = (float)cpuMemory[4*i+0];
			p.y = (float)cpuMemory[4*i+1];
			p.z = (float)cpuMemory[4*i+2];
			points.push_back(p/m_VirtualVoxelResolutionScalar);
			colors.push_back(c);
		}

		std::cout << "Dumping voxel grid " << filename <<  " ( " << points.size() << " ) ...";
		PointCloudIOf::saveToFile(filename, &points, NULL, &colors);
		std::cout << " done!" << std::endl;
		SAFE_RELEASE(debugbuf);
		SAFE_DELETE_ARRAY(cpuMemory);

		return hr;
	}

private:

	mat4f						m_LastRigidTransform;
	unsigned int				m_HashNumBuckets;
	unsigned int				m_HashBucketSize;
	float						m_VirtualVoxelResolutionScalar;

	ID3D11Buffer*				m_VoxelHashBuffer;
	ID3D11ShaderResourceView*	m_VoxelHashBufferSRV;
	ID3D11UnorderedAccessView*	m_VoxelHashBufferUAV;

	unsigned int				m_NumIntegratedImages;

	static ID3D11Buffer*			s_VoxelHashCB;
	static ID3D11ComputeShader*		s_VoxelHashIntegrate;
	static ID3D11ComputeShader*		s_VoxelHashReset;
	static ID3D11ComputeShader*		s_VoxelHashStarve;
	static ID3D11ComputeShader*		s_VoxelHashStarveVisible;
	static ID3D11ComputeShader*		s_VoxelHashStarveRegulizeNoise;

	static ID3D11ComputeShader*		s_VoxelHashRemoveAndIntegrateInFrustum;
	static ID3D11ComputeShader*		s_VoxelHashRemoveAndIntegrateOutFrustum;
	static ID3D11ComputeShader*		s_VoxelHashIntegrateFromOther;
	static ID3D11ComputeShader*		s_VoxelHashRemoveFromOther;

	static Timer m_timer;

};

