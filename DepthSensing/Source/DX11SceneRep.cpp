
#include "DX11SceneRep.h"

ID3D11Buffer*					DX11SceneRep::s_VoxelHashCB = NULL;
ID3D11ComputeShader*			DX11SceneRep::s_VoxelHashIntegrate = NULL;
ID3D11ComputeShader*			DX11SceneRep::s_VoxelHashReset = NULL;
ID3D11ComputeShader*			DX11SceneRep::s_VoxelHashStarve = NULL;
ID3D11ComputeShader*			DX11SceneRep::s_VoxelHashStarveVisible = NULL;
ID3D11ComputeShader*			DX11SceneRep::s_VoxelHashStarveRegulizeNoise = NULL;

ID3D11ComputeShader*			DX11SceneRep::s_VoxelHashRemoveAndIntegrateInFrustum = NULL;
ID3D11ComputeShader*			DX11SceneRep::s_VoxelHashRemoveAndIntegrateOutFrustum = NULL;
ID3D11ComputeShader*			DX11SceneRep::s_VoxelHashIntegrateFromOther = NULL;
ID3D11ComputeShader*			DX11SceneRep::s_VoxelHashRemoveFromOther = NULL;

Timer DX11SceneRep::m_timer;






DX11SceneRep::DX11SceneRep( void )
{
	m_LastRigidTransform.setIdentity();
	m_VoxelHashBuffer = NULL;
	m_VoxelHashBufferSRV = NULL;
	m_VoxelHashBufferUAV = NULL;
	m_NumIntegratedImages = 0;
}

DX11SceneRep::~DX11SceneRep( void )
{
	Destroy();
}

HRESULT DX11SceneRep::OnD3D11CreateDevice( ID3D11Device* pd3dDevice )
{
	HRESULT hr = S_OK;

	D3D11_BUFFER_DESC bDesc;
	ZeroMemory(&bDesc, sizeof(D3D11_BUFFER_DESC));
	bDesc.BindFlags	= D3D11_BIND_CONSTANT_BUFFER;
	bDesc.Usage	= D3D11_USAGE_DYNAMIC;
	bDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	bDesc.MiscFlags	= 0;
	bDesc.ByteWidth	= sizeof(CB_VOXEL_HASH);
	V_RETURN(pd3dDevice->CreateBuffer(&bDesc, NULL, &s_VoxelHashCB));

	char blockSize[5];
	sprintf_s(blockSize, "%d", THREAD_GROUP_SIZE_SCENE_REP);
	D3D_SHADER_MACRO macro[] = {{"groupthreads", blockSize}, {0}};

	ID3DBlob* pBlob = NULL;
	V_RETURN(CompileShaderFromFile(L"Shaders\\SceneRep.hlsl", "integrateCS", "cs_5_0", &pBlob, macro));
	V_RETURN(pd3dDevice->CreateComputeShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &s_VoxelHashIntegrate));

	V_RETURN(CompileShaderFromFile(L"Shaders\\SceneRep.hlsl", "resetCS", "cs_5_0", &pBlob, macro));
	V_RETURN(pd3dDevice->CreateComputeShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &s_VoxelHashReset));

	V_RETURN(CompileShaderFromFile(L"Shaders\\SceneRep.hlsl", "starveCS", "cs_5_0", &pBlob, macro));
	V_RETURN(pd3dDevice->CreateComputeShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &s_VoxelHashStarve));

	V_RETURN(CompileShaderFromFile(L"Shaders\\SceneRep.hlsl", "starveVisibleCS", "cs_5_0", &pBlob, macro));
	V_RETURN(pd3dDevice->CreateComputeShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &s_VoxelHashStarveVisible));

	V_RETURN(CompileShaderFromFile(L"Shaders\\SceneRep.hlsl", "starveRegulizeNoiseCS", "cs_5_0", &pBlob, macro));
	V_RETURN(pd3dDevice->CreateComputeShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &s_VoxelHashStarveRegulizeNoise));

	D3D_SHADER_MACRO macro_out = {"MOVE_OUT_FRUSTUM", "1"};
	D3D_SHADER_MACRO macro_in = {"MOVE_IN_FRUSTUM", "1"};

	D3D_SHADER_MACRO macro_and_out[10];
	D3D_SHADER_MACRO macro_and_in[10];
	AddDefinitionToMacro(macro, macro_and_out, macro_out);
	AddDefinitionToMacro(macro, macro_and_in, macro_in);

	V_RETURN(CompileShaderFromFile(L"Shaders\\SceneRep.hlsl", "removeAndIntegrateCS", "cs_5_0", &pBlob, macro_and_out));
	V_RETURN(pd3dDevice->CreateComputeShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &s_VoxelHashRemoveAndIntegrateOutFrustum));

	V_RETURN(CompileShaderFromFile(L"Shaders\\SceneRep.hlsl", "removeAndIntegrateCS", "cs_5_0", &pBlob, macro_and_in));
	V_RETURN(pd3dDevice->CreateComputeShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &s_VoxelHashRemoveAndIntegrateInFrustum));

	V_RETURN(CompileShaderFromFile(L"Shaders\\SceneRep.hlsl", "integrateFromOtherCS", "cs_5_0", &pBlob, macro));
	V_RETURN(pd3dDevice->CreateComputeShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &s_VoxelHashIntegrateFromOther));

	V_RETURN(CompileShaderFromFile(L"Shaders\\SceneRep.hlsl", "removeFromOtherCS", "cs_5_0", &pBlob, macro));
	V_RETURN(pd3dDevice->CreateComputeShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &s_VoxelHashRemoveFromOther));

	SAFE_RELEASE(pBlob);

	return hr;
}

void DX11SceneRep::OnD3D11DestroyDevice()
{
	SAFE_RELEASE(s_VoxelHashCB);
	SAFE_RELEASE(s_VoxelHashIntegrate);
	SAFE_RELEASE(s_VoxelHashReset);
	SAFE_RELEASE(s_VoxelHashStarve);
	SAFE_RELEASE(s_VoxelHashStarveVisible);
	SAFE_RELEASE(s_VoxelHashStarveRegulizeNoise);

	SAFE_RELEASE(s_VoxelHashRemoveAndIntegrateInFrustum);
	SAFE_RELEASE(s_VoxelHashRemoveAndIntegrateOutFrustum);	
	SAFE_RELEASE(s_VoxelHashIntegrateFromOther);
	SAFE_RELEASE(s_VoxelHashRemoveFromOther);
}

HRESULT DX11SceneRep::Init( ID3D11Device* pd3dDevice, unsigned int hashNumBuckets /*= 1000000*/, unsigned int hashBucketSize /*= 10*/, float voxelSize /*= 0.002f*/ )
{
	HRESULT hr = S_OK;

	//m_HashNumBuckets = 3300000;	//max
	////m_HashNumBuckets = 1000000;
	//m_HashBucketSize = 10;
	//m_VirtualVoxelResolutionScalar = 500.0f;	//maps from meters to voxel res 0.2cm 
	
	m_HashNumBuckets = hashNumBuckets;
	m_HashBucketSize = hashBucketSize;
	m_VirtualVoxelResolutionScalar = 1.0f/voxelSize;

	m_LastRigidTransform.setIdentity();

	D3D11_BUFFER_DESC bDesc;
	ZeroMemory(&bDesc, sizeof(D3D11_BUFFER_DESC));
	bDesc.BindFlags	= D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
	bDesc.Usage	= D3D11_USAGE_DEFAULT;
	bDesc.CPUAccessFlags = 0;
	bDesc.MiscFlags	= 0;
	bDesc.ByteWidth	= sizeof(int) * 4 * m_HashBucketSize * m_HashNumBuckets;
	bDesc.StructureByteStride = sizeof(int);

	D3D11_SHADER_RESOURCE_VIEW_DESC SRVDesc;
	ZeroMemory( &SRVDesc, sizeof(D3D11_SHADER_RESOURCE_VIEW_DESC) );
	SRVDesc.Format = DXGI_FORMAT_R32_SINT;
	SRVDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
	SRVDesc.Buffer.FirstElement = 0;
	SRVDesc.Buffer.NumElements = 4 * m_HashBucketSize * m_HashNumBuckets;

	D3D11_UNORDERED_ACCESS_VIEW_DESC UAVDesc;
	ZeroMemory( &UAVDesc, sizeof(D3D11_UNORDERED_ACCESS_VIEW_DESC) );
	UAVDesc.Format = DXGI_FORMAT_R32_SINT;;
	UAVDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
	UAVDesc.Buffer.FirstElement = 0;
	UAVDesc.Buffer.NumElements =  4 * m_HashBucketSize * m_HashNumBuckets;

	D3D11_SUBRESOURCE_DATA InitData;
	ZeroMemory( &InitData, sizeof(D3D11_SUBRESOURCE_DATA) );
	int* cpuNull = new int[4 * m_HashBucketSize * m_HashNumBuckets];
	ZeroMemory(cpuNull, sizeof(int) * 4 * m_HashBucketSize * m_HashNumBuckets);
	InitData.pSysMem = cpuNull;
	V_RETURN(pd3dDevice->CreateBuffer(&bDesc, &InitData, &m_VoxelHashBuffer));
	SAFE_DELETE_ARRAY(cpuNull);

	V_RETURN(pd3dDevice->CreateShaderResourceView(m_VoxelHashBuffer, &SRVDesc, &m_VoxelHashBufferSRV));
	V_RETURN(pd3dDevice->CreateUnorderedAccessView(m_VoxelHashBuffer, &UAVDesc, &m_VoxelHashBufferUAV));

	return hr;
}

void DX11SceneRep::Destroy()
{
	SAFE_RELEASE(m_VoxelHashBuffer);
	SAFE_RELEASE(m_VoxelHashBufferSRV);
	SAFE_RELEASE(m_VoxelHashBufferUAV);
}
