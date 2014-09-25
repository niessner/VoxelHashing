#include "stdafx.h"

#include "DX11MarchingCubesHashSDF.h"

#include "GlobalAppState.h"

unsigned int DX11MarchingCubesHashSDF::s_maxNumberOfTriangles = 5000000;

ID3D11ComputeShader* DX11MarchingCubesHashSDF::m_pComputeShader = NULL;
ID3D11Buffer* DX11MarchingCubesHashSDF::m_constantBuffer = NULL;
ID3D11Buffer* DX11MarchingCubesHashSDF::s_pTriangles = NULL;
ID3D11UnorderedAccessView* DX11MarchingCubesHashSDF::s_pTrianglesUAV = NULL;
ID3D11Buffer* DX11MarchingCubesHashSDF::s_pOutputFloatCPU = NULL;
ID3D11Buffer* DX11MarchingCubesHashSDF::s_BuffCountTriangles = NULL;

//std::vector<Vertex> DX11MarchingCubesHashSDF::s_vertices;
//std::vector<unsigned int> DX11MarchingCubesHashSDF::s_indices;
MeshDataf DX11MarchingCubesHashSDF::s_meshData;

Timer DX11MarchingCubesHashSDF::m_timer;

HRESULT DX11MarchingCubesHashSDF::OnD3D11CreateDevice( ID3D11Device* pd3dDevice )
{
	HRESULT hr = S_OK;

	V_RETURN(initialize(pd3dDevice));

	return  hr;
}

void DX11MarchingCubesHashSDF::OnD3D11DestroyDevice()
{
	destroy();
}

void DX11MarchingCubesHashSDF::clearMeshBuffer() {
	s_meshData.clear();
}

void DX11MarchingCubesHashSDF::saveMesh( const std::string& filename, const mat4f *transform /*= NULL*/ )
{
	std::string dir = util::directoryFromPath(filename);
	if (!util::directoryExists(dir)) util::makeDirectory(dir);

	//create index buffer (required for merging the triangle soup)
	s_meshData.m_FaceIndicesVertices.resize(s_meshData.m_Vertices.size(), std::vector<unsigned int>(3));
	for (unsigned int i = 0; i < (unsigned int)s_meshData.m_FaceIndicesVertices.size()/3; i++) {
		s_meshData.m_FaceIndicesVertices[i][0] = 3*i+0;
		s_meshData.m_FaceIndicesVertices[i][1] = 3*i+1;
		s_meshData.m_FaceIndicesVertices[i][2] = 3*i+2;
	}


	//s_meshData.removeDuplicateVertices();
	//s_meshData.mergeCloseVertices(0.00001f);
	s_meshData.mergeCloseVertices(0.00001f, true);
	s_meshData.removeDuplicateFaces();

	if (transform) {
		s_meshData.applyTransform(*transform);
	}

	std::cout << "saving mesh (" << filename << ") ...";
	MeshIOf::writeToFile(filename, s_meshData);
	std::cout << "done!" << std::endl;

	clearMeshBuffer();
}

HRESULT DX11MarchingCubesHashSDF::extractIsoSurface( ID3D11DeviceContext* context, ID3D11ShaderResourceView* hash, ID3D11ShaderResourceView* sdfBlocksSDF, ID3D11ShaderResourceView* sdfBlocksRGBW, ID3D11Buffer* CBsceneRepSDF, unsigned int hashNumBuckets, unsigned int hashBucketSize, vec3f& minCorner /*= vec3f(0.0f, 0.0f, 0.0f)*/, vec3f& maxCorner /*= vec3f(0.0f, 0.0f, 0.0f)*/, bool boxEnabled /*= false*/ )
{
	HRESULT hr = S_OK;

	// Initialize constant buffer
	D3D11_MAPPED_SUBRESOURCE mappedResource;
	V_RETURN(context->Map(m_constantBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource));

	CBuffer *cbuffer = (CBuffer*)mappedResource.pData;
	cbuffer->boxEnabled = boxEnabled ? 1 : 0;
	memcpy(&cbuffer->minCorner, &minCorner.array[0], sizeof(vec3f));
	memcpy(&cbuffer->maxCorner, &maxCorner.array[0], sizeof(vec3f));

	context->Unmap(m_constantBuffer, 0);

	// Setup pipeline
	unsigned int initialCount = 0;
	context->CSSetUnorderedAccessViews(0, 1, &s_pTrianglesUAV, &initialCount);
	context->CSSetShaderResources(0, 1, &hash);
	context->CSSetShaderResources(1, 1, &sdfBlocksSDF);
	context->CSSetShaderResources(2, 1, &sdfBlocksRGBW);
	context->CSSetConstantBuffers(0, 1, &CBsceneRepSDF);
	context->CSSetConstantBuffers(2, 1, &m_constantBuffer);
	ID3D11Buffer* CBGlobalAppState = GlobalAppState::getInstance().MapAndGetConstantBuffer(context);
	context->CSSetConstantBuffers(8, 1, &CBGlobalAppState);
	context->CSSetShader(m_pComputeShader, 0, 0);

	// Run compute shader
	unsigned int dimX = NUM_GROUPS_X;
	unsigned int dimY = (hashNumBuckets*hashBucketSize + NUM_GROUPS_X - 1) / NUM_GROUPS_X;
	assert(dimX <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);
	assert(dimY <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);

	context->Dispatch(dimX, dimY, 1);

	// Cleanup
	ID3D11UnorderedAccessView* nullUAV[] = {NULL};
	ID3D11ShaderResourceView* nullSRV[] = {NULL, NULL, NULL};
	ID3D11Buffer* nullB[] = {NULL, NULL, NULL};

	context->CSSetUnorderedAccessViews(0, 1, nullUAV, 0);
	context->CSSetShaderResources(0, 3, nullSRV);
	context->CSSetConstantBuffers(0, 3, nullB);
	context->CSSetConstantBuffers(8, 1, nullB);
	context->CSSetShader(0, 0, 0);

	// Copy to CPU
	//D3D11_MAPPED_SUBRESOURCE mappedResource;
	context->CopyStructureCount(s_BuffCountTriangles, 0, s_pTrianglesUAV);
	V_RETURN(context->Map(s_BuffCountTriangles, 0, D3D11_MAP_READ, 0, &mappedResource));
	unsigned int nTriangles = ((unsigned int*)mappedResource.pData)[0];
	context->Unmap(s_BuffCountTriangles, 0);

	if (nTriangles != 0) {
		context->CopyResource(s_pOutputFloatCPU, s_pTriangles);
		V_RETURN(context->Map(s_pOutputFloatCPU, 0, D3D11_MAP_READ, 0, &mappedResource));

		unsigned int baseIdx = (unsigned int)s_meshData.m_Vertices.size();
		s_meshData.m_Vertices.resize(baseIdx + 3*nTriangles);
		s_meshData.m_Colors.resize(baseIdx + 3*nTriangles);

		vec3f* vc = (vec3f*)mappedResource.pData;
		for (unsigned int i = 0; i < 3*nTriangles; i++) {
			s_meshData.m_Vertices[baseIdx + i] = vc[2*i+0];
			s_meshData.m_Colors[baseIdx + i] = vc[2*i+1];
		}

		context->Unmap(s_pOutputFloatCPU, 0);
	}

	return hr;
}

HRESULT DX11MarchingCubesHashSDF::initialize( ID3D11Device* pd3dDevice )
{
	HRESULT hr = S_OK;

	ID3DBlob* pBlob = NULL;

	char SDFBLOCKSIZE[5];
	sprintf_s(SDFBLOCKSIZE, "%d", SDF_BLOCK_SIZE);

	char HANDLECOLLISIONS[5];
	sprintf_s(HANDLECOLLISIONS, "%d", GlobalAppState::getInstance().s_HANDLE_COLLISIONS);

	D3D_SHADER_MACRO shaderDefines[] = { { "SDF_BLOCK_SIZE", SDFBLOCKSIZE }, {"HANDLE_COLLISIONS", HANDLECOLLISIONS }, { 0 } };
	D3D_SHADER_MACRO shaderDefinesWithout[] = { { "SDF_BLOCK_SIZE", SDFBLOCKSIZE }, { 0 } };

	D3D_SHADER_MACRO* validDefines = shaderDefines;
	if(GlobalAppState::getInstance().s_HANDLE_COLLISIONS == 0)
	{
		validDefines = shaderDefinesWithout;
	}

	V_RETURN(CompileShaderFromFile(L"Shaders\\ExtractIsoSurfaceHashSDF.hlsl", "extractIsoSurfaceHashSDFCS", "cs_5_0", &pBlob, validDefines));
	V_RETURN(pd3dDevice->CreateComputeShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &m_pComputeShader))
		SAFE_RELEASE(pBlob);

	D3D11_BUFFER_DESC bDesc;
	bDesc.BindFlags	= D3D11_BIND_CONSTANT_BUFFER;
	bDesc.Usage	= D3D11_USAGE_DYNAMIC;
	bDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	bDesc.MiscFlags	= 0;

	bDesc.ByteWidth	= sizeof(CBuffer);
	V_RETURN(pd3dDevice->CreateBuffer(&bDesc, NULL, &m_constantBuffer));

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

	return  hr;
}

