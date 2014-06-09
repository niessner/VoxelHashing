#pragma once

/*****************************************************************************/
/* Marching cubes for a HashSDF on the GPU, needs to be run for every chunk  */
/*****************************************************************************/

#include "stdafx.h"

#include <D3D11.h>
#include "DXUT.h"
#include "DX11Utils.h"
#include "TimingLog.h"
#include "removeDuplicate.h"

#include <vector>
#include <string>

#define NUM_GROUPS_X 1024 // to be in-sync with the define in the shader

class DX11MarchingCubesHashSDF
{
	public:
		
		static HRESULT OnD3D11CreateDevice(ID3D11Device* pd3dDevice);

		static void OnD3D11DestroyDevice();

		static void clearMeshBuffer();

		static void saveMesh(const std::string& filename, const mat4f *transform = NULL);

		static HRESULT extractIsoSurface(ID3D11DeviceContext* context, ID3D11ShaderResourceView* hash, ID3D11ShaderResourceView* sdfBlocksSDF, ID3D11ShaderResourceView* sdfBlocksRGBW, ID3D11Buffer* CBsceneRepSDF, unsigned int hashNumBuckets, unsigned int hashBucketSize, vec3f& minCorner = vec3f(0.0f, 0.0f, 0.0f), vec3f& maxCorner = vec3f(0.0f, 0.0f, 0.0f), bool boxEnabled = false);

	private:

		static HRESULT initialize(ID3D11Device* pd3dDevice);


		static void destroy()
		{
			SAFE_RELEASE(m_pComputeShader);
			SAFE_RELEASE(m_constantBuffer);

			SAFE_RELEASE(s_pTriangles);
			SAFE_RELEASE(s_pTrianglesUAV);
			SAFE_RELEASE(s_pOutputFloatCPU);
			SAFE_RELEASE(s_BuffCountTriangles);
		}
		
		// State
		struct CBuffer
		{
			unsigned int boxEnabled;
			float3 minCorner;

			unsigned int align;
			float3 maxCorner;		
		};
										
		static unsigned int m_blockSize;
		static unsigned int s_maxNumberOfTriangles;

		static ID3D11ComputeShader* m_pComputeShader;
		static ID3D11Buffer* m_constantBuffer;

		static ID3D11Buffer* s_pTriangles;
		static ID3D11UnorderedAccessView* s_pTrianglesUAV;
		static ID3D11Buffer* s_pOutputFloatCPU;
		static ID3D11Buffer* s_BuffCountTriangles;

		//static std::vector<Vertex> s_vertices;
		//static std::vector<unsigned int> s_indices;
		static MeshDataf s_meshData;

		static Timer m_timer;
};
