#pragma once

#include <D3D11.h>
#include "DX11Utils.h"

class DX11VoxelGrid
{
	public:

		static HRESULT OnD3D11CreateDevice(ID3D11Device* pd3dDevice)
		{
			HRESULT hr = S_OK;

			V_RETURN(initialize(pd3dDevice));

			return  hr;
		}

		static void OnD3D11DestroyDevice()
		{
			destroy();
		}


		static ID3D11ShaderResourceView* getBufferDataSRV()
		{
			return m_voxelGridSRV;
		}

		static ID3D11UnorderedAccessView* const getBufferDataUAV()
		{
			return m_voxelGridUAV;
		}

		static D3DXVECTOR3* getPosition()
		{
			return &m_position;
		}

		static int3* getGridDimensions()
		{
			return &m_gridDimensions;
		}

		static D3DXVECTOR3* getVoxelExtends()
		{
			return &m_voxelExtends;
		}
		
	private:

		static HRESULT initialize(ID3D11Device* pd3dDevice)
		{
			HRESULT hr = S_OK;

			D3D11_BUFFER_DESC bDesc;
			ZeroMemory(&bDesc, sizeof(D3D11_BUFFER_DESC));
			bDesc.BindFlags	= D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
			bDesc.Usage	= D3D11_USAGE_DEFAULT;
			bDesc.CPUAccessFlags = 0;
			bDesc.MiscFlags	= 0;
			bDesc.ByteWidth	= 2*sizeof(int)*m_gridDimensions.x*m_gridDimensions.y*m_gridDimensions.z;

			D3D11_SHADER_RESOURCE_VIEW_DESC SRVDesc;
			ZeroMemory(&SRVDesc, sizeof(D3D11_SHADER_RESOURCE_VIEW_DESC));
			SRVDesc.Format = DXGI_FORMAT_R32_SINT;
			SRVDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
			SRVDesc.Buffer.FirstElement = 0;
			SRVDesc.Buffer.NumElements = 2*m_gridDimensions.x*m_gridDimensions.y*m_gridDimensions.z;

			D3D11_UNORDERED_ACCESS_VIEW_DESC UAVDesc;
			ZeroMemory(&UAVDesc, sizeof(D3D11_UNORDERED_ACCESS_VIEW_DESC));
			UAVDesc.Format = DXGI_FORMAT_R32_SINT;
			UAVDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
			UAVDesc.Buffer.FirstElement = 0;
			UAVDesc.Buffer.NumElements = 2*m_gridDimensions.x*m_gridDimensions.y*m_gridDimensions.z;

			V_RETURN(pd3dDevice->CreateBuffer(&bDesc, NULL, &m_voxelGrid));
			V_RETURN(pd3dDevice->CreateShaderResourceView(m_voxelGrid, &SRVDesc, &m_voxelGridSRV));
			V_RETURN(pd3dDevice->CreateUnorderedAccessView(m_voxelGrid, &UAVDesc, &m_voxelGridUAV));
			
			return  hr;
		}
		
		static void destroy()
		{
			SAFE_RELEASE(m_voxelGrid);
			SAFE_RELEASE(m_voxelGridSRV);
			SAFE_RELEASE(m_voxelGridUAV);
		}

		static D3DXVECTOR3 m_position; // position of voxel grid
		static D3DXVECTOR3 m_voxelExtends;  // length of voxel in meters
		static int3 m_gridDimensions; // number of voxels in each dimension
		
		static ID3D11Buffer* m_voxelGrid;
		static ID3D11ShaderResourceView* m_voxelGridSRV;
		static ID3D11UnorderedAccessView* m_voxelGridUAV;
};
