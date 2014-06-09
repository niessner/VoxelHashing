#include "DX11VoxelGrid.h"

D3DXVECTOR3 DX11VoxelGrid::m_position = D3DXVECTOR3(-1.0f, -1.0f, 0.3f); // position in meters
D3DXVECTOR3 DX11VoxelGrid::m_voxelExtends = D3DXVECTOR3(0.0078125f, 0.0078125f, 0.0078125f); // extend of the voxels in meters
int3 DX11VoxelGrid::m_gridDimensions = int3(256, 256, 256); // number of voxels in each dimension

ID3D11Buffer* DX11VoxelGrid::m_voxelGrid = NULL;
ID3D11ShaderResourceView* DX11VoxelGrid::m_voxelGridSRV = NULL;
ID3D11UnorderedAccessView* DX11VoxelGrid::m_voxelGridUAV = NULL;
