#include "DX11VoxelGridOperations.h"

//---------------------------------------------------
// Reset
//---------------------------------------------------

unsigned int DX11VoxelGridOperations::m_blockSizeReset = 8;

ID3D11ComputeShader* DX11VoxelGridOperations::m_pComputeShaderReset = NULL;
ID3D11Buffer* DX11VoxelGridOperations::m_constantBufferReset = NULL;

//---------------------------------------------------
// Set distance function ellipsoid
//---------------------------------------------------

unsigned int DX11VoxelGridOperations::m_blockSizeSetDistanceFunctionEllipsoid = 8;

ID3D11ComputeShader* DX11VoxelGridOperations::m_pComputeShaderSetDistanceFunctionEllipsoid = NULL;
ID3D11Buffer* DX11VoxelGridOperations::m_constantBufferSetDistanceFunctionEllipsoid = NULL;

//---------------------------------------------------
// Integrate depth frame
//---------------------------------------------------

mat4f DX11VoxelGridOperations::m_LastRigidTransform;
unsigned int DX11VoxelGridOperations::m_NumIntegratedImages;

unsigned int DX11VoxelGridOperations::m_blockSizeIntegrateDepthFrame = 8;

ID3D11ComputeShader* DX11VoxelGridOperations::m_pComputeShaderIntegrateDepthFrame = NULL;
ID3D11Buffer* DX11VoxelGridOperations::m_constantBufferIntegrateDepthFrame = NULL;

//---------------------------------------------------
// Extract Iso-Surface (Marching Cubes)
//---------------------------------------------------

unsigned int DX11VoxelGridOperations::s_maxNumberOfTriangles = 1000000;
unsigned int DX11VoxelGridOperations::m_blockSizeExtractIsoSurface = 8;

ID3D11ComputeShader* DX11VoxelGridOperations::m_pComputeShaderExtractIsoSurface = NULL;
ID3D11Buffer* DX11VoxelGridOperations::m_constantBufferExtractIsoSurface = NULL;

ID3D11Buffer* DX11VoxelGridOperations::s_pTriangles = NULL;
ID3D11UnorderedAccessView* DX11VoxelGridOperations::s_pTrianglesUAV = NULL;
ID3D11Buffer* DX11VoxelGridOperations::s_BuffCountTriangles = NULL;
ID3D11Buffer* DX11VoxelGridOperations::s_pOutputFloatCPU = NULL;

//---------------------------------------------------------------------------------------------------------------
// Other operation
//---------------------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------------------
// Timer
//---------------------------------------------------------------------------------------------------------------

Timer DX11VoxelGridOperations::m_timer;
