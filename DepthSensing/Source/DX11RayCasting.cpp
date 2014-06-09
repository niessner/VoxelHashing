#include "DX11RayCasting.h"

unsigned int DX11RayCasting::m_blockSize = 16;

ID3D11ComputeShader* DX11RayCasting::m_pComputeShader = NULL;
ID3D11Buffer* DX11RayCasting::m_constantBuffer = NULL;

ID3D11Texture2D* DX11RayCasting::m_pOutputImage2D = NULL;
ID3D11ShaderResourceView* DX11RayCasting::m_pOutputImage2DSRV = NULL;
ID3D11UnorderedAccessView* DX11RayCasting::m_pOutputImage2DUAV = NULL;

// Output
ID3D11Texture2D* DX11RayCasting::s_pColors;
ID3D11ShaderResourceView* DX11RayCasting::s_pColorsSRV;
ID3D11UnorderedAccessView* DX11RayCasting::s_pColorsUAV;

ID3D11Texture2D* DX11RayCasting::s_pPositions;
ID3D11ShaderResourceView* DX11RayCasting::s_pPositionsSRV;
ID3D11UnorderedAccessView* DX11RayCasting::s_pPositionsUAV;

ID3D11Texture2D* DX11RayCasting::s_pNormals;
ID3D11ShaderResourceView* DX11RayCasting::s_pNormalsSRV;
ID3D11UnorderedAccessView* DX11RayCasting::s_pNormalsUAV;


Timer DX11RayCasting::m_timer;
