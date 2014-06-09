#pragma once

#include <windows.h>

// This file requires the installation of the DirectX SDK, a link for which is included in the Toolkit Browser
#include <d3d11.h>
#include <xnamath.h>

#include "DX11Utils.h"


#include "Filter.h"
#include "StdOutputLogger.h"
#include "DX11QuadDrawer.h"
#include "DXUT.h"
#include "DX11BuildLinearSystem.h"
#include "GlobalAppState.h"

#include "DXUT.h"
#include "DXUTcamera.h"
#include "DXUTgui.h"
#include "DXUTsettingsDlg.h"
#include "SDKmisc.h"

#include "GlobalAppState.h"
#include "TimingLog.h"

#include "DX11PhongLighting.h"
#include "DX11CameraTrackingMultiRes.h"
#include "DX11NormalizeReduction.h"


#include "DX11RayCastingHashSDF.h"
#include "DX11SceneRepHashSDF.h"
#include "DX11MarchingCubesHashSDF.h"
#include "DX11HistogramHashSDF.h"
#include "DX11SceneRepChunkGrid.h"
#include "DX11MarchingCubesChunkGrid.h"

#include "DX11RayMarchingStepsSplatting.h"
#include "DepthSensor.h"
#include "DX11DepthSensor.h"
#include "TrajectoryLogReader.h"

#include <algorithm>
#include <process.h>



//--------------------------------------------------------------------------------------
// UI control IDs
//--------------------------------------------------------------------------------------
#define IDC_TOGGLEFULLSCREEN      1
#define IDC_TOGGLEREF             3
#define IDC_CHANGEDEVICE          4


//--------------------------------------------------------------------------------------
// Forward declarations 
//--------------------------------------------------------------------------------------
bool CALLBACK		ModifyDeviceSettings( DXUTDeviceSettings* pDeviceSettings, void* pUserContext );
void CALLBACK		OnFrameMove( double fTime, float fElapsedTime, void* pUserContext );
LRESULT CALLBACK	MsgProc( HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam, bool* pbNoFurtherProcessing,	void* pUserContext );
void CALLBACK		OnGUIEvent( UINT nEvent, int nControlID, CDXUTControl* pControl, void* pUserContext );
void CALLBACK		OnKeyboard( UINT nChar, bool bKeyDown, bool bAltDown, void* pUserContext );
bool CALLBACK		IsD3D11DeviceAcceptable( const CD3D11EnumAdapterInfo *AdapterInfo, UINT Output, const CD3D11EnumDeviceInfo *DeviceInfo, DXGI_FORMAT BackBufferFormat, bool bWindowed, void* pUserContext );
HRESULT CALLBACK	OnD3D11CreateDevice( ID3D11Device* pd3dDevice, const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc, void* pUserContext );
HRESULT CALLBACK	OnD3D11ResizedSwapChain( ID3D11Device* pd3dDevice, IDXGISwapChain* pSwapChain,	const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc, void* pUserContext );
void CALLBACK		OnD3D11ReleasingSwapChain( void* pUserContext );
void CALLBACK		OnD3D11DestroyDevice( void* pUserContext );
void CALLBACK		OnD3D11FrameRender( ID3D11Device* pd3dDevice, ID3D11DeviceContext* pd3dImmediateContext, double fTime, float fElapsedTime, void* pUserContext );

void InitApp();
void RenderText();
void RenderHelp();