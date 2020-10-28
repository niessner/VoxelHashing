#pragma once

#include <windows.h>
#include <d3d11.h>
#include <xnamath.h>
#include "DX11Utils.h"
#include "resource.h"

#include "GlobalAppState.h"
#include "GlobalCameraTrackingState.h"
#include "TimingLog.h"
#include "StdOutputLogger.h"
#include "Util.h"

#include "KinectSensor.h"
#include "KinectOneSensor.h"
#include "PrimeSenseSensor.h"
#include "BinaryDumpReader.h"
#include "NetworkSensor.h"
#include "IntelSensor.h"
#include "RealSenseSensor.h"


#include "DXUT.h"
#include "DXUTcamera.h"
#include "DXUTgui.h"
#include "DXUTsettingsDlg.h"
#include "SDKmisc.h"

#include "DX11RGBDRenderer.h"
#include "DX11QuadDrawer.h"
#include "DX11CustomRenderTarget.h"
#include "DX11PhongLighting.h"

#include "CUDARGBDAdapter.h"
#include "CUDARGBDSensor.h"
#include "CUDACameraTrackingMultiRes.h"
#include "CUDACameraTrackingMultiResRGBD.h"
#include "CUDASceneRepHashSDF.h"
#include "CUDARayCastSDF.h"
#include "CUDAMarchingCubesHashSDF.h"
#include "CUDAHistogramHashSDF.h"
#include "CUDASceneRepChunkGrid.h"

#include <iomanip>
#pragma comment(lib, "legacy_stdio_definitions.lib")

#ifdef KINECT
#pragma comment(lib, "Kinect10.lib")
#endif

#ifdef KINECT_ONE
#pragma comment(lib, "Kinect20.lib")
#endif

#ifdef OPEN_NI
#pragma comment(lib, "OpenNI2.lib")
#endif

#ifdef INTEL_SENSOR
#ifdef _DEBUG
#pragma comment(lib, "DSAPI.dbg.lib")
#else
#pragma comment(lib, "DSAPI.lib")
#endif
#endif

#ifdef REAL_SENSE
#ifdef _DEBUG
#pragma comment(lib, "libpxc_d.lib")
#pragma comment(lib, "libpxcutils_d.lib")
#else
#pragma comment(lib, "libpxc.lib")
#pragma comment(lib, "libpxcutils.lib")
#endif
#endif

#ifdef STRUCTURE_SENSOR
#pragma comment(lib, "gdiplus.lib")
#endif

#ifdef OBJECT_SENSING
#include "../../ObjectSensing/ObjectSensing/include/object_sensing.h"
#ifdef _DEBUG
#pragma comment(lib, "../../ObjectSensing/build/lib/ObjectSensing_debug.lib")
#else
#pragma comment(lib, "../../ObjectSensing/build/lib/ObjectSensing_release.lib")
#endif
#endif //OBJECT_SENSING

//--------------------------------------------------------------------------------------
// UI control IDs
//--------------------------------------------------------------------------------------
#define IDC_TOGGLEFULLSCREEN      1
#define IDC_TOGGLEREF             3
#define IDC_CHANGEDEVICE          4
#define IDC_TEST                  5

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

void renderToFile(ID3D11DeviceContext* pd3dImmediateContext);

void InitApp();
void RenderText();
void RenderHelp();



RGBDSensor* getRGBDSensor();
void ResetDepthSensing();
void StopScanningAndExtractIsoSurfaceMC(const std::string& filename = "./Scans/scan.ply", bool overwriteExistingFile = false);
