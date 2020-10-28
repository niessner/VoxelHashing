﻿
#include "stdafx.h"
#include "DepthSensing.h"
#include "StructureSensor.h"
#include "SensorDataReader.h"
#include "SensorDataReaderZhou.h"

//--------------------------------------------------------------------------------------
// Global variables
//--------------------------------------------------------------------------------------

CDXUTDialogResourceManager	g_DialogResourceManager; // manager for shared resources of dialogs
CDXUTTextHelper*            g_pTxtHelper = NULL;
bool						g_renderText = true;
bool						g_bRenderHelp = true;

CModelViewerCamera          g_Camera;               // A model viewing camera
CUDARGBDSensor				g_CudaDepthSensor;
CUDARGBDAdapter				g_RGBDAdapter;
DX11RGBDRenderer			g_RGBDRenderer;
DX11CustomRenderTarget		g_CustomRenderTarget;
DX11CustomRenderTarget		g_RenderToFileTarget;

CUDACameraTrackingMultiRes*		g_cameraTracking	 = NULL;
CUDACameraTrackingMultiResRGBD*	g_cameraTrackingRGBD = NULL;

CUDASceneRepHashSDF*		g_sceneRep			= NULL;
CUDARayCastSDF*				g_rayCast			= NULL;
CUDAMarchingCubesHashSDF*	g_marchingCubesHashSDF = NULL;
CUDAHistrogramHashSDF*		g_historgram = NULL;
CUDASceneRepChunkGrid*		g_chunkGrid = NULL;


RGBDSensor* getRGBDSensor()
{
	static RGBDSensor* g_sensor = NULL;
	if (g_sensor != NULL)	return g_sensor;

	if (GlobalAppState::get().s_sensorIdx == GlobalAppState::Sensor_Kinect) {
#ifdef KINECT
		//static KinectSensor s_kinect;
		//return &s_kinect;
		g_sensor = new KinectSensor;
		return g_sensor;
#else 
		throw MLIB_EXCEPTION("Requires KINECT V1 SDK and enable KINECT macro");
#endif
	}

	if (GlobalAppState::get().s_sensorIdx == GlobalAppState::Sensor_PrimeSense)	{
#ifdef OPEN_NI
		//static PrimeSenseSensor s_primeSense;
		//return &s_primeSense;
		g_sensor = new PrimeSenseSensor;
		return g_sensor;
#else 
		throw MLIB_EXCEPTION("Requires OpenNI 2 SDK and enable OPEN_NI macro");
#endif
	}
	else if (GlobalAppState::getInstance().s_sensorIdx == GlobalAppState::Sensor_KinectOne) {
#ifdef KINECT_ONE
		//static KinectOneSensor s_kinectOne;
		//return &s_kinectOne;
		g_sensor = new KinectOneSensor;
		return g_sensor;
#else
		throw MLIB_EXCEPTION("Requires Kinect 2.0 SDK and enable KINECT_ONE macro");
#endif
	}
	if (GlobalAppState::get().s_sensorIdx == GlobalAppState::Sensor_BinaryDumpReader) {
#ifdef BINARY_DUMP_READER
		//static BinaryDumpReader s_binaryDump;
		//return &s_binaryDump;
		g_sensor = new BinaryDumpReader;
		return g_sensor;
#else 
		throw MLIB_EXCEPTION("Requires BINARY_DUMP_READER macro");
#endif
	}
	if (GlobalAppState::get().s_sensorIdx == GlobalAppState::Sensor_NetworkSensor) {
		//static NetworkSensor s_networkSensor;
		//return &s_networkSensor;
		g_sensor = new NetworkSensor;
		return g_sensor;
	}
	if (GlobalAppState::get().s_sensorIdx == GlobalAppState::Sensor_IntelSensor) {
#ifdef INTEL_SENSOR
		//static IntelSensor s_intelSensor;
		//return &s_intelSensor;
		g_sensor = new IntelSensor;
		return g_sensor;
#else 
		throw MLIB_EXCEPTION("Requires INTEL_SENSOR macro");
#endif
	}
	if (GlobalAppState::get().s_sensorIdx == GlobalAppState::Sensor_RealSense) {
#ifdef REAL_SENSE
		//static RealSenseSensor s_realSenseSensor;
		//return &s_realSenseSensor;
		g_sensor = RealSenseSensor;
		return g_sensor;
#else
		throw MLIB_EXCEPTION("Requires Real Sense SDK and REAL_SENSE macro");
#endif
	}
	if (GlobalAppState::get().s_sensorIdx == GlobalAppState::Sensor_StructureSensor) {
#ifdef STRUCTURE_SENSOR
		//static StructureSensor s_structureSensor;
		//return &s_structureSensor;
		g_sensor = new StructureSensor;
		return g_sensor;
#else
		throw MLIB_EXCEPTION("Requires STRUCTURE_SENSOR macro");
#endif
	}
	if (GlobalAppState::get().s_sensorIdx == GlobalAppState::Sensor_SensorDataReader) {
#ifdef SENSOR_DATA_READER
		//static SensorDataReader s_sensorDataReader;
		//return &s_sensorDataReader;
		g_sensor = new SensorDataReader;
		return g_sensor;
#else
		throw MLIB_EXCEPTION("Requires SENSOR_DATA_READER macro");
#endif
	}
	if (GlobalAppState::get().s_sensorIdx == GlobalAppState::Sensor_SensorDataFileReader) {
#ifdef SENSOR_DATA_FILE_READER
		g_sensor = new SensorDataReaderZhou(1, GlobalAppState::get().s_nVideoFrame, GlobalAppState::get().s_startFrame);
		return g_sensor;
#else
		throw MLIB_EXCEPTION("Requires SENSOR_DATA_FILE_READER macro");
#endif
	}

	throw MLIB_EXCEPTION("unkown sensor id " + std::to_string(GlobalAppState::get().s_sensorIdx));

	return NULL;
}


//--------------------------------------------------------------------------------------
// Initialize the app 
//--------------------------------------------------------------------------------------
void InitApp()
{
}

//--------------------------------------------------------------------------------------
// Called right before creating a D3D9 or D3D10 device, allowing the app to modify the device settings as needed
//--------------------------------------------------------------------------------------
bool CALLBACK ModifyDeviceSettings( DXUTDeviceSettings* pDeviceSettings, void* pUserContext )
{
	// For the first device created if its a REF device, optionally display a warning dialog box
	static bool s_bFirstTime = true;
	if( s_bFirstTime )
	{
		s_bFirstTime = false;
		if( ( DXUT_D3D9_DEVICE == pDeviceSettings->ver && pDeviceSettings->d3d9.DeviceType == D3DDEVTYPE_REF ) ||
			( DXUT_D3D11_DEVICE == pDeviceSettings->ver &&
			pDeviceSettings->d3d11.DriverType == D3D_DRIVER_TYPE_REFERENCE ) )
		{
			DXUTDisplaySwitchingToREFWarning( pDeviceSettings->ver );
		}
	}

	return true;
}

//--------------------------------------------------------------------------------------
// Handle updates to the scene
//--------------------------------------------------------------------------------------
void CALLBACK OnFrameMove( double fTime, float fElapsedTime, void* pUserContext )
{
	g_Camera.FrameMove( fElapsedTime );
	// Update the camera's position based on user input 
}

//--------------------------------------------------------------------------------------
// Render the statistics text
//--------------------------------------------------------------------------------------
void RenderText()
{
	g_pTxtHelper->Begin();
	g_pTxtHelper->SetInsertionPos( 2, 0 );
	g_pTxtHelper->SetForegroundColor( D3DXCOLOR( 1.0f, 1.0f, 0.0f, 1.0f ) );
	g_pTxtHelper->DrawTextLine( DXUTGetFrameStats( DXUTIsVsyncEnabled() ) );
	g_pTxtHelper->DrawTextLine( DXUTGetDeviceStats() );
	if (!g_bRenderHelp) {
		g_pTxtHelper->SetForegroundColor( D3DXCOLOR( 1.0f, 1.0f, 1.0f, 1.0f ) );
		g_pTxtHelper->DrawTextLine(L"\tPress F1 for help");
	}
	g_pTxtHelper->End();


	if (g_bRenderHelp) {
		RenderHelp();
	}
}

void RenderHelp() 
{
	g_pTxtHelper->Begin();
	g_pTxtHelper->SetInsertionPos( 2, 40 );
	g_pTxtHelper->SetForegroundColor( D3DXCOLOR( 1.0f, 0.0f, 0.0f, 1.0f ) );
	g_pTxtHelper->DrawTextLine( L"Controls " );
	g_pTxtHelper->DrawTextLine(L"  \tF1:\t Hide help");
	g_pTxtHelper->DrawTextLine(L"  \tF2:\t Screenshot");
	g_pTxtHelper->DrawTextLine(L"  \t'R':\t Reset scan");
	g_pTxtHelper->DrawTextLine(L"  \t'9':\t Extract geometry (Marching Cubes)");
	g_pTxtHelper->DrawTextLine(L"  \t'8':\t Save recorded input data to sensor file (if enabled)");
	g_pTxtHelper->DrawTextLine(L"  \t'<tab>':\t Switch to free-view mode");
	g_pTxtHelper->DrawTextLine(L"  \t");
	g_pTxtHelper->DrawTextLine(L"  \t'1':\t Visualize reconstruction (default)");
	g_pTxtHelper->DrawTextLine(L"  \t'2':\t Visualize input depth");
	g_pTxtHelper->DrawTextLine(L"  \t'3':\t Visualize input color");
	g_pTxtHelper->DrawTextLine(L"  \t'4':\t Visualize input normals");
	g_pTxtHelper->DrawTextLine(L"  \t'5':\t Visualize phong shaded");
	g_pTxtHelper->DrawTextLine(L"  \t'H':\t GPU hash statistics");
	g_pTxtHelper->DrawTextLine(L"  \t'T':\t Print detailed timings");
	g_pTxtHelper->DrawTextLine(L"  \t'M':\t Debug hash");
	g_pTxtHelper->DrawTextLine(L"  \t'N':\t Save hash to file");
	g_pTxtHelper->DrawTextLine(L"  \t'N':\t Load hash from file");
	g_pTxtHelper->End();
}


//--------------------------------------------------------------------------------------
// Handle messages to the application
//--------------------------------------------------------------------------------------
LRESULT CALLBACK MsgProc( HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam, bool* pbNoFurtherProcessing,
						 void* pUserContext )
{
	// Pass messages to dialog resource manager calls so GUI state is updated correctly
	*pbNoFurtherProcessing = g_DialogResourceManager.MsgProc( hWnd, uMsg, wParam, lParam );
	if( *pbNoFurtherProcessing )
		return 0;

	g_Camera.HandleMessages( hWnd, uMsg, wParam, lParam );

	return 0;
}


void StopScanningAndExtractIsoSurfaceMC(const std::string& filename, bool overwriteExistingFile /*= false*/)
{
	//g_sceneRep->debugHash();
	//g_chunkGrid->debugCheckForDuplicates();

	if (GlobalAppState::get().s_sensorIdx == 7) { //! hack for structure sensor
		std::cout << "[marching cubes] stopped receiving frames from structure sensor" << std::endl;
		getRGBDSensor()->stopReceivingFrames();
	}

	Timer t;

	vec4f posWorld = g_sceneRep->getLastRigidTransform()*GlobalAppState::get().s_streamingPos; // trans lags one frame
	vec3f p(posWorld.x, posWorld.y, posWorld.z);


	g_marchingCubesHashSDF->clearMeshBuffer();
	if (!GlobalAppState::get().s_streamingEnabled) {
		//g_chunkGrid->stopMultiThreading();
		//g_chunkGrid->streamInToGPUAll();
		g_marchingCubesHashSDF->extractIsoSurface(g_sceneRep->getHashData(), g_sceneRep->getHashParams(), g_rayCast->getRayCastData());
		//g_chunkGrid->startMultiThreading();
	} else {
		g_marchingCubesHashSDF->extractIsoSurface(*g_chunkGrid, g_rayCast->getRayCastData(), p, GlobalAppState::getInstance().s_streamingRadius);
	}

	//const mat4f& rigidTransform = g_sceneRep->getLastRigidTransform();
	g_marchingCubesHashSDF->saveMesh(filename, NULL, overwriteExistingFile);

	std::cout << "Mesh generation time " << t.getElapsedTime() << " seconds" << std::endl;

	//g_sceneRep->debugHash();
	//g_chunkGrid->debugCheckForDuplicates();
}

void ResetDepthSensing()
{
	g_sceneRep->reset();
	g_RGBDAdapter.reset();
	g_chunkGrid->reset();
	g_Camera.Reset();
}


void StopScanningAndSaveSDFHash(const std::string& filename = "test.hashgrid") {
	//g_sceneRep->debugHash();
	//g_chunkGrid->debugCheckForDuplicates();

	Timer t;
	std::cout << "saving hash grid to file " << filename << "... ";

	vec4f posWorld = g_sceneRep->getLastRigidTransform()*GlobalAppState::get().s_streamingPos; // trans lags one frame
	vec3f p(posWorld.x, posWorld.y, posWorld.z);

	g_chunkGrid->saveToFile(filename, g_rayCast->getRayCastData(), p, GlobalAppState::getInstance().s_streamingRadius);

	std::cout << "Done!" << std::endl;
	std::cout << "Saving Time " << t.getElapsedTime() << " seconds" << std::endl;

	//g_sceneRep->debugHash();
	//g_chunkGrid->debugCheckForDuplicates();
}


void StopScanningAndLoadSDFHash(const std::string& filename = "test.hashgrid") {
	//g_sceneRep->debugHash();
	//g_chunkGrid->debugCheckForDuplicates();

	Timer t;

	vec4f posWorld = g_sceneRep->getLastRigidTransform()*GlobalAppState::get().s_streamingPos; // trans lags one frame
	vec3f p(posWorld.x, posWorld.y, posWorld.z);

	ResetDepthSensing();
	g_chunkGrid->loadFromFile(filename, g_rayCast->getRayCastData(), p, GlobalAppState::getInstance().s_streamingRadius);

	std::cout << "Loading Time " << t.getElapsedTime() << " seconds" << std::endl;

	GlobalAppState::get().s_integrationEnabled = false;
	std::cout << "Integration enabled == false" << std::endl; 
	GlobalAppState::get().s_trackingEnabled = false;
	std::cout << "Tracking enabled == false" << std::endl;

	//g_sceneRep->debugHash();
	//g_chunkGrid->debugCheckForDuplicates();
}

//--------------------------------------------------------------------------------------
// Handle key presses
//--------------------------------------------------------------------------------------
static int whichScreenshot = 0;


void CALLBACK OnKeyboard( UINT nChar, bool bKeyDown, bool bAltDown, void* pUserContext )
{

	if( bKeyDown ) {
		wchar_t sz[200];

		switch( nChar )
		{
		case VK_F1:
			g_bRenderHelp = !g_bRenderHelp;
			break;
		case VK_F2:
			swprintf_s(sz, 200, L"screenshot%d.bmp", whichScreenshot++);
			DXUTSnapD3D11Screenshot(sz, D3DX11_IFF_BMP);
			std::wcout << std::wstring(sz) << std::endl;
			break;
		case '\t':
			g_renderText = !g_renderText;
			break;
		case '1':
			GlobalAppState::get().s_RenderMode = 1;
			break;
		case '2':
			GlobalAppState::get().s_RenderMode = 2;
			break;
		case '3':
			GlobalAppState::get().s_RenderMode = 3;
			break;
		case '4':
			GlobalAppState::get().s_RenderMode = 4;
			break;
		case '5':
			GlobalAppState::get().s_RenderMode = 5;
			break;
		case '6':
			GlobalAppState::get().s_RenderMode = 6;
			break;
		case '7':
			GlobalAppState::get().s_RenderMode = 7;
			break;
			//case '8':
			//GlobalAppState::get().s_RenderMode = 8;
		case '8':
			{
				if (GlobalAppState::getInstance().s_recordData) {
					if (GlobalAppState::get().s_sensorIdx == GlobalAppState::Sensor_StructureSensor) { //! hack for structure sensor
						std::cout << "[dump frames] stopped receiving frames from structure sensor" << std::endl;
						getRGBDSensor()->stopReceivingFrames();
					}
					g_RGBDAdapter.saveRecordedFramesToFile(GlobalAppState::getInstance().s_recordDataFile);
				} else {
					std::cout << "Cannot save recording: enable \"s_recordData\" in parameter file" << std::endl;
				}
				break;
			}
			break;
		case '9':
			StopScanningAndExtractIsoSurfaceMC();
			break;
		case '0':
			GlobalAppState::get().s_RenderMode = 0;
			break;
		case 'T':
			GlobalAppState::get().s_timingsDetailledEnabled = !GlobalAppState::get().s_timingsDetailledEnabled;
			break;
		case 'Z':
			GlobalAppState::get().s_timingsTotalEnabled = !GlobalAppState::get().s_timingsTotalEnabled;
			break;
			//case VK_F3:
			//	GlobalAppState::get().s_texture_threshold += 0.02;
			//	std::cout<<GlobalAppState::get().s_texture_threshold<<std::endl;
			//	if(GlobalAppState::get().s_texture_threshold>1.0f)
			//		GlobalAppState::get().s_texture_threshold = 1.0f;
			//	break;
			//case VK_F4:
			//	GlobalAppState::get().s_texture_threshold -= 0.02;
			//	std::cout<<GlobalAppState::get().s_texture_threshold<<std::endl;
			//	if(GlobalAppState::get().s_texture_threshold<0.0f)
			//		GlobalAppState::get().s_texture_threshold = 0.0f;
			//	break;
		case 'R':
			ResetDepthSensing();
			break;
		case 'H':
			g_historgram->computeHistrogram(g_sceneRep->getHashData(), g_sceneRep->getHashParams());
			break;
		case 'M':
			g_sceneRep->debugHash();
			if (g_chunkGrid)	g_chunkGrid->debugCheckForDuplicates();
			break;
		case 'D':
			g_RGBDAdapter.getRGBDSensor()->savePointCloud("test.ply");
			break;
		case 'Y':
			{
				float* h_rawDepth = g_RGBDAdapter.getRGBDSensor()->getDepthFloat();
				DepthImage dRawImage(g_RGBDAdapter.getRGBDSensor()->getDepthHeight(), g_RGBDAdapter.getRGBDSensor()->getDepthWidth(), h_rawDepth);
				ColorImageRGB cRawImage(dRawImage);
				FreeImageWrapper::saveImage("raw.png", cRawImage);

				Util::writeToImage(g_RGBDAdapter.getRawDepthMap(), g_RGBDAdapter.getRGBDSensor()->getDepthWidth(), g_RGBDAdapter.getRGBDSensor()->getDepthHeight(), "aRaw.png");
				Util::writeToImage(g_RGBDAdapter.getDepthMapResampledFloat(), g_RGBDAdapter.getWidth(), g_RGBDAdapter.getHeight(), "aResampled.png");
				Util::writeToImage(g_CudaDepthSensor.getDepthCameraData().d_depthData, g_CudaDepthSensor.getDepthCameraParams().m_imageWidth, g_CudaDepthSensor.getDepthCameraParams().m_imageHeight, "depth.png");
				Util::writeToImage(g_rayCast->getRayCastData().d_depth, g_rayCast->getRayCastParams().m_width, g_rayCast->getRayCastParams().m_height, "raycast.png");

				break;
			}
		case 'N':
			StopScanningAndSaveSDFHash("test.hashgrid");
			break;
		case 'B':
			StopScanningAndLoadSDFHash("test.hashgrid");
			break;
		case 'I':
			{
				GlobalAppState::get().s_integrationEnabled = !GlobalAppState::get().s_integrationEnabled;
				if (GlobalAppState::get().s_integrationEnabled)		std::cout << "integration enabled" << std::endl;
				else std::cout << "integration disabled" << std::endl;
			}

		default:
			break;
		}
	}
}

//--------------------------------------------------------------------------------------
// Handles the GUI events
//--------------------------------------------------------------------------------------
void CALLBACK OnGUIEvent( UINT nEvent, int nControlID, CDXUTControl* pControl, void* pUserContext )
{
	switch( nControlID )
	{
		// Standard DXUT controls
	case IDC_TOGGLEFULLSCREEN:
		DXUTToggleFullScreen(); 
		break;
	case IDC_TOGGLEREF:
		DXUTToggleREF(); 
		break;
	case IDC_TEST:
		break;
	}
}

//--------------------------------------------------------------------------------------
// Reject any D3D11 devices that aren't acceptable by returning false
//--------------------------------------------------------------------------------------
bool CALLBACK IsD3D11DeviceAcceptable( const CD3D11EnumAdapterInfo *AdapterInfo, UINT Output, const CD3D11EnumDeviceInfo *DeviceInfo, DXGI_FORMAT BackBufferFormat, bool bWindowed, void* pUserContext )
{
	return true;
}

//--------------------------------------------------------------------------------------
// Create any D3D11 resources that aren't dependent on the back buffer
//--------------------------------------------------------------------------------------
HRESULT CALLBACK OnD3D11CreateDevice(ID3D11Device* pd3dDevice, const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc, void* pUserContext)
{
	Util::printMemoryUseMB("init");

	HRESULT hr = S_OK;

	V_RETURN(GlobalAppState::get().OnD3D11CreateDevice(pd3dDevice));

	ID3D11DeviceContext* pd3dImmediateContext = DXUTGetD3D11DeviceContext();

	V_RETURN( g_DialogResourceManager.OnD3D11CreateDevice( pd3dDevice, pd3dImmediateContext ) );
	g_pTxtHelper = new CDXUTTextHelper( pd3dDevice, pd3dImmediateContext, &g_DialogResourceManager, 15 );

	if (getRGBDSensor() == NULL)
	{
		std::cout << "No RGBD Sensor specified" << std::endl;
		while(1);
	}

	if ( FAILED( getRGBDSensor()->createFirstConnected() ) )
	{
		MessageBox(NULL, L"No ready Depth Sensor found!", L"Error", MB_ICONHAND | MB_OK);
		return S_FALSE;
	}
	//Util::printMemoryUseMB("rgbdsensor");

	//static init
	V_RETURN(g_RGBDAdapter.OnD3D11CreateDevice(pd3dDevice, getRGBDSensor(), GlobalAppState::get().s_adapterWidth, GlobalAppState::get().s_adapterHeight));
	//Util::printMemoryUseMB("rgbdadapter");
	V_RETURN(g_CudaDepthSensor.OnD3D11CreateDevice(pd3dDevice, &g_RGBDAdapter));
	//Util::printMemoryUseMB("cudadepthsensor");

	V_RETURN(DX11QuadDrawer::OnD3D11CreateDevice(pd3dDevice));
	V_RETURN(DX11PhongLighting::OnD3D11CreateDevice(pd3dDevice));

	TimingLog::init();

	std::vector<DXGI_FORMAT> formats;
	formats.push_back(DXGI_FORMAT_R32_FLOAT);
	formats.push_back(DXGI_FORMAT_R32G32B32A32_FLOAT);
	formats.push_back(DXGI_FORMAT_R32G32B32A32_FLOAT);
	formats.push_back(DXGI_FORMAT_R32G32B32A32_FLOAT);

	V_RETURN(g_RGBDRenderer.OnD3D11CreateDevice(pd3dDevice, GlobalAppState::get().s_adapterWidth, GlobalAppState::get().s_adapterHeight));
	V_RETURN(g_CustomRenderTarget.OnD3D11CreateDevice(pd3dDevice, GlobalAppState::get().s_adapterWidth, GlobalAppState::get().s_adapterHeight, formats));
	if (GlobalAppState::get().s_renderToFile) {
		std::vector<DXGI_FORMAT> rtfFormat;
		rtfFormat.push_back(DXGI_FORMAT_R8G8B8A8_UNORM); // _SRGB
		V_RETURN(g_RenderToFileTarget.OnD3D11CreateDevice(pd3dDevice, GlobalAppState::get().s_adapterWidth, GlobalAppState::get().s_adapterHeight, rtfFormat));
	}

	D3DXVECTOR3 vecEye ( 0.0f, 0.0f, 0.0f );
	D3DXVECTOR3 vecAt ( 0.0f, 0.0f, 1.0f );
	g_Camera.SetViewParams( &vecEye, &vecAt );

	//Util::printMemoryUseMB("renderer");

	g_cameraTracking = new CUDACameraTrackingMultiRes(g_RGBDAdapter.getWidth(), g_RGBDAdapter.getHeight(), GlobalCameraTrackingState::get().s_maxLevels);
	//g_cameraTrackingRGBD = new CUDACameraTrackingMultiResRGBD(g_RGBDAdapter.getWidth(), g_RGBDAdapter.getHeight(), GlobalCameraTrackingState::get().s_maxLevels);

	//g_CUDASolverSFS = new CUDAPatchSolverSFS();
	//g_CUDASolverSHLighting = new CUDASolverSHLighting(GlobalAppState::get().s_adapterWidth, GlobalAppState::get().s_adapterHeight);
	//Util::printMemoryUseMB("cameratracking");

	g_sceneRep = new CUDASceneRepHashSDF(CUDASceneRepHashSDF::parametersFromGlobalAppState(GlobalAppState::get()));
	//Util::printMemoryUseMB("scenerep");
	g_rayCast = new CUDARayCastSDF(CUDARayCastSDF::parametersFromGlobalAppState(GlobalAppState::get(), g_RGBDAdapter.getColorIntrinsics(), g_RGBDAdapter.getColorIntrinsicsInv()));
	//Util::printMemoryUseMB("raycast");
	g_marchingCubesHashSDF = new CUDAMarchingCubesHashSDF(CUDAMarchingCubesHashSDF::parametersFromGlobalAppState(GlobalAppState::get()));
	//Util::printMemoryUseMB("marchingcubes");
	g_historgram = new CUDAHistrogramHashSDF(g_sceneRep->getHashParams());
	//Util::printMemoryUseMB("histogram");

	g_chunkGrid = new CUDASceneRepChunkGrid(g_sceneRep, 
		GlobalAppState::get().s_streamingVoxelExtents, 
		GlobalAppState::get().s_streamingGridDimensions,
		GlobalAppState::get().s_streamingMinGridPos,
		GlobalAppState::get().s_streamingInitialChunkListSize,
		GlobalAppState::get().s_streamingEnabled,
		GlobalAppState::get().s_streamingOutParts);
	//Util::printMemoryUseMB("chunkgrid");

	g_sceneRep->bindDepthCameraTextures(g_CudaDepthSensor.getDepthCameraData());

	if (!GlobalAppState::get().s_reconstructionEnabled) {
		GlobalAppState::get().s_RenderMode = 2;
	}

	if (GlobalAppState::get().s_sensorIdx == GlobalAppState::Sensor_StructureSensor) { // structure sensor
		getRGBDSensor()->startReceivingFrames();
	}

	Util::printMemoryUseMB("post-init");

	//std::cout << "waiting..." << std::endl;
	//getchar();
	return hr;
}

//--------------------------------------------------------------------------------------
// Release D3D11 resources created in OnD3D10CreateDevice 
//--------------------------------------------------------------------------------------
void CALLBACK OnD3D11DestroyDevice( void* pUserContext )
{
	g_DialogResourceManager.OnD3D11DestroyDevice();
	DXUTGetGlobalResourceCache().OnDestroyDevice();
	SAFE_DELETE( g_pTxtHelper );

	DX11QuadDrawer::OnD3D11DestroyDevice();
	DX11PhongLighting::OnD3D11DestroyDevice();
	GlobalAppState::get().OnD3D11DestroyDevice();

	g_CudaDepthSensor.OnD3D11DestroyDevice();
	g_RGBDAdapter.OnD3D11DestroyDevice();

	g_RGBDRenderer.OnD3D11DestroyDevice();
	g_CustomRenderTarget.OnD3D11DestroyDevice();
	g_RenderToFileTarget.OnD3D11DestroyDevice();

	SAFE_DELETE(g_cameraTracking);
	SAFE_DELETE(g_cameraTrackingRGBD);

	SAFE_DELETE(g_sceneRep);
	SAFE_DELETE(g_rayCast);
	SAFE_DELETE(g_marchingCubesHashSDF);
	SAFE_DELETE(g_historgram);
	SAFE_DELETE(g_chunkGrid);

	TimingLog::destroy();
}

//--------------------------------------------------------------------------------------
// Create any D3D11 resources that depend on the back buffer
//--------------------------------------------------------------------------------------
HRESULT CALLBACK OnD3D11ResizedSwapChain( ID3D11Device* pd3dDevice, IDXGISwapChain* pSwapChain,
										 const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc, void* pUserContext )
{
	HRESULT hr = S_OK;

	V_RETURN( g_DialogResourceManager.OnD3D11ResizedSwapChain( pd3dDevice, pBackBufferSurfaceDesc ) );

	// Setup the camera's projection parameters
	g_Camera.SetWindow( pBackBufferSurfaceDesc->Width, pBackBufferSurfaceDesc->Height );
	g_Camera.SetButtonMasks( MOUSE_MIDDLE_BUTTON, MOUSE_WHEEL, MOUSE_LEFT_BUTTON );

	//g_Camera.SetRotateButtons(true, false, false);

	float fAspectRatio = pBackBufferSurfaceDesc->Width / ( FLOAT )pBackBufferSurfaceDesc->Height;
	//D3DXVECTOR3 vecEye ( 0.0f, 0.0f, 0.0f );
	//D3DXVECTOR3 vecAt ( 0.0f, 0.0f, 1.0f );
	//g_Camera.SetViewParams( &vecEye, &vecAt );
	g_Camera.SetProjParams( D3DX_PI / 4, fAspectRatio, 0.1f, 10.0f );


	V_RETURN(DX11PhongLighting::OnResize(pd3dDevice, pBackBufferSurfaceDesc->Width, pBackBufferSurfaceDesc->Height));

	return hr;
}

//--------------------------------------------------------------------------------------
// Release D3D11 resources created in OnD3D10ResizedSwapChain 
//--------------------------------------------------------------------------------------
void CALLBACK OnD3D11ReleasingSwapChain( void* pUserContext )
{
	g_DialogResourceManager.OnD3D11ReleasingSwapChain();
}

void reconstruction()
{

	//only if binary dump
	if (GlobalAppState::get().s_sensorIdx == GlobalAppState::Sensor_BinaryDumpReader || GlobalAppState::get().s_sensorIdx == GlobalAppState::Sensor_SensorDataReader || GlobalAppState::get().s_sensorIdx == GlobalAppState::Sensor_SensorDataFileReader) {
		unsigned int heapFreeCount = g_sceneRep->getHeapFreeCount();
		std::cout << "[ frame " << g_RGBDAdapter.getFrameNumber() << " ] " << " [Free SDFBlocks " << heapFreeCount << " ] " << std::endl;
		if (heapFreeCount < 5000) std::cout << "WARNING: Heap Free Count is low!  if crash, increase s_hashNumSDFBlocks" << std::endl;
	}
	
	mat4f transformation = mat4f::identity();
	if ((GlobalAppState::get().s_sensorIdx == GlobalAppState::Sensor_BinaryDumpReader || GlobalAppState::get().s_sensorIdx == GlobalAppState::Sensor_SensorDataReader) 
		&& GlobalAppState::get().s_binaryDumpSensorUseTrajectory) {
		transformation = g_RGBDAdapter.getRigidTransform();

		if (transformation[0] == -std::numeric_limits<float>::infinity() || isnan(transformation[0])) {
			std::cout << "INVALID FRAME" << std::endl;
			return;
		}
		//std::cout << transformation << std::endl;
	}

	if (g_RGBDAdapter.getFrameNumber() > 1) {
		mat4f renderTransform = g_sceneRep->getLastRigidTransform();
		//if we have a pre-recorded trajectory; use it as an init (if specificed to do so)
		if ((GlobalAppState::get().s_sensorIdx == GlobalAppState::Sensor_BinaryDumpReader || GlobalAppState::get().s_sensorIdx == GlobalAppState::Sensor_SensorDataReader)
			&& GlobalAppState::get().s_binaryDumpSensorUseTrajectory
			&& GlobalAppState::get().s_binaryDumpSensorUseTrajectoryOnlyInit) {
			//deltaTransformEstimate = lastTransform.getInverse() * transformation;
			mat4f deltaTransformEstimate = g_RGBDAdapter.getRigidTransform(-1).getInverse() * transformation;
			renderTransform = renderTransform * deltaTransformEstimate;
			g_sceneRep->setLastRigidTransformAndCompactify(renderTransform, g_CudaDepthSensor.getDepthCameraData());
			//TODO if this is enabled there is a problem with the ray interval splatting
		}

		g_rayCast->render(g_sceneRep->getHashData(), g_sceneRep->getHashParams(), g_CudaDepthSensor.getDepthCameraData(), renderTransform);

 
		if (GlobalAppState::get().s_sensorIdx == GlobalAppState::Sensor_NetworkSensor)
		{
			mat4f rigid_transform_from_tango = g_RGBDAdapter.getRigidTransform();
			transformation = g_cameraTracking->applyCT(
				g_CudaDepthSensor.getCameraSpacePositionsFloat4(), g_CudaDepthSensor.getNormalMapFloat4(), g_CudaDepthSensor.getColorMapFilteredFloat4(),
				//g_rayCast->getRayCastData().d_depth4Transformed, g_CudaDepthSensor.getNormalMapNoRefinementFloat4(), g_CudaDepthSensor.getColorMapFilteredFloat4(),
				g_rayCast->getRayCastData().d_depth4, g_rayCast->getRayCastData().d_normals, g_rayCast->getRayCastData().d_colors,
				g_sceneRep->getLastRigidTransform(),
				GlobalCameraTrackingState::getInstance().s_maxInnerIter, GlobalCameraTrackingState::getInstance().s_maxOuterIter,
				GlobalCameraTrackingState::getInstance().s_distThres,	 GlobalCameraTrackingState::getInstance().s_normalThres,
				100.0f, 3.0f,
				g_sceneRep->getLastRigidTransform().getInverse()*rigid_transform_from_tango,
				GlobalCameraTrackingState::getInstance().s_residualEarlyOut,
				g_RGBDAdapter.getDepthIntrinsics(), g_CudaDepthSensor.getDepthCameraData(), 
				NULL);
			if (transformation(0, 0) == -std::numeric_limits<float>::infinity()) {
				std::cout << "Tracking lost in DepthSensing..." << std::endl;
				transformation = rigid_transform_from_tango;
			}
		}
		else
		{
			//static ICPErrorLog errorLog;
			if (!GlobalAppState::get().s_trackingEnabled) {
				transformation.setIdentity();
			}
			else if (
				(GlobalAppState::get().s_sensorIdx == GlobalAppState::Sensor_BinaryDumpReader || GlobalAppState::get().s_sensorIdx == GlobalAppState::Sensor_SensorDataReader)
				&& GlobalAppState::get().s_binaryDumpSensorUseTrajectory
				&& !GlobalAppState::get().s_binaryDumpSensorUseTrajectoryOnlyInit) {
							
				//actually: nothing to do here; transform is already set: just don't do icp and use pre-recorded trajectory

				//transformation = g_RGBDAdapter.getRigidTransform();
				//if (transformation[0] == -std::numeric_limits<float>::infinity()) {
				//	std::cout << "INVALID FRAME" << std::endl;
				//	return;					
				//}
			} else {
				mat4f lastTransform = g_sceneRep->getLastRigidTransform();
				mat4f deltaTransformEstimate = mat4f::identity();
				//if we have a pre-recorded trajectory; use it as an init (if specificed to do so)
				if ((GlobalAppState::get().s_sensorIdx == GlobalAppState::Sensor_BinaryDumpReader || GlobalAppState::get().s_sensorIdx == GlobalAppState::Sensor_SensorDataReader)
					&& GlobalAppState::get().s_binaryDumpSensorUseTrajectory
					&& GlobalAppState::get().s_binaryDumpSensorUseTrajectoryOnlyInit) {
					//deltaTransformEstimate = lastTransform.getInverse() * transformation;	//simple case; not ideal in case of drift
					//deltaTransformEstimate = g_RGBDAdapter.getRigidTransform(-1).getInverse() * transformation;

					/*mat4f lastTransform = g_sceneRep->getLastRigidTransform();
					mat4f deltaTransformEstimate = mat4f::identity();

					transformation = g_cameraTracking->applyCT(
					g_CudaDepthSensor.getCameraSpacePositionsFloat4(), g_CudaDepthSensor.getNormalMapFloat4(), g_CudaDepthSensor.getColorMapFilteredFloat4(),
					//g_rayCast->getRayCastData().d_depth4Transformed, g_CudaDepthSensor.getNormalMapNoRefinementFloat4(), g_CudaDepthSensor.getColorMapFilteredFloat4(),
					g_rayCast->getRayCastData().d_depth4, g_rayCast->getRayCastData().d_normals, g_rayCast->getRayCastData().d_colors,
					lastTransform,
					GlobalCameraTrackingState::getInstance().s_maxInnerIter, GlobalCameraTrackingState::getInstance().s_maxOuterIter,
					GlobalCameraTrackingState::getInstance().s_distThres, GlobalCameraTrackingState::getInstance().s_normalThres,
					100.0f, 3.0f,
					deltaTransformEstimate,
					GlobalCameraTrackingState::getInstance().s_residualEarlyOut,
					g_RGBDAdapter.getDepthIntrinsics(), g_CudaDepthSensor.getDepthCameraData(),
					NULL);

					transformation = g_RGBDAdapter.getRigidTransform();
					if (transformation[0] == -std::numeric_limits<float>::infinity()) {
						std::cout << "INVALID FRAME" << std::endl;
						return;					
					}*/
				}

				const bool useRGBDTracking = false;	//Depth vs RGBD
				if (!useRGBDTracking) {
					transformation = g_cameraTracking->applyCT(
						g_CudaDepthSensor.getCameraSpacePositionsFloat4(), g_CudaDepthSensor.getNormalMapFloat4(), g_CudaDepthSensor.getColorMapFilteredFloat4(),
						//g_rayCast->getRayCastData().d_depth4Transformed, g_CudaDepthSensor.getNormalMapNoRefinementFloat4(), g_CudaDepthSensor.getColorMapFilteredFloat4(),
						g_rayCast->getRayCastData().d_depth4, g_rayCast->getRayCastData().d_normals, g_rayCast->getRayCastData().d_colors,
						lastTransform,
						GlobalCameraTrackingState::getInstance().s_maxInnerIter, GlobalCameraTrackingState::getInstance().s_maxOuterIter,
						GlobalCameraTrackingState::getInstance().s_distThres,	 GlobalCameraTrackingState::getInstance().s_normalThres,
						100.0f, 3.0f,
						deltaTransformEstimate,
						GlobalCameraTrackingState::getInstance().s_residualEarlyOut,
						g_RGBDAdapter.getDepthIntrinsics(), g_CudaDepthSensor.getDepthCameraData(), 
						NULL);
				} else {
					transformation = g_cameraTrackingRGBD->applyCT(
						//g_rayCast->getRayCastData().d_depth4Transformed, g_CudaDepthSensor.getColorMapFilteredFloat4(),
						g_CudaDepthSensor.getCameraSpacePositionsFloat4(), g_CudaDepthSensor.getNormalMapFloat4(), g_CudaDepthSensor.getColorMapFilteredFloat4(),
						g_rayCast->getRayCastData().d_depth4, g_rayCast->getRayCastData().d_normals,  g_rayCast->getRayCastData().d_colors, //g_CudaDepthSensor.getColorMapFilteredLastFrameFloat4(), // g_rayCast->getRayCastData().d_colors,
						lastTransform,
						GlobalCameraTrackingState::getInstance().s_maxInnerIter, GlobalCameraTrackingState::getInstance().s_maxOuterIter,
						GlobalCameraTrackingState::getInstance().s_distThres,	 GlobalCameraTrackingState::getInstance().s_normalThres,
						GlobalCameraTrackingState::getInstance().s_colorGradientMin, GlobalCameraTrackingState::getInstance().s_colorThres,
						100.0f, 3.0f,
						deltaTransformEstimate,
						GlobalCameraTrackingState::getInstance().s_weightsDepth,
						GlobalCameraTrackingState::getInstance().s_weightsColor,
						GlobalCameraTrackingState::getInstance().s_residualEarlyOut,
						g_RGBDAdapter.getDepthIntrinsics(), g_CudaDepthSensor.getDepthCameraData(), 
						NULL);
				}
			}
		}
	}


	if (GlobalAppState::getInstance().s_recordData) {
		g_RGBDAdapter.recordTrajectory(transformation);
	}

	//std::cout << transformation << std::endl;
	//std::cout << g_CudaDepthSensor.getRigidTransform() << std::endl << std::endl;

	if (transformation(0, 0) == -std::numeric_limits<float>::infinity()) {
		std::cout << "!!! TRACKING LOST !!!" << std::endl;
		GlobalAppState::get().s_reconstructionEnabled = false;
		return;
	}

	if (GlobalAppState::get().s_streamingEnabled) {
		vec4f posWorld = transformation*GlobalAppState::getInstance().s_streamingPos; // trans laggs one frame *trans
		vec3f p(posWorld.x, posWorld.y, posWorld.z);

		if (GlobalAppState::get().s_offlineProcessing) {
			for (unsigned int i = 0; i < GlobalAppState::get().s_streamingOutParts; i++) {
				g_chunkGrid->streamOutToCPUPass0GPU(p, GlobalAppState::get().s_streamingRadius, g_chunkGrid->s_useParts, true);
			}
			//g_chunkGrid->streamInToGPUPass1GPU(true);
			unsigned int nStreamedBlocks;		//this calls both streamInToGPUPass0GPU and streamInToGPUPass1GPU
			g_chunkGrid->streamInToGPUAll(p, GlobalAppState::get().s_streamingRadius, g_chunkGrid->s_useParts, nStreamedBlocks);
		}
		else {
			g_chunkGrid->streamOutToCPUPass0GPU(p, GlobalAppState::get().s_streamingRadius, g_chunkGrid->s_useParts, true);
			g_chunkGrid->streamInToGPUPass1GPU(true);
		}


		//g_chunkGrid->debugCheckForDuplicates();
	}

	if (GlobalAppState::get().s_integrationEnabled) {
		g_sceneRep->integrate(transformation, g_CudaDepthSensor.getDepthCameraData(), g_CudaDepthSensor.getDepthCameraParams(), g_chunkGrid->getBitMaskGPU());
	} else {
		//compactification is required for the raycast splatting
		g_sceneRep->setLastRigidTransformAndCompactify(transformation, g_CudaDepthSensor.getDepthCameraData());
	}

	//{
	//	Util::saveScreenshot(g_rayCast->getRayCastData().d_depth, g_rayCast->getRayCastParams().m_width, g_rayCast->getRayCastParams().m_height, "ray/raycast_" + std::to_string(g_RGBDAdapter.getFrameNumber()) + ".png");
	//	g_sceneRep->debug("ray/hash_", g_RGBDAdapter.getFrameNumber());
	//}

	//if (g_RGBDAdapter.getFrameNumber() >= 2800) {
	//	g_RGBDAdapter.saveRecordedFramesToFile(GlobalAppState::getInstance().s_recordDataFile);
	//	StopScanningAndExtractIsoSurfaceMC();
	//	getchar();
	//}
}

//--------------------------------------------------------------------------------------
// Render the scene using the D3D11 device
//--------------------------------------------------------------------------------------

void CALLBACK OnD3D11FrameRender( ID3D11Device* pd3dDevice, ID3D11DeviceContext* pd3dImmediateContext, double fTime, float fElapsedTime, void* pUserContext )
{
	//g_historgram->computeHistrogram(g_sceneRep->getHashData(), g_sceneRep->getHashParams());

	// If the settings dialog is being shown, then render it instead of rendering the app's scene
	//if(g_D3DSettingsDlg.IsActive())
	//{
	//	g_D3DSettingsDlg.OnRender(fElapsedTime);
	//	return;
	//}

	// Clear the back buffer
	static float ClearColor[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
	ID3D11RenderTargetView* pRTV = DXUTGetD3D11RenderTargetView();
	ID3D11DepthStencilView* pDSV = DXUTGetD3D11DepthStencilView();
	pd3dImmediateContext->ClearRenderTargetView(pRTV, ClearColor);
	pd3dImmediateContext->ClearDepthStencilView(pDSV, D3D11_CLEAR_DEPTH, 1.0f, 0);

#if defined(SENSOR_DATA_READER) || defined(SENSOR_DATA_FILE_READER)
	//only if sensor data reader
	if ((GlobalAppState::get().s_sensorIdx == GlobalAppState::Sensor_SensorDataReader || GlobalAppState::get().s_sensorIdx == GlobalAppState::Sensor_SensorDataFileReader) && GlobalAppState::get().s_playData) {
		SensorDataReader* sensor = (SensorDataReader*)getRGBDSensor();

		if (sensor->getCurrFrame() >= sensor->getNumFrames()) {
			//recreate adapter and cuda sensor to use new intrinsics			
			sensor->loadNextSensFile();

			if (GlobalAppState::get().s_playData) {	//this if is a bit of a hack to avoid an overflow...			
				g_RGBDAdapter.OnD3D11DestroyDevice();
				g_CudaDepthSensor.OnD3D11DestroyDevice();
				SAFE_DELETE(g_rayCast);

				g_RGBDAdapter.OnD3D11CreateDevice(DXUTGetD3D11Device(), getRGBDSensor(), GlobalAppState::get().s_adapterWidth, GlobalAppState::get().s_adapterHeight);
				g_CudaDepthSensor.OnD3D11CreateDevice(DXUTGetD3D11Device(), &g_RGBDAdapter);
				g_rayCast = new CUDARayCastSDF(CUDARayCastSDF::parametersFromGlobalAppState(GlobalAppState::get(), g_RGBDAdapter.getColorIntrinsics(), g_RGBDAdapter.getColorIntrinsicsInv()));
				g_sceneRep->bindDepthCameraTextures(g_CudaDepthSensor.getDepthCameraData());
			}
		}
	}
#endif

	// if we have received any valid new depth data we may need to draw
	HRESULT bGotDepth = g_CudaDepthSensor.process(pd3dImmediateContext);

	// Filtering
	g_CudaDepthSensor.setFiterDepthValues(GlobalAppState::get().s_depthFilter, GlobalAppState::get().s_depthSigmaD, GlobalAppState::get().s_depthSigmaR);
	g_CudaDepthSensor.setFiterIntensityValues(GlobalAppState::get().s_colorFilter, GlobalAppState::get().s_colorSigmaD, GlobalAppState::get().s_colorSigmaR);

	HRESULT hr = S_OK;

	///////////////////////////////////////
	// Render
	///////////////////////////////////////

	//Start Timing
	if (GlobalAppState::get().s_timingsDetailledEnabled) { GlobalAppState::get().WaitForGPU(); GlobalAppState::get().s_Timer.start(); }


	mat4f view = MatrixConversion::toMlib(*g_Camera.GetViewMatrix());

	mat4f t = mat4f::identity();
	t(1,1) *= -1.0f;	view = t * view * t;	//t is self-inverse

	if (bGotDepth == S_OK) {
		if (GlobalAppState::getInstance().s_recordData) {
			g_RGBDAdapter.recordFrame();
			if (!GlobalAppState::get().s_reconstructionEnabled) {
				g_RGBDAdapter.recordTrajectory(mat4f::zero());
			}
		}

		if (GlobalAppState::get().s_reconstructionEnabled) {
			reconstruction();
		}
	}

	if(GlobalAppState::get().s_RenderMode == 0) {
		const mat4f renderIntrinsics = g_RGBDAdapter.getColorIntrinsics();

		g_CustomRenderTarget.Clear(pd3dImmediateContext);
		g_CustomRenderTarget.Bind(pd3dImmediateContext);
		g_RGBDRenderer.RenderDepthMap(pd3dImmediateContext, g_CudaDepthSensor.getDepthMapRawFloat(), g_CudaDepthSensor.getColorMapFilteredFloat4(), g_RGBDAdapter.getWidth(), g_RGBDAdapter.getHeight(), g_RGBDAdapter.getColorIntrinsicsInv(), view, renderIntrinsics, g_CustomRenderTarget.getWidth(), g_CustomRenderTarget.getHeight(), GlobalAppState::get().s_renderingDepthDiscontinuityThresOffset, GlobalAppState::get().s_renderingDepthDiscontinuityThresLin);
		g_CustomRenderTarget.Unbind(pd3dImmediateContext);
		DX11PhongLighting::render(pd3dImmediateContext, g_CustomRenderTarget.GetSRV(1), g_CustomRenderTarget.GetSRV(2), g_CustomRenderTarget.GetSRV(3), GlobalAppState::get().s_useColorForRendering, g_CustomRenderTarget.getWidth(), g_CustomRenderTarget.getHeight());
		DX11QuadDrawer::RenderQuad(pd3dImmediateContext, DX11PhongLighting::GetColorsSRV(), 1.0f);
	}
	else if(GlobalAppState::get().s_RenderMode == 1)
	{
		//default render mode
		const mat4f& renderIntrinsics = g_RGBDAdapter.getColorIntrinsics();

		//always render, irrespective whether there is a new depth frame available
		g_CustomRenderTarget.Clear(pd3dImmediateContext);
		g_CustomRenderTarget.Bind(pd3dImmediateContext);
		g_RGBDRenderer.RenderDepthMap(pd3dImmediateContext, g_rayCast->getRayCastData().d_depth, g_rayCast->getRayCastData().d_colors, g_rayCast->getRayCastParams().m_width, g_rayCast->getRayCastParams().m_height, MatrixConversion::toMlib(g_rayCast->getRayCastParams().m_intrinsicsInverse), view, renderIntrinsics, g_CustomRenderTarget.getWidth(), g_CustomRenderTarget.getHeight(), GlobalAppState::get().s_renderingDepthDiscontinuityThresOffset, GlobalAppState::get().s_renderingDepthDiscontinuityThresLin);
		g_CustomRenderTarget.Unbind(pd3dImmediateContext);

		DX11PhongLighting::render(pd3dImmediateContext, g_CustomRenderTarget.GetSRV(1), g_CustomRenderTarget.GetSRV(2), g_CustomRenderTarget.GetSRV(3), GlobalAppState::get().s_useColorForRendering, g_CustomRenderTarget.getWidth(), g_CustomRenderTarget.getHeight());		
		DX11QuadDrawer::RenderQuad(pd3dImmediateContext, DX11PhongLighting::GetColorsSRV(), 1.0f);
#ifdef STRUCTURE_SENSOR
		if (GlobalAppState::get().s_sensorIdx == GlobalAppState::Sensor_StructureSensor) {
			ID3D11Texture2D* pSurface;
			HRESULT hr = DXUTGetDXGISwapChain()->GetBuffer( 0, __uuidof( ID3D11Texture2D ), reinterpret_cast< void** >( &pSurface ) );
			if (pSurface) {
				float* tex = (float*)CreateAndCopyToDebugTexture2D(pd3dDevice, pd3dImmediateContext, pSurface, true); //!!! TODO just copy no create
				((StructureSensor*)getRGBDSensor())->updateFeedbackImage((BYTE*)tex);
				SAFE_DELETE_ARRAY(tex);
			}
		}
#endif
	}
	else if(GlobalAppState::get().s_RenderMode == 2) {
		//DX11QuadDrawer::RenderQuadDynamic(DXUTGetD3D11Device(), pd3dImmediateContext, (float*)g_CudaDepthSensor.getCameraSpacePositionsFloat4(), 4, g_CudaDepthSensor.getColorWidth(), g_CudaDepthSensor.getColorHeight());
		DX11QuadDrawer::RenderQuadDynamic(DXUTGetD3D11Device(), pd3dImmediateContext, (float*)g_CudaDepthSensor.getAndComputeDepthHSV(), 4, g_CudaDepthSensor.getColorWidth(), g_CudaDepthSensor.getColorHeight());
	}
	else if(GlobalAppState::get().s_RenderMode == 3) {
		DX11QuadDrawer::RenderQuadDynamic(DXUTGetD3D11Device(), pd3dImmediateContext, (float*)g_CudaDepthSensor.getColorMapFilteredFloat4(), 4, g_CudaDepthSensor.getColorWidth(), g_CudaDepthSensor.getColorHeight());
	}
	else if(GlobalAppState::get().s_RenderMode == 4) {
		DX11QuadDrawer::RenderQuadDynamic(DXUTGetD3D11Device(), pd3dImmediateContext, (float*)g_CudaDepthSensor.getNormalMapFloat4(), 4, g_CudaDepthSensor.getColorWidth(), g_CudaDepthSensor.getColorHeight());
	}
	else if(GlobalAppState::get().s_RenderMode == 5) {
		//DX11QuadDrawer::RenderQuadDynamic(DXUTGetD3D11Device(), pd3dImmediateContext, (float*)g_rayCast->getRayCastData().d_colors, 4, g_rayCast->getRayCastParams().m_width, g_rayCast->getRayCastParams().m_height);

		//default render mode
		const mat4f& renderIntrinsics = g_RGBDAdapter.getColorIntrinsics();

		g_CustomRenderTarget.Clear(pd3dImmediateContext);
		g_CustomRenderTarget.Bind(pd3dImmediateContext);
		g_RGBDRenderer.RenderDepthMap(pd3dImmediateContext, g_rayCast->getRayCastData().d_depth, g_rayCast->getRayCastData().d_colors, g_rayCast->getRayCastParams().m_width, g_rayCast->getRayCastParams().m_height, MatrixConversion::toMlib(g_rayCast->getRayCastParams().m_intrinsicsInverse), view, renderIntrinsics, g_CustomRenderTarget.getWidth(), g_CustomRenderTarget.getHeight(), GlobalAppState::get().s_renderingDepthDiscontinuityThresOffset, GlobalAppState::get().s_renderingDepthDiscontinuityThresLin);
		g_CustomRenderTarget.Unbind(pd3dImmediateContext);

		DX11PhongLighting::render(pd3dImmediateContext, g_CustomRenderTarget.GetSRV(1), g_CustomRenderTarget.GetSRV(2), g_CustomRenderTarget.GetSRV(3), !GlobalAppState::get().s_useColorForRendering, g_CustomRenderTarget.getWidth(), g_CustomRenderTarget.getHeight());
		DX11QuadDrawer::RenderQuad(pd3dImmediateContext, DX11PhongLighting::GetColorsSRV(), 1.0f);
	}
	else if(GlobalAppState::get().s_RenderMode == 6) {
		//DX11QuadDrawer::RenderQuadDynamic(DXUTGetD3D11Device(), pd3dImmediateContext, (float*)g_rayCast->getRayCastData().d_depth4DV, 4, g_rayCast->getRayCastParams().m_width, g_rayCast->getRayCastParams().m_height, 500.0f);	
	}
	else if(GlobalAppState::get().s_RenderMode == 8) {
		//DX11QuadDrawer::RenderQuadDynamic(DXUTGetD3D11Device(), pd3dImmediateContext, (float*)g_CudaDepthSensor.getColorMapFilteredLastFrameFloat4(), 4, g_CudaDepthSensor.getColorWidth(), g_CudaDepthSensor.getColorHeight());
	}
	else if(GlobalAppState::get().s_RenderMode == 9) {
		const mat4f& renderIntrinsics = g_RGBDAdapter.getColorIntrinsics();

		g_CustomRenderTarget.Clear(pd3dImmediateContext);
		g_CustomRenderTarget.Bind(pd3dImmediateContext);
		g_RGBDRenderer.RenderDepthMap(pd3dImmediateContext, g_CudaDepthSensor.getDepthMapColorSpaceFloat(), g_CudaDepthSensor.getColorMapFilteredFloat4(), g_RGBDAdapter.getWidth(), g_RGBDAdapter.getHeight(), g_RGBDAdapter.getColorIntrinsicsInv(), view, renderIntrinsics, g_CustomRenderTarget.getWidth(), g_CustomRenderTarget.getHeight(), GlobalAppState::get().s_renderingDepthDiscontinuityThresOffset, GlobalAppState::get().s_renderingDepthDiscontinuityThresLin);
		g_CustomRenderTarget.Unbind(pd3dImmediateContext);

		DX11PhongLighting::render(pd3dImmediateContext, g_CustomRenderTarget.GetSRV(1), g_CustomRenderTarget.GetSRV(2), g_CustomRenderTarget.GetSRV(3), GlobalAppState::get().s_useColorForRendering, g_CustomRenderTarget.getWidth(), g_CustomRenderTarget.getHeight());
		DX11QuadDrawer::RenderQuad(pd3dImmediateContext, DX11PhongLighting::GetColorsSRV(), 1.0f);
	}

	// Stop Timing
	if (GlobalAppState::get().s_timingsDetailledEnabled) { GlobalAppState::get().WaitForGPU(); GlobalAppState::get().s_Timer.stop(); TimingLog::totalTimeRender += GlobalAppState::get().s_Timer.getElapsedTimeMS(); TimingLog::countTimeRender++; }


	TimingLog::printTimings();
	if (g_renderText) RenderText();

#ifdef OBJECT_SENSING
	if (bGotDepth == S_OK) ObjectSensing::getInstance()->processFrame(g_sceneRep);
#endif // OBJECT_SENSING

	//debug
	//if (g_RGBDAdapter.getFrameNumber() == 5) GlobalAppState::get().s_playData = false;

	// for scannet
	if (!GlobalAppState::get().s_playData && GlobalAppState::get().s_offlineProcessing) {
		std::string filename = GlobalAppState::get().s_binaryDumpSensorFile[0];
		filename = ml::util::removeExtensions(filename) + "_vh.ply";
		StopScanningAndExtractIsoSurfaceMC(filename, true);
		exit(0);
	}

	if (GlobalAppState::get().s_renderToFile) {
		renderToFile(pd3dImmediateContext);
	}

	DXUT_EndPerfEvent();
}



void renderToFile(ID3D11DeviceContext* pd3dImmediateContext) {

	std::string baseFolder = GlobalAppState::get().s_renderToFileDir;

	const mat4f& lastRigidTransform = g_sceneRep->getLastRigidTransform();
	//static unsigned int frameNumber = 0;
	unsigned int frameNumber = g_RGBDAdapter.getFrameNumber();
	if (!util::directoryExists(baseFolder)) util::makeDirectory(baseFolder);

	const std::string inputColorDir = baseFolder + "/" + "input_color/"; if (!util::directoryExists(inputColorDir)) util::makeDirectory(inputColorDir);
	const std::string inputDepthDir = baseFolder + "/" + "input_depth/"; if (!util::directoryExists(inputDepthDir)) util::makeDirectory(inputDepthDir);
	const std::string reconstructionDir = baseFolder + "/" + "reconstruction/"; if (!util::directoryExists(reconstructionDir)) util::makeDirectory(reconstructionDir);
	const std::string reconstructColorDir = baseFolder + "/" + "reconstruction_color/"; if (!util::directoryExists(reconstructColorDir)) util::makeDirectory(reconstructColorDir);

	g_sceneRep->setLastRigidTransformAndCompactify(lastRigidTransform, g_CudaDepthSensor.getDepthCameraData());
	g_rayCast->render(g_sceneRep->getHashData(), g_sceneRep->getHashParams(), g_CudaDepthSensor.getDepthCameraData(), lastRigidTransform);

	std::stringstream ssFrameNumber;	unsigned int numCountDigits = 6;
	for (unsigned int i = std::max(1u, (unsigned int)std::ceilf(std::log10f((float)frameNumber + 1))); i < numCountDigits; i++) ssFrameNumber << "0";
	ssFrameNumber << frameNumber;

	mat4f view = mat4f::identity();
	{	// reconstruction
		const mat4f& renderIntrinsics = g_RGBDAdapter.getColorIntrinsics();

		g_CustomRenderTarget.Clear(pd3dImmediateContext);
		g_CustomRenderTarget.Bind(pd3dImmediateContext);
		g_RGBDRenderer.RenderDepthMap(pd3dImmediateContext, g_rayCast->getRayCastData().d_depth, g_rayCast->getRayCastData().d_colors, g_rayCast->getRayCastParams().m_width, g_rayCast->getRayCastParams().m_height, MatrixConversion::toMlib(g_rayCast->getRayCastParams().m_intrinsicsInverse), view, renderIntrinsics, g_CustomRenderTarget.getWidth(), g_CustomRenderTarget.getHeight(), GlobalAppState::get().s_renderingDepthDiscontinuityThresOffset, GlobalAppState::get().s_renderingDepthDiscontinuityThresLin);
		g_CustomRenderTarget.Unbind(pd3dImmediateContext);

		bool colored = false;
		DX11PhongLighting::render(pd3dImmediateContext, g_CustomRenderTarget.GetSRV(1), g_CustomRenderTarget.GetSRV(2), g_CustomRenderTarget.GetSRV(3), colored, g_CustomRenderTarget.getWidth(), g_CustomRenderTarget.getHeight());
		
		g_RenderToFileTarget.Clear(pd3dImmediateContext);
		g_RenderToFileTarget.Bind(pd3dImmediateContext);
		DX11QuadDrawer::RenderQuad(pd3dImmediateContext, DX11PhongLighting::GetColorsSRV(), 1.0f);
		g_RenderToFileTarget.Unbind(pd3dImmediateContext);
		
		BYTE* data; unsigned int bytesPerElement;
		g_RenderToFileTarget.copyToHost(data, bytesPerElement);
		ColorImageR8G8B8A8 image(g_RenderToFileTarget.getHeight(), g_RenderToFileTarget.getWidth(), (vec4uc*)data);
		for (unsigned int i = 0; i < image.getWidth()*image.getHeight(); i++) {
			if (image.getDataPointer()[i].x > 0 || image.getDataPointer()[i].y > 0 || image.getDataPointer()[i].z > 0)
				image.getDataPointer()[i].w = 255;
		}
		FreeImageWrapper::saveImage(reconstructionDir + ssFrameNumber.str() + ".jpg", image);
		SAFE_DELETE_ARRAY(data);
	}
	{	// reconstruction color
		const mat4f& renderIntrinsics = g_RGBDAdapter.getColorIntrinsics();


		g_CustomRenderTarget.Clear(pd3dImmediateContext);
		g_CustomRenderTarget.Bind(pd3dImmediateContext);
		g_RGBDRenderer.RenderDepthMap(pd3dImmediateContext, g_rayCast->getRayCastData().d_depth, g_rayCast->getRayCastData().d_colors, g_rayCast->getRayCastParams().m_width, g_rayCast->getRayCastParams().m_height, MatrixConversion::toMlib(g_rayCast->getRayCastParams().m_intrinsicsInverse), view, renderIntrinsics, g_CustomRenderTarget.getWidth(), g_CustomRenderTarget.getHeight(), GlobalAppState::get().s_renderingDepthDiscontinuityThresOffset, GlobalAppState::get().s_renderingDepthDiscontinuityThresLin);
		g_CustomRenderTarget.Unbind(pd3dImmediateContext);

		bool colored = true;
		DX11PhongLighting::render(pd3dImmediateContext, g_CustomRenderTarget.GetSRV(1), g_CustomRenderTarget.GetSRV(2), g_CustomRenderTarget.GetSRV(3), colored, g_CustomRenderTarget.getWidth(), g_CustomRenderTarget.getHeight());

		g_RenderToFileTarget.Clear(pd3dImmediateContext);
		g_RenderToFileTarget.Bind(pd3dImmediateContext);
		DX11QuadDrawer::RenderQuad(pd3dImmediateContext, DX11PhongLighting::GetColorsSRV(), 1.0f);
		g_RenderToFileTarget.Unbind(pd3dImmediateContext);

		BYTE* data; unsigned int bytesPerElement;
		g_RenderToFileTarget.copyToHost(data, bytesPerElement);
		ColorImageR8G8B8A8 image(g_RenderToFileTarget.getHeight(), g_RenderToFileTarget.getWidth(), (vec4uc*)data);
		for (unsigned int i = 0; i < image.getWidth()*image.getHeight(); i++) {
			if (image.getDataPointer()[i].x > 0 || image.getDataPointer()[i].y > 0 || image.getDataPointer()[i].z > 0)
				image.getDataPointer()[i].w = 255;
		}
		FreeImageWrapper::saveImage(reconstructColorDir + ssFrameNumber.str() + ".jpg", image);
		SAFE_DELETE_ARRAY(data);
	}

	// for input color/depth
	{	// input color
		ColorImageR8G8B8A8 image(getRGBDSensor()->getColorHeight(), getRGBDSensor()->getColorWidth(), (vec4uc*)getRGBDSensor()->getColorRGBX());
		for (unsigned int i = 0; i < image.getWidth()*image.getHeight(); i++) {
			if (image.getDataPointer()[i].x > 0 || image.getDataPointer()[i].y > 0 || image.getDataPointer()[i].z > 0)
				image.getDataPointer()[i].w = 255;
		}
		FreeImageWrapper::saveImage(inputColorDir + ssFrameNumber.str() + ".jpg", image);
	}
	{	// input depth
		DepthImage depthImage(getRGBDSensor()->getDepthHeight(), getRGBDSensor()->getDepthWidth(), getRGBDSensor()->getDepthFloat());
		ColorImageR32G32B32A32 image(depthImage);
		for (unsigned int i = 0; i < image.getWidth()*image.getHeight(); i++) {
			if (image.getDataPointer()[i].x > 0 || image.getDataPointer()[i].y > 0 || image.getDataPointer()[i].z > 0)
				image.getDataPointer()[i].w = 1.0f;
			if (depthImage.getDataPointer()[i] == 0.0f || depthImage.getDataPointer()[i] == -std::numeric_limits<float>::infinity())
				image.getDataPointer()[i] = vec4f(0.0f);
		}
		FreeImageWrapper::saveImage(inputDepthDir + ssFrameNumber.str() + ".jpg", image);
	}
}


//--------------------------------------------------------------------------------------
// Entry point to the program. Initializes everything and goes into a message processing 
// loop. Idle time is used to render the scene.
//--------------------------------------------------------------------------------------
//int WINAPI main(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPWSTR lpCmdLine, int nCmdShow)
int main(int argc, char** argv)
{
	// Enable run-time memory check for debug builds.
#if defined(DEBUG) | defined(_DEBUG)
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif

#ifdef OBJECT_SENSING
	ObjectSensing::getInstance()->initQtApp(false);
	ObjectSensing::getInstance()->detach();
#endif // OBJECT_SENSING

	try {
		std::string fileNameDescGlobalApp;
		std::string fileNameDescGlobalTracking;
		if (argc >= 3) {
			fileNameDescGlobalApp = std::string(argv[1]);
			fileNameDescGlobalTracking = std::string(argv[2]);
		}
		else {
			std::cout << "usage: DepthSensing [fileNameDescGlobalApp] [fileNameDescGlobalTracking]" << std::endl;
			//fileNameDescGlobalApp = "zParametersDefault.txt";
			fileNameDescGlobalApp = "zParametersManolisScan.txt";
			
			fileNameDescGlobalTracking = "zParametersTrackingDefault.txt";
		}
		std::cout << VAR_NAME(fileNameDescGlobalApp) << " = " << fileNameDescGlobalApp << std::endl;
		std::cout << VAR_NAME(fileNameDescGlobalTracking) << " = " << fileNameDescGlobalTracking << std::endl;
		std::cout << std::endl;

		//Read the global app state
		ParameterFile parameterFileGlobalApp(fileNameDescGlobalApp);
		std::ofstream out;
		if (argc >= 4) //for scan net: overwrite .sens file
		{
			for (unsigned int i = 0; i < (unsigned int)argc - 3; i++) {
				const std::string filename = std::string(argv[i+3]);
				const std::string paramName = "s_binaryDumpSensorFile[" + std::to_string(i) + "]";
				parameterFileGlobalApp.overrideParameter(paramName, filename);
				std::cout << "Overwriting s_binaryDumpSensorFile; now set to " << filename << std::endl;

				if (i == 0) {
					//redirect stdout to file
					out.open(util::removeExtensions(filename) + ".voxelhashing.log");
					if (!out.is_open()) throw MLIB_EXCEPTION("unable to open log file " + util::removeExtensions(filename) + ".voxelhashing.log");
					//std::streambuf *coutbuf = std::cout.rdbuf(); //save old buf
					std::cout.rdbuf(out.rdbuf()); //redirect std::cout to out.txt!
				}
			}
		}

		GlobalAppState::getInstance().readMembers(parameterFileGlobalApp);
		//GlobalAppState::getInstance().print();

		//Read the global camera tracking state
		ParameterFile parameterFileGlobalTracking(fileNameDescGlobalTracking);
		GlobalCameraTrackingState::getInstance().readMembers(parameterFileGlobalTracking);
		//GlobalCameraTrackingState::getInstance().print();

		{
			//OVERWRITE streaming radius and streaming pose
			auto& gas = GlobalAppState::get();

			float chunkExt = std::max(std::max(gas.s_streamingVoxelExtents.x, gas.s_streamingVoxelExtents.y), gas.s_streamingVoxelExtents.z);
			float chunkRadius = 0.5f*chunkExt*sqrt(3.0f);

			float frustExt = gas.s_SDFMaxIntegrationDistance - gas.s_sensorDepthMin;
			float frustRadius = 0.5f*frustExt*sqrt(3.0f);	//this assumes that the fov is less than 90 degree

			gas.s_streamingPos = vec3f(0.0f, 0.0f, gas.s_sensorDepthMin + 0.5f*frustExt);
			gas.s_streamingRadius = frustRadius + chunkRadius;

			std::cout << "overwriting s_streamingPos,\t now " << gas.s_streamingPos << std::endl;
			std::cout << "overwriting s_streamingRadius,\t now " << gas.s_streamingRadius << std::endl;
		}


		// Set DXUT callbacks
		DXUTSetCallbackDeviceChanging(ModifyDeviceSettings);
		DXUTSetCallbackMsgProc(MsgProc);
		DXUTSetCallbackKeyboard(OnKeyboard);
		DXUTSetCallbackFrameMove(OnFrameMove);

		DXUTSetCallbackD3D11DeviceAcceptable(IsD3D11DeviceAcceptable);
		DXUTSetCallbackD3D11DeviceCreated(OnD3D11CreateDevice);
		DXUTSetCallbackD3D11SwapChainResized(OnD3D11ResizedSwapChain);
		DXUTSetCallbackD3D11FrameRender(OnD3D11FrameRender);
		DXUTSetCallbackD3D11SwapChainReleasing(OnD3D11ReleasingSwapChain);
		DXUTSetCallbackD3D11DeviceDestroyed(OnD3D11DestroyDevice);

		InitApp();
		bool bShowMsgBoxOnError = false;
		DXUTInit(true, bShowMsgBoxOnError); // Parse the command line, show msgboxes on error, and an extra cmd line param to force REF for now
		DXUTSetCursorSettings(true, true); // Show the cursor and clip it when in full screen
		DXUTCreateWindow(GlobalAppState::get().s_windowWidth, GlobalAppState::get().s_windowHeight, L"VoxelHashing", false);

		DXUTSetIsInGammaCorrectMode(false);	//gamma fix (for kinect)

		DXUTCreateDevice(D3D_FEATURE_LEVEL_11_0, true, GlobalAppState::get().s_windowWidth, GlobalAppState::get().s_windowHeight);
		DXUTMainLoop(); // Enter into the DXUT render loop

	}
	catch (const std::exception& e)
	{
		MessageBoxA(NULL, e.what(), "Exception caught", MB_ICONERROR);
		exit(EXIT_FAILURE);
	}
	catch (...)
	{
		MessageBoxA(NULL, "UNKNOWN EXCEPTION", "Exception caught", MB_ICONERROR);
		exit(EXIT_FAILURE);
	}

	//this is a bit of a hack due to a bug in std::thread (a static object cannot join if the main thread exists)
	auto* s = getRGBDSensor();
	SAFE_DELETE(s);

	return DXUTGetExitCode();
}
