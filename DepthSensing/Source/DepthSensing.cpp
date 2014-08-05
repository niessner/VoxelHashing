#include "stdafx.h"
#include <iomanip>

#include "DepthSensing.h"

//--------------------------------------------------------------------------------------
// Global variables
//--------------------------------------------------------------------------------------
CDXUTDialogResourceManager	g_DialogResourceManager; // manager for shared resources of dialogs
CD3DSettingsDlg             g_D3DSettingsDlg;        // Device settings dialog
CDXUTDialog                 g_HUD;                   // manages the 3D   
CDXUTDialog                 g_SampleUI;              // dialog for sample specific controls

CDXUTTextHelper*            g_pTxtHelper = NULL;

CModelViewerCamera          g_Camera;                // A model viewing camera
DX11SceneRepHashSDF			g_SceneRepSDFLocal;
DX11SceneRepHashSDF			g_SceneRepSDFGlobal;	// we had the idea once to use two hashes on the GPU, one inside, and one outside of the frustm; but not used atm
DX11SceneRepChunkGrid		g_SceneRepChunkGrid;
TrajectoryLogReader			g_TrajectoryLogReader;
DX11Sensor					g_Sensor;
bool						g_bRenderHelp = true;



//--------------------------------------------------------------------------------------
// Entry point to the program. Initializes everything and goes into a message processing 
// loop. Idle time is used to render the scene.
//--------------------------------------------------------------------------------------
int main(int argc, char** argv) {

	// Enable run-time memory check for debug builds.
#if defined(DEBUG) | defined(_DEBUG)
	_CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );
#endif

	try {
		StdOutputLogger::start(StdOutputLogger::LOGDEVICE_CONSOLE, StdOutputLogger::LOGDEVICE_CONSOLE);

		std::string fileNameDescGlobalApp;
		std::string fileNameDescGlobalTracking;
		if (argc == 3) {
			fileNameDescGlobalApp = std::string(argv[1]);
			fileNameDescGlobalTracking = std::string(argv[2]);
		} else {
			std::cout << "usage: DepthSensing [fileNameDescGlobalApp] [fileNameDescGlobalTracking]" << std::endl;
			fileNameDescGlobalApp = "zParametersDefault.txt";
			//fileNameDescGlobalApp = "zParametersManolisScan.txt";
			//fileNameDescGlobalApp = "zParametersCSDZFile.txt";
			fileNameDescGlobalTracking = "zParametersTrackingDefault.txt";
		}
		std::cout << VAR_NAME(fileNameDescGlobalApp) << " = " << fileNameDescGlobalApp << std::endl;
		std::cout << VAR_NAME(fileNameDescGlobalTracking) << " = " << fileNameDescGlobalTracking << std::endl;
		
		//Read the global app state
		ParameterFile parameterFileGlobalApp(fileNameDescGlobalApp);
		GlobalAppState::getInstance().readMembers(parameterFileGlobalApp);
		//GlobalAppState::getInstance().print();

		//Read the global camera tracking state
		ParameterFile parameterFileGlobalTracking(fileNameDescGlobalTracking);
		GlobalCameraTrackingState::getInstance().readMembers(parameterFileGlobalTracking);
		//GlobalCameraTrackingState::getInstance().print();

		// Set DXUT callbacks
		DXUTSetCallbackDeviceChanging( ModifyDeviceSettings );
		DXUTSetCallbackMsgProc( MsgProc );
		DXUTSetCallbackKeyboard( OnKeyboard );
		DXUTSetCallbackFrameMove( OnFrameMove );

		DXUTSetCallbackD3D11DeviceAcceptable( IsD3D11DeviceAcceptable );
		DXUTSetCallbackD3D11DeviceCreated( OnD3D11CreateDevice );
		DXUTSetCallbackD3D11SwapChainResized( OnD3D11ResizedSwapChain );
		DXUTSetCallbackD3D11FrameRender( OnD3D11FrameRender );
		DXUTSetCallbackD3D11SwapChainReleasing( OnD3D11ReleasingSwapChain );
		DXUTSetCallbackD3D11DeviceDestroyed( OnD3D11DestroyDevice );


		InitApp();

		DXUTInit( true, true );					// Parse the command line, show msgboxes on error, and an extra cmd line param to force REF for now
		DXUTSetCursorSettings( true, true );	// Show the cursor and clip it when in full screen
		DXUTCreateWindow( L"DepthSensing", false);

		//TODO MADDI FIGURE OUT WHEN THAT IS ACTUALLY REQUIRED
		//if(GlobalAppState::getInstance().s_sensorIdx == 0 || GlobalAppState::getInstance().s_sensorIdx == 3 || GlobalAppState::getInstance().s_sensorIdx == 7)	//this is just a weird thing...
		//if (GlobalAppState::getInstance().s_sensorIdx == 0)
		//{
		//	DXUTSetIsInGammaCorrectMode(false); // Gamma fix for Kinect 4 windows
		//}
		//else {
		//	DXUTSetIsInGammaCorrectMode(true);
		//}

		DXUTSetIsInGammaCorrectMode(false); // Gamma fix for Kinect 4 windows

		//DXUTCreateDevice( D3D_FEATURE_LEVEL_11_0,  true, GlobalAppState::getInstance().s_windowWidth, GlobalAppState::getInstance().s_windowHeight);
		DXUTCreateDevice( D3D_FEATURE_LEVEL_11_0,  true, GlobalAppState::getInstance().s_outputWindowWidth, GlobalAppState::getInstance().s_outputWindowHeight);
		DXUTMainLoop(); // Enter into the DXUT render loop

		StdOutputLogger::stop();

	}
	catch(const std::exception& e)
	{
		MessageBoxA(NULL, e.what(), "Exception caught", MB_ICONERROR);
		exit(EXIT_FAILURE);
	}
	catch(...)
	{
		MessageBoxA(NULL, "UNKNOWN EXCEPTION", "Exception caught", MB_ICONERROR);
		exit(EXIT_FAILURE);
	}
	return DXUTGetExitCode();
}


//--------------------------------------------------------------------------------------
// Initialize the app 
//--------------------------------------------------------------------------------------
void InitApp()
{
	// Initialize dialogs
	g_D3DSettingsDlg.Init( &g_DialogResourceManager );
	g_HUD.Init( &g_DialogResourceManager );
	g_SampleUI.Init( &g_DialogResourceManager );

	g_HUD.SetCallback( OnGUIEvent ); int iY = 20;
	g_HUD.AddButton( IDC_TOGGLEFULLSCREEN, L"Toggle full screen", 0, iY, 170, 22 );
	g_HUD.AddButton( IDC_TOGGLEREF, L"Toggle REF (F3)", 0, iY += 26, 170, 22, VK_F3 );
	g_HUD.AddButton( IDC_CHANGEDEVICE, L"Change device (F2)", 0, iY += 26, 170, 22, VK_F2 );

	g_SampleUI.SetCallback( OnGUIEvent ); iY = 10;


	D3DXVECTOR3 camPos(0.0f, 0.0f, 0.0f);
	D3DXVECTOR3 lookAt(0.0, 0.0f, 1.0f);

	g_Camera.SetProjParams(D3DX_PI/4.0f, 1.3, 0.3, 5.0f);
	g_Camera.SetViewParams(&camPos, &lookAt);

	TimingLog::resetTimings();
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
	// Update the camera's position based on user input 
	g_Camera.FrameMove( fElapsedTime );
}


//--------------------------------------------------------------------------------------
// Render the help and statistics text
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
	g_pTxtHelper->DrawTextLine(L"  \t'<tab>':\t Switch to free-view mode");
	g_pTxtHelper->DrawTextLine(L"  \t'1':\t Visualize input depth");
	g_pTxtHelper->DrawTextLine(L"  \t'2':\t Visualize input normals");
	g_pTxtHelper->DrawTextLine(L"  \t'3':\t Visualize reconstruction (default)");
	g_pTxtHelper->DrawTextLine(L"  \t'4':\t Visualize input color");
	g_pTxtHelper->DrawTextLine(L"  \t'5':\t Visualize phong shaded");
	g_pTxtHelper->DrawTextLine(L"  \t'6':\t Pause/continue the application");
	g_pTxtHelper->DrawTextLine(L"  \t'7':\t GPU hash statistics");

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

	// Pass messages to settings dialog if its active
	if( g_D3DSettingsDlg.IsActive() )
	{
		g_D3DSettingsDlg.MsgProc( hWnd, uMsg, wParam, lParam );
		return 0;
	}

	// Give the dialogs a chance to handle the message first
	*pbNoFurtherProcessing = g_HUD.MsgProc( hWnd, uMsg, wParam, lParam );
	if( *pbNoFurtherProcessing )
		return 0;
	*pbNoFurtherProcessing = g_SampleUI.MsgProc( hWnd, uMsg, wParam, lParam );
	if( *pbNoFurtherProcessing )
		return 0;

	// Pass all remaining windows messages to camera so it can respond to user input
	g_Camera.HandleMessages( hWnd, uMsg, wParam, lParam );

	return 0;
}

//--------------------------------------------------------------------------------------
// Handle key presses
//--------------------------------------------------------------------------------------
void StopScanningAndExtractIsoSurfaceMC()
{
	g_SceneRepChunkGrid.stopMultiThreading();

	Timer t;

	vec4f posWorld = g_SceneRepSDFLocal.GetLastRigidTransform()*GlobalAppState::getInstance().s_StreamingPos; // trans lags one frame
	vec3f p(posWorld.x, posWorld.y, posWorld.z);

	mat4f rigidTransform = g_SceneRepSDFLocal.GetLastRigidTransform();
	DX11MarchingCubesChunkGrid::extractIsoSurface(DXUTGetD3D11DeviceContext(), g_SceneRepChunkGrid, g_SceneRepSDFLocal, p, GlobalAppState::getInstance().s_StreamingRadius, "./Scans/scan.ply", &rigidTransform);



	std::cout << "Mesh Processing Time " << t.getElapsedTime() << " seconds" << std::endl;

	g_SceneRepChunkGrid.startMultiThreading(DXUTGetD3D11DeviceContext());
}

void StopScanningAndDumpVoxelHash() {

	//std::cout << "dumping (local) voxel grid... ";
	//vec4f pos = g_SceneRepSDFLocal.GetLastRigidTransform()*GlobalAppState::getInstance().s_StreamingPos;
	//g_SceneRepSDFLocal.DumpHashToDisk(GlobalAppState::getInstance().s_DumpVoxelGridFile + std::to_string(g_Sensor.GetFrameNumberDepth()) + ".dump", 
	//	GlobalAppState::getInstance().s_StreamingRadius - sqrt(3.0f), pos.getPoint3d());
	//std::cout << "done!" << std::endl;
	//break; 

	g_SceneRepChunkGrid.stopMultiThreading();

	Timer t;
	std::cout << "dumping (local) voxel grid... ";

	vec4f posWorld = g_SceneRepSDFLocal.GetLastRigidTransform()*GlobalAppState::getInstance().s_StreamingPos; // trans lags one frame
	vec3f p(posWorld.x, posWorld.y, posWorld.z);
	mat4f rigidTransform = g_SceneRepSDFLocal.GetLastRigidTransform();

	g_SceneRepChunkGrid.DumpVoxelHash(DXUTGetD3D11DeviceContext(), 
		GlobalAppState::getInstance().s_DumpVoxelGridFile + ".dump", 
		g_SceneRepSDFLocal, p, GlobalAppState::getInstance().s_StreamingRadius);
	std::cout << "done!" << std::endl;

	std::cout << "Dump Voxel Hash Processing Time " << t.getElapsedTime() << " seconds" << std::endl;

	g_SceneRepChunkGrid.startMultiThreading(DXUTGetD3D11DeviceContext());
}

void CALLBACK OnKeyboard( UINT nChar, bool bKeyDown, bool bAltDown, void* pUserContext )
{
	static int whichShot = 0;
	GlobalAppState::getInstance().s_bRenderModeChanged = true;

	if( bKeyDown ) {
		wchar_t sz[200];

		switch( nChar )
		{
		case VK_F1:
			g_bRenderHelp = !g_bRenderHelp;
			break;
		case VK_F2:
			swprintf_s(sz, 200, L"screenshot%d.bmp", whichShot++);
			DXUTSnapD3D11Screenshot(sz, D3DX11_IFF_BMP);

			//swprintf_s(sz, 200, L"depth%d.bmp", whichShot);
			//D3DX11SaveTextureToFileW(DXUTGetD3D11DeviceContext(), g_KinectSensor.GetHSVDepthFloatTexture(), D3DX11_IFF_BMP, sz); 
			//swprintf_s(sz, 200, L"color%d.bmp", whichShot++);
			//D3DX11SaveTextureToFileW(DXUTGetD3D11DeviceContext(), g_KinectSensor.GetColorTexture(), D3DX11_IFF_BMP, sz);

			break;
		case '\t':
			GlobalAppState::getInstance().s_RenderMode = (GlobalAppState::getInstance().s_RenderMode + 1) % 2;
			g_Camera.Reset();
			break;
		case 'B':
			GlobalAppState::getInstance().s_timingsStepsEnabled = !GlobalAppState::getInstance().s_timingsStepsEnabled;
			break;
		case 'N':
			GlobalAppState::getInstance().s_timingsTotalEnabled = !GlobalAppState::getInstance().s_timingsTotalEnabled;
			break;
		case 'K':
			g_SceneRepChunkGrid.printStatistics();
			break;
		case 'L':
			GlobalAppState::getInstance().s_bEnableGlobalLocalStreaming = !GlobalAppState::getInstance().s_bEnableGlobalLocalStreaming;
			std::cout << "toggled global-local streaming" << std::endl;
			break;
		case 'M':
			std::cout << "Debugging SceneRepHash..." << std::endl;
			g_SceneRepSDFLocal.DebugHash();
			g_SceneRepChunkGrid.checkForDuplicates();
			break;
		case 'R':
			TimingLog::resetTimings();

			g_SceneRepSDFLocal.Reset(DXUTGetD3D11DeviceContext());
			g_SceneRepSDFGlobal.Reset(DXUTGetD3D11DeviceContext());

			g_SceneRepChunkGrid.Reset(DXUTGetD3D11DeviceContext());

			g_Sensor.reset();
			g_Camera.Reset();
			break;
		case 'P':	
			GlobalAppState::getInstance().s_useGradients = !GlobalAppState::getInstance().s_useGradients;
			if(GlobalAppState::getInstance().s_useGradients)
			{
				std::cout << "using gradients" << std::endl;
			}
			else
			{
				std::cout << "not using gradients" << std::endl;
			}
			break;
		case 'O':
			break;
		case 'T':
			GlobalAppState::getInstance().s_timingsDetailledEnabled = !GlobalAppState::getInstance().s_timingsDetailledEnabled;
			break;
		case 'I':
			GlobalAppState::getInstance().s_enableMultiLayerSplatting = !GlobalAppState::getInstance().s_enableMultiLayerSplatting;
			if (GlobalAppState::getInstance().s_enableMultiLayerSplatting) {
				std::cout << "using multi layer splatting" << std::endl;
			} else {
				std::cout << "not using multi layer splatting" << std::endl;
			}
			break;
		case 'F':
			GlobalAppState::getInstance().s_bFilterKinectInputData = !GlobalAppState::getInstance().s_bFilterKinectInputData;
			std::cout << "Toggled filter kinect input data" << std::endl;
			break;
		case 'Y':
			GlobalAppState::getInstance().getDepthSensor()->toggleNearMode();
			std::cout << "Toggled near mode" << std::endl;
			break;
		case 'H':
			GlobalAppState::getInstance().getDepthSensor()->toggleAutoWhiteBalance();
			std::cout << "Toggled auto white balance" << std::endl;
			break;
		case 'C':
			GlobalAppState::getInstance().s_bRegistrationEnabled = !GlobalAppState::getInstance().s_bRegistrationEnabled;
			std::cout << "Toggled registration" << std::endl;
			break;
		case '0':
			GlobalAppState::getInstance().s_DisplayTexture = 0;
			std::cout << "Standard rendering" << std::endl;
			break;
		case '1':
			GlobalAppState::getInstance().s_DisplayTexture = 1;
			std::cout << "Kinect Input Depth Float3" << std::endl;
			break;
		case '2':
			GlobalAppState::getInstance().s_DisplayTexture = 2;
			std::cout << "Kinect Input Normal Float3" << std::endl;
			break;
		case '3':
			GlobalAppState::getInstance().s_DisplayTexture = 3;
			std::cout << "Renered Depth Float3" << std::endl;
			break;
		case '4':
			GlobalAppState::getInstance().s_DisplayTexture = 4;
			std::cout << "Kinect Input Colors" << std::endl;
			break;
		case '5':
			GlobalAppState::getInstance().s_DisplayTexture = 5;
			std::cout << "Renered Phong Shaded" << std::endl;
			break;
		case '6':
			{
				GlobalAppState::getInstance().s_bApplicationEnabled = !GlobalAppState::getInstance().s_bApplicationEnabled;
				if(GlobalAppState::getInstance().s_bApplicationEnabled) {
					std::cout << "application started" << std::endl;
				} else {
					std::cout << "application stopped" << std::endl;
				}
				break;
			}
		case '7':
			//g_SceneRepSDFGlobal.RunCompactifyForView(DXUTGetD3D11DeviceContext());
			g_SceneRepSDFLocal.RunCompactifyForView(DXUTGetD3D11DeviceContext());
			//std::cout << "global before: " << g_SceneRepSDFGlobal.GetNumOccupiedHashEntries() << std::endl;
			std::cout << "local before: " << g_SceneRepSDFLocal.GetNumOccupiedHashEntries() << std::endl;
			//DX11HistogramHashSDF::computeHistrogram(DXUTGetD3D11DeviceContext(), g_SceneRepSDFGlobal.GetHashSRV(), g_SceneRepSDFGlobal.MapAndGetConstantBuffer(DXUTGetD3D11DeviceContext()), g_SceneRepSDFGlobal.GetHashNumBuckets(), g_SceneRepSDFGlobal.GetHashBucketSize(), "globalHash");
			DX11HistogramHashSDF::computeHistrogram(DXUTGetD3D11DeviceContext(), g_SceneRepSDFLocal.GetHashSRV(), g_SceneRepSDFLocal.MapAndGetConstantBuffer(DXUTGetD3D11DeviceContext()), g_SceneRepSDFLocal.GetHashNumBuckets(), g_SceneRepSDFLocal.GetHashBucketSize(), "localHash");
			break;

		//case '8':

		//	DX11MarchingCubesHashSDF::clearMeshBuffer();
		//	DX11MarchingCubesHashSDF::extractIsoSurface(DXUTGetD3D11DeviceContext(), g_SceneRepSDFLocal.GetHashSRV(), g_SceneRepSDFLocal.GetSDFBlocksSDFSRV(), g_SceneRepSDFLocal.GetSDFBlocksRGBWSRV(), g_SceneRepSDFLocal.MapAndGetConstantBuffer(DXUTGetD3D11DeviceContext()), g_SceneRepSDFLocal.GetHashNumBuckets(), g_SceneRepSDFLocal.GetHashBucketSize());
		//	DX11MarchingCubesHashSDF::saveMesh("dumpLocal.off");

		//	DX11MarchingCubesHashSDF::clearMeshBuffer();
		//	DX11MarchingCubesHashSDF::extractIsoSurface(DXUTGetD3D11DeviceContext(), g_SceneRepSDFGlobal.GetHashSRV(), g_SceneRepSDFLocal.GetSDFBlocksSDFSRV(), g_SceneRepSDFLocal.GetSDFBlocksRGBWSRV(), g_SceneRepSDFGlobal.MapAndGetConstantBuffer(DXUTGetD3D11DeviceContext()), g_SceneRepSDFGlobal.GetHashNumBuckets(), g_SceneRepSDFGlobal.GetHashBucketSize());
		//	DX11MarchingCubesHashSDF::saveMesh("dumpGlobal.off");

		//	//g_SceneRepSDFLocal.RunCompactifyForView(DXUTGetD3D11DeviceContext());
		//	//g_SceneRepSDFGlobal.RunCompactifyForView(DXUTGetD3D11DeviceContext());
		//	//std::cout << "global before: " << g_SceneRepSDFGlobal.GetNumOccupiedHashEntries() << std::endl;
		//	//std::cout << "local before: " << g_SceneRepSDFLocal.GetNumOccupiedHashEntries() << std::endl;

		//	g_SceneRepSDFLocal.RemoveAndIntegrateToOther(DXUTGetD3D11DeviceContext(), &g_SceneRepSDFGlobal, NULL, false);

		//	//g_SceneRepSDFLocal.RunCompactifyForView(DXUTGetD3D11DeviceContext());
		//	//g_SceneRepSDFGlobal.RunCompactifyForView(DXUTGetD3D11DeviceContext());
		//	//std::cout << "global after: " << g_SceneRepSDFGlobal.GetNumOccupiedHashEntries() << std::endl;
		//	//std::cout << "local after: " << g_SceneRepSDFLocal.GetNumOccupiedHashEntries() << std::endl;
		//	
		//	DX11MarchingCubesHashSDF::clearMeshBuffer();
		//	DX11MarchingCubesHashSDF::extractIsoSurface(DXUTGetD3D11DeviceContext(), g_SceneRepSDFGlobal.GetHashSRV(), g_SceneRepSDFLocal.GetSDFBlocksSDFSRV(), g_SceneRepSDFLocal.GetSDFBlocksRGBWSRV(), g_SceneRepSDFGlobal.MapAndGetConstantBuffer(DXUTGetD3D11DeviceContext()), g_SceneRepSDFGlobal.GetHashNumBuckets(), g_SceneRepSDFGlobal.GetHashBucketSize());
		//	DX11MarchingCubesHashSDF::saveMesh("dumpBoth.off");

		//	//DX11VoxelGridOperations::extractIsoSurface(DXUTGetD3D11DeviceContext(), DX11VoxelGrid::getBufferDataSRV(), DX11VoxelGrid::getPosition(), DX11VoxelGrid::getGridDimensions(), DX11VoxelGrid::getVoxelExtends());
		//	
		//	g_SceneRepSDFGlobal.RemoveAndIntegrateToOther(DXUTGetD3D11DeviceContext(), &g_SceneRepSDFLocal, NULL, false);

		//	//g_SceneRepSDFLocal.RunCompactifyForView(DXUTGetD3D11DeviceContext());
		//	//g_SceneRepSDFGlobal.RunCompactifyForView(DXUTGetD3D11DeviceContext());
		//	//std::cout << "global after after : " << g_SceneRepSDFGlobal.GetNumOccupiedHashEntries() << std::endl;
		//	//std::cout << "local after after : " << g_SceneRepSDFLocal.GetNumOccupiedHashEntries() << std::endl;
		//	break;
		case '8':
			{
				if (GlobalAppState::getInstance().s_RecordData) {
					g_Sensor.saveRecordedFramesToFile(GlobalAppState::getInstance().s_RecordDataFile);
				}
				break;
			}
		case '9':
			{
				StopScanningAndExtractIsoSurfaceMC();
				break;
			}
		case 'G':
			GlobalAppState::getInstance().s_bEnableGarbageCollection = !GlobalAppState::getInstance().s_bEnableGarbageCollection;
			std::cout << "Toggled enable garbage collect" << std::endl;
		case 'D':
			{				
				std::string pointOut = "pointcloud" + std::to_string(g_Sensor.GetFrameNumberDepth()) + ".ply";
				g_Sensor.savePointCloud(pointOut);
			}
			break;
		case 'V':
			{
				//std::cout << "dumping (local) voxel grid... ";
				//vec4f pos = g_SceneRepSDFLocal.GetLastRigidTransform()*GlobalAppState::getInstance().s_StreamingPos;
				//g_SceneRepSDFLocal.DumpHashToDisk(GlobalAppState::getInstance().s_DumpVoxelGridFile + std::to_string(g_Sensor.GetFrameNumberDepth()) + ".dump", 
				//	GlobalAppState::getInstance().s_StreamingRadius - sqrt(3.0f), pos.getPoint3d());
				//std::cout << "done!" << std::endl;
				//break; 
				StopScanningAndDumpVoxelHash();
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
	GlobalAppState::getInstance().s_bRenderModeChanged = true;

	switch( nControlID )
	{
		// Standard DXUT controls
	case IDC_TOGGLEFULLSCREEN:
		DXUTToggleFullScreen(); 
		break;
	case IDC_TOGGLEREF:
		DXUTToggleREF(); 
		break;
	case IDC_CHANGEDEVICE:
		g_D3DSettingsDlg.SetActive( !g_D3DSettingsDlg.IsActive() ); 
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
HRESULT CALLBACK OnD3D11CreateDevice( ID3D11Device* pd3dDevice, const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc, void* pUserContext )
{
	HRESULT hr = S_OK;

	ID3D11DeviceContext* pd3dImmediateContext = DXUTGetD3D11DeviceContext();
	V_RETURN( g_DialogResourceManager.OnD3D11CreateDevice( pd3dDevice, pd3dImmediateContext ) );
	V_RETURN( g_D3DSettingsDlg.OnD3D11CreateDevice( pd3dDevice ) );
	g_pTxtHelper = new CDXUTTextHelper( pd3dDevice, pd3dImmediateContext, &g_DialogResourceManager, 15 );

	if ( FAILED( GlobalAppState::getInstance().getDepthSensor()->createFirstConnected() ) )
	{
		MessageBox(NULL, L"No ready Depth Sensor found!", L"Error", MB_ICONHAND | MB_OK);
		throw MLIB_EXCEPTION("failed to create device");
		return S_FALSE;
	}

	V_RETURN(g_Sensor.OnD3D11CreateDevice(pd3dDevice, GlobalAppState::getInstance().getDepthSensor())); 


	V_RETURN(g_SceneRepSDFLocal.Init(pd3dDevice, false, GlobalAppState::getInstance().s_hashNumBucketsLocal, GlobalAppState::getInstance().s_hashBucketSizeLocal, GlobalAppState::getInstance().s_hashNumSDFBlocks, GlobalAppState::getInstance().s_virtualVoxelSize));
	V_RETURN(g_SceneRepSDFGlobal.Init(pd3dDevice, true, GlobalAppState::getInstance().s_hashNumBucketsGlobal, GlobalAppState::getInstance().s_hashBucketSizeGlobal, 0, GlobalAppState::getInstance().s_virtualVoxelSize));	//ATTENTION TAKE CARE FOR SCAN SIZE

	V_RETURN(g_SceneRepChunkGrid.Init(pd3dDevice, DXUTGetD3D11DeviceContext(), GlobalAppState::getInstance().s_voxelExtends, GlobalAppState::getInstance().s_gridDimensions, GlobalAppState::getInstance().s_minGridPos, GlobalAppState::getInstance().s_virtualVoxelSize, GlobalAppState::getInstance().s_initialChunkListSize));
	V_RETURN(g_SceneRepChunkGrid.Init());


	//static init
	V_RETURN(DX11QuadDrawer::OnD3D11CreateDevice(pd3dDevice));
	V_RETURN(DX11ImageHelper::OnD3D11CreateDevice(pd3dDevice));
	V_RETURN(DX11BuildLinearSystem::OnD3D11CreateDevice(pd3dDevice));
	V_RETURN(DX11PhongLighting::OnD3D11CreateDevice(pd3dDevice));
	V_RETURN(DX11CameraTrackingMultiRes::OnD3D11CreateDevice(pd3dDevice));
	V_RETURN(DX11NormalizeReduction::OnD3D11CreateDevice(pd3dDevice));
	V_RETURN(DX11RayCastingHashSDF::OnD3D11CreateDevice(pd3dDevice));
	V_RETURN(DX11SceneRepHashSDF::OnD3D11CreateDevice(pd3dDevice));
	V_RETURN(DX11MarchingCubesHashSDF::OnD3D11CreateDevice(pd3dDevice));
	V_RETURN(DX11HistogramHashSDF::OnD3D11CreateDevice(pd3dDevice));
	V_RETURN(DX11RayMarchingStepsSplatting::OnD3D11CreateDevice(pd3dDevice));

	V_RETURN(GlobalAppState::getInstance().OnD3D11CreateDevice(pd3dDevice));
	//if(GlobalAppState::getInstance().s_usePreComputedCameraTrajectory) g_TrajectoryLogReader.Init(GlobalAppState::getInstance().s_PreComputedCameraTrajectoryPath);

	return hr;
}

//--------------------------------------------------------------------------------------
// Release D3D11 resources created in OnD3D10CreateDevice 
//--------------------------------------------------------------------------------------
void CALLBACK OnD3D11DestroyDevice( void* pUserContext )
{
	g_DialogResourceManager.OnD3D11DestroyDevice();
	g_D3DSettingsDlg.OnD3D11DestroyDevice();
	DXUTGetGlobalResourceCache().OnDestroyDevice();
	SAFE_DELETE( g_pTxtHelper );

	g_Sensor.OnD3D11DestroyDevice();
	g_SceneRepSDFLocal.Destroy();
	g_SceneRepSDFGlobal.Destroy();

	g_SceneRepChunkGrid.Destroy(DXUTGetD3D11DeviceContext());

	//static delete
	DX11QuadDrawer::OnD3D11DestroyDevice();
	DX11ImageHelper::OnD3D11DestroyDevice();
	DX11BuildLinearSystem::OnD3D11DestroyDevice();
	DX11PhongLighting::OnD3D11DestroyDevice();
	DX11CameraTrackingMultiRes::OnD3D11DestroyDevice();
	DX11NormalizeReduction::OnD3D11DestroyDevice();
	DX11SceneRepHashSDF::OnD3D11DestroyDevice();
	DX11RayCastingHashSDF::OnD3D11DestroyDevice();
	DX11MarchingCubesHashSDF::OnD3D11DestroyDevice();
	DX11HistogramHashSDF::OnD3D11DestroyDevice();
	DX11RayMarchingStepsSplatting::OnD3D11DestroyDevice();


	GlobalAppState::getInstance().OnD3D11DestroyDevice();
}

//--------------------------------------------------------------------------------------
// Create any D3D11 resources that depend on the back buffer
//--------------------------------------------------------------------------------------
HRESULT CALLBACK OnD3D11ResizedSwapChain( ID3D11Device* pd3dDevice, IDXGISwapChain* pSwapChain,
	const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc, void* pUserContext )
{
	HRESULT hr = S_OK;

	V_RETURN( g_DialogResourceManager.OnD3D11ResizedSwapChain( pd3dDevice, pBackBufferSurfaceDesc ) );
	V_RETURN( g_D3DSettingsDlg.OnD3D11ResizedSwapChain( pd3dDevice, pBackBufferSurfaceDesc ) );

	// Setup the camera's projection parameters
	float fAspectRatio = pBackBufferSurfaceDesc->Width / ( FLOAT )pBackBufferSurfaceDesc->Height;
	g_Camera.SetProjParams( D3DX_PI / 4, fAspectRatio, 0.3f, 5.0f );
	g_Camera.SetWindow( pBackBufferSurfaceDesc->Width, pBackBufferSurfaceDesc->Height );
	g_Camera.SetButtonMasks( MOUSE_MIDDLE_BUTTON, MOUSE_WHEEL, MOUSE_LEFT_BUTTON );

	g_HUD.SetLocation( pBackBufferSurfaceDesc->Width - 170, 0 );
	g_HUD.SetSize( 170, 170 );
	g_SampleUI.SetLocation( pBackBufferSurfaceDesc->Width - 170, pBackBufferSurfaceDesc->Height - 300 );
	g_SampleUI.SetSize( 170, 300 );

	return hr;
}

//--------------------------------------------------------------------------------------
// Release D3D11 resources created in OnD3D10ResizedSwapChain 
//--------------------------------------------------------------------------------------
void CALLBACK OnD3D11ReleasingSwapChain( void* pUserContext )
{
	g_DialogResourceManager.OnD3D11ReleasingSwapChain();
}

void DumpAllRendering( ID3D11DeviceContext* pd3dImmediateContext )
{
	unsigned int frameNumber = g_Sensor.GetFrameNumberDepth();

	std::string baseFolder = "output\\";
	CreateDirectory(L"output", NULL);
	CreateDirectory(L"output\\reconstruction", NULL);
	CreateDirectory(L"output\\reconstructionColor", NULL);
	CreateDirectory(L"output\\color", NULL);
	CreateDirectory(L"output\\normal", NULL);
	CreateDirectory(L"output\\depth", NULL);

	std::stringstream ssFrameNumber;	unsigned int numCountDigits = 6;
	for (unsigned int i = std::max(1u, (unsigned int)std::ceilf(std::log10f((float)frameNumber+1))); i < numCountDigits; i++) ssFrameNumber << "0";
	ssFrameNumber << frameNumber;

	{
		//reconstruction (normal)
		static float ClearColor[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
		ID3D11RenderTargetView* pRTV = DXUTGetD3D11RenderTargetView();
		ID3D11DepthStencilView* pDSV = DXUTGetD3D11DepthStencilView();
		pd3dImmediateContext->ClearRenderTargetView(pRTV, ClearColor);
		pd3dImmediateContext->ClearDepthStencilView(pDSV, D3D11_CLEAR_DEPTH, 1.0f, 0);
		DX11PhongLighting::render(pd3dImmediateContext, DX11RayCastingHashSDF::getPositonsImageSRV(), DX11RayCastingHashSDF::getNormalsImageSRV(), DX11RayCastingHashSDF::getColorsImageSRV(), DX11RayCastingHashSDF::getSSAOMapFilteredSRV(), false, false);

		std::string fileName = baseFolder + "reconstruction\\" + ssFrameNumber.str() + ".png";
		std::wstring fileNameW(fileName.begin(), fileName.end());
		DXUTSnapD3D11Screenshot(fileNameW.c_str(), D3DX11_IFF_PNG);
	}
	{
		//reconstruction (color)
		static float ClearColor[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
		ID3D11RenderTargetView* pRTV = DXUTGetD3D11RenderTargetView();
		ID3D11DepthStencilView* pDSV = DXUTGetD3D11DepthStencilView();
		pd3dImmediateContext->ClearRenderTargetView(pRTV, ClearColor);
		pd3dImmediateContext->ClearDepthStencilView(pDSV, D3D11_CLEAR_DEPTH, 1.0f, 0);
		DX11PhongLighting::render(pd3dImmediateContext, DX11RayCastingHashSDF::getPositonsImageSRV(), DX11RayCastingHashSDF::getNormalsImageSRV(), DX11RayCastingHashSDF::getColorsImageSRV(), DX11RayCastingHashSDF::getSSAOMapFilteredSRV(), true, true);

		std::string fileName = baseFolder + "reconstructionColor\\" + ssFrameNumber.str() + ".png";
		std::wstring fileNameW(fileName.begin(), fileName.end());
		DXUTSnapD3D11Screenshot(fileNameW.c_str(), D3DX11_IFF_PNG);
	}
	{
		//input color
		static float ClearColor[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
		ID3D11RenderTargetView* pRTV = DXUTGetD3D11RenderTargetView();
		ID3D11DepthStencilView* pDSV = DXUTGetD3D11DepthStencilView();
		pd3dImmediateContext->ClearRenderTargetView(pRTV, ClearColor);
		pd3dImmediateContext->ClearDepthStencilView(pDSV, D3D11_CLEAR_DEPTH, 1.0f, 0);
		DX11QuadDrawer::RenderQuad(pd3dImmediateContext, g_Sensor.GetColorSRV());
		std::string fileName = baseFolder + "color\\" + ssFrameNumber.str() + ".png";
		std::wstring fileNameW(fileName.begin(), fileName.end());
		DXUTSnapD3D11Screenshot(fileNameW.c_str(), D3DX11_IFF_PNG);
	}
	{
		//input depth
		static float ClearColor[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
		ID3D11RenderTargetView* pRTV = DXUTGetD3D11RenderTargetView();
		ID3D11DepthStencilView* pDSV = DXUTGetD3D11DepthStencilView();
		pd3dImmediateContext->ClearRenderTargetView(pRTV, ClearColor);
		pd3dImmediateContext->ClearDepthStencilView(pDSV, D3D11_CLEAR_DEPTH, 1.0f, 0);
		DX11QuadDrawer::RenderQuad(pd3dImmediateContext, g_Sensor.GetHSVDepthFloat4SRV(), 1.0f);
		std::string fileName = baseFolder + "depth\\" + ssFrameNumber.str() + ".png";
		std::wstring fileNameW(fileName.begin(), fileName.end());
		DXUTSnapD3D11Screenshot(fileNameW.c_str(), D3DX11_IFF_PNG);
	}
	{
		//input normal
		static float ClearColor[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
		ID3D11RenderTargetView* pRTV = DXUTGetD3D11RenderTargetView();
		ID3D11DepthStencilView* pDSV = DXUTGetD3D11DepthStencilView();
		pd3dImmediateContext->ClearRenderTargetView(pRTV, ClearColor);
		pd3dImmediateContext->ClearDepthStencilView(pDSV, D3D11_CLEAR_DEPTH, 1.0f, 0);
		DX11QuadDrawer::RenderQuad(pd3dImmediateContext, g_Sensor.GetNormalFloat4SRV(), 1.0f);
		std::string fileName = baseFolder + "normal\\" + ssFrameNumber.str() + ".png";
		std::wstring fileNameW(fileName.begin(), fileName.end());
		DXUTSnapD3D11Screenshot(fileNameW.c_str(), D3DX11_IFF_PNG);
	}
}


//--------------------------------------------------------------------------------------
// Render the scene using the D3D11 device
//--------------------------------------------------------------------------------------
void CALLBACK OnD3D11FrameRender( ID3D11Device* pd3dDevice, ID3D11DeviceContext* pd3dImmediateContext, double fTime, float fElapsedTime, void* pUserContext )
{
	if(GlobalAppState::getInstance().s_bApplicationEnabled) {
		g_SceneRepSDFLocal.SetEnableGarbageCollect(GlobalAppState::getInstance().s_bEnableGarbageCollection);
	
	
		if (GlobalAppState::getInstance().s_bRenderModeChanged) {
			TimingLog::resetTimings();
			GlobalAppState::getInstance().s_bRenderModeChanged = false;
		}

		// If the settings dialog is being shown, then render it instead of rendering the app's scene
		/*if( g_D3DSettingsDlg.IsActive() )
		{
			g_D3DSettingsDlg.OnRender( fElapsedTime );
			return;
		}*/

		g_Sensor.setFiterDepthValues(GlobalAppState::getInstance().s_bFilterKinectInputData, 2.5f, 0.03f);

		// Clear the back buffer
		static float ClearColor[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
		ID3D11RenderTargetView* pRTV = DXUTGetD3D11RenderTargetView();
		ID3D11DepthStencilView* pDSV = DXUTGetD3D11DepthStencilView();
		pd3dImmediateContext->ClearRenderTargetView(pRTV, ClearColor);
		pd3dImmediateContext->ClearDepthStencilView(pDSV, D3D11_CLEAR_DEPTH, 1.0f, 0);

		// if we have received any valid new depth data we may need to draw
		if (GlobalAppState::getInstance().s_timingsTotalEnabled) {
			GlobalAppState::getInstance().WaitForGPU();
			GlobalAppState::getInstance().s_Timer.start();
		}

		if (GlobalAppState::getInstance().s_timingsStepsEnabled) {
			GlobalAppState::getInstance().WaitForGPU();
			GlobalAppState::getInstance().s_Timer.start();
		}

		HRESULT hr0 = g_Sensor.processDepth(pd3dImmediateContext); // shouldn't hr0 and h1 be used to check if new work has to be done? registration etc
		HRESULT hr1 = g_Sensor.processColor(pd3dImmediateContext);

		if (hr0 == S_OK) {
			//bool printFrameNumbers = true;
			//if (printFrameNumbers) {
			//	std::cout << g_Sensor.GetFrameNumberDepth() << std::endl;
			//}
			if (GlobalAppState::getInstance().s_DataDumpDepthData) {
				std::stringstream ss;	ss << GlobalAppState::getInstance().s_DataDumpPath;
				for (unsigned int i = std::max(1u,g_Sensor.GetFrameNumberDepth()); i < 1000000; i *= 10) ss << "0";
				ss << g_Sensor.GetFrameNumberDepth() << ".mbindepth";
				std::cout << "Dumping " << ss.str() << std::endl;
				g_Sensor.writeDepthDataToFile(ss.str());
			}
			if (GlobalAppState::getInstance().s_DataDumpColorData) {
				std::stringstream ss;	ss << GlobalAppState::getInstance().s_DataDumpPath;
				for (unsigned int i = std::max(1u,g_Sensor.GetFrameNumberDepth()); i < 1000000; i *= 10) ss << "0";
				ss << g_Sensor.GetFrameNumberDepth() << ".binRGB";
				std::cout << "Dumping " << ss.str() << std::endl;
				g_Sensor.writeColorDataToFile(ss.str());
			}

			if (GlobalAppState::getInstance().s_RecordData) {
				g_Sensor.recordFrame();
			}
		}

		//dumps data at a specific frame number (if specified)
		if (GlobalAppState::getInstance().s_ImageReaderSensorNumFrames > 0) {
			if (g_Sensor.GetFrameNumberDepth() == GlobalAppState::getInstance().s_ImageReaderSensorNumFrames - 1)	{
				StopScanningAndExtractIsoSurfaceMC();	
			}
		}

		if (GlobalAppState::getInstance().s_timingsStepsEnabled) {
			GlobalAppState::getInstance().WaitForGPU();
			GlobalAppState::getInstance().s_Timer.stop();
			TimingLog::countTimeMisc++;
			TimingLog::totalTimeMisc += GlobalAppState::getInstance().s_Timer.getElapsedTimeMS();
		}

		HRESULT hr = S_OK;

		if (GlobalAppState::getInstance().s_DisplayTexture == 0 || GlobalAppState::getInstance().s_DisplayTexture == 3 || GlobalAppState::getInstance().s_DisplayTexture == 5) {

			// Voxel Hashing
			mat4f trans = g_SceneRepSDFLocal.GetLastRigidTransform();
			if (GlobalAppState::getInstance().s_RenderMode == RENDERMODE_VIEW) {
				D3DXMATRIX view = *g_Camera.GetViewMatrix();
				D3DXMatrixInverse(&view, NULL, &view);
				D3DXMatrixTranspose(&view, &view);
				trans = trans * *(mat4f*)&view;

				vec4f posWorld = trans*GlobalAppState::getInstance().s_StreamingPos; // trans laggs one frame *trans
				vec3f p(posWorld.x, posWorld.y, posWorld.z);


				//g_SceneRepChunkGrid.StreamOutToCPUPass0GPU(DXUTGetD3D11DeviceContext(), g_SceneRepSDFLocal, p, GlobalAppState::getInstance().s_StreamingRadius, true, true);
				//g_SceneRepChunkGrid.StreamInToGPUPass1GPU(DXUTGetD3D11DeviceContext(), g_SceneRepSDFLocal, true);

				//Timer timer;
				//// Start query for timing
				////if(GlobalAppState::getInstance().s_timingsEnabled)
				//{
				//	GlobalAppState::getInstance().WaitForGPU();
				//	g_timer.start();
				//}
		
				//// Wait for query
				////if(GlobalAppState::getInstance().s_timingsEnabled)
				//{
				//	GlobalAppState::getInstance().WaitForGPU();
				//	//TimingLog::totalTimeStreamIn+=timer.getElapsedTimeMS();
				//	//TimingLog::countStreamIn++;

				//	double time = g_timer.getElapsedTimeMS();
				//	//std::cout << time << std::endl;
				//}
			}

			if (!(GlobalAppState::getInstance().s_RenderMode == RENDERMODE_INTEGRATE && GlobalAppState::getInstance().s_bRegistrationEnabled && hr0 == S_OK)) {
				if (GlobalAppState::getInstance().s_timingsDetailledEnabled) {
					GlobalAppState::getInstance().WaitForGPU();
					GlobalAppState::getInstance().s_Timer.start();
				}

				if (GlobalAppState::getInstance().s_bEnableGlobalLocalStreaming) {
					g_SceneRepSDFLocal.RemoveAndIntegrateToOther(pd3dImmediateContext, &g_SceneRepSDFGlobal, &trans, true);
					g_SceneRepSDFGlobal.RemoveAndIntegrateToOther(pd3dImmediateContext, &g_SceneRepSDFLocal, &trans, false);
				}

				if (GlobalAppState::getInstance().s_timingsDetailledEnabled) {
					GlobalAppState::getInstance().WaitForGPU();
					GlobalAppState::getInstance().s_Timer.stop();
					TimingLog::countRemoveAndIntegrate++;
					TimingLog::totalTimeRemoveAndIntegrate += GlobalAppState::getInstance().s_Timer.getElapsedTimeMS();
				}
			}

			if (GlobalAppState::getInstance().s_RenderMode == RENDERMODE_VIEW) {
				g_SceneRepSDFLocal.RunCompactifyForView(pd3dImmediateContext);
				g_SceneRepSDFGlobal.RunCompactifyForView(pd3dImmediateContext);
				//std::cout << "Occupied Local  Entries: " << g_SceneRepSDFLocal.GetNumOccupiedHashEntries() << std::endl;
				//std::cout << "Occupied Global Entries: " << g_SceneRepSDFGlobal.GetNumOccupiedHashEntries() << std::endl;
				//UINT numOccupied = g_SceneRepSDFLocal.GetNumOccupiedHashEntries() + g_SceneRepSDFGlobal.GetNumOccupiedHashEntries();
				//std::cout << "Occupied + free: " << numOccupied + g_SceneRepSDFLocal.GetHeapFreeCount(pd3dImmediateContext) << std::endl;
			}

			if (GlobalAppState::getInstance().s_timingsStepsEnabled) {
				GlobalAppState::getInstance().WaitForGPU();
				GlobalAppState::getInstance().s_Timer.start();
			}
			DX11RayCastingHashSDF::Render(pd3dImmediateContext, g_SceneRepSDFLocal.GetHashSRV(), g_SceneRepSDFLocal.GetHashCompactifiedSRV(), g_SceneRepSDFLocal.GetSDFBlocksSDFSRV(), g_SceneRepSDFLocal.GetSDFBlocksRGBWSRV(), g_SceneRepSDFLocal.GetNumOccupiedHashEntries(), DXUTGetWindowWidth(), DXUTGetWindowHeight(), &trans, g_SceneRepSDFLocal.MapAndGetConstantBuffer(pd3dImmediateContext));

			DX11RayCastingHashSDF::RenderStereo(pd3dImmediateContext, g_SceneRepSDFLocal.GetHashSRV(), g_SceneRepSDFLocal.GetHashCompactifiedSRV(), g_SceneRepSDFLocal.GetSDFBlocksSDFSRV(), g_SceneRepSDFLocal.GetSDFBlocksRGBWSRV(), g_SceneRepSDFLocal.GetNumOccupiedHashEntries(), GlobalAppState::getInstance().s_windowWidthStereo, GlobalAppState::getInstance().s_windowHeightStereo, &trans, g_SceneRepSDFLocal.MapAndGetConstantBuffer(pd3dImmediateContext));

			if (GlobalAppState::getInstance().s_timingsStepsEnabled) {
				GlobalAppState::getInstance().WaitForGPU();
				GlobalAppState::getInstance().s_Timer.stop();
				TimingLog::countTimeRender++;
				TimingLog::totalTimeRender += GlobalAppState::getInstance().s_Timer.getElapsedTimeMS();
			}

			if(GlobalAppState::getInstance().s_DisplayTexture == 0)
			{
				DX11PhongLighting::render(pd3dImmediateContext, DX11RayCastingHashSDF::getPositonsImageSRV(), DX11RayCastingHashSDF::getNormalsImageSRV(), DX11RayCastingHashSDF::getColorsImageSRV(), DX11RayCastingHashSDF::getSSAOMapFilteredSRV(), false, false);
			}
			else if(GlobalAppState::getInstance().s_DisplayTexture == 3)
			{
				DX11PhongLighting::render(pd3dImmediateContext, DX11RayCastingHashSDF::getPositonsImageSRV(), DX11RayCastingHashSDF::getNormalsImageSRV(), DX11RayCastingHashSDF::getColorsImageSRV(), DX11RayCastingHashSDF::getSSAOMapFilteredSRV(), false, true);
			}
			else if(GlobalAppState::getInstance().s_DisplayTexture == 5)
			{
				DX11PhongLighting::render(pd3dImmediateContext, DX11RayCastingHashSDF::getPositonsImageSRV(), DX11RayCastingHashSDF::getNormalsImageSRV(), DX11RayCastingHashSDF::getColorsImageSRV(), DX11RayCastingHashSDF::getSSAOMapFilteredSRV(), true, true);
			}

			bool trackingLost = false;

			if (GlobalAppState::getInstance().s_RenderMode == RENDERMODE_INTEGRATE && GlobalAppState::getInstance().s_bRegistrationEnabled && hr0 == S_OK)
			{
				mat4f transformation; transformation.setIdentity();
		
				if (g_SceneRepSDFLocal.GetNumIntegratedImages() > 0)
				{
					if (GlobalAppState::getInstance().s_timingsStepsEnabled) {
						GlobalAppState::getInstance().WaitForGPU();
						GlobalAppState::getInstance().s_Timer.start();
					}
					if (GlobalAppState::getInstance().s_usePreComputedCameraTrajectory) {
						//transformation = g_TrajectoryLogReader.getNextTransform();
						transformation = g_Sensor.getRigidTransform();	//should map from the current frame to the base frame 
						std::cout << transformation << std::endl;
					} else {
						mat4f deltaEstimate; deltaEstimate.setIdentity();
						transformation = DX11CameraTrackingMultiRes::applyCT
							(pd3dImmediateContext, 
							g_Sensor.GetDepthFloat4SRV(), 
							g_Sensor.GetNormalFloat4SRV(), 
							g_Sensor.GetColorSRV(), 
							DX11RayCastingHashSDF::getPositonsImageSRV(), 
							DX11RayCastingHashSDF::getNormalsImageSRV(), DX11RayCastingHashSDF::getColorsImageSRV(), 
							g_SceneRepSDFLocal.GetLastRigidTransform(), 
							GlobalCameraTrackingState::getInstance().s_maxInnerIter, 
							GlobalCameraTrackingState::getInstance().s_maxOuterIter, 
							GlobalCameraTrackingState::getInstance().s_distThres, 
							GlobalCameraTrackingState::getInstance().s_normalThres, 
							100.0f, 3.0f,
							deltaEstimate,
							GlobalCameraTrackingState::getInstance().s_residualEarlyOut,
							NULL //&g_ICPErrorLog with an error log, it will be much slower since all steps are executed twice)
							);
		 			}
					if (GlobalAppState::getInstance().s_timingsStepsEnabled) {
						GlobalAppState::getInstance().WaitForGPU();
						GlobalAppState::getInstance().s_Timer.stop();
						TimingLog::countTimeTrack++;
						TimingLog::totalTimeTrack += GlobalAppState::getInstance().s_Timer.getElapsedTimeMS();
					}

					if(transformation(0, 0) == -std::numeric_limits<float>::infinity())
					{
						//TODO MADDI do something reasonable here... (also check whether tracking lost is accurate)
						std::cout << "Tracking lost" << std::endl;
						while(1);

						transformation = g_SceneRepSDFLocal.GetLastRigidTransform();
					
						g_SceneRepSDFLocal.Reset(DXUTGetD3D11DeviceContext());
						g_SceneRepSDFGlobal.Reset(DXUTGetD3D11DeviceContext());
						g_SceneRepChunkGrid.Reset(DXUTGetD3D11DeviceContext());
						g_Camera.Reset();

						return;
					}
				}

				if (GlobalAppState::getInstance().s_DataDumpRigidTransform) {
					std::stringstream ss;	ss << GlobalAppState::getInstance().s_DataDumpPath;
					for (unsigned int i = std::max(1u,g_Sensor.GetFrameNumberDepth()); i < 1000000; i *= 10) ss << "0";
					ss << g_Sensor.GetFrameNumberDepth() << ".matrix";
					std::cout << "Dumping " << ss.str() << std::endl;
					transformation.saveMatrixToFile(ss.str());
				}
				if (GlobalAppState::getInstance().s_RecordData) {
					g_Sensor.recordTrajectory(transformation);
				}

				if (GlobalAppState::getInstance().s_timingsDetailledEnabled) {
					GlobalAppState::getInstance().WaitForGPU();
					GlobalAppState::getInstance().s_Timer.start();
				}

				if (GlobalAppState::getInstance().s_bEnableGlobalLocalStreaming) {
					g_SceneRepSDFLocal.RemoveAndIntegrateToOther(pd3dImmediateContext, &g_SceneRepSDFGlobal, &transformation, true);
					g_SceneRepSDFGlobal.RemoveAndIntegrateToOther(pd3dImmediateContext, &g_SceneRepSDFLocal, &transformation, false);
				}
				if (GlobalAppState::getInstance().s_timingsDetailledEnabled) {
					GlobalAppState::getInstance().WaitForGPU();
					GlobalAppState::getInstance().s_Timer.stop();
					TimingLog::countRemoveAndIntegrate++;
					TimingLog::totalTimeRemoveAndIntegrate += GlobalAppState::getInstance().s_Timer.getElapsedTimeMS();
				}
						
				vec4f posWorld = transformation*GlobalAppState::getInstance().s_StreamingPos; // trans laggs one frame *trans
				vec3f p(posWorld.x, posWorld.y, posWorld.z);
				//g_SceneRepChunkGrid.setPositionAndRadius(p, GlobalAppState::getInstance().s_StreamingRadius, true);

				//unsigned int nStreamedBlocks;
				//g_SceneRepChunkGrid.StreamOutToCPU(DXUTGetD3D11DeviceContext(), g_SceneRepSDFLocal, p, GlobalAppState::getInstance().s_StreamingRadius, true, nStreamedBlocks);
				//g_SceneRepChunkGrid.StreamInToGPU(DXUTGetD3D11DeviceContext(), g_SceneRepSDFLocal, p, GlobalAppState::getInstance().s_StreamingRadius, true, nStreamedBlocks);
				
				if (GlobalAppState::getInstance().s_timingsStepsEnabled) {
					GlobalAppState::getInstance().WaitForGPU();
					GlobalAppState::getInstance().s_Timer.start();
				}
				
				g_SceneRepChunkGrid.StreamOutToCPUPass0GPU(DXUTGetD3D11DeviceContext(), g_SceneRepSDFLocal, p, GlobalAppState::getInstance().s_StreamingRadius, true, true);
				g_SceneRepChunkGrid.StreamInToGPUPass1GPU(DXUTGetD3D11DeviceContext(), g_SceneRepSDFLocal, true);
				
				
				if (GlobalAppState::getInstance().s_timingsStepsEnabled) {
					GlobalAppState::getInstance().WaitForGPU();
					GlobalAppState::getInstance().s_Timer.stop();
					//TimingLog::countTimeMisc++;	//will already be incremeneted before!!!!!
					TimingLog::totalTimeMisc += GlobalAppState::getInstance().s_Timer.getElapsedTimeMS();
				}

				//V_RETURN(StreamInToGPUPass1GPU(DXUTGetD3D11DeviceContext(), g_SceneRepSDFLocal, nStreamedBlocks, true));
			
				if(!GlobalAppState::getInstance().s_bDisableIntegration)
				{
					if (GlobalAppState::getInstance().s_timingsStepsEnabled) {
						GlobalAppState::getInstance().WaitForGPU();
						GlobalAppState::getInstance().s_Timer.start();
					}
					g_SceneRepSDFLocal.Integrate(pd3dImmediateContext, g_Sensor.GetDepthFErodedSRV(), g_Sensor.GetColorSRV(), g_SceneRepChunkGrid.getBitMask(pd3dImmediateContext), &transformation);
					if (GlobalAppState::getInstance().s_timingsStepsEnabled) {
						GlobalAppState::getInstance().WaitForGPU();
						GlobalAppState::getInstance().s_Timer.stop();
						TimingLog::countTimeSceneUpdate++;
						TimingLog::totalTimeSceneUpdate += GlobalAppState::getInstance().s_Timer.getElapsedTimeMS();
					}
					//std::cout << "NumFree: " << g_SceneRepSDFLocal.GetHeapFreeCount(pd3dImmediateContext) << std::endl;
				}
				else
				{
					g_SceneRepSDFLocal.RunCompactifyForView(pd3dImmediateContext);
				}

				//vec4f posWorld = transformation*GlobalAppState::getInstance().s_StreamingPos; // trans laggs one frame *trans
				//vec3f p(posWorld.x, posWorld.y, posWorld.z);
				//g_SceneRepChunkGrid.StreamOutToCPUPass0GPU(DXUTGetD3D11DeviceContext(), g_SceneRepSDFLocal, p, GlobalAppState::getInstance().s_StreamingRadius, true, true);

				//if (doneItfirst == false) {
				//	doneItfirst = true;
				//	g_SceneRepSDF.ToggleEnableGarbageCollect();
				//}


				/////TODO MAKE FLAG (angie dump)
				//if (hr0 == S_OK) {
				//	unsigned int frameNumber = g_Sensor.GetFrameNumberDepth();
				//	if (frameNumber == 400) g_Sensor.savePointCloud("singleFrame.ply");
				//	if (frameNumber > 400 && frameNumber < 500) {
				//		if (frameNumber % 20 == 0)
				//			g_Sensor.recordPointCloud(transformation);
				//	}
				//	if (frameNumber == 500) {
				//		g_Sensor.saveRecordedPointCloud("multiFrame.ply");
				//	}
				//}

			}

			// Uniform Voxel Grid
			/*mat4f trans = DX11VoxelGridOperations::GetLastRigidTransform();
			if (GlobalAppState::getInstance().s_RenderMode == RENDERMODE_VIEW) {
				D3DXMATRIX view = *g_Camera.GetViewMatrix();
				D3DXMatrixInverse(&view, NULL, &view);
				D3DXMatrixTranspose(&view, &view);
				trans = *(mat4f*)&view;
			}

			DX11RayCasting::render(pd3dImmediateContext, DX11VoxelGrid::getBufferDataSRV(), DX11VoxelGrid::getPosition(), DX11VoxelGrid::getGridDimensions(), DX11VoxelGrid::getVoxelExtends(), &trans, DXUTGetWindowWidth(), DXUTGetWindowHeight());
			DX11PhongLighting::render(pd3dImmediateContext, DX11RayCasting::getPositonsImageSRV(), DX11RayCasting::getNormalsImageSRV(), DX11RayCasting::getColorsImageSRV(), false);

			if (GlobalAppState::getInstance().s_RenderMode == RENDERMODE_INTEGRATE && (g_registrationEnabled  || g_SceneRepLocal.GetNumIntegratedImages() <= 100) && hr0 == S_OK) {
				mat4f transformation; transformation.setIdentity();
		
				if (DX11VoxelGridOperations::GetNumIntegratedImages() == 0)
				{
					DX11VoxelGridOperations::reset(pd3dImmediateContext, DX11VoxelGrid::getBufferDataUAV(), DX11VoxelGrid::getPosition(), DX11VoxelGrid::getGridDimensions(), DX11VoxelGrid::getVoxelExtends());
				}

				if (DX11VoxelGridOperations::GetNumIntegratedImages() > 0)
				{
					transformation = DX11CameraTrackingMultiRes::applyCT(pd3dImmediateContext, g_KinectSensor.GetDepthFloat4SRV(), g_KinectSensor.GetNormalFloat4SRV(), g_KinectSensor.GetColorSRV(), DX11RayCasting::getPositonsImageSRV(), DX11RayCasting::getNormalsImageSRV(), DX11RayCasting::getColorsImageSRV(), DX11VoxelGridOperations::GetLastRigidTransform(), GlobalCameraTrackingState::s_maxInnerIter, GlobalCameraTrackingState::s_maxOuterIter, GlobalCameraTrackingState::s_distThres, GlobalCameraTrackingState::s_normalThres, 100.0f, 3.0f);
				}
			
				DX11VoxelGridOperations::integrateDepthFrame(pd3dImmediateContext, DX11VoxelGrid::getBufferDataUAV(), DX11VoxelGrid::getPosition(), DX11VoxelGrid::getGridDimensions(), DX11VoxelGrid::getVoxelExtends(), g_KinectSensor.GetDepthFErodedSRV(), g_KinectSensor.GetColorSRV(), &transformation, DXUTGetWindowWidth(), DXUTGetWindowHeight());
			}*/

		}
		else if (GlobalAppState::getInstance().s_DisplayTexture == 1)
		{
			//DX11QuadDrawer::RenderQuad(pd3dImmediateContext, g_KinectSensor.GetDepthFSRV(), 1.0f);
			DX11QuadDrawer::RenderQuad(pd3dImmediateContext, g_Sensor.GetHSVDepthFloat4SRV(), 1.0f);
			//DX11QuadDrawer::RenderQuad(pd3dImmediateContext, g_KinectSensor.GetDepthFloat4SRV(), 1.0f/2.0f);
		}
		else if (GlobalAppState::getInstance().s_DisplayTexture == 2)
		{
			DX11QuadDrawer::RenderQuad(pd3dImmediateContext, g_Sensor.GetNormalFloat4SRV());
		}
		else if (GlobalAppState::getInstance().s_DisplayTexture == 4)
		{
			//DX11QuadDrawer::RenderQuad(pd3dImmediateContext, DX11RayCastingHashSDF::getColorsImageSRV());
			DX11QuadDrawer::RenderQuad(pd3dImmediateContext, g_Sensor.GetColorSRV());
		}

		if (GlobalAppState::getInstance().s_timingsTotalEnabled) {
			GlobalAppState::getInstance().WaitForGPU();
			GlobalAppState::getInstance().s_Timer.stop();

			if(hr0 == S_OK)		{
				double elapsedTime = GlobalAppState::getInstance().s_Timer.getElapsedTimeMS();
				TimingLog::totalTimeAllAvgArray[TimingLog::countTotalTimeAll % BENCHMARK_SAMPLES] = elapsedTime;
				TimingLog::totalTimeAllWorst = std::max(elapsedTime, TimingLog::totalTimeAllWorst);
				TimingLog::totalTimeAll += elapsedTime;
				TimingLog::totalTimeSquaredAll += (elapsedTime*elapsedTime);
				TimingLog::countTotalTimeAll = (TimingLog::countTotalTimeAll + 1);
			}
		}

		if (hr0 == S_OK && GlobalAppState::getInstance().s_DumpAllRendering) {
			DumpAllRendering(pd3dImmediateContext);
		}

	}




	if (GlobalAppState::getInstance().s_DumpVoxelGridFrames > 0 && 
		(g_Sensor.GetFrameNumberDepth() % GlobalAppState::getInstance().s_DumpVoxelGridFrames) == 0) 
	{	
		std::cout << "dumping voxel grid... ";
		vec4f pos = g_SceneRepSDFLocal.GetLastRigidTransform()*GlobalAppState::getInstance().s_StreamingPos;
		g_SceneRepSDFLocal.DumpHashToDisk(GlobalAppState::getInstance().s_DumpVoxelGridFile + std::to_string(g_Sensor.GetFrameNumberDepth()) + ".dump", 
			GlobalAppState::getInstance().s_StreamingRadius - sqrt(3.0f), pos.getPoint3d());
		std::cout << "done!" << std::endl;

	}

	TimingLog::printTimings();
		
	if (true)
	{	
		DXUT_BeginPerfEvent( DXUT_PERFEVENTCOLOR, L"HUD / Stats" );
		g_HUD.OnRender( fElapsedTime );
		g_SampleUI.OnRender( fElapsedTime );
		RenderText();
		DXUT_EndPerfEvent();
	}





}
