

unsigned int GlobalAppState::s_sensorIdx = 4; // 0 = Kinect, 1 = PrimeSense, 2 = SoftKinect, 3 = BinaryDump, 4 = ImageReader
std::string GlobalAppState::s_ImageReaderSesnorSourcePath = "../stanfordData/copyroom_png/";
unsigned int GlobalAppState::s_ImageReaderSesnorNumFrames = 4700; //5490;

bool GlobalAppState::s_timingsDetailledEnabled = false;
bool GlobalAppState::s_timingsStepsEnabled = false;
bool GlobalAppState::s_timingsTotalEnabled = false;

unsigned int GlobalAppState::s_windowWidth = 640;
unsigned int GlobalAppState::s_windowHeight = 480;
unsigned int GlobalAppState::s_RenderMode = 0;
unsigned int GlobalAppState::s_DisplayTexture = 3;

Timer	GlobalAppState::s_Timer;
bool	GlobalAppState::s_bRenderModeChanged = true;
bool	GlobalAppState::s_bFilterKinectInputData = true;

bool	GlobalAppState::s_bEnableGlobalLocalStreaming = false;
bool	GlobalAppState::s_bEnableGarbageCollection = false;

ID3D11Query* GlobalAppState::s_pQuery = NULL;

unsigned int GlobalAppState::s_hashNumBucketsLocal = 1000000;
unsigned int GlobalAppState::s_hashNumBucketsGlobal = 1337;		// Is not used in the current implementation
unsigned int GlobalAppState::s_hashNumSDFBlocks = 0x1 << 18;	//17 on GTX 480; 18 on GTX TITAN

unsigned int GlobalAppState::s_initialChunkListSize = 2000;

unsigned int GlobalAppState::s_hashStreamOutParts = 80;

unsigned int GlobalAppState::s_hashBucketSizeLocal = 2;
unsigned int GlobalAppState::s_hashBucketSizeGlobal = 1337;		// Is not used in the current implementation

float GlobalAppState::s_virtualVoxelSize = 0.004;	//4 mm in extracted data
float GlobalAppState::s_thresMarchingCubes = 10.0f*s_virtualVoxelSize;
float GlobalAppState::s_thresMarchingCubes2 = 10.0f*s_virtualVoxelSize;

bool GlobalAppState::s_applicationDisabled = false;

float GlobalAppState::s_StreamingRadius = 5.0f; // Depends on DepthMin and DepthMax
vec4f GlobalAppState::s_StreamingPos = vec4f(0.0f, 0.0f, 3.0f, 1.0f); // Depends on DepthMin and DepthMax

vec3f GlobalAppState::s_voxelExtends = vec3f(1.0f, 1.0f, 1.0f);
vec3i GlobalAppState::s_gridDimensions = vec3i(257, 257, 257); // dimensions have to be odd (number of samples)
vec3i GlobalAppState::s_minGridPos = -s_gridDimensions/2;
int GlobalAppState::s_nBitsInT = 32;

unsigned int GlobalAppState::s_SDF_BLOCK_SIZE = 8;

unsigned int GlobalAppState::s_WeightSample = 8;
unsigned int GlobalAppState::s_WeightMax = 255;

float GlobalAppState::s_Truncation = 0.02f;
float GlobalAppState::s_TruncScale = 0.01f;

float GlobalAppState::s_maxIntegrationDistance = 2.0f;
float GlobalAppState::s_SensorDepthWorldMin = 0.3f;
float GlobalAppState::s_SensorDepthWorldMax = 5.0f;

float GlobalAppState::s_rayIncrement = 0.8f*s_Truncation;
float GlobalAppState::s_thresSampleDist = 50.5f*s_rayIncrement;
float GlobalAppState::s_thresDist = 50.5f*s_rayIncrement;

unsigned int GlobalAppState::s_HANDLE_COLLISIONS = 1; // 1 with collision handling, 0 without, also defined in the shader

//float GlobalAppState::s_materialShininess = 3.0f;
//vec4f GlobalAppState::s_materialSpecular = vec4f(1.0f, 1.0f, 1.0f, 1.0f);
//vec4f GlobalAppState::s_lightAmbient = vec4f(0.4f, 0.4f, 0.4f, 1.0f);
//vec4f GlobalAppState::s_lightDiffuse = vec4f(0.5f, 0.5f, 0.5f, 1.0f);
//vec4f GlobalAppState::s_lightSpecular = vec4f(0.05f, 0.05f, 0.05f, 1.0f);
//vec3f GlobalAppState::s_lightDir = vec3f(0.0f, 0.0f, 1.0f);

//copied from hierarchy
float GlobalAppState::s_materialShininess = 16.0f;
vec4f GlobalAppState::s_materialDiffuse = vec4f(1.0f, 0.9f, 0.7f, 1.0f);
vec4f GlobalAppState::s_materialSpecular = vec4f(1.0f, 1.0f, 1.0f, 1.0f);
vec4f GlobalAppState::s_lightAmbient = vec4f(0.1f, 0.1f, 0.1f, 1.0f)*4.0f;
vec4f GlobalAppState::s_lightDiffuse = vec4f(1.0000f, 0.8824f ,0.761f, 1.0f)*0.6f;
vec4f GlobalAppState::s_lightSpecular = vec4f(0.3f, 0.3f, 0.3f, 1.0f);
vec4f GlobalAppState::s_lightDir = vec4f(0.0f, 1.0f, -2.0f, 1.0f);

unsigned int GlobalAppState::s_MaxLoopIterCount = 1024;

unsigned int GlobalAppState::s_MaxCollisionLinkedListSize = 7;

ID3D11Buffer* GlobalAppState::m_constantBuffer = NULL;

bool GlobalAppState::s_useGradients = true;
bool GlobalAppState::s_enableMultiLayerSplatting = false;
bool GlobalAppState::s_usePreComputedCameraTrajectory = false;
std::string GlobalAppState::s_PreComputedCameraTrajectoryPath = "../stanfordData/copyroom_trajectory.log";

bool GlobalAppState::s_stereoEnabled = false;
unsigned int GlobalAppState::s_windowWidthStereo = 1920;
unsigned int GlobalAppState::s_windowHeightStereo = 1080;

D3DXMATRIX GlobalAppState::s_intrinsics; // Automatic Initialization
D3DXMATRIX GlobalAppState::s_intrinsicsInv; // Automatic Initialization


// Stereo

D3DXMATRIX GlobalAppState::s_intrinsicsStereo; // Automatic Initialization
D3DXMATRIX GlobalAppState::s_intrinsicsInvStereo; // Automatic Initialization

D3DXMATRIX GlobalAppState::s_intrinsicsStereoOther;  // Automatic Initialization
D3DXMATRIX GlobalAppState::s_intrinsicsInvStereoOther;  // Automatic Initialization

D3DXMATRIX GlobalAppState::s_worldToCamStereo;  // Automatic Initialization
D3DXMATRIX GlobalAppState::s_camToWorldStereo;  // Automatic Initialization

D3DXMATRIX GlobalAppState::s_worldToCamStereoOther;  // Automatic Initialization
D3DXMATRIX GlobalAppState::s_camToWorldStereoOther;  // Automatic Initialization

bool GlobalAppState::s_currentlyInStereoMode = false; // Default state
