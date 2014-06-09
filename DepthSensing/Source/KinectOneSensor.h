#pragma once

/****************************************************************************/
/* KinectOneSensor (the new Kinect)... be aware of the naming ambiguity :)  */
/****************************************************************************/

#include "GlobalAppState.h"

//Only working with Kinect 2.0 SDK (which wants to run on Win8)
#ifdef KINECT_ONE

#include <Kinect.h>
#include "DepthSensor.h"

// Safe release for interfaces
template<class Interface>
inline void SafeRelease(Interface *& pInterfaceToRelease)
{
    if (pInterfaceToRelease != NULL) {
        pInterfaceToRelease->Release();
        pInterfaceToRelease = NULL;
    }
}

class KinectOneSensor : public DepthSensor
{
public:

	KinectOneSensor()
	{
		DepthSensor::init(cDepthWidth, cDepthHeight, cDepthWidth, cDepthHeight);

		m_pKinectSensor = NULL;

		HRESULT hr = createFirstConnected();
		if (hr != S_OK)	throw EXCEPTION("failed to initialize kinect");

		// create heap storage for color pixel data in RGBX format
		m_pColorRGBX = new RGBQUAD[cColorWidth * cColorHeight];

		m_pColorCoordinates = new ColorSpacePoint[cDepthWidth * cDepthHeight];

		m_depthPointCount = cDepthWidth*cDepthHeight;
		m_depthSpacePoints = new DepthSpacePoint[m_depthPointCount];
		m_cameraSpacePoints = new CameraSpacePoint[m_depthPointCount];

		for(unsigned int i = 0; i<cDepthWidth; i++)
		{
			for(unsigned int j = 0; j<cDepthHeight; j++)
			{
				m_depthSpacePoints[j*cDepthWidth+i].X = (float)i;
				m_depthSpacePoints[j*cDepthWidth+i].Y = (float)j;
			}
		}

		IMultiSourceFrame* pMultiSourceFrame = NULL;
		IDepthFrame* pDepthFrame = NULL;
		IColorFrame* pColorFrame = NULL;

		hr = S_FALSE;
		while(hr != S_OK) hr = m_pMultiSourceFrameReader->AcquireLatestFrame(&pMultiSourceFrame);

		if(SUCCEEDED(hr))
		{
			IDepthFrameReference* pDepthFrameReference = NULL;

			hr = pMultiSourceFrame->get_DepthFrameReference(&pDepthFrameReference);
			if (SUCCEEDED(hr))
			{
				hr = pDepthFrameReference->AcquireFrame(&pDepthFrame);
			}

			SafeRelease(pDepthFrameReference);
		}

		if (SUCCEEDED(hr))
		{
			IColorFrameReference* pColorFrameReference = NULL;

			hr = pMultiSourceFrame->get_ColorFrameReference(&pColorFrameReference);
			if (SUCCEEDED(hr))
			{
				hr = pColorFrameReference->AcquireFrame(&pColorFrame);
			}

			SafeRelease(pColorFrameReference);
		}

		if (SUCCEEDED(hr))
		{
			INT64 nDepthTime = 0;
			IFrameDescription* pDepthFrameDescription = NULL;
			int nDepthWidth = 0;
			int nDepthHeight = 0;
			UINT nDepthBufferSize = 0;
			UINT16 *pDepthBuffer = NULL;

			USHORT nDepthMinReliableDistance = 0;
			USHORT nDepthMaxReliableDistance = 0;
			float fovX;
			float fovY;

			IFrameDescription* pColorFrameDescription = NULL;
			int nColorWidth = 0;
			int nColorHeight = 0;
			ColorImageFormat imageFormat = ColorImageFormat_None;
			UINT nColorBufferSize = 0;
			RGBQUAD *pColorBuffer = NULL;

			// get depth frame data
			hr = pDepthFrame->get_RelativeTime(&nDepthTime);

			if (SUCCEEDED(hr))
			{
				hr = pDepthFrame->get_FrameDescription(&pDepthFrameDescription);
				hr = pDepthFrameDescription->get_Width(&nDepthWidth);
				hr = pDepthFrameDescription->get_Height(&nDepthHeight);
				hr = pDepthFrame->AccessUnderlyingBuffer(&nDepthBufferSize, &pDepthBuffer);            
				hr = pColorFrame->get_FrameDescription(&pColorFrameDescription);
				hr = pColorFrameDescription->get_Width(&nColorWidth);
				hr = pColorFrameDescription->get_Height(&nColorHeight);
				hr = pColorFrame->get_RawColorImageFormat(&imageFormat);
				hr = pDepthFrame->get_DepthMinReliableDistance(&nDepthMinReliableDistance);
				hr = pDepthFrame->get_DepthMaxReliableDistance(&nDepthMaxReliableDistance);
			}

			if (SUCCEEDED(hr))
			{
				hr = pDepthFrameDescription->get_Width(&nDepthWidth);
				hr = pDepthFrameDescription->get_Height(&nDepthHeight);

				hr = pDepthFrameDescription->get_HorizontalFieldOfView(&fovX);
				hr = pDepthFrameDescription->get_VerticalFieldOfView(&fovY);

				hr = pDepthFrame->get_DepthMinReliableDistance(&nDepthMinReliableDistance);
				hr = pDepthFrame->get_DepthMaxReliableDistance(&nDepthMaxReliableDistance);
			
				float centerX = nDepthWidth/2.0f;
				float centerY = nDepthHeight/2.0f;

				float focalLengthX = centerX/tan(fovX/2.0f*(float)M_PI/180.0f);
				float focalLengthY = centerY/tan(fovY/2.0f*(float)M_PI/180.0f);

				double k1 = 1.5355725262415776e-001;
				double k2 = -3.3172043290283648e-001;
				double p1 = -8.6523355025079577e-003;
				double p2 = 1.3118055542282476e-002;
				double k3 = 1.5561334862418680e-001;
				initializeIntrinsics(focalLengthX, focalLengthY, centerX, centerY, k1, k2, k3, p1, p2);
			}
			SafeRelease(pDepthFrameDescription);
			SafeRelease(pColorFrameDescription);
		}

		SafeRelease(pDepthFrame);
		SafeRelease(pColorFrame);
		SafeRelease(pMultiSourceFrame);
	}

	~KinectOneSensor()
	{
		if (m_pKinectSensor)			m_pKinectSensor->Release();
		if (m_pMultiSourceFrameReader)	m_pMultiSourceFrameReader->Release();
		if (m_pCoordinateMapper)		m_pCoordinateMapper->Release();
		if (m_pColorCoordinates)		delete [] m_pColorCoordinates;

		if (m_depthSpacePoints)			delete [] m_depthSpacePoints;
		if (m_cameraSpacePoints)		delete [] m_cameraSpacePoints;

		if (m_pColorRGBX)
		{
			delete [] m_pColorRGBX;
			m_pColorRGBX = NULL;
		}
	}

	HRESULT createFirstConnected()
	{
		HRESULT hr;

		hr = GetDefaultKinectSensor(&m_pKinectSensor);
		if (FAILED(hr))
		{
			return hr;
		}

		if (m_pKinectSensor)
		{
			// Initialize the Kinect and get coordinate mapper and the frame reader
			if (SUCCEEDED(hr))
			{
				hr = m_pKinectSensor->get_CoordinateMapper(&m_pCoordinateMapper);
			}

			hr = m_pKinectSensor->Open();

			if (SUCCEEDED(hr))
			{
				hr = m_pKinectSensor->OpenMultiSourceFrameReader(
					FrameSourceTypes::FrameSourceTypes_Depth | FrameSourceTypes::FrameSourceTypes_Color,
					&m_pMultiSourceFrameReader);
			}
		}

		return hr;
	}

	HRESULT processDepth()
	{
		IMultiSourceFrame* pMultiSourceFrame = NULL;
		IDepthFrame* pDepthFrame = NULL;
		IColorFrame* pColorFrame = NULL;

		HRESULT hr = m_pMultiSourceFrameReader->AcquireLatestFrame(&pMultiSourceFrame);

		if(SUCCEEDED(hr))
		{
			IDepthFrameReference* pDepthFrameReference = NULL;

			hr = pMultiSourceFrame->get_DepthFrameReference(&pDepthFrameReference);
			if (SUCCEEDED(hr))
			{
				hr = pDepthFrameReference->AcquireFrame(&pDepthFrame);
			}

			SafeRelease(pDepthFrameReference);
		}

		if (SUCCEEDED(hr))
		{
			IColorFrameReference* pColorFrameReference = NULL;

			hr = pMultiSourceFrame->get_ColorFrameReference(&pColorFrameReference);
			if (SUCCEEDED(hr))
			{
				hr = pColorFrameReference->AcquireFrame(&pColorFrame);
			}

			SafeRelease(pColorFrameReference);
		}

		if (SUCCEEDED(hr))
		{
			INT64 nDepthTime = 0;
			IFrameDescription* pDepthFrameDescription = NULL;
			
			UINT nDepthBufferSize = 0;
			UINT16 *pDepthBuffer = NULL;

			IFrameDescription* pColorFrameDescription = NULL;
		
			ColorImageFormat imageFormat = ColorImageFormat_None;
			UINT nColorBufferSize = 0;
			RGBQUAD *pColorBuffer = NULL;

			// get depth frame data
			if (SUCCEEDED(hr)) hr = pDepthFrame->get_RelativeTime(&nDepthTime);
			if (SUCCEEDED(hr)) hr = pDepthFrame->get_FrameDescription(&pDepthFrameDescription);
			if (SUCCEEDED(hr)) hr = pDepthFrame->AccessUnderlyingBuffer(&nDepthBufferSize, &pDepthBuffer);

			// get color frame data
			if (SUCCEEDED(hr)) hr = pColorFrame->get_FrameDescription(&pColorFrameDescription);
			if (SUCCEEDED(hr)) hr = pColorFrame->get_RawColorImageFormat(&imageFormat);

			if (SUCCEEDED(hr))
			{
				if (m_pColorRGBX)
				{
					pColorBuffer = m_pColorRGBX;
					nColorBufferSize = cColorWidth * cColorHeight * sizeof(RGBQUAD);
					hr = pColorFrame->CopyConvertedFrameDataToArray(nColorBufferSize, reinterpret_cast<BYTE*>(pColorBuffer), ColorImageFormat_Rgba);
				}
				else
				{
					hr = E_FAIL;
				}
			}

			if (SUCCEEDED(hr))
			{	
				const UINT16* const depthFrame = pDepthBuffer;
				m_pCoordinateMapper->MapDepthFrameToColorSpace(m_depthPointCount, depthFrame, m_depthPointCount, m_pColorCoordinates);
					
				// Make sure we've received valid data
				if (m_pCoordinateMapper && m_pColorCoordinates && pDepthBuffer && pColorBuffer)
				{
					#pragma omp parallel for
					for(int i = 0; i<(int)(cDepthWidth*cDepthHeight); i++)
					{
						m_depthD16[i] = depthFrame[i];

						ColorSpacePoint colorPoint = m_pColorCoordinates[i];
						
						int colorX = (int)(floor(colorPoint.X + 0.5));
						int colorY = (int)(floor(colorPoint.Y + 0.5));
						if ((colorX >= 0) && (colorX < cColorWidth) && (colorY >= 0) && (colorY < cColorHeight))
						{
							RGBQUAD q = m_pColorRGBX[colorX + (colorY * cColorWidth)];
							m_colorRGBX[4*i+0] = q.rgbRed;
							m_colorRGBX[4*i+1] = q.rgbGreen;
							m_colorRGBX[4*i+2] = q.rgbBlue;
							m_colorRGBX[4*i+3] = 255;
						}
						else
						{
							m_colorRGBX[4*i+0] = 0;
							m_colorRGBX[4*i+1] = 0;
							m_colorRGBX[4*i+2] = 0;
							m_colorRGBX[4*i+3] = 0;
						}
					}
				}				
			}

			SafeRelease(pDepthFrameDescription);
			SafeRelease(pColorFrameDescription);
		}

		SafeRelease(pDepthFrame);
		SafeRelease(pColorFrame);
		SafeRelease(pMultiSourceFrame);

		return hr;
	}

	HRESULT processColor() {
		HRESULT hr = S_OK;
		return hr;
	}

	HRESULT toggleAutoWhiteBalance() {
		HRESULT hr = S_OK;
		return hr;
	}

private:

	static const unsigned int cDepthHeight = 424;
	static const unsigned int cDepthWidth  = 512;
	static const unsigned int cColorHeight = 1080;
	static const unsigned int cColorWidth  = 1920;

    // Current Kinect
    IKinectSensor*          m_pKinectSensor;

	RGBQUAD*                m_pColorRGBX;

	// Frame reader
	IMultiSourceFrameReader* m_pMultiSourceFrameReader;

	// Mapping
	ICoordinateMapper*		 m_pCoordinateMapper; 
	ColorSpacePoint*         m_pColorCoordinates; 
		
	unsigned int m_depthPointCount;
	DepthSpacePoint* m_depthSpacePoints;
	CameraSpacePoint* m_cameraSpacePoints;
};

#endif