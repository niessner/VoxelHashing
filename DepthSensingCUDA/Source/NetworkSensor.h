#pragma once

#include <iomanip>

#include "RGBDSensor.h"
#include "DepthSensing.h"
#include "NetworkServer.h"

enum class ClientType
{
	CLIENT_UNKNOWN = 0,

	// Kinect
	CLIENT_KINECT = 1,
	CLIENT_PRIME_SENSE = 2,
	CLIENT_KINECT_ONE = 3,

	// Intel
	CLIENT_INTEL = 1024+1,

	// Tango
	CLIENT_TANGO_YELLOW_STONE = 2048+1
};

enum class PacketType
{
	UNKNOWN = 0,

	// packets from client to server
	CLIENT_2_SERVER_CALIBRATION = 1,
	CLIENT_2_SERVER_FRAME_DATA = 2,
	CLIENT_2_SERVER_TRANSFORMATION = 3,
	CLIENT_2_SERVER_DISCONNECT = 4,

	// packets from server to client
	SERVER_2_CLIENT_PROCESSED = 1024+1,
	SERVER_2_CLIENT_RESET = 1024+2
};

struct PacketHeader 
{
	ClientType client_type;
	PacketType packet_type;
	int packet_size;
	int packet_size_decompressed;
};

struct Calibration {
	unsigned int	m_DepthImageWidth;
	unsigned int	m_DepthImageHeight;
	unsigned int	m_ColorImageWidth;
	unsigned int	m_ColorImageHeight;
	CalibrationData m_CalibrationDepth;
	CalibrationData m_CalibrationColor;
	bool			m_bUseTrajectory;
};

class NetworkSensor : public RGBDSensor
{
public:
	NetworkSensor() {
		m_rigidTransform.setIdentity();
		m_bUseTrajectory = false;
		m_iFrame = 0;
		packet_type_status = PacketType::SERVER_2_CLIENT_PROCESSED;
	}
	~NetworkSensor() {
		m_networkServer.close();
	}

	virtual void reset()
	{
		RGBDSensor::reset();
		m_iFrame = 0;
		packet_type_status = PacketType::SERVER_2_CLIENT_RESET;
	}

	void waitForConnection()
	{
		m_networkServer.close();

		const unsigned int defaultPort = 1337;
		std::string client("unknown client");

		std::cout << "waiting for network connection" << std::endl;
		if (!m_networkServer.open(defaultPort, client)) {
			throw MLIB_EXCEPTION("could not open network server");
		}
		std::cout << "connected to " << client << std::endl;

		PacketHeader packet_header;
		int byte_size_received = m_networkServer.receiveDataBlocking((BYTE*)(&packet_header), sizeof(PacketHeader));
		if (byte_size_received != sizeof(PacketHeader)) throw MLIB_EXCEPTION("invalid size reading packet header");
		if (packet_header.packet_type != PacketType::CLIENT_2_SERVER_CALIBRATION)
			throw MLIB_EXCEPTION("expecting calibration packet");

		Calibration calibration;
		byte_size_received = m_networkServer.receiveDataBlocking((BYTE*)(&calibration), sizeof(Calibration));
		if (byte_size_received != sizeof(Calibration)) throw MLIB_EXCEPTION("invalid size reading parameters");
		m_bUseTrajectory = calibration.m_bUseTrajectory;

		init(calibration.m_DepthImageWidth, calibration.m_DepthImageHeight, calibration.m_ColorImageWidth, calibration.m_ColorImageHeight);
		m_depthIntrinsics = calibration.m_CalibrationDepth.m_Intrinsic;
		m_depthIntrinsicsInv = calibration.m_CalibrationDepth.m_IntrinsicInverse;
		m_colorIntrinsics = calibration.m_CalibrationDepth.m_Intrinsic;
		m_colorIntrinsicsInv = calibration.m_CalibrationColor.m_IntrinsicInverse;
		initializeDepthExtrinsics(calibration.m_CalibrationDepth.m_Extrinsic);
		initializeColorExtrinsics(calibration.m_CalibrationColor.m_Extrinsic);

		return;
	}

	HRESULT createFirstConnected() {
		m_iFrame = 0;
		waitForConnection();
		for (unsigned int i = 0; i < getColorWidth()*getColorHeight(); i++) {
			m_colorRGBX[i] = vec4uc(255,255,255,255);
		}

		return S_OK;
	}

	HRESULT processDepth();

	HRESULT processColor() {

		return S_OK;
	}

	mat4f getRigidTransform() const {
		return m_rigidTransform;
	}

private:


	NetworkServer		m_networkServer;
	bool				m_bUseTrajectory;
	mat4f				m_rigidTransform;
	int m_iFrame;
	PacketType packet_type_status;
};

