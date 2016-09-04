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

	// Virtual Scan
	CLIENT_VIRTUAL_SCAN = 4,

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

	void waitForConnection();

	HRESULT createFirstConnected() {
		m_iFrame = 0;
		waitForConnection();

		return S_OK;
	}

	HRESULT processDepth();

	HRESULT processColor() {
		return S_OK;
	}

	std::string getSensorName() const {
		return "NetworkSensor";
	}

	mat4f getRigidTransform(int offset) const {
		return m_rigidTransform;
	}

private:


	NetworkServer		m_networkServer;
	bool				m_bUseTrajectory;
	mat4f				m_rigidTransform;
	int m_iFrame;
	PacketType packet_type_status;
};

