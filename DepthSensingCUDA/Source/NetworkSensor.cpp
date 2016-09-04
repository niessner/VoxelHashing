
#include "stdafx.h"

#include "NetworkSensor.h"

#include "sensorData/sensorData.h"


struct Calibration {
	unsigned int	m_DepthImageWidth;
	unsigned int	m_DepthImageHeight;
	unsigned int	m_ColorImageWidth;
	unsigned int	m_ColorImageHeight;
	ml::SensorData::CalibrationData m_CalibrationDepth;
	ml::SensorData::CalibrationData m_CalibrationColor;
	bool			m_bUseTrajectory;
};


void NetworkSensor::waitForConnection()
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
	m_depthIntrinsics = calibration.m_CalibrationDepth.m_intrinsic;
	m_depthIntrinsicsInv = m_depthIntrinsics.getInverse();
	m_colorIntrinsics = calibration.m_CalibrationDepth.m_intrinsic;
	m_colorIntrinsicsInv = m_colorIntrinsics.getInverse();
	initializeDepthExtrinsics(calibration.m_CalibrationDepth.m_extrinsic);
	initializeColorExtrinsics(calibration.m_CalibrationColor.m_extrinsic);

	std::cout << "depth intrinsics: " << std::endl << m_depthIntrinsics << std::endl;
	std::cout << "color intrinsics: " << std::endl << m_colorIntrinsics << std::endl;
	std::cout << "depth extrinsics: " << std::endl << m_depthExtrinsics << std::endl;
	std::cout << "color extrinsics: " << std::endl << m_colorExtrinsics << std::endl;

	for (unsigned int i = 0; i < getColorWidth()*getColorHeight(); i++) {
		m_colorRGBX[i] = vec4uc(255, 255, 255, 255);
	}

}

HRESULT NetworkSensor::processDepth()
{
	int byte_size_received = 0;

	PacketHeader packet_header;
	byte_size_received = m_networkServer.receiveDataBlocking((BYTE*)(&packet_header), sizeof(PacketHeader));
	if (byte_size_received != sizeof(PacketHeader)) throw MLIB_EXCEPTION("invalid size reading packet header");
	if (packet_header.packet_type == PacketType::CLIENT_2_SERVER_DISCONNECT)
	{
		StopScanningAndExtractIsoSurfaceMC();
		ResetDepthSensing();
		waitForConnection();
		byte_size_received = m_networkServer.receiveDataBlocking((BYTE*)(&packet_header), sizeof(PacketHeader));
		if (byte_size_received != sizeof(PacketHeader)) throw MLIB_EXCEPTION("invalid size reading packet header");

		if (packet_type_status == PacketType::SERVER_2_CLIENT_RESET) {
			packet_type_status = PacketType::SERVER_2_CLIENT_PROCESSED;
		}
	}

	if (packet_header.packet_type != PacketType::CLIENT_2_SERVER_FRAME_DATA)
		throw MLIB_EXCEPTION("expecting frame data packet");

	std::vector<BYTE> dataCompressed(packet_header.packet_size);
	byte_size_received = m_networkServer.receiveDataBlocking((BYTE*)(&dataCompressed[0]), packet_header.packet_size);
	if (byte_size_received != packet_header.packet_size) throw MLIB_EXCEPTION("invalid size reading frame data");
	std::cout << byte_size_received << " bytes received in frame " << m_iFrame << std::endl;

	if (packet_header.client_type == ClientType::CLIENT_TANGO_YELLOW_STONE)
	{
		int width = getDepthWidth();
		int height = getDepthHeight();
		// Decompress
		std::vector<USHORT> depth_image(width*height);
		ZLibWrapper::DecompressStreamFromMemory((BYTE*)(&dataCompressed[0]), dataCompressed.size(), (BYTE*)(&depth_image[0]), depth_image.size()*sizeof(USHORT));

		// Smooth
		float* data = getDepthFloat();
		int filter_size = 1;
		for (int u = 0; u < width; ++u) {
			for (int v = 0; v < height; ++v) {
				int idx = v * width + u;
				if (depth_image[idx] != 0)
				{
					data[idx] = depth_image[idx]*0.001f;
					continue;
				}

				USHORT value = 0;
				int u_begin = std::max((int) (0), (int) (u - filter_size));
				int u_end = std::min((int) (width - 1), (int) (u + filter_size));
				int v_begin = std::max((int) (0), (int) (v - filter_size));
				int v_end = std::min((int) (height - 1), (int) (v + filter_size));
				int non_zero_neighbor_count = 0;
				for (int uu = u_begin; uu <= u_end; ++uu) {
					for (int vv = v_begin; vv <= v_end; ++vv) {
						USHORT neighbor = depth_image[vv * width + uu];
						if (neighbor == 0)
							continue;
						non_zero_neighbor_count ++;
						if (value == 0)
							value = neighbor;
						else
							value = std::min(value, neighbor);
					}
				}
				data[idx] = (non_zero_neighbor_count>1)?(value*0.001f):(0.0f);
			}
		}
	}
	else
	{
		USHORT* dataUSHORT = new USHORT[getDepthWidth()*getDepthHeight()];
		ZLibWrapper::DecompressStreamFromMemory((BYTE*)(&dataCompressed[0]), packet_header.packet_size, (BYTE*)dataUSHORT, getDepthWidth()*getDepthHeight()*sizeof(USHORT));
		float* data = getDepthFloat();
		for (unsigned int i = 0; i < getDepthWidth()*getDepthHeight(); i++) {
			data[i] = (float)dataUSHORT[i] * 0.001f;
		}
		SAFE_DELETE_ARRAY(dataUSHORT);
	}

	//DepthImage di(getDepthHeight(), getDepthWidth(), getDepthFloat());
	//ColorImageRGB ci(di);
	//std::ostringstream ss;
	//ss << std::setw(4) << std::setfill('0') << m_iFrame;
	//FreeImageWrapper::saveImage("Dump\\socket_frame_"+ss.str()+".png", ci, true);

	//getchar();
	//exit(1);

	if (m_bUseTrajectory) {
		byte_size_received = m_networkServer.receiveDataBlocking((BYTE*)(&packet_header), sizeof(PacketHeader));
		if (byte_size_received != sizeof(PacketHeader)) throw MLIB_EXCEPTION("invalid size reading packet header");
		if (packet_header.packet_type != PacketType::CLIENT_2_SERVER_TRANSFORMATION)
		{
			throw MLIB_EXCEPTION("expecting transformation packet");
			return S_FALSE;
		}
		byte_size_received = m_networkServer.receiveDataBlocking((BYTE*)&m_rigidTransform, packet_header.packet_size);
		if (byte_size_received != packet_header.packet_size) throw MLIB_EXCEPTION("invalid size reading transformation");

		//m_rigidTransform.transpose();
		//std::cout << "NetworkSensor: " <<  m_rigidTransform << std::endl;
	}

	packet_header.packet_type = packet_type_status;
	packet_header.packet_size = 0;
	int byte_size_sent = m_networkServer.sendDataBlocking((BYTE*)(&packet_header), sizeof(PacketHeader));
	if (byte_size_sent != sizeof(PacketHeader)) throw MLIB_EXCEPTION("invalid size writing packet header");
	
	if (packet_type_status == PacketType::SERVER_2_CLIENT_RESET) {
		packet_type_status = PacketType::SERVER_2_CLIENT_PROCESSED;
	}
	m_iFrame++;

	return S_OK;
}
