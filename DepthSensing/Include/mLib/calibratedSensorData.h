
#ifndef EXT_DEPTHCAMERA_CALIBRATEDSENSORDATA_H_
#define EXT_DEPTHCAMERA_CALIBRATEDSENSORDATA_H_

namespace ml
{

class CalibrationData {
public:
	CalibrationData() {
		setIdentity();
	}

	void setIdentity() {
		m_Intrinsic.setIdentity();
		m_IntrinsicInverse.setIdentity();
		m_Extrinsic.setIdentity();
		m_ExtrinsicInverse.setIdentity();
	}

	void setMatrices(const mat4f& intrinsic, const mat4f& extrinsic) {
		m_Intrinsic = intrinsic;
		m_Extrinsic = extrinsic;
		m_IntrinsicInverse = m_Intrinsic.getInverse();
		m_ExtrinsicInverse = m_Extrinsic.getInverse();
	}

	//! Camera-to-Proj matrix
	mat4f m_Intrinsic;

	//! Proj-to-Camera matrix
	mat4f m_IntrinsicInverse;

	//! World-to-Camera matrix
	mat4f m_Extrinsic;

	//! Camera-to-World matrix
	mat4f m_ExtrinsicInverse;

	//TODO MATTHIAS get rid of the inverse matrices
};

//! write to binary stream overload
template<class BinaryDataBuffer, class BinaryDataCompressor>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator<<(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, const CalibrationData& calibrationData) {
	s << calibrationData.m_Intrinsic;
	s << calibrationData.m_IntrinsicInverse;
	s << calibrationData.m_Extrinsic;
	s << calibrationData.m_ExtrinsicInverse;
	return s;
}

//! read from binary stream overload
template<class BinaryDataBuffer, class BinaryDataCompressor>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator>>(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, CalibrationData& calibrationData) {
	s >> calibrationData.m_Intrinsic;
	s >> calibrationData.m_IntrinsicInverse;
	s >> calibrationData.m_Extrinsic;
	s >> calibrationData.m_ExtrinsicInverse;
	return s;
}


class CalibratedSensorData {
public:
	CalibratedSensorData() {
		#define M_CALIBRATED_SENSOR_DATA_VERSION 2
		m_VersionNumber = M_CALIBRATED_SENSOR_DATA_VERSION;
		m_SensorName = "Unknown";

		m_DepthNumFrames = 0;
		m_DepthImageWidth = 0;
		m_DepthImageHeight = 0;

		m_ColorNumFrames = 0;
		m_ColorImageWidth = 0;
		m_ColorImageHeight = 0;
	}
	~CalibratedSensorData() {
		deleteData();
	}

	void assertVersionNumber() {
		if (m_VersionNumber != M_CALIBRATED_SENSOR_DATA_VERSION)	throw MLIB_EXCEPTION("Invalid file version");
	}

	void deleteData() {
		for (unsigned int i = 0; i < m_DepthImages.size(); i++) {
			SAFE_DELETE_ARRAY(m_DepthImages[i])
		}
		for (unsigned int i = 0; i < m_ColorImages.size(); i++) {
			SAFE_DELETE_ARRAY(m_ColorImages[i]);
		}
		m_DepthImages.clear();
		m_ColorImages.clear();
	}

	float getDepth(unsigned int ux, unsigned int uy, unsigned int frame) const {
		return m_DepthImages[frame][ux + uy*m_DepthImageWidth];
	}

	vec3f getWorldPos(unsigned int ux, unsigned int uy, unsigned int frame) const {
		const float depth = getDepth(ux, uy, frame);
		vec4f world = m_CalibrationDepth.m_IntrinsicInverse*vec4f((float)ux*depth, (float)uy*depth, depth, 0.0f);
		return world.getPoint3d();

		//const float fx = m_CalibrationDepth.m_Intrinsic(0,0);
		//const float fy = m_CalibrationDepth.m_Intrinsic(1,1);
		//const float mx = m_CalibrationDepth.m_Intrinsic(0,2);
		//const float my = m_CalibrationDepth.m_Intrinsic(1,2);
		//float x = ((float)ux-mx) / fx;
		//float y = (my-(float)uy) / fy;
		//return vec3f(depth*x, depth*y, depth);
	}

	void savePointCloud(const std::string& filename, unsigned int frame) const {
		PointCloudf pc;
		for (unsigned int i = 0; i < m_DepthImageWidth*m_DepthImageHeight; i++) {
			float depth = getDepth(i%m_DepthImageWidth, i/m_DepthImageWidth, frame);
			if (depth != -std::numeric_limits<float>::infinity() && depth != -FLT_MAX && depth != 0.0f) {

				vec3f p = getWorldPos(i%m_DepthImageWidth, i/m_DepthImageWidth, frame);
				pc.m_points.push_back(p);
				if (m_ColorImageWidth == m_DepthImageWidth && m_ColorImageHeight == m_DepthImageHeight) {
					vec4uc c = m_ColorImages[frame][i];
					pc.m_colors.push_back(vec4f(c)/255.0f);
				}
			}
		}
		PointCloudIOf::saveToFile(filename, pc);
	} 

	unsigned int	m_VersionNumber;
	std::string		m_SensorName;

	unsigned int m_DepthNumFrames;
	unsigned int m_DepthImageWidth;
	unsigned int m_DepthImageHeight;

	unsigned int m_ColorNumFrames;
	unsigned int m_ColorImageWidth;
	unsigned int m_ColorImageHeight;

	CalibrationData m_CalibrationDepth;
	CalibrationData m_CalibrationColor;

	std::vector<float*>		m_DepthImages;	//in meters
	std::vector<vec4uc*>	m_ColorImages;	//in [0,255]^4

	std::vector<UINT64>	m_DepthImagesTimeStamps;
	std::vector<UINT64>	m_ColorImagesTimeStamps;

	std::vector<mat4f> m_trajectory;
};


#define VAR_STR_LINE(x) '\t' << #x << '=' << x << '\n'

inline std::ostream& operator<<(std::ostream& s, const CalibratedSensorData& sensorData) {
	s << "CalibratedSensorData:\n";
	s << VAR_STR_LINE(sensorData.m_VersionNumber);
	s << VAR_STR_LINE(sensorData.m_SensorName);
	s << VAR_STR_LINE(sensorData.m_DepthNumFrames);
	s << VAR_STR_LINE(sensorData.m_DepthImageWidth);
	s << VAR_STR_LINE(sensorData.m_DepthImageHeight);
	s << VAR_STR_LINE(sensorData.m_ColorNumFrames);
	s << VAR_STR_LINE(sensorData.m_ColorImageWidth);
	s << VAR_STR_LINE(sensorData.m_ColorImageHeight);
	//s << VAR_STR_LINE(sensorData.m_CalibrationDepth);
	//s << VAR_STR_LINE(sensorData.m_CalibrationColor);
	s << VAR_STR_LINE(sensorData.m_DepthImages.size());
	s << VAR_STR_LINE(sensorData.m_ColorImages.size());
	s << VAR_STR_LINE(sensorData.m_DepthImagesTimeStamps.size());
	s << VAR_STR_LINE(sensorData.m_ColorImagesTimeStamps.size());
	s << VAR_STR_LINE(sensorData.m_trajectory.size());
	return s;
}


//! write to binary stream overload
template<class BinaryDataBuffer, class BinaryDataCompressor>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator<<(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, const CalibratedSensorData& sensorData) {
	s << sensorData.m_VersionNumber;
	s << sensorData.m_SensorName;
	s << sensorData.m_DepthNumFrames;
	s << sensorData.m_DepthImageWidth;
	s << sensorData.m_DepthImageHeight;
	s << sensorData.m_ColorNumFrames;
	s << sensorData.m_ColorImageWidth;
	s << sensorData.m_ColorImageHeight;
	s << sensorData.m_CalibrationDepth;
	s << sensorData.m_CalibrationColor;

	assert(sensorData.m_ColorNumFrames == sensorData.m_ColorImages.size());
	assert(sensorData.m_DepthNumFrames == sensorData.m_DepthImages.size());
	assert(sensorData.m_DepthNumFrames == sensorData.m_trajectory.size());

	for (unsigned int i = 0; i < sensorData.m_DepthImages.size(); i++) {
		s.writeData((BYTE*)sensorData.m_DepthImages[i], sizeof(float)*sensorData.m_DepthImageWidth*sensorData.m_DepthImageHeight);
	}
	for (unsigned int i = 0; i < sensorData.m_ColorImages.size(); i++) {
		s.writeData((BYTE*)sensorData.m_ColorImages[i], sizeof(vec4uc)*sensorData.m_ColorImageWidth*sensorData.m_ColorImageHeight);
	}

	s << sensorData.m_ColorImagesTimeStamps;
	s << sensorData.m_DepthImagesTimeStamps;

	s << sensorData.m_trajectory;

	return s;
}


//! read from binary stream overload
template<class BinaryDataBuffer, class BinaryDataCompressor>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator>>(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, CalibratedSensorData& sensorData) {

	sensorData.deleteData();
	s >> sensorData.m_VersionNumber;

	if (sensorData.m_VersionNumber == 1) {
		s >> sensorData.m_SensorName;
		s >> sensorData.m_DepthNumFrames;
		s >> sensorData.m_DepthImageWidth;
		s >> sensorData.m_DepthImageHeight;
		s >> sensorData.m_ColorNumFrames;
		s >> sensorData.m_ColorImageWidth;
		s >> sensorData.m_ColorImageHeight;
		s >> sensorData.m_CalibrationDepth;
		s >> sensorData.m_CalibrationColor;

		sensorData.m_DepthImages.resize(sensorData.m_DepthNumFrames);
		sensorData.m_ColorImages.resize(sensorData.m_ColorNumFrames);

		for (size_t i = 0; i < sensorData.m_DepthImages.size(); i++) {
			sensorData.m_DepthImages[i] = new float[sensorData.m_DepthImageWidth*sensorData.m_DepthImageHeight];
			s.readData((BYTE*)sensorData.m_DepthImages[i], sizeof(float)*sensorData.m_DepthImageWidth*sensorData.m_DepthImageHeight);
		}
		for (size_t i = 0; i < sensorData.m_ColorImages.size(); i++) {
			sensorData.m_ColorImages[i] = new vec4uc[sensorData.m_ColorImageWidth*sensorData.m_ColorImageHeight];
			s.readData((BYTE*)sensorData.m_ColorImages[i], sizeof(vec4uc)*sensorData.m_ColorImageWidth*sensorData.m_ColorImageHeight);
		}

		s >> sensorData.m_ColorImagesTimeStamps;
		s >> sensorData.m_DepthImagesTimeStamps;
	}
	else if (sensorData.m_VersionNumber == M_CALIBRATED_SENSOR_DATA_VERSION) {
		s >> sensorData.m_SensorName;
		s >> sensorData.m_DepthNumFrames;
		s >> sensorData.m_DepthImageWidth;
		s >> sensorData.m_DepthImageHeight;
		s >> sensorData.m_ColorNumFrames;
		s >> sensorData.m_ColorImageWidth;
		s >> sensorData.m_ColorImageHeight;
		s >> sensorData.m_CalibrationDepth;
		s >> sensorData.m_CalibrationColor;

		sensorData.m_DepthImages.resize(sensorData.m_DepthNumFrames);
		sensorData.m_ColorImages.resize(sensorData.m_ColorNumFrames);

		for (size_t i = 0; i < sensorData.m_DepthImages.size(); i++) {
			sensorData.m_DepthImages[i] = new float[sensorData.m_DepthImageWidth*sensorData.m_DepthImageHeight];
			s.readData((BYTE*)sensorData.m_DepthImages[i], sizeof(float)*sensorData.m_DepthImageWidth*sensorData.m_DepthImageHeight);
		}
		for (size_t i = 0; i < sensorData.m_ColorImages.size(); i++) {
			sensorData.m_ColorImages[i] = new vec4uc[sensorData.m_ColorImageWidth*sensorData.m_ColorImageHeight];
			s.readData((BYTE*)sensorData.m_ColorImages[i], sizeof(vec4uc)*sensorData.m_ColorImageWidth*sensorData.m_ColorImageHeight);
		}

		s >> sensorData.m_ColorImagesTimeStamps;
		s >> sensorData.m_DepthImagesTimeStamps;

		s >> sensorData.m_trajectory;
	} else {
		throw MLIB_EXCEPTION("Calibrated Sensor Data: Invalid file version");
	}

	return s;
}

/*
class CalibratedSensorDataArray {
public:
	CalibratedSensorDataArray() {
#define M_CALIBRATED_SENSOR_DATA_ARRAY_VERSION 1
		m_VersionNumber = M_CALIBRATED_SENSOR_DATA_ARRAY_VERSION;
	}
	~CalibratedSensorDataArray() {
		for (size_t i = 0; i < m_CalibratedSensorData.size(); i++) {
			m_CalibratedSensorData[i].deleteData();
		}
	}

	void assertVersionNumber() {
		if (m_VersionNumber != M_CALIBRATED_SENSOR_DATA_ARRAY_VERSION)	throw MLIB_EXCEPTION("Invalid file version");
	}

	unsigned int m_VersionNumber;
	unsigned int m_NumSensors;
	std::vector<CalibratedSensorData>	m_CalibratedSensorData;
};



//cannot overload via template since it is not supposed to work for complex types


//! write to binary stream overload
template<class BinaryDataBuffer, class BinaryDataCompressor>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator<<(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, const CalibratedSensorData& sensorData) {
	s << sensorData.m_VersionNumber;
	s << sensorData.m_ColorNumFrames;
	s << sensorData.m_ColorNumCameras;
	s << sensorData.m_ColorImageWidth;
	s << sensorData.m_ColorImageHeight;
	s << sensorData.m_ColorCalibration;
	s << sensorData.m_DepthNumFrames;
	s << sensorData.m_DepthNumCameras;
	s << sensorData.m_DepthImageWidth;
	s << sensorData.m_DepthImageHeight;
	s << sensorData.m_DepthCalibration;

	assert(sensorData.m_ColorNumFrames*sensorData.m_ColorNumCameras == sensorData.m_ColorImages.size());
	assert(sensorData.m_DepthNumFrames*sensorData.m_DepthNumCameras == sensorData.m_DepthImages.size());

	for (unsigned int i = 0; i < sensorData.m_DepthImages.size(); i++) {
		s.writeData((BYTE*)sensorData.m_DepthImages[i], sizeof(float)*sensorData.m_DepthImageWidth*sensorData.m_DepthImageHeight);
	}
	for (unsigned int i = 0; i < sensorData.m_ColorImages.size(); i++) {
		s.writeData((BYTE*)sensorData.m_ColorImages[i], sizeof(vec4uc)*sensorData.m_ColorImageWidth*sensorData.m_ColorImageHeight);
	}

	s << sensorData.m_ColorImagesTimeStamps;
	s << sensorData.m_DepthImagesTimeStamps;

	return s;
}

//! read from binary stream overload
template<class BinaryDataBuffer, class BinaryDataCompressor>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator>>(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s,CalibratedSensorData& sensorData) {

	sensorData.deleteData();
	s >> sensorData.m_VersionNumber;

	if (sensorData.m_VersionNumber == M_SENSOR_DATA_VERSION_NUMBER_ - 1) {
		MLIB_WARNING("CalibratedSensor Data Version: Obsolte version number; still works but should be updated");
		s >> sensorData.m_ColorNumFrames;
		s >> sensorData.m_ColorNumCameras;
		s >> sensorData.m_ColorImageWidth;
		s >> sensorData.m_ColorImageHeight;
		s >> sensorData.m_ColorCalibration;
		s >> sensorData.m_DepthNumFrames;
		s >> sensorData.m_DepthNumCameras;
		s >> sensorData.m_DepthImageWidth;
		s >> sensorData.m_DepthImageHeight;
		s >> sensorData.m_DepthCalibration;

		sensorData.m_DepthImages.resize(sensorData.m_DepthNumFrames*sensorData.m_DepthNumCameras);
		sensorData.m_ColorImages.resize(sensorData.m_ColorNumFrames*sensorData.m_ColorNumCameras);

		for (unsigned int i = 0; i < sensorData.m_DepthImages.size(); i++) {
			sensorData.m_DepthImages[i] = new float[sensorData.m_DepthImageWidth*sensorData.m_DepthImageHeight];
			s.readData((BYTE*)sensorData.m_DepthImages[i], sizeof(float)*sensorData.m_DepthImageWidth*sensorData.m_DepthImageHeight);
		}
		for (unsigned int i = 0; i < sensorData.m_ColorImages.size(); i++) {
			sensorData.m_ColorImages[i] = new vec4uc[sensorData.m_ColorImageWidth*sensorData.m_ColorImageHeight];
			s.readData((BYTE*)sensorData.m_ColorImages[i], sizeof(vec4uc)*sensorData.m_ColorImageWidth*sensorData.m_ColorImageHeight);
		}
	} else if (sensorData.m_VersionNumber == M_SENSOR_DATA_VERSION_NUMBER_) {
		s >> sensorData.m_ColorNumFrames;
		s >> sensorData.m_ColorNumCameras;
		s >> sensorData.m_ColorImageWidth;
		s >> sensorData.m_ColorImageHeight;
		s >> sensorData.m_ColorCalibration;
		s >> sensorData.m_DepthNumFrames;
		s >> sensorData.m_DepthNumCameras;
		s >> sensorData.m_DepthImageWidth;
		s >> sensorData.m_DepthImageHeight;
		s >> sensorData.m_DepthCalibration;

		sensorData.m_DepthImages.resize(sensorData.m_DepthNumFrames*sensorData.m_DepthNumCameras);
		sensorData.m_ColorImages.resize(sensorData.m_ColorNumFrames*sensorData.m_ColorNumCameras);

		for (unsigned int i = 0; i < sensorData.m_DepthImages.size(); i++) {
			sensorData.m_DepthImages[i] = new float[sensorData.m_DepthImageWidth*sensorData.m_DepthImageHeight];
			s.readData((BYTE*)sensorData.m_DepthImages[i], sizeof(float)*sensorData.m_DepthImageWidth*sensorData.m_DepthImageHeight);
		}
		for (unsigned int i = 0; i < sensorData.m_ColorImages.size(); i++) {
			sensorData.m_ColorImages[i] = new vec4uc[sensorData.m_ColorImageWidth*sensorData.m_ColorImageHeight];
			s.readData((BYTE*)sensorData.m_ColorImages[i], sizeof(vec4uc)*sensorData.m_ColorImageWidth*sensorData.m_ColorImageHeight);
		}

		s >> sensorData.m_ColorImagesTimeStamps;
		s >> sensorData.m_DepthImagesTimeStamps;
	} else {
		throw MLIB_EXCEPTION("Calibrated Sensor Data: Invalid file version");
	}

	return s;
}
*/

}  // namespace ml

#endif  // EXT_DEPTHCAMERA_CALIBRATEDSENSORDATA_H_
