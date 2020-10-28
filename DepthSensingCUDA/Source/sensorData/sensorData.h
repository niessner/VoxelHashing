

#ifndef _SENSOR_FILE_H_
#define _SENSOR_FILE_H_

///////////////////////////////////////////////////////////////////////////////////
////////////////// ADJUST THESE DEFINES TO YOUR NEEDS /////////////////////////////
///////////////////////////////////////////////////////////////////////////////////

#define _HAS_MLIB
#define _USE_UPLINK_COMPRESSION

///////////////////////////////////////////////////////////////////////////////////
/////////////////// DON'T TOUCH THE FILE BLOW THIS LINE ///////////////////////////
///////////////////////////////////////////////////////////////////////////////////

namespace stb {
//#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
//#undef STB_IMAGE_IMPLEMENTATION

//#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
//#undef STB_IMAGE_WRITE_IMPLEMENTATION
}

#ifdef _USE_UPLINK_COMPRESSION
	#if _WIN32
	#pragma comment(lib, "gdiplus.lib")
	#pragma comment(lib, "Shlwapi.lib")
	#include <winsock2.h>
	#include <Ws2tcpip.h>
	#include "sensorData/uplinksimple.h"
	#else 
	#undef _USE_UPLINK_COMPRESSION
	#endif 
#endif
#include <vector>
#include <string>
#include <exception>
#include <fstream>
#include <cassert>
#include <iostream>
#include <atomic>


namespace ml {

#ifndef _HAS_MLIB

#ifndef _MSC_VER
#define NOEXCEPT noexcept
#else
#define NOEXCEPT
#endif

	class MLibException : public std::exception {
	public:
		MLibException(const std::string& what) : std::exception() {
			m_msg = what;
		}
		MLibException(const char* what) : std::exception() {
			m_msg = std::string(what);
		}
		const char* what() const NOEXCEPT{
			return m_msg.c_str();
		}
	private:
		std::string m_msg;
	};



#ifndef MLIB_EXCEPTION
#define MLIB_EXCEPTION(s) ml::MLibException(std::string(__FUNCTION__).append(":").append(std::to_string(__LINE__)).append(": ").append(s).c_str())
#endif

#ifndef SAFE_DELETE
#define SAFE_DELETE(p)       { if (p) { delete (p);     (p)=nullptr; } }
#endif

#ifndef SAFE_DELETE_ARRAY
#define SAFE_DELETE_ARRAY(p) { if (p) { delete[] (p);   (p)=nullptr; } }
#endif

#ifndef UINT64
#ifdef WIN32
	typedef unsigned __int64 UINT64;
#else
	typedef uint64_t UINT64;
#endif
#endif

	class vec3uc {
		union
		{
			struct
			{
				unsigned char x, y, z; // standard names for components
			};
			unsigned char array[3];     // array access
		};
	};

	class vec4uc {
		union
		{
			struct
			{
				unsigned char x, y, z, w; // standard names for components
			};
			unsigned char array[4];     // array access
		};
	};

	class mat4f {
	public:
		mat4f() { }

		Matrix4x4(	
			const float& m00, const float& m01, const float& m02, const float& m03,
			const float& m10, const float& m11, const float& m12, const float& m13,
			const float& m20, const float& m21, const float& m22, const float& m23,
			const float& m30, const float& m31, const float& m32, const float& m33) 
		{
			_m00 = m00;	_m01 = m01;	_m02 = m02;	_m03 = m03;
			_m10 = m10;	_m11 = m11;	_m12 = m12;	_m13 = m13;
			_m20 = m20;	_m21 = m21;	_m22 = m22;	_m23 = m23;
			_m30 = m30;	_m31 = m31;	_m32 = m32;	_m33 = m33;
		}

		void setIdentity() {
			matrix[0] = 1.0;	matrix[1] = 0.0f;	matrix[2] = 0.0f; matrix[3] = 0.0f;
			matrix[4] = 0.0f;	matrix[5] = 1.0;	matrix[6] = 0.0f; matrix[7] = 0.0f;
			matrix[8] = 0.0f;	matrix[9] = 0.0f;	matrix[10] = 1.0; matrix[11] = 0.0f;
			matrix[12] = 0.0f;	matrix[13] = 0.0f;	matrix[14] = 0.0f; matrix[15] = 1.0;
		}

		static mat4f identity() {
			mat4f res;	res.setIdentity();
			return res;
		}
		union {
			//! access matrix using a single array
			float matrix[16];
			//! access matrix using a two-dimensional array
			float matrix2[4][4];
			//! access matrix using single elements
			struct {
				float
					_m00, _m01, _m02, _m03,
					_m10, _m11, _m12, _m13,
					_m20, _m21, _m22, _m23,
					_m30, _m31, _m32, _m33;
			};
		};
	};
#endif //_NO_MLIB_


	class SensorData {
	public:
		class CalibrationData {
		public:
			CalibrationData() {
				setIdentity();
			}

			CalibrationData(const mat4f& intrinsic, const mat4f& extrinsic = mat4f::identity()) {
				m_intrinsic = intrinsic;
				m_extrinsic = extrinsic;
			}

			static mat4f makeIntrinsicMatrix(float fx, float fy, float mx, float my) {
				return mat4f(
					fx, 0.0f,   mx,   0.0f,
					0.0f, fy,   my,   0.0f,
					0.0f, 0.0f, 1.0f, 0.0f,
					0.0f, 0.0f, 0.0f, 1.0f
					);
			}

			void setIdentity() {
				m_intrinsic.setIdentity();
				m_extrinsic.setIdentity();
			}

			void setMatrices(mat4f& intrinsic, mat4f& extrinsic = mat4f::identity()) {
				m_intrinsic = intrinsic;
				m_extrinsic = extrinsic;
				//m_intr = m_intrinsic.getInverse();
				//m_ExtrinsicInverse = m_extrinsic.getInverse();
			}

			void saveToFile(std::ofstream& out) const {
				out.write((const char*)&m_intrinsic, sizeof(mat4f));
				out.write((const char*)&m_extrinsic, sizeof(mat4f));
			}

			void loadFromFile(std::ifstream& in) {
				in.read((char*)&m_intrinsic, sizeof(mat4f));
				in.read((char*)&m_extrinsic, sizeof(mat4f));
			}

			bool operator==(const CalibrationData& other) const {
				for (unsigned int i = 0; i < 16; i++) {
					if (m_intrinsic[i] != other.m_intrinsic[i]) return false;
					if (m_extrinsic[i] != other.m_extrinsic[i]) return false;
				}
				return true;
			}

			bool operator!=(const CalibrationData& other) const {
				return !((*this) == other);
			}

			//! Camera-to-Proj matrix
			mat4f m_intrinsic;

			//! World-to-Camera matrix (accumulated R|t mapping back to the first frame))
			mat4f m_extrinsic;
		};

		enum COMPRESSION_TYPE_COLOR {
			TYPE_RAW = 0,
			TYPE_PNG = 1,
			TYPE_JPEG = 2
		};
		enum COMPRESSION_TYPE_DEPTH {
			TYPE_RAW_USHORT = 0,
			TYPE_ZLIB_USHORT = 1,
			TYPE_OCCI_USHORT = 2
		};


		class RGBDFrame {
		public:
			RGBDFrame() {
				m_colorCompressed = NULL;
				m_depthCompressed = NULL;
				m_colorSizeBytes = 0;
				m_depthSizeBytes = 0;
				m_cameraToWorld.setIdentity();
				m_timeStampColor = 0;
				m_timeStampDepth = 0;
			}


			unsigned char* getColorCompressed() const {
				return m_colorCompressed;
			}
			unsigned char* getDepthCompressed() const {
				return m_depthCompressed;
			}
			size_t getColorSizeBytes() const {
				return m_colorSizeBytes;
			}
			size_t getDepthSizeBytes() const {
				return m_depthSizeBytes;
			}

			const mat4f& getCameraToWorld() const {
				return m_cameraToWorld;
			}

			void setCameraToWorld(const mat4f& cameraToWorld) {
				m_cameraToWorld = cameraToWorld;
			}

			//! typically in microseconds
			void setTimeStampColor(UINT64 t) {
				m_timeStampColor = t;
			}
			//! typically in microseconds
			void setTimeStampDepth(UINT64 t) {
				m_timeStampDepth = t;
			}

			//! returns the color time stamp; typically in microseconds
			UINT64 getTimeStampColor() const {
				return m_timeStampColor;
			}

			//! returns the depth time stamp; typically in microseconds
			UINT64 getTimeStampDepth() const {
				return m_timeStampDepth;
			}

		private:
			friend class SensorData;
			friend class SensorDataZhou;

			RGBDFrame(
				const vec3uc* color, unsigned int colorWidth, unsigned int colorHeight,
				const unsigned short*  depth, unsigned int depthWidth, unsigned int depthHeight,
				const mat4f& cameraToWorld = mat4f::identity(),
				COMPRESSION_TYPE_COLOR colorType = TYPE_JPEG,
				COMPRESSION_TYPE_DEPTH depthType = TYPE_ZLIB_USHORT,	
				UINT64 timeStampColor = 0,
				UINT64 timeStampDepth = 0)
			{
				m_colorCompressed = NULL;
				m_depthCompressed = NULL;
				m_colorSizeBytes = 0;
				m_depthSizeBytes = 0;
				m_cameraToWorld = cameraToWorld;
				m_timeStampColor = timeStampColor;
				m_timeStampDepth = timeStampDepth;

				if (color) {
					//Timer t;
					compressColor(color, colorWidth, colorHeight, colorType);
					//std::cout << "compressColor " << t.getElapsedTimeMS() << " [ms] " << std::endl;
				}
				if (depth) {
					//Timer t;
					compressDepth(depth, depthWidth, depthHeight, depthType);
					//std::cout << "compressDepth " << t.getElapsedTimeMS() << " [ms] " << std::endl;
				}
			}

			void free() {
				freeColor();
				freeDepth();
				m_cameraToWorld.setIdentity();
			}


			void freeColor() {
				if (m_colorCompressed) std::free(m_colorCompressed);
				m_colorCompressed = NULL;
				m_colorSizeBytes = 0;
				m_timeStampColor = 0;
			}
			void freeDepth() {
				if (m_depthCompressed) std::free(m_depthCompressed);
				m_depthCompressed = NULL;
				m_depthSizeBytes = 0;
				m_timeStampDepth = 0;
			}


			void compressColor(const vec3uc* color, unsigned int width, unsigned int height, COMPRESSION_TYPE_COLOR type) {

				if (type == TYPE_RAW) {
					if (m_colorSizeBytes != width*height) {
						freeColor();
						m_colorSizeBytes = width*height*sizeof(vec3uc);
						m_colorCompressed = (unsigned char*)std::malloc(m_colorSizeBytes);
					}
					std::memcpy(m_colorCompressed, color, m_colorSizeBytes);
				}
				else if (type == TYPE_PNG || type == TYPE_JPEG) {
					freeColor();

#ifdef _USE_UPLINK_COMPRESSION
					uplinksimple::graphics_PixelFormat format = uplinksimple::graphics_PixelFormat_RGB;
					uplinksimple::graphics_ImageCodec codec = uplinksimple::graphics_ImageCodec_JPEG;
					if (type == TYPE_PNG) codec = uplinksimple::graphics_ImageCodec_PNG;

					uplinksimple::MemoryBlock block;
					float quality = uplinksimple::defaultQuality;
					//float quality = 1.0f;
					uplinksimple::encode_image(codec, (const uint8_t*)color, width*height, format, width, height, block, quality);
					m_colorCompressed = block.Data;
					m_colorSizeBytes = block.Size;
					block.relinquishOwnership();
#else
					throw MLIB_EXCEPTION("need UPLINK_COMPRESSION");
#endif
				}
				else {
					throw MLIB_EXCEPTION("unknown compression type");
				}
			}

			vec3uc* decompressColorAlloc(COMPRESSION_TYPE_COLOR type) const {
				if (type == TYPE_RAW)	return decompressColorAlloc_raw(type);
#ifdef _USE_UPLINK_COMPRESSION
				else return decompressColorAlloc_occ(type);	//this handles all image formats;
#else
				else return decompressColorAlloc_stb(type);	// this handles all image formats
#endif
			}

			vec3uc* decompressColorAlloc_stb(COMPRESSION_TYPE_COLOR type) const {	//can handle PNG, JPEG etc.
				if (type != TYPE_JPEG && type != TYPE_PNG) throw MLIB_EXCEPTION("invliad type");
				if (m_colorCompressed == NULL || m_colorSizeBytes == 0) throw MLIB_EXCEPTION("decompression error");
				int channels = 3;
				int width, height;
				unsigned char* raw = stb::stbi_load_from_memory(m_colorCompressed, (int)m_colorSizeBytes, &width, &height, NULL, channels);
				return (vec3uc*)raw;
			}

			vec3uc* decompressColorAlloc_occ(COMPRESSION_TYPE_COLOR type) const {
#ifdef _USE_UPLINK_COMPRESSION
				if (type != TYPE_JPEG && type != TYPE_PNG) throw MLIB_EXCEPTION("invliad type");
				if (m_colorCompressed == NULL || m_colorSizeBytes == 0) throw MLIB_EXCEPTION("decompression error");
				uplinksimple::graphics_PixelFormat format = uplinksimple::graphics_PixelFormat_RGB;
				uplinksimple::graphics_ImageCodec codec = uplinksimple::graphics_ImageCodec_JPEG;
				if (type == TYPE_PNG) codec = uplinksimple::graphics_ImageCodec_PNG;

				uplinksimple::MemoryBlock block;
				float quality = uplinksimple::defaultQuality;
				size_t width = 0;
				size_t height = 0;
				uplinksimple::decode_image(codec, (const uint8_t*)m_colorCompressed, m_colorSizeBytes, format, width, height, block);
				vec3uc* res = (vec3uc*)block.Data;
				block.relinquishOwnership();
				return res;
#else
				throw MLIB_EXCEPTION("need UPLINK_COMPRESSION");
				return NULL;
#endif
			}

			vec3uc* decompressColorAlloc_raw(COMPRESSION_TYPE_COLOR type) const {
				if (type != TYPE_RAW) throw MLIB_EXCEPTION("invliad type");
				if (m_colorCompressed == NULL || m_colorSizeBytes == 0) throw MLIB_EXCEPTION("invalid data");
				vec3uc* res = (vec3uc*)std::malloc(m_colorSizeBytes);
				memcpy(res, m_colorCompressed, m_colorSizeBytes);
				return res;
			}

			void compressDepth(const unsigned short* depth, unsigned int width, unsigned int height, COMPRESSION_TYPE_DEPTH type) {
				freeDepth();

				if (type == TYPE_RAW_USHORT) {
					if (m_depthSizeBytes != width*height) {
						freeDepth();
						m_depthSizeBytes = width*height*sizeof(unsigned short);
						m_depthCompressed = (unsigned char*)std::malloc(m_depthSizeBytes);
					}
					std::memcpy(m_depthCompressed, depth, m_depthSizeBytes);
				}
				else if (type == TYPE_ZLIB_USHORT) {
					freeDepth();

					int out_len = 0;
					int quality = 8;
					int n = 2;
					unsigned char* tmpBuff = (unsigned char *)std::malloc((width*n + 1) * height);
					std::memcpy(tmpBuff, depth, width*height*sizeof(unsigned short));
					m_depthCompressed = stb::stbi_zlib_compress(tmpBuff, width*height*sizeof(unsigned short), &out_len, quality);
					std::free(tmpBuff);
					m_depthSizeBytes = out_len;
				} 
				else if (type == TYPE_OCCI_USHORT) {
					freeDepth();
#ifdef _USE_UPLINK_COMPRESSION
					//TODO fix the shift here
					int out_len = 0;
					int n = 2;
					unsigned int tmpBuffSize = (width*n + 1) * height;
					unsigned char* tmpBuff = (unsigned char *)std::malloc(tmpBuffSize);
					out_len = uplinksimple::encode(depth, width*height, tmpBuff, tmpBuffSize);
					m_depthSizeBytes = out_len;
					m_depthCompressed = (unsigned char*)std::malloc(out_len);
					std::memcpy(m_depthCompressed, tmpBuff, out_len);
					std::free(tmpBuff);
#else
					throw MLIB_EXCEPTION("need UPLINK_COMPRESSION");
#endif
				}
				else {
					throw MLIB_EXCEPTION("unknown compression type");
				}
			}

			unsigned short* decompressDepthAlloc(unsigned int width, unsigned int height, COMPRESSION_TYPE_DEPTH type) const {
				if (type == TYPE_RAW_USHORT)	return decompressDepthAlloc_raw(type);
				else if (type == TYPE_ZLIB_USHORT) return decompressDepthAlloc_stb(type);
				else if (type == TYPE_OCCI_USHORT) return decompressDepthAlloc_occ(width, height, type);
				else {
					throw MLIB_EXCEPTION("invalid type");
					return NULL;
				}
			}

			unsigned short* decompressDepthAlloc_stb(COMPRESSION_TYPE_DEPTH type) const {
				if (type != TYPE_ZLIB_USHORT) throw MLIB_EXCEPTION("invliad type");
				unsigned short* res;
				int len;
				res = (unsigned short*)stb::stbi_zlib_decode_malloc((const char*)m_depthCompressed, (int)m_depthSizeBytes, &len);
				return res;
			}

			unsigned int short* decompressDepthAlloc_occ(unsigned int width, unsigned int height, COMPRESSION_TYPE_DEPTH type) const {
#ifdef _USE_UPLINK_COMPRESSION
				if (type != TYPE_OCCI_USHORT) throw MLIB_EXCEPTION("invliad type");
				unsigned short* res = (unsigned short*)std::malloc(width*height * 2);
				uplinksimple::decode(m_depthCompressed, (unsigned int)m_depthSizeBytes, width*height, res);
				//uplinksimple::shift2depth(res, m_depthWidth*m_depthHeight); //this is a stupid idea i think
				return res;
#else
				throw MLIB_EXCEPTION("need UPLINK_COMPRESSION");
				return NULL;
#endif
			}

			unsigned short* decompressDepthAlloc_raw(COMPRESSION_TYPE_DEPTH type) const {
				if (type != TYPE_RAW_USHORT) throw MLIB_EXCEPTION("invliad type");
				if (m_depthCompressed == NULL || m_depthSizeBytes == 0) throw MLIB_EXCEPTION("invalid data");
				unsigned short* res = (unsigned short*)std::malloc(m_depthSizeBytes);
				memcpy(res, m_depthCompressed, m_depthSizeBytes);
				return res;
			}


			void saveToFile(std::ofstream& out) const {
				out.write((const char*)&m_cameraToWorld, sizeof(mat4f));
				out.write((const char*)&m_timeStampColor, sizeof(UINT64));
				out.write((const char*)&m_timeStampDepth, sizeof(UINT64));
				out.write((const char*)&m_colorSizeBytes, sizeof(UINT64));
				out.write((const char*)&m_depthSizeBytes, sizeof(UINT64));
				out.write((const char*)m_colorCompressed, m_colorSizeBytes);
				out.write((const char*)m_depthCompressed, m_depthSizeBytes);
			}

			void loadFromFile(std::ifstream& in) {
				free();
				in.read((char*)&m_cameraToWorld, sizeof(mat4f));
				in.read((char*)&m_timeStampColor, sizeof(UINT64));
				in.read((char*)&m_timeStampDepth, sizeof(UINT64));
				in.read((char*)&m_colorSizeBytes, sizeof(UINT64));
				in.read((char*)&m_depthSizeBytes, sizeof(UINT64));
				m_colorCompressed = (unsigned char*)std::malloc(m_colorSizeBytes);
				in.read((char*)m_colorCompressed, m_colorSizeBytes);
				m_depthCompressed = (unsigned char*)std::malloc(m_depthSizeBytes);
				in.read((char*)m_depthCompressed, m_depthSizeBytes);
			}

			bool operator==(const RGBDFrame& other) const {
				if (m_colorSizeBytes != other.m_colorSizeBytes) return false;
				if (m_depthSizeBytes != other.m_depthSizeBytes) return false;
				if (m_timeStampColor != other.m_timeStampColor) return false;
				if (m_timeStampDepth != other.m_timeStampDepth) return false;
				for (unsigned int i = 0; i < 16; i++) {
					if (m_cameraToWorld[i] != other.m_cameraToWorld[i]) return false;
				}
				for (UINT64 i = 0; i < m_colorSizeBytes; i++) {
					if (m_colorCompressed[i] != other.m_colorCompressed[i]) return false;
				}
				for (UINT64 i = 0; i < m_depthSizeBytes; i++) {
					if (m_depthCompressed[i] != other.m_depthCompressed[i]) return false;
				}
				return true;
			}

			bool operator!=(const RGBDFrame& other) const {
				return !((*this) == other);
			}

			UINT64 m_colorSizeBytes;					//compressed byte size
			UINT64 m_depthSizeBytes;					//compressed byte size
			unsigned char* m_colorCompressed;			//compressed color data
			unsigned char* m_depthCompressed;			//compressed depth data
			UINT64 m_timeStampColor;					//time stamp color (convection: in microseconds)
			UINT64 m_timeStampDepth;					//time stamp depth (convention: in microseconds)
			mat4f m_cameraToWorld;						//camera trajectory: from current frame to base frame
		};

		struct IMUFrame {
			IMUFrame() {
				rotationRate = vec3d(0.0);
				acceleration = vec3d(0.0);
				magneticField = vec3d(0.0);
				attitude = vec3d(0.0);
				gravity = vec3d(0.0);
				timeStamp = 0;
			}

			void loadFromFile(std::ifstream& in) {
				in.read((char*)&rotationRate, sizeof(vec3d));
				in.read((char*)&acceleration, sizeof(vec3d));
				in.read((char*)&magneticField, sizeof(vec3d));
				in.read((char*)&attitude, sizeof(vec3d));
				in.read((char*)&gravity, sizeof(vec3d));
				in.read((char*)&timeStamp, sizeof(UINT64));				
			}
			void saveToFile(std::ofstream& out) const {
				out.write((const char*)&rotationRate, sizeof(vec3d));
				out.write((const char*)&acceleration, sizeof(vec3d));
				out.write((const char*)&magneticField, sizeof(vec3d));
				out.write((const char*)&attitude, sizeof(vec3d));
				out.write((const char*)&gravity, sizeof(vec3d));
				out.write((const char*)&timeStamp, sizeof(UINT64));
			}

			bool operator==(const IMUFrame& other) const {
				if (rotationRate != other.rotationRate) return false;
				if (acceleration != other.acceleration) return false;
				if (magneticField != other.magneticField) return false;
				if (attitude != other.attitude) return false;
				if (gravity != other.gravity) return false;
				if (timeStamp != other.timeStamp) return false;
			}

			bool operator!=(const IMUFrame& other) const {
				return *this != other;
			}
			
			vec3d rotationRate;		//angular velocity (raw data)
			vec3d acceleration;		//acceleration in x,y,z direction (raw data)
			vec3d magneticField;	//magnetometer data (raw data)
			vec3d attitude;			//roll, pitch, yaw estimate (inferred)			
			vec3d gravity;			//gravitation dir estimate (inferred)	
			UINT64 timeStamp;		//timestamp (typically in microseconds)
		};

		//////////////////////////////////
		// SensorData Class starts here //
		//////////////////////////////////

#define M_SENSOR_DATA_VERSION 4
		///version 3 was missing the IMUFrame vector
		//the first 3 versions [0,1,2] are reserved for the old .sensor files


		SensorData() {
			m_versionNumber = M_SENSOR_DATA_VERSION;
			m_sensorName = "Unknown";
			m_colorWidth = 0;
			m_colorHeight = 0;
			m_depthWidth = 0;
			m_depthHeight = 0;
		}

		SensorData(const std::string& filename) {
			m_versionNumber = M_SENSOR_DATA_VERSION;
			m_sensorName = "Unknown";
			loadFromFile(filename);
		}

		~SensorData() {
			free();
		}

		//! frees all allocated memory (all data will be lost)
		void free() {
			for (size_t i = 0; i < m_frames.size(); i++) {
				m_frames[i].free();
			}
			m_frames.clear();
			m_calibrationColor.setIdentity();
			m_calibrationDepth.setIdentity();
			m_colorWidth = 0;
			m_colorHeight = 0;
			m_depthWidth = 0;
			m_depthHeight = 0;
			m_IMUFrames.clear();
		}

		//! checks the version number
		void assertVersionNumber() const {
			if (m_versionNumber != M_SENSOR_DATA_VERSION)	
				throw MLIB_EXCEPTION("Invalid file version -- found " + std::to_string(m_versionNumber) + " but expectd " + std::to_string(M_SENSOR_DATA_VERSION));
		}

		//! this function must be called if not read from a file
		void initDefault(
			unsigned int colorWidth,
			unsigned int colorHeight,
			unsigned int depthWidth,
			unsigned int depthHeight,
			const CalibrationData& calibrationColor,
			const CalibrationData& calibrationDepth,
			COMPRESSION_TYPE_COLOR colorType = TYPE_JPEG,
			COMPRESSION_TYPE_DEPTH depthType = TYPE_ZLIB_USHORT,
			float depthShift = 1000.0f,
			const std::string sensorName = "Unknown") 
		{
			m_sensorName = sensorName;
			m_colorCompressionType = colorType;
			m_depthCompressionType = depthType;
			m_colorWidth = colorWidth;
			m_colorHeight = colorHeight;
			m_depthWidth = depthWidth;
			m_depthHeight = depthHeight;
			m_depthShift = depthShift;
			m_calibrationColor = calibrationColor;
			m_calibrationDepth = calibrationDepth;
		}

		RGBDFrame& addFrame(const vec3uc* color, const unsigned short* depth, const mat4f& cameraToWorld = mat4f::identity(), UINT64 timeStampColor = 0, UINT64 timeStampDepth = 0) {
			m_frames.push_back(RGBDFrame(color, m_colorWidth, m_colorHeight, depth, m_depthWidth, m_depthHeight, cameraToWorld, m_colorCompressionType, m_depthCompressionType, timeStampColor, timeStampDepth));
			return m_frames.back();
		}

		IMUFrame& addIMUFrame(const IMUFrame& frame) {
			m_IMUFrames.push_back(frame);
			return m_IMUFrames.back();
		}
		
		//! decompresses the frame and allocates the memory -- needs std::free afterwards!
		vec3uc* decompressColorAlloc(const RGBDFrame& f) const {
			return f.decompressColorAlloc(m_colorCompressionType);
		}
		//! decompresses the frame and allocates the memory -- needs std::free afterwards!
		vec3uc* decompressColorAlloc(size_t frameIdx) const {
			if (frameIdx > m_frames.size()) throw MLIB_EXCEPTION("out of bounds");
			return decompressColorAlloc(m_frames[frameIdx]);
		}

		//! decompresses the frame and allocates the memory -- needs std::free afterwards!
		unsigned short* decompressDepthAlloc(const RGBDFrame& f) const {
			return f.decompressDepthAlloc(m_depthWidth, m_depthHeight, m_depthCompressionType);
		}
		//! decompresses the frame and allocates the memory -- needs std::free afterwards!
		unsigned short* decompressDepthAlloc(size_t frameIdx) const {
			if (frameIdx > m_frames.size()) throw MLIB_EXCEPTION("out of bounds");
			return decompressDepthAlloc(m_frames[frameIdx]);
		}

		//! returns the closest IMUFrame for a given RGBDFrame (time-wise)
		const IMUFrame& findClosestIMUFrame(const RGBDFrame& f, bool basedOnRGB = true) const {

			if (m_IMUFrames.size() == 0) throw MLIB_EXCEPTION("no imu data available");

			UINT64 t = f.m_timeStampColor;
			if (!basedOnRGB) t = f.m_timeStampDepth;
			
			size_t begin = 0;
			size_t end = m_IMUFrames.size();
			const UINT64 key = t;

			if (key < m_IMUFrames[0].timeStamp) return m_IMUFrames[0];
			if (key > m_IMUFrames.back().timeStamp) return m_IMUFrames.back();

			while (begin + 1 < end) {
				size_t middle = begin + ((end - begin) / 2);

				//std::cout << "range (" << *begin << ", " << *middle << ", ";
				//if (end != invalid) std::cout << *end << ")" << std::endl;
				//else std::cout << "END)" << std::endl;

				if (m_IMUFrames[middle].timeStamp == t) {	// in that case we exactly found the value
					return m_IMUFrames[middle];
				}
				else if (m_IMUFrames[middle].timeStamp > key) {
					end = middle;
				}
				else {
					begin = middle;
				}
			}

			// still possible that begin == key or end == key; otherwise return the closest
			if (key - m_IMUFrames[begin].timeStamp < m_IMUFrames[end].timeStamp - key) {
				return m_IMUFrames[begin];
			}
			else {
				return m_IMUFrames[end];
			}
		}

		//! returns the closest IMUFrame for a given RGBDFrame (time-wise)
		const IMUFrame& findClosestIMUFrame(size_t frameIdx, bool basedOnRGB = true) const {
			return findClosestIMUFrame(m_frames[frameIdx], basedOnRGB);
		}


		//! saves a .sens file
		void saveToFile(const std::string& filename) const {
			std::ofstream out(filename, std::ios::binary);

			out.write((const char*)&m_versionNumber, sizeof(unsigned int));
			UINT64 strLen = m_sensorName.size();
			out.write((const char*)&strLen, sizeof(UINT64));
			out.write((const char*)&m_sensorName[0], strLen*sizeof(char));

			m_calibrationColor.saveToFile(out);
			m_calibrationDepth.saveToFile(out);

			out.write((const char*)&m_colorCompressionType, sizeof(COMPRESSION_TYPE_COLOR));
			out.write((const char*)&m_depthCompressionType, sizeof(COMPRESSION_TYPE_DEPTH));
			out.write((const char*)&m_colorWidth, sizeof(unsigned int));
			out.write((const char*)&m_colorHeight, sizeof(unsigned int));
			out.write((const char*)&m_depthWidth, sizeof(unsigned int));
			out.write((const char*)&m_depthHeight, sizeof(unsigned int));
			out.write((const char*)&m_depthShift, sizeof(float));

			UINT64 numFrames = m_frames.size();
			out.write((const char*)&numFrames, sizeof(UINT64));
			for (size_t i = 0; i < m_frames.size(); i++) {
				m_frames[i].saveToFile(out);
			}

			UINT64 numIMUFrames = m_frames.size();
			out.write((const char*)&numIMUFrames, sizeof(UINT64));
			for (size_t i = 0; i < m_IMUFrames.size(); i++) {
				m_IMUFrames[i].saveToFile(out);
			}
		}

		//! loads a .sens file
		void loadFromFile(const std::string& filename) {
			std::ifstream in(filename, std::ios::binary);

			if (!in.is_open()) {
				throw MLIB_EXCEPTION("could not open file " + filename);
			}

			in.read((char*)&m_versionNumber, sizeof(unsigned int));
			assertVersionNumber();
			UINT64 strLen = 0;
			in.read((char*)&strLen, sizeof(UINT64));
			m_sensorName.resize(strLen);
			in.read((char*)&m_sensorName[0], strLen*sizeof(char));

			m_calibrationColor.loadFromFile(in);
			m_calibrationDepth.loadFromFile(in);

			in.read((char*)&m_colorCompressionType, sizeof(COMPRESSION_TYPE_COLOR));
			in.read((char*)&m_depthCompressionType, sizeof(COMPRESSION_TYPE_DEPTH));
			in.read((char*)&m_colorWidth, sizeof(unsigned int));
			in.read((char*)&m_colorHeight, sizeof(unsigned int));
			in.read((char*)&m_depthWidth, sizeof(unsigned int));
			in.read((char*)&m_depthHeight, sizeof(unsigned int));
			in.read((char*)&m_depthShift, sizeof(unsigned int));

			UINT64 numFrames = 0;
			in.read((char*)&numFrames, sizeof(UINT64));
			m_frames.resize(numFrames);
			for (size_t i = 0; i < m_frames.size(); i++) {
				m_frames[i].loadFromFile(in);
			}

			UINT64 numIMUFrames = 0;
			in.read((char*)&numIMUFrames, sizeof(UINT64));
			m_IMUFrames.resize(numIMUFrames);
			for (size_t i = 0; i < m_IMUFrames.size(); i++) {
				m_IMUFrames[i].loadFromFile(in);
			}
		}

		class StringCounter {
		public:
			StringCounter(const std::string& base, const std::string fileEnding, unsigned int numCountDigits = 0, unsigned int initValue = 0) {
				m_base = base;
				if (fileEnding[0] != '.') {
					m_fileEnding = ".";
					m_fileEnding.append(fileEnding);
				}
				else {
					m_fileEnding = fileEnding;
				}
				m_numCountDigits = numCountDigits;
				m_initValue = initValue;
				resetCounter();
			}

			~StringCounter() {
			}

			std::string getNext() {
				std::string curr = getCurrent();
				m_current++;
				return curr;
			}

			std::string getCurrent() {
				std::stringstream ss;
				ss << m_base;
				for (unsigned int i = std::max(1u, (unsigned int)std::ceilf(std::log10f((float)m_current + 1))); i < m_numCountDigits; i++) ss << "0";
				ss << m_current;
				ss << m_fileEnding;
				return ss.str();
			}

			void resetCounter() {
				m_current = m_initValue;
			}
		private:
			std::string		m_base;
			std::string		m_fileEnding;
			unsigned int	m_numCountDigits;
			unsigned int	m_current;
			unsigned int	m_initValue;
		};

		
#ifdef 	_HAS_MLIB	//needs free image to write out data
		//! 7-scenes format
		void saveToImages(const std::string& outputFolder, const std::string& basename = "frame-") const {
			if (!ml::util::directoryExists(outputFolder)) ml::util::makeDirectory(outputFolder);

			{
				//write meta information
				const std::string& metaData = "info.txt";
				std::ofstream outMeta(outputFolder + "/" + metaData);

				outMeta << "m_versionNumber" << " = " << m_versionNumber << '\n';
				outMeta << "m_sensorName" << " = " << m_sensorName << '\n';
				outMeta << "m_colorWidth" << " = " << m_colorWidth << '\n';
				outMeta << "m_colorHeight" << " = " << m_colorHeight << '\n';
				outMeta << "m_depthWidth" << " = " << m_depthWidth << '\n';
				outMeta << "m_depthHeight" << " = " << m_depthHeight << '\n';
				outMeta << "m_depthShift" << " = " << m_depthShift << '\n';
				outMeta << "m_calibrationColorIntrinsic" << " = ";
				for (unsigned int i = 0; i < 16; i++) outMeta << m_calibrationColor.m_intrinsic[i] << " ";	outMeta << "\n";
				outMeta << "m_calibrationColorExtrinsic" << " = ";
				for (unsigned int i = 0; i < 16; i++) outMeta << m_calibrationColor.m_extrinsic[i] << " ";	outMeta << "\n";
				outMeta << "m_calibrationDepthIntrinsic" << " = ";
				for (unsigned int i = 0; i < 16; i++) outMeta << m_calibrationDepth.m_intrinsic[i] << " ";	outMeta << "\n";
				outMeta << "m_calibrationDepthExtrinsic" << " = ";
				for (unsigned int i = 0; i < 16; i++) outMeta << m_calibrationDepth.m_extrinsic[i] << " ";	outMeta << "\n";
				UINT64 numFrames = m_frames.size();
				outMeta << "m_frames.size" << " = " << numFrames << "\n";

				if (m_IMUFrames.size() > 0) MLIB_WARNING("warning sensor has imu frames; but writing is not implemented here");
			}

			if (m_frames.size() == 0) return;	//nothing to do
			std::string colorFormatEnding = "png";	//default is png			
			if (m_colorCompressionType == TYPE_PNG) colorFormatEnding = "png";
			else if (m_colorCompressionType == TYPE_JPEG) colorFormatEnding = "jpg";


			StringCounter scColor(outputFolder + "/" + basename, "color." + colorFormatEnding, 6);
			StringCounter scDepth(outputFolder + "/" + basename, "depth.png", 6);
			StringCounter scPose(outputFolder + "/" + basename, ".pose.txt", 6);

			for (size_t i = 0; i < m_frames.size(); i++) {
				std::string colorFile = scColor.getNext();
				std::string depthFile = scDepth.getNext();
				std::string poseFile = scPose.getNext();

				const RGBDFrame& f = m_frames[i];

				//color data
				if (m_colorCompressionType == TYPE_RAW)
				{
					RGBDFrame frameCompressed((vec3uc*)f.getColorCompressed(), m_colorWidth, m_colorHeight, NULL, 0, 0, mat4f::identity(), TYPE_PNG);
					FILE* file = fopen(colorFile.c_str(), "wb");
					if (!file) throw MLIB_EXCEPTION("cannot open file " + colorFile);
					fwrite(frameCompressed.getColorCompressed(), 1, frameCompressed.getColorSizeBytes(), file);
					fclose(file);
					frameCompressed.free();
				}
				else if (m_colorCompressionType == TYPE_PNG || m_colorCompressionType == TYPE_JPEG)
				{
					FILE* file = fopen(colorFile.c_str(), "wb");
					if (!file) throw MLIB_EXCEPTION("cannot open file " + colorFile);
					fwrite(f.getColorCompressed(), 1, f.getColorSizeBytes(), file);
					fclose(file);
				}
				else {
					throw MLIB_EXCEPTION("unknown format");
				}

				//depth data
				const bool writeDepthData = true;
				if (writeDepthData)
				{
					unsigned short* depth = decompressDepthAlloc(f);
					DepthImage16 image(m_depthWidth, m_depthHeight, depth);
					FreeImageWrapper::saveImage(depthFile, image);					
					std::free(depth);
				}

				savePoseFile(poseFile, f.m_cameraToWorld);
			}
		}

		//! 7-scenes format
		void loadFromImages(const std::string& sourceFolder, const std::string& basename = "frame-", const std::string& colorEnding = "png") 
		{
			if (colorEnding != "png" && colorEnding != "jpg") throw MLIB_EXCEPTION("invalid color format " + colorEnding);
			

			{
				//write meta information
				const std::string& metaData = "info.txt";
				std::ifstream inMeta(sourceFolder + "/" + metaData);

				std::string varName; std::string seperator;
				inMeta >> varName; inMeta >> seperator; inMeta >> m_versionNumber;
				inMeta >> varName; inMeta >> seperator; inMeta >> m_sensorName;
				inMeta >> varName; inMeta >> seperator; inMeta >> m_colorWidth;
				inMeta >> varName; inMeta >> seperator; inMeta >> m_colorHeight;
				inMeta >> varName; inMeta >> seperator; inMeta >> m_depthWidth;
				inMeta >> varName; inMeta >> seperator; inMeta >> m_depthHeight;
				inMeta >> varName; inMeta >> seperator; inMeta >> m_depthShift;

				inMeta >> varName; inMeta >> seperator;
				for (unsigned int i = 0; i < 16; i++) inMeta >> m_calibrationColor.m_intrinsic[i];
				inMeta >> varName; inMeta >> seperator;
				for (unsigned int i = 0; i < 16; i++) inMeta >> m_calibrationColor.m_extrinsic[i];
				inMeta >> varName; inMeta >> seperator;
				for (unsigned int i = 0; i < 16; i++) inMeta >> m_calibrationDepth.m_intrinsic[i];
				inMeta >> varName; inMeta >> seperator;
				for (unsigned int i = 0; i < 16; i++) inMeta >> m_calibrationDepth.m_extrinsic[i];
				UINT64 numFrames;
				inMeta >> varName; inMeta >> numFrames;
			}


			StringCounter scColor(sourceFolder + "/" + basename, "color." + colorEnding, 6);
			StringCounter scDepth(sourceFolder + "/" + basename, "depth.png", 6);
			StringCounter scPose(sourceFolder + "/" + basename, ".pose.txt", 6);

			for (unsigned int i = 0;; i++) {
				std::string colorFile = scColor.getNext();
				std::string depthFile = scDepth.getNext();
				std::string poseFile = scPose.getNext();

				if (!ml::util::fileExists(colorFile) || !ml::util::fileExists(depthFile) || !ml::util::fileExists(poseFile)) {
					std::cout << "DONE" << std::endl;
					break;
				}


				//ColorImageR8G8B8 colorImage;	ml::FreeImageWrapper::loadImage(colorFile, colorImage, false);
				//vec3uc* colorData = new vec3uc[m_colorWidth*m_colorHeight];
				//memcpy(colorData, colorImage.getPointer(), sizeof(vec3uc)*m_colorWidth*m_colorHeight);
				std::ifstream colorStream(colorFile, std::ios::binary | std::ios::ate);
				size_t colorSizeBytes = colorStream.tellg();
				colorStream.seekg(0, std::ios::beg);
				unsigned char* colorData = (unsigned char*)std::malloc(colorSizeBytes);
				colorStream.read((char*)colorData, colorSizeBytes);
				colorStream.close();


				DepthImage16 depthImage;			ml::FreeImageWrapper::loadImage(depthFile, depthImage, false);
				unsigned short*	depthData = new unsigned short[m_depthWidth*m_depthHeight];
				memcpy(depthData, depthImage.getDataPointer(), sizeof(unsigned short)*m_depthWidth*m_depthHeight);
				

				ml::SensorData::COMPRESSION_TYPE_COLOR compressionColor = ml::SensorData::COMPRESSION_TYPE_COLOR::TYPE_PNG;
				if (colorEnding == "png")		compressionColor = ml::SensorData::COMPRESSION_TYPE_COLOR::TYPE_PNG;
				else if (colorEnding == "jpg")	compressionColor = ml::SensorData::COMPRESSION_TYPE_COLOR::TYPE_JPEG;
				else throw MLIB_EXCEPTION("invalid color format " + compressionColor);

				//by default use TYPE_OCCI_USHORT (it's just the best)
				ml::SensorData::COMPRESSION_TYPE_DEPTH compressionDepth = ml::SensorData::COMPRESSION_TYPE_DEPTH::TYPE_OCCI_USHORT;

				RGBDFrame& f = addFrame(NULL, depthData);
				f.m_colorSizeBytes = colorSizeBytes;
				f.m_colorCompressed = colorData;

				//! camera trajectory (from base to current frame)
				mat4f m_frameToWorld;

				mat4f pose = loadPoseFile(poseFile);
				f.m_cameraToWorld = pose;

				////debug
				//{
				//	unsigned short* depth = m_frames.back().decompressDepth();
				//	ml::DepthImage16 depth16(m_depthWidth, m_depthHeight, depth);
				//	ml::FreeImageWrapper::saveImage(sourceFolder + "/" + "depth" + std::to_string(i) + ".png", ml::ColorImageR32G32B32A32(depth16));
				//}

				SAFE_DELETE_ARRAY(depthData);
			}
		}
#endif //_HAS_MLIB

		//! appends another SensorData object
		void append(const SensorData& second) {
			if (m_colorWidth != second.m_colorWidth &&
				m_colorHeight != second.m_colorHeight &&
				m_depthWidth != second.m_depthWidth &&
				m_depthHeight != second.m_depthHeight &&
				m_colorCompressionType != second.m_colorCompressionType &&
				m_depthCompressionType != second.m_depthCompressionType) {
				throw MLIB_EXCEPTION("sensor data incompatible");
			}

			for (size_t i = 0; i < second.m_frames.size(); i++) {
				m_frames.push_back(second.m_frames[i]);	//this is a bit of hack (relying on the fact that no copy-operator is implemented)
				RGBDFrame& f = m_frames.back();
				f.m_colorCompressed = (unsigned char*)std::malloc(f.m_colorSizeBytes);
				f.m_depthCompressed = (unsigned char*)std::malloc(f.m_depthSizeBytes);
				std::memcpy(f.m_colorCompressed, second.m_frames[i].m_colorCompressed, f.m_colorSizeBytes);
				std::memcpy(f.m_depthCompressed, second.m_frames[i].m_depthCompressed, f.m_depthSizeBytes);
			}
		}
		
		bool operator==(const SensorData& other) const {
			if (m_versionNumber != other.m_versionNumber) return false;
			if (m_sensorName != other.m_sensorName) return false;
			if (m_calibrationColor != other.m_calibrationColor) return false;
			if (m_calibrationDepth != other.m_calibrationDepth) return false;
			if (m_colorCompressionType != other.m_colorCompressionType) return false;
			if (m_depthCompressionType != other.m_depthCompressionType) return false;
			if (m_colorWidth != other.m_colorWidth) return false;
			if (m_colorHeight != other.m_colorHeight) return false;
			if (m_depthWidth != other.m_depthWidth) return false;
			if (m_depthHeight != other.m_depthHeight) return false;
			if (m_depthShift != other.m_depthShift) return false;
			if (m_frames.size() != other.m_frames.size()) return false;
			for (size_t i = 0; i < m_frames.size(); i++) {
				if (m_frames[i] != other.m_frames[i])	return false;
			}
			if (m_IMUFrames.size() != other.m_IMUFrames.size()) return false;
			for (size_t i = 0; i < m_IMUFrames.size(); i++) {
				if (m_IMUFrames[i] != other.m_IMUFrames[i]) return false;
			}
			return true;
		}

		bool operator!=(const SensorData& other) const {
			return !((*this) == other);
		}

		struct SensorNames {
			const std::string StructureSensor = "StructureSensor";
			const std::string Kinect_V1 = "Kinect.V1";
			const std::string Kinect_V2 = "Kinect.V2";
			const std::string PrimeSense = "PrimeSense Carmine";	//there are different versions 1.08 / 1.09
			const std::string Asus_Xtion = "Asus Xtion Pro";
		};
		static const SensorNames& getName() {
			static SensorNames sn;
			return sn;
		}

		///////////////////////////////
		//MEMBER VARIABLES START HERE//
		///////////////////////////////

		unsigned int	m_versionNumber;
		std::string		m_sensorName;

		CalibrationData m_calibrationColor;
		CalibrationData m_calibrationDepth;

		COMPRESSION_TYPE_COLOR m_colorCompressionType;
		COMPRESSION_TYPE_DEPTH m_depthCompressionType;

		unsigned int m_colorWidth;
		unsigned int m_colorHeight;
		unsigned int m_depthWidth;
		unsigned int m_depthHeight;
		float m_depthShift;	//conversion from float[m] to ushort (typically 1000.0f)

		std::vector<RGBDFrame> m_frames;
		std::vector<IMUFrame> m_IMUFrames;

		/////////////////////////////
		//MEMBER VARIABLES END HERE//
		/////////////////////////////


		static mat4f loadPoseFile(const std::string& filename) {
			std::ifstream file(filename);
			if (!file.is_open()) throw MLIB_EXCEPTION("file not found " + filename);

			mat4f m;
			file >>
				m._m00 >> m._m01 >> m._m02 >> m._m03 >>
				m._m10 >> m._m11 >> m._m12 >> m._m13 >>
				m._m20 >> m._m21 >> m._m22 >> m._m23 >>
				m._m30 >> m._m31 >> m._m32 >> m._m33;
			file.close();
			return m;
		}

		static void savePoseFile(const std::string& filename, const mat4f& m) {
			std::ofstream file(filename);
			file <<
				m._m00 << " " << m._m01 << " " << m._m02 << " " << m._m03 << "\n" <<
				m._m10 << " " << m._m11 << " " << m._m12 << " " << m._m13 << "\n" <<
				m._m20 << " " << m._m21 << " " << m._m22 << " " << m._m23 << "\n" <<
				m._m30 << " " << m._m31 << " " << m._m32 << " " << m._m33;
			file.close();
		}

	};
	
			class RGBDFrameCacheRead {
		public:
			struct FrameState {
				FrameState() {
					m_bIsReady = false;
					m_colorFrame = NULL;
					m_depthFrame = NULL;
				}
				~FrameState() {
					//NEEDS MANUAL FREE
				}
				void free() {
					if (m_colorFrame) std::free(m_colorFrame);
					if (m_depthFrame) std::free(m_depthFrame);
				}
				bool			m_bIsReady;
				vec3uc*			m_colorFrame;
				unsigned short*	m_depthFrame;
			};
			RGBDFrameCacheRead(SensorData* sensorData, unsigned int cacheSize) : m_decompThread() {
				m_sensorData = sensorData;
				m_cacheSize = cacheSize;
				m_bTerminateThread = false;
				m_nextFromSensorCache = 0;
				m_nextFromSensorData = 0;
				startDecompBackgroundThread();
			}

			~RGBDFrameCacheRead() {
				m_bTerminateThread = true;
				if (m_decompThread.joinable()) {
					m_decompThread.join();
				}

				for (auto& fs : m_data) {
					fs.free();
				}
			}

			FrameState getNext() {
				while (1) {
					if (m_nextFromSensorCache >= m_sensorData->m_frames.size()) {
						m_bTerminateThread = true;	// should be already true anyway
						break; //we're done
					}
					if (m_data.size() > 0 && m_data.front().m_bIsReady) {
						m_mutexList.lock();
						FrameState fs = m_data.front();
						m_data.pop_front();
						m_mutexList.unlock();
						m_nextFromSensorCache++;
						return fs;
					}
					else {
						Sleep(0);
					}
				}
				return FrameState();
			}

			FrameState getNextOurs() {
				while (1) {
					//if (m_nextFromSensorCache >= m_sensorData->m_frames.size()) {
					//	m_bTerminateThread = true;	// should be already true anyway
					//	break; //we're done
					//}
					if (m_data.size() > 0 && m_data.front().m_bIsReady) {
						m_mutexList.lock();
						FrameState fs = m_data.front();
						m_data.erase(m_data.begin());
						m_mutexList.unlock();
						m_nextFromSensorCache++;
						return fs;
					}
					else {
						Sleep(0);
					}
				}
				return FrameState();
			}

			bool getCheck() {
				return check;
			}

		private:
			void startDecompBackgroundThread() {
				m_decompThread = std::thread(decompFunc, this);
			}

			static void decompFunc(RGBDFrameCacheRead* cache) {
				while (1) {
					if (cache->m_bTerminateThread) break;
					if (cache->m_nextFromSensorData >= cache->m_sensorData->m_frames.size()) break;	//we're done

					if (cache->m_data.size() < cache->m_cacheSize) {	//need to fill the cache
						cache->m_mutexList.lock();
						cache->m_data.push_back(FrameState());
						cache->m_mutexList.unlock();

						SensorData* sensorData = cache->m_sensorData;
						SensorData::RGBDFrame& frame = sensorData->m_frames[cache->m_nextFromSensorData];
						FrameState& fs = cache->m_data.back();

						//std::cout << "decompressing frame " << cache->m_nextFromSensorData << std::endl;
						fs.m_colorFrame = sensorData->decompressColorAlloc(frame);
						fs.m_depthFrame = sensorData->decompressDepthAlloc(frame);
						fs.m_bIsReady = true;
						cache->m_nextFromSensorData++;
					}
				}
			}

			SensorData* m_sensorData;
			unsigned int m_cacheSize;

			std::list<FrameState> m_data;
			std::thread m_decompThread;
			std::mutex m_mutexList;
			std::atomic<bool> m_bTerminateThread;

			bool check;

			unsigned int m_nextFromSensorData;
			unsigned int m_nextFromSensorCache;
		};

		class RGBDFrameCacheWrite {
		private:
			struct FrameState {
				FrameState() {
					m_bIsReady = false;
					m_colorFrame = NULL;
					m_depthFrame = NULL;
				}
				~FrameState() {
					//NEEDS MANUAL FREE
				}
				void free() {
					if (m_colorFrame) std::free(m_colorFrame);
					if (m_depthFrame) std::free(m_depthFrame);
				}
				bool			m_bIsReady;
				vec3uc*			m_colorFrame;
				unsigned short*	m_depthFrame;
			};
		public:
			RGBDFrameCacheWrite(SensorData* sensorData, unsigned int cacheSize) : m_compThread() {
				m_sensorData = sensorData;
				m_cacheSize = cacheSize;
				m_bTerminateThread = false;
				startCompBackgroundThread();
			}

			~RGBDFrameCacheWrite() {
				m_bTerminateThread = true;
				if (m_compThread.joinable()) {
					m_compThread.join();
				}

				for (auto& fs : m_data) {
					fs.free();
				}
			}

			//! appends the data to the cache for process AND frees the memory
			void writeNextAndFree(vec3uc* color, unsigned short* depth) {
				FrameState fs;
				fs.m_colorFrame = color;
				fs.m_depthFrame = depth;
				while (m_data.size() >= m_cacheSize) {
					Sleep(0);	//wait until we have space in our cache
				}
				m_mutexList.lock();
				m_data.push_back(fs);
				m_mutexList.unlock();
			}

		private:
			void startCompBackgroundThread() {
				m_compThread = std::thread(compFunc, this);
			}

			static void compFunc(RGBDFrameCacheWrite* cache) {
				while (1) {
					if (cache->m_bTerminateThread && cache->m_data.size() == 0) break;	//if terminated AND compression is done
					if (cache->m_data.size() > 0) {
						cache->m_mutexList.lock();
						FrameState fs = cache->m_data.front();
						cache->m_data.pop_front();
						cache->m_mutexList.unlock();

						//cache->m_sensorData->m_frames.push_back(SensorData::RGBDFrame(
						//	fs.m_colorFrame,
						//	cache->m_sensorData->m_colorWidth,
						//	cache->m_sensorData->m_colorHeight,
						//	fs.m_depthFrame,
						//	cache->m_sensorData->m_depthWidth,
						//	cache->m_sensorData->m_depthHeight));
						cache->m_sensorData->addFrame(fs.m_colorFrame, fs.m_depthFrame);
						fs.free();

					}
				}
			}

			SensorData* m_sensorData;
			unsigned int m_cacheSize;

			std::list<FrameState> m_data;
			std::thread m_compThread;
			std::mutex m_mutexList;
			std::atomic<bool> m_bTerminateThread;
		};

#ifndef VAR_STR_LINE
#define VAR_STR_LINE(x) '\t' << #x << '=' << x << '\n'
#endif
	inline std::ostream& operator<<(std::ostream& s, const SensorData& sensorData) {
		s << "CalibratedSensorData:\n";
		s << VAR_STR_LINE(sensorData.m_versionNumber);
		s << VAR_STR_LINE(sensorData.m_sensorName);
		s << VAR_STR_LINE(sensorData.m_colorWidth);
		s << VAR_STR_LINE(sensorData.m_colorHeight);
		s << VAR_STR_LINE(sensorData.m_depthWidth);
		s << VAR_STR_LINE(sensorData.m_depthHeight);
		//s << VAR_STR_LINE(sensorData.m_CalibrationDepth);
		//s << VAR_STR_LINE(sensorData.m_CalibrationColor);
		s << VAR_STR_LINE(sensorData.m_frames.size());
		s << VAR_STR_LINE(sensorData.m_IMUFrames.size());
		return s;
	}

}	//namespace ml

#endif //_SENSOR_FILE_H_

