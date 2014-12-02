#pragma once

/************************************************************************/
/* Parent class of all sensors (sensors MUST inherit from that class)   */
/************************************************************************/

#include "stdafx.h"

#include <cassert>

struct Intrinsics {
	union
	{
		struct {
			float fx;	
			float fy;
			float mx;
			float my;
			float k1;
			float k2;
			float k3;
			float p1;
			float p2;
		};
		float coeff[9];
	};

	//! returns an intrinsic matrix (warning: doesn't account for radial distortion)
	mat4f converToMatrix() {
		return mat4f(
			fx  , 0.0f, mx  , 0.0f,
			0.0f, fy  , my  , 0.0f,
			0.0f, 0.0f, 1.0f, 0.0f,
			0.0f, 0.0f, 0.0f, 1.0f);
	}

	void print() {
		std::cout << "intrinsics:" << std::endl <<
			"\tfx " << fx << std::endl <<
			"\tfy " << fy << std::endl <<
			"\tmx " << mx << std::endl <<
			"\tmy " << my << std::endl;
	}
};

class DepthSensor
{
	public:

		// Default constructor
		DepthSensor();

		//! Init
		void init(unsigned int depthWidth, unsigned int depthHeight, unsigned int colorWidth, unsigned int colorHeight);

		//! Destructor; releases allocated ressources
		virtual ~DepthSensor();

		//! Connected to Depth Sensor
		virtual HRESULT createFirstConnected() = 0;

		//! Processes the depth data 
		virtual HRESULT processDepth() = 0;

		//! Processes the color data
		virtual HRESULT processColor() = 0;

		//! Toggles the near-mode if available (only some cameras can do that)
		virtual HRESULT toggleNearMode() { return S_OK; }

		//! Toggle enable auto white balance (only some cameras can do that)
		virtual HRESULT toggleAutoWhiteBalance() { return S_OK; }

		//! Get the intrinsics of the sensor
		const Intrinsics& getIntrinsics() const {
			return m_intrinsics;
		}

		//const Intrinsics& getIntrinsicsInv() const {
		//	return m_intrinsicsInv;
		//}

		void initializeIntrinsics(
			float fovX, float fovY, float centerX, float centerY, 
			float k1 = 0.0f, float k2 = 0.0f, float k3 = 0.0f, float p1 = 0.0f, float p2 = 0.0f);

		//! returns the depth value at ux / uy, does consider clamps
		float getDepth(unsigned int ux, unsigned int uy) const {

			const unsigned int minDepth = 0;
			const unsigned int maxDepth = UINT_MAX;

			if (m_depthD16) {
				const unsigned int pixel = m_depthD16[ux + uy * getDepthWidth()];
				if (pixel >= minDepth && pixel <= maxDepth){
					return pixel*0.001f;	//conversion to meters
				}
			}
			return -FLT_MAX;
		}

		vec3f getNormal(unsigned int x, unsigned int y) const {
			vec3f ret(-FLT_MAX,-FLT_MAX,-FLT_MAX);
			if (x > 0 && y > 0 && x < getDepthWidth() - 1 && y < getDepthWidth() - 1) {
				vec3f cc = depthToSkeleton(x,y);
				vec3f pc = depthToSkeleton(x+1,y+0);
				vec3f cp = depthToSkeleton(x+0,y+1);
				vec3f mc = depthToSkeleton(x-1,y+0);
				vec3f cm = depthToSkeleton(x+0,y-1);

				if (cc.x != -FLT_MAX && pc.x != -FLT_MAX && cp.x != -FLT_MAX && mc.x != -FLT_MAX && cm.x != -FLT_MAX)
				{
					vec3f n = (pc - mc) ^ (cp - cm);
					float l = n.length();
					if (l > 0.0f) {
						ret = n/l;
					}
				}
			}
			return ret;
		}

		// Functions below are untested but might be useful
		vec3f depthToSkeleton(unsigned int ux, unsigned int uy) const {
			return depthToSkeleton(ux, uy, getDepth(ux,uy));
		}
		
		vec3f depthToSkeleton(unsigned int ux, unsigned int uy, float depth) const {
			//vec4f camera = m_intrinsicsInv*vec4f((float)ux*depth, (float)uy*depth, 0.0f, depth);
			//return vec3f(camera.x, camera.y, camera.w);
			float x = ((float)ux-m_intrinsics.mx) / m_intrinsics.fx;
			float y = (m_intrinsics.my-(float)uy) / m_intrinsics.fy;
			return vec3f(depth*x, depth*y, depth);
		}

		//! gets the pointer to depth array
		USHORT *getDepthD16() { return m_depthD16; }

		//! gets the pointer to color array
		BYTE *getColorRGBX() { return m_colorRGBX; }

		unsigned int getColorWidth()  const { return m_colorWidth; }
		unsigned int getColorHeight() const { return m_colorHeight; }
		unsigned int getDepthWidth()  const { return m_depthWidth; }
		unsigned int getDepthHeight() const { return m_depthHeight; }
		unsigned int getColorBytesPerPixel() const { return cBytesPerPixel; }

		USHORT getMaxDepthUSHORT() const {
			USHORT maxDepth = 0;
			if (m_depthD16) {
				for (unsigned int i = 0; i < getDepthWidth()*getDepthHeight(); i++) { 
					if (maxDepth < m_depthD16[i])	maxDepth = m_depthD16[i];
				}
			}
			return maxDepth;
		}
		USHORT getMinDepthUSHORT() const {
			USHORT minDepth = USHRT_MAX;
			if (m_depthD16) {
				for (unsigned int i = 0; i < getDepthWidth()*getDepthHeight(); i++) {
					if (m_depthD16[i] > 0 && minDepth > m_depthD16[i])	minDepth = m_depthD16[i];
				}
			}
			return minDepth;
		}

		//! writes the raw depth data to a file
		bool writeDepthDataToFile(const std::string& filename) const {
			return writeDepthDataToFile(filename, m_depthD16, m_depthHeight, m_depthWidth);

		}

		//! writes the raw color data to a file
		bool writeColorDataToFile(const std::string& filename) const {
			return writeColorDataToFile(filename, m_colorRGBX, m_colorHeight, m_colorWidth);
		}

		//! saves the current frame's input data as a point cloud
		void savePointCloud(const std::string& filename, const mat4f& transform = mat4f::identity()) const;


		//! resets the cache of the sensor (mainly setting the frame counter back to zero, and frees potentially cached memory)
		void reset() {
			for (auto iter = m_RecordedColor.begin(); iter != m_RecordedColor.end(); iter++)
				SAFE_DELETE_ARRAY(*iter);
			for (auto iter = m_RecordedDepth.begin(); iter != m_RecordedDepth.end(); iter++) {
				SAFE_DELETE_ARRAY(*iter);
			}
			m_RecordedDepth.clear();
			m_RecordedColor.clear();

			m_RecordedTrajectory.clear();

			m_accumulatedPoints.clear();
		}

		void recordPointCloud(const mat4f& transform = mat4f::identity()) {
			m_accumulatedPoints.push_back(PointCloudf());
			computePointCurrentPointCloud(m_accumulatedPoints.back(), transform);
		}

		void saveRecordedPointCloud(const std::string& filename) {

			PointCloudf pc;
			
			for (const auto& p : m_accumulatedPoints ) {
				pc.m_points.insert(pc.m_points.end(), p.m_points.begin(), p.m_points.end());
				pc.m_colors.insert(pc.m_colors.end(), p.m_colors.begin(), p.m_colors.end());
				pc.m_normals.insert(pc.m_normals.end(), p.m_normals.begin(), p.m_normals.end());
			}

			PointCloudIOf::saveToFile(filename,pc);
			//if (colors.size() > 0) {
			//	assert(points.size() == colors.size());
			//	PointCloudIOf::saveToFile(filename, &points, NULL, &colors);
			//} else {
			//	PointCloudIOf::saveToFile(filename, &points, NULL, NULL);
			//}

			m_accumulatedPoints.clear();

		}

		//! records the current frame and allocates memory accordingly
		void recordFrame() {
			if (m_depthD16) {
				m_RecordedDepth.push_back(m_depthD16);
				m_depthD16 = new USHORT[m_depthWidth*m_depthHeight];
			}
			if (m_colorRGBX) {
				m_RecordedColor.push_back(m_colorRGBX);
				m_colorRGBX = new BYTE[getColorBytesPerPixel()*m_colorWidth*m_colorHeight];
			}
		}
		void recordTrajectory(const mat4f& transformation) {
			m_RecordedTrajectory.push_back(transformation);
		}

		//TODO continue here
		void saveRecordedFramesToFile(const std::string& filename) {
			CalibratedSensorData cs;
			cs.m_DepthImageWidth = m_depthWidth;
			cs.m_DepthImageHeight = m_depthHeight;
			cs.m_ColorImageWidth = m_colorWidth;
			cs.m_ColorImageHeight = m_colorHeight;
			cs.m_DepthNumFrames = (unsigned int)m_RecordedDepth.size();
			cs.m_ColorNumFrames = (unsigned int)m_RecordedColor.size();
			cs.m_CalibrationDepth.m_Intrinsic = m_intrinsicsOriginal.converToMatrix();
			cs.m_CalibrationDepth.m_Extrinsic.setIdentity();
			cs.m_CalibrationDepth.m_IntrinsicInverse = cs.m_CalibrationDepth.m_Intrinsic.getInverse();
			cs.m_CalibrationDepth.m_ExtrinsicInverse = cs.m_CalibrationDepth.m_Extrinsic.getInverse();
			cs.m_CalibrationColor = cs.m_CalibrationDepth;

			cs.m_DepthImages.resize(cs.m_DepthNumFrames);
			cs.m_ColorImages.resize(cs.m_ColorNumFrames);
			unsigned int cFrame = 0;
			for (auto a : m_RecordedColor) {
				cs.m_ColorImages[cFrame] = new vec4uc[cs.m_ColorImageWidth*cs.m_ColorImageHeight];
				memcpy(cs.m_ColorImages[cFrame], a, sizeof(vec4uc)*cs.m_ColorImageWidth*cs.m_ColorImageHeight);
				SAFE_DELETE_ARRAY(a);
				cFrame++;
			}
			unsigned int dFrame = 0;
			for (auto a : m_RecordedDepth) {
				cs.m_DepthImages[dFrame] = new float[cs.m_DepthImageWidth*cs.m_DepthImageHeight];
				for (unsigned int i = 0; i < cs.m_DepthImageWidth*cs.m_DepthImageHeight; i++) {
					cs.m_DepthImages[dFrame][i] = (float)a[i] / 1000.0f;
				}
				SAFE_DELETE_ARRAY(a);
				dFrame++;
			}

			cs.m_trajectory = m_RecordedTrajectory;

			std::string dir = util::directoryFromPath(filename);
			if (!util::directoryExists(dir)) util::makeDirectory(dir);

			std::cout << cs << std::endl;
			std::cout << "dumping recorded frames... ";
			BinaryDataStreamFile outStream(filename, true);
			//BinaryDataStreamZLibFile outStream(filename, true);
			outStream << cs;
			std::cout << "done" << std::endl;

			m_RecordedDepth.clear();
			m_RecordedColor.clear();
			m_RecordedTrajectory.clear();
		}

		//! returns the current rigid transform (if available; designed for binary dump sensor);
		virtual mat4f getRigidTransform() const {
			return mat4f::identity();
		}
	protected:
		//! depth and color data filled by respective child classes
		USHORT* m_depthD16;
		BYTE*   m_colorRGBX;

	private:

		std::list<PointCloudf> m_accumulatedPoints;


		void computePointCurrentPointCloud(PointCloudf& pc, const mat4f& transform = mat4f::identity()) const {

			if (!(getColorWidth() == getDepthWidth() && getColorHeight() == getDepthHeight()))	throw MLIB_EXCEPTION("invalid dimensions");

			for (unsigned int i = 0; i < getDepthWidth()*getDepthHeight(); i++) {
				unsigned int x = i % getDepthWidth();
				unsigned int y = i / getDepthWidth();
				vec3f p = depthToSkeleton(x,y);
				if (p.x != -FLT_MAX && p.x != 0.0f)	{
					vec3f n = getNormal(x,y);
					if (n.x != -FLT_MAX) {
						pc.m_points.push_back(p);
						pc.m_normals.push_back(n);
						vec4ui c = vec4ui(m_colorRGBX[4*i+0],m_colorRGBX[4*i+1],m_colorRGBX[4*i+2],m_colorRGBX[4*i+3]);
						pc.m_colors.push_back(vec4f(c.z/255.0f, c.y/255.0f, c.x/255.0f, 1.0f));	//there's a swap... dunno why really
					}
				}
			}

			//if (getColorWidth() == getDepthWidth() && getColorHeight() == getDepthHeight()) {
			//	for (unsigned int i = 0; i < getColorWidth()*getColorHeight(); i++) {
			//		vec3f p = depthToSkeleton(i % getDepthWidth(), i / getDepthWidth());
			//		if (p.x != -FLT_MAX && p.x != 0.0f)	{
			//			vec4ui c = vec4ui(m_colorRGBX[4*i+0],m_colorRGBX[4*i+1],m_colorRGBX[4*i+2],m_colorRGBX[4*i+3]);
			//			colors.push_back(vec3f(c.z/255.0f, c.y/255.0f, c.x/255.0f));	//there's a swap... dunno why really
			//		}
			//	}
			//}  
			for (auto& p : pc.m_points) {
				p = transform * p;
			}
			mat4f invTranspose = transform.getInverse().getTranspose();
			for (auto& n : pc.m_normals) {
				n = invTranspose * n;
				n.normalize();
			}
		}


		bool writeDepthDataToFile( const std::string& filename, USHORT* depthData, unsigned int height, unsigned int width) const {
			if (!depthData)	return false;
			BaseImage<unsigned short>::saveBinaryMImage(filename, depthData, height, width);
			return true;
		}

		bool writeColorDataToFile( const std::string& filename,  BYTE* colorData, unsigned int height, unsigned int width) const {
			if (!colorData)	return false;
			BaseImage<vec4uc>::saveBinaryMImage(filename, m_colorRGBX, m_depthHeight, m_depthWidth);
			return true;
		}

		//! intrinsic matrix and its inverse of the sensor
		//mat4f m_intrinsics;
		//mat4f m_intrinsicsInv;
		Intrinsics m_intrinsics;
		Intrinsics m_intrinsicsOriginal;

		//! width and height of depth input data
		LONG    m_depthWidth;
		LONG    m_depthHeight;

		//! width and height of color input data
		LONG    m_colorWidth;
		LONG	m_colorHeight;

		//! bytes in each color pixel
		static const int cBytesPerPixel = 4;

		std::list<USHORT*>	m_RecordedDepth;
		std::list<BYTE*>	m_RecordedColor;
		std::vector<mat4f>	m_RecordedTrajectory;
};
