

#ifndef _SENSOR_FILE_ZHOU_H_
#define _SENSOR_FILE_ZHOU_H_

#include "mlib.h"
//#include "opencv2\core.hpp"
#include "opencv2\opencv.hpp"
#include "sensorData\sensorData.h"

void read_directory(const std::string& name, std::vector<std::string>& v);

namespace ml {

	class SensorDataZhou : public SensorData {
	public:

		//the input is a path!
		//load Zhou's dataset.
		//1234567 - 123456789ABC
		//0004443 - 000148226209 : dept
		//0000001 - 000000000000 : image
		//frame num - timestamp

		void loadFromFile(const std::string& filename, int nFrame) {

			std::cout << filename + "\\camerainfo.txt" << std::endl;

			//camera info

			//	filename
			std::ifstream in(filename + "\\camerainfo.txt", std::ios::in);

			if (!in.is_open()) {
				throw MLIB_EXCEPTION("could not open file " + filename);
			}

			float rgbwidth, rgbheight;
			float dwidth, dheight;
			float rgbfx, rgbfy;
			float dfx, dfy;
			float rgbcx, rgbcy, dcx, dcy;
			float dshift;
			in >> rgbwidth >> rgbheight;
			in >> rgbfx >> rgbfy >> rgbcx >> rgbcy;
			in >> dwidth >> dheight;
			in >> dfx >> dfy >> dcx >> dcy;
			in >> dshift;
			//debugging

			//rgb
			printf("rgb w: %f h: %f\n", rgbwidth, rgbheight);
			printf("rgb fx: %f fy: %f\n", rgbfx, rgbfy);
			printf("rgb cx: %f cy: %f\n", rgbcx, rgbcy);

			//depth
			printf("depth w: %f h: %f\n", dwidth, dheight);
			printf("depth fx: %f fy: %f\n", dfx, dfy);
			printf("depth cx: %f cy: %f\n", dcx, dcy);
			printf("depth shift: %f\n", dshift);

			//set intrinsic matrix
			m_calibrationColor.setMatrices(m_calibrationColor.makeIntrinsicMatrix(rgbfx, rgbfy, rgbcx, rgbcy));
			m_calibrationDepth.setMatrices(m_calibrationDepth.makeIntrinsicMatrix(dfx, dfy, dcx, dcy));

			m_colorCompressionType = ml::SensorData::COMPRESSION_TYPE_COLOR::TYPE_RAW;
			m_depthCompressionType = ml::SensorData::COMPRESSION_TYPE_DEPTH::TYPE_RAW_USHORT;

			m_colorWidth = rgbwidth;
			m_colorHeight = rgbheight;
			m_depthWidth = dwidth;
			m_depthHeight = dheight;
			m_depthShift = dshift;

			std::cout << filename + "\\depth_vga.match" << std::endl;

			//	filename
			in = std::ifstream(filename + "\\depth_vga.match", std::ios::in);

			if (!in.is_open()) {
				throw MLIB_EXCEPTION("could not open file " + filename);
			}

			//read rgbd data
			std::vector<uint8_t> color(m_colorWidth * m_colorHeight * 3);
			std::vector<unsigned short> depth(m_depthWidth * m_depthHeight);

			int i;
			std::string dfilename, vgafilename;
			//char timestamp[30], frame[30];
			int timestamp, frame;
			RGBDFrame* rgbdframe;

			m_frames.resize(nFrame);
			int framecnt = 0;

			for (i = 0; !in.eof() && framecnt < nFrame; i++) {

				rgbdframe = &m_frames[framecnt];

				//read match
				in >> dfilename >> vgafilename;
				std::cout << dfilename << " " << vgafilename << std::endl;

				framecnt++;

				//read rgb image
				std::cout << filename + "\\" + vgafilename << " ";

				sscanf(vgafilename.data(), "vga\\%07d-000%09d.jpg", &frame, &timestamp);
				printf("%d %d\n", frame, timestamp);

				cv::Mat bgr = cv::imread(filename + "\\" + vgafilename, cv::IMREAD_COLOR), bgra;
				if (bgr.empty())
					continue;

				size_t nbytes = bgr.cols * bgr.rows * 3;
				if (nbytes != color.size())
					continue;

				cv::cvtColor(bgr, bgr, cv::COLOR_BGR2RGB);
				memcpy_s(color.data(), nbytes, bgr.data, nbytes);

				//set rgb timestamp and data
				rgbdframe->setTimeStampColor(timestamp);
				rgbdframe->m_colorSizeBytes = m_colorWidth * m_colorHeight * 3;

				rgbdframe->compressColor((vec3uc*)color.data(), m_colorWidth, m_colorHeight, TYPE_RAW);

				//read depth image
				std::cout << filename + "\\" + dfilename << " ";

				sscanf(dfilename.data(), "depth\\%07d-000%09d.png", &frame, &timestamp);
				printf("%d %d\n", frame, timestamp);

				cv::Mat depthmat = cv::imread(filename + "\\" + dfilename, cv::IMREAD_ANYDEPTH);
				if (depthmat.empty())
					continue;
				nbytes = depthmat.cols * depthmat.rows * 2;
				if (nbytes != depth.size() * 2)
					continue;

				depthmat.convertTo(depthmat, CV_16UC1);
				memcpy_s(depth.data(), nbytes, depthmat.data, nbytes);

				//set depth time stemp 
				rgbdframe->setTimeStampDepth(timestamp);
				rgbdframe->m_depthSizeBytes = m_depthWidth * m_depthHeight * 4;
				rgbdframe->compressDepth((unsigned short*)depth.data(), m_depthWidth, m_depthHeight, TYPE_RAW_USHORT);

			}
		}

		void loadFromFileObject(const std::string& filename, int nFrame, int startFrame) {

			std::cout << filename + "\\camerainfo.txt" << std::endl;

			//camera info

			//	filename
			std::ifstream in(filename + "\\camerainfo.txt", std::ios::in);

			if (!in.is_open()) {
				throw MLIB_EXCEPTION("could not open file " + filename);
			}

			float rgbwidth, rgbheight;
			float dwidth, dheight;
			float rgbfx, rgbfy;
			float dfx, dfy;
			float rgbcx, rgbcy, dcx, dcy;
			float dshift;
			in >> rgbwidth >> rgbheight;
			in >> rgbfx >> rgbfy >> rgbcx >> rgbcy;
			in >> dwidth >> dheight;
			in >> dfx >> dfy >> dcx >> dcy;
			in >> dshift;
			//debugging

			//rgb
			printf("rgb w: %f h: %f\n", rgbwidth, rgbheight);
			printf("rgb fx: %f fy: %f\n", rgbfx, rgbfy);
			printf("rgb cx: %f cy: %f\n", rgbcx, rgbcy);

			//depth
			printf("depth w: %f h: %f\n", dwidth, dheight);
			printf("depth fx: %f fy: %f\n", dfx, dfy);
			printf("depth cx: %f cy: %f\n", dcx, dcy);
			printf("depth shift: %f\n", dshift);

			//set intrinsic matrix
			m_calibrationColor.setMatrices(m_calibrationColor.makeIntrinsicMatrix(rgbfx, rgbfy, rgbcx, rgbcy));
			m_calibrationDepth.setMatrices(m_calibrationDepth.makeIntrinsicMatrix(dfx, dfy, dcx, dcy));

			m_colorCompressionType = ml::SensorData::COMPRESSION_TYPE_COLOR::TYPE_RAW;
			m_depthCompressionType = ml::SensorData::COMPRESSION_TYPE_DEPTH::TYPE_RAW_USHORT;

			m_colorWidth = rgbwidth;
			m_colorHeight = rgbheight;
			m_depthWidth = dwidth;
			m_depthHeight = dheight;
			m_depthShift = dshift;

			std::cout << filename + "\\depth_vga.match" << std::endl;

			//	filename
			in = std::ifstream(filename + "\\depth_vga.match", std::ios::in);

			if (!in.is_open()) {
				throw MLIB_EXCEPTION("could not open file " + filename);
			}

			//read rgbd data
			std::vector<uint8_t> color(m_colorWidth * m_colorHeight * 3);
			std::vector<unsigned short> depth(m_depthWidth * m_depthHeight);

			int i;
			std::string dfilename, vgafilename;
			//char timestamp[30], frame[30];
			int timestamp, frame;
			RGBDFrame* rgbdframe;

			m_frames.resize(nFrame);
			int framecnt = 0;

			for (i = 0; !in.eof() && framecnt < nFrame; i++) {

				//rgbdframe = new RGBDFrame;
				rgbdframe = &m_frames[framecnt];

				//read match
				in >> dfilename >> vgafilename;
				std::cout << dfilename << " " << vgafilename << std::endl;

				if (i < startFrame)
					continue;
				framecnt++;
				//read rgb image
				std::cout << filename + "/" + vgafilename << " ";

				//sscanf(vgafilename.data(), "vga/%07d-000%09d.jpg", &frame, &timestamp);
				printf("%d %d\n", frame, timestamp);

				cv::Mat bgr = cv::imread(filename + "/" + vgafilename, cv::IMREAD_COLOR);
				if (bgr.empty())
					continue;

				printf("%d %d\n", bgr.rows, bgr.cols);

				size_t nbytes = bgr.cols * bgr.rows * 3;
				if (nbytes != color.size())
					continue;

				cv::cvtColor(bgr, bgr, cv::COLOR_BGR2RGB);
				memcpy_s(color.data(), nbytes, bgr.data, nbytes);

				//set rgb timestamp and data
				rgbdframe->setTimeStampColor(timestamp);
				rgbdframe->m_colorSizeBytes = m_colorWidth * m_colorHeight * 3;

				rgbdframe->compressColor((vec3uc*)color.data(), m_colorWidth, m_colorHeight, TYPE_RAW);


				//read depth image
				std::cout << filename + "/" + dfilename << " ";

				sscanf(dfilename.data(), "depth/%07d-000%09d.png", &frame, &timestamp);
				printf("%d %d\n", frame, timestamp);

				cv::Mat depthmat = cv::imread(filename + "\\" + dfilename, cv::IMREAD_ANYDEPTH);
				if (depthmat.empty())
					continue;
				nbytes = depthmat.cols * depthmat.rows * 2;
				if (nbytes != depth.size() * 2)
					continue;

				depthmat.convertTo(depthmat, CV_16UC1);
				memcpy_s(depth.data(), nbytes, depthmat.data, nbytes);

				//set depth time stemp 
				rgbdframe->setTimeStampDepth(timestamp);
				rgbdframe->m_depthSizeBytes = m_depthWidth * m_depthHeight * 4;
				rgbdframe->compressDepth((unsigned short*)depth.data(), m_depthWidth, m_depthHeight, TYPE_RAW_USHORT);

			}
		}

	};
};

#endif //_SENSOR_FILE_ZHOU_H_
