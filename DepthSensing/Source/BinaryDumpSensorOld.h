#pragma once

#include "DepthSensor.h"

#include <NuiApi.h>
#include <NuiSkeleton.h>

#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <string>

class BinaryDumpSensorOld : public DepthSensor
{
public:

	//! Constructor; allocates CPU memory and creates handles
	BinaryDumpSensorOld::BinaryDumpSensorOld()
	{
		m_numFrames = 0;

		//unsigned int depthWidth = 640;
		//unsigned int depthHeight = 480;

		//unsigned int colorWidth = 640;
		//unsigned int colorHeight = 480;

		//DepthSensor::init(depthWidth, depthHeight, colorWidth, colorHeight);

		//initializeIntrinsics(2.0f*NUI_CAMERA_SKELETON_TO_DEPTH_IMAGE_MULTIPLIER_320x240, 2.0f*NUI_CAMERA_SKELETON_TO_DEPTH_IMAGE_MULTIPLIER_320x240, 320.0f, 240.0f);

		//ALL PARAMETERS READ FROM DUMP FILE
	}

	//! Destructor; releases allocated ressources
	~BinaryDumpSensorOld()
	{
		m_infile.close();
	}

	static void writeBinaryDump(
		const std::string &filename, unsigned int imageWidth, unsigned int imageHeight, unsigned int numCameras, const mat4f* intrinsics, const mat4f* extrinsics, std::list<USHORT*> depthFrames, std::list<BYTE*> colorFrames) 
	{
		std::ofstream out(filename.c_str(), std::ofstream::binary);
		
		assert(depthFrames.size() == colorFrames.size());
		unsigned int numFrames = (unsigned int)depthFrames.size();
		assert(numFrames > 0);

		writeBinaryDumpHeader(out,imageWidth, imageHeight, numFrames, numCameras, intrinsics, extrinsics);
		
		std::list<USHORT*>::const_iterator depthIter = depthFrames.begin();
		std::list<BYTE*>::const_iterator colIter = colorFrames.begin();
		for (unsigned int i = 0; i < numFrames; i++) {
			out.write((char*)*colIter, sizeof(BYTE)*imageWidth*imageHeight*4);
			out.write((char*)*depthIter, sizeof(USHORT)*imageWidth*imageHeight);
			depthIter++;
			colIter++;
		}
		out.close();
	}

#define BINARY_DUMP_HEADER_VERSION 1
	static void writeBinaryDumpHeader(std::ofstream &out, unsigned int imageWidth, unsigned int imageHeight, unsigned int numFrames, unsigned int numCameras, const mat4f* intrinsics, const mat4f* extrinsics) {
		unsigned int headerVersion = BINARY_DUMP_HEADER_VERSION;
		out.write((char*)&headerVersion, sizeof(unsigned int));
		out.write((char*)&imageWidth, sizeof(unsigned int));
		out.write((char*)&imageHeight, sizeof(unsigned int));
		out.write((char*)&numFrames, sizeof(unsigned int));
		out.write((char*)&numCameras, sizeof(unsigned int));
		for (unsigned int i = 0; i < numCameras; i++) {
			out.write((char*)intrinsics[i].getRawData(), sizeof(float)*16);
			out.write((char*)extrinsics[i].getRawData(), sizeof(float)*16);
		}
	}
	static void readBinaryDumpHeader(std::ifstream &in, unsigned int& imageWidth, unsigned int& imageHeight, unsigned int& numFrames, unsigned int &numCameras, std::vector<mat4f>& intrinsics, std::vector<mat4f>& extrinsics, bool printDetails = true) {
		intrinsics.clear();
		extrinsics.clear();

		unsigned int headerVersion;
		in.read((char*)&headerVersion, sizeof(unsigned int));
		if (headerVersion != BINARY_DUMP_HEADER_VERSION) {
			std::cerr << "ERRROR header version invalid!" << std::endl;
			while(1);
		}
		in.read((char*)&imageWidth, sizeof(unsigned int));
		in.read((char*)&imageHeight, sizeof(unsigned int));
		in.read((char*)&numFrames, sizeof(unsigned int));
		in.read((char*)&numCameras, sizeof(unsigned int));
		for (unsigned int i = 0; i < numCameras; i++) {
			float intr[16], extr[16];
			in.read((char*)intr, sizeof(float)*16);
			in.read((char*)extr, sizeof(float)*16);
			intrinsics.push_back(mat4f(intr));
			extrinsics.push_back(mat4f(extr));
		}

		if (printDetails) {
			std::cout << "Binary Dump Header (Version " << headerVersion << "):" << std::endl;
			std::cout << "\tImage size: \t" << imageWidth << " / " << imageHeight << std::endl;
			std::cout << "\tNum Frames: \t" << numFrames << std::endl;
			std::cout << "\tNum Cameras: \t" << numCameras << std::endl;
		}
	}

	HRESULT BinaryDumpSensorOld::createFirstConnected()
	{
		// Open File
		//std::string filename = ".\\sequence_bookshop.bdump";
		std::string filename = "sequence.bdump";
		m_infile = std::ifstream(filename.c_str(), std::ofstream::binary);

		unsigned int imageWidth;
		unsigned int imageHeight;
		unsigned int numFrames;
		unsigned int numCams;
		std::vector<mat4f> intrinsics, extrinsics;
		readBinaryDumpHeader(m_infile, imageWidth, imageHeight, numFrames, numCams, intrinsics, extrinsics);

		DepthSensor::init(imageWidth, imageHeight, imageWidth, imageHeight);
		initializeIntrinsics(intrinsics[0](0,0),intrinsics[0](1,1),intrinsics[0](0,2),intrinsics[0](1,2));

		m_numFrames = numFrames;

		if (m_infile.fail() || numCams != 1)
		{
			std::cout << "file " << filename << " not found or invalid parameters" << std::endl;
			return S_FALSE;
		}

		//for(unsigned int i = 0; i<0; i++)
		//{
		//	m_infile.read((char*)m_colorRGBX, 4*sizeof(BYTE)*imageWidth*imageHeight);
		//	m_infile.read((char*)m_depthD16, sizeof(unsigned short)*imageWidth*imageHeight);
		//}

		return S_OK;
	}

	HRESULT BinaryDumpSensorOld::processDepth()
	{
		m_infile.read((char*)m_colorRGBX, sizeof(BYTE)*getColorWidth()*getColorHeight()*getColorBytesPerPixel());
		m_infile.read((char*)m_depthD16, sizeof(unsigned short)*getDepthWidth()*getDepthHeight());
		return S_OK;
	}

	HRESULT processColor()
	{
		//everything done in process depth since order is relevant (color must be read first)
		return S_OK;
	}

	HRESULT BinaryDumpSensorOld::toggleNearMode()
	{
		return S_OK;
	}

	//! Toggle enable auto white balance
	HRESULT toggleAutoWhiteBalance()
	{
		return S_OK;
	}

	bool isKinect4Windows()
	{
		return true;
	}

private:
	std::ifstream m_infile;
	unsigned int m_numFrames;
};
