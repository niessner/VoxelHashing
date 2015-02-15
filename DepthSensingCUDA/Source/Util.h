#pragma once

#include "cudaUtil.h"
#include "mLib.h"

namespace Util
{
	static void writeToImage(float* d_buffer, unsigned int width, unsigned int height, const std::string& filename) {
		//bool minfInvalid = false;

		float* h_buffer = new float[width * height];
		cudaMemcpy(h_buffer, d_buffer, sizeof(float)*width*height, cudaMemcpyDeviceToHost);

		//for (unsigned int i = 0; i < width*height; i++) {
		//	if (h_buffer[i] == -std::numeric_limits<float>::infinity()) {
		//		minfInvalid = true;
		//		break;
		//	}
		//}
		//if (!minfInvalid) std::cout << "invalid valid != MINF" << std::endl;
		//else std::cout << "MINF invalid value" << std::endl;

		DepthImage dimage(height, width, h_buffer);
		ColorImageRGB cImage(dimage);
		FreeImageWrapper::saveImage(filename, cImage);

		SAFE_DELETE_ARRAY(h_buffer);
	}

	static void writeToImage(float4* d_buffer, float min, float max, unsigned int width, unsigned int height, const std::string& filename) {
		unsigned int size = width * height;
		float* h_buffer = new float[4 * size];
		cudaMemcpy(h_buffer, d_buffer, 4*sizeof(float)*size, cudaMemcpyDeviceToHost);

		vec3f* data = new vec3f[size];
		for (unsigned int i = 0; i < size; i++) {
			data[i] = vec3f(h_buffer[i*4], h_buffer[i*4+1], h_buffer[i*4+2]);
			if (data[i].x != -std::numeric_limits<float>::infinity()) {
				data[i] = (data[i] - min) / (max - min);
			}
		}
		std::cout << "range: [" << min << ", " << max << "]" << std::endl;
		ColorImageRGB cImage(height, width, data);
		FreeImageWrapper::saveImage(filename, cImage);

		SAFE_DELETE_ARRAY(h_buffer);
		SAFE_DELETE_ARRAY(data);
	}

	static void writeToImage(float4* d_buffer, unsigned int width, unsigned int height, const std::string& filename) {
		unsigned int size = width * height;
		float* h_buffer = new float[4 * size];
		cudaMemcpy(h_buffer, d_buffer, 4*sizeof(float)*size, cudaMemcpyDeviceToHost);

		vec3f* data = new vec3f[size];
		float min = std::numeric_limits<float>::infinity(), max = -std::numeric_limits<float>::infinity();
		for (unsigned int i = 0; i < size; i++) {
			data[i] = vec3f(h_buffer[i*4], h_buffer[i*4+1], h_buffer[i*4+2]);
			if (data[i].x != -std::numeric_limits<float>::infinity()) {
				if (data[i].x < min) min = data[i].x;
				if (data[i].y < min) min = data[i].y;
				if (data[i].z < min) min = data[i].z;

				if (data[i].x > max) max = data[i].x;
				if (data[i].y > max) max = data[i].y;
				if (data[i].z > max) max = data[i].z;
			}
		}
		for (unsigned int i = 0; i < size; i++) {
			if (data[i].x != -std::numeric_limits<float>::infinity()) {
				data[i] = (data[i] - min) / (max - min);
			}
		}
		std::cout << "range: [" << min << ", " << max << "]" << std::endl;
		ColorImageRGB cImage(height, width, data);
		FreeImageWrapper::saveImage(filename, cImage);

		SAFE_DELETE_ARRAY(h_buffer);
		SAFE_DELETE_ARRAY(data);
	}

	static float* loadFloat4FromBinary(const std::string& filename, unsigned int& width, unsigned int& height, unsigned int& numChannels)
	{
		BinaryDataStreamFile s(filename, false);
		s >> width;
		s >> height;
		s >> numChannels;
		float * result = new float[numChannels*width*height];
		s.readData((BYTE*)result, width*height*sizeof(float)*numChannels);
		s.closeStream();
		return result;
	}

	static float* loadFloatFromBinary(const std::string& filename, unsigned int& size)
	{
		BinaryDataStreamFile s(filename, false);
		s >> size;
		float * result = new float[size];
		s.readData((BYTE*)result, size*sizeof(float));
		s.closeStream();
		return result;
	}

	static void saveFloat4ToBinary(float4* d_buffer, unsigned int width, unsigned int height, const std::string& filename) 
	{
		unsigned int size = width * height;
		float* h_buffer = new float[4 * size];
		cudaMemcpy(h_buffer, d_buffer, 4*sizeof(float)*size, cudaMemcpyDeviceToHost);

		BinaryDataStreamFile s(filename, true);
		s << width << height << 4;
		s.writeData((BYTE*)h_buffer, size*sizeof(float)*4);
		s.closeStream();

		SAFE_DELETE_ARRAY(h_buffer);
	}

	static void getRange(float* d_buffer, unsigned int size, float& min, float& max)
	{
		min = std::numeric_limits<float>::infinity();
		max = -std::numeric_limits<float>::infinity();

		float* h_buffer = new float[size];
		cudaMemcpy(h_buffer, d_buffer, sizeof(float)*size, cudaMemcpyDeviceToHost);

		for (unsigned int i = 0; i < size; i++) {
			if (h_buffer[i] != -std::numeric_limits<float>::infinity()) {
				if (h_buffer[i] < min) min = h_buffer[i];
				if (h_buffer[i] > max) max = h_buffer[i];
			}
		}

		SAFE_DELETE_ARRAY(h_buffer);
	}
}
