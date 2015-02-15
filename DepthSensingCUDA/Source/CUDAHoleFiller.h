#pragma once

#include "cudaUtil.h"
#include "cuda_SimpleMatrixUtil.h"

#include <vector>

extern "C" void pull(float* d_input, float* d_output, unsigned int inputWidth, unsigned int inputHeight, unsigned int outputWidth, unsigned int outputHeight);
extern "C" void push(float* d_input, float* d_output, unsigned int inputWidth, unsigned int inputHeight, unsigned int outputWidth, unsigned int outputHeight);
extern "C" void laplaceInterpolate(float* d_input, float* d_constraint, unsigned int width, unsigned int height);
extern "C" void clearImage(float* d_inout, unsigned int imageWidth, unsigned int imageHeight, float clearValue);

template<class T> T 
	inline log2IntegerCeil(T x) 
{
	T r = 0;
	while (x) {
		r++;
		x >>= 1;
	}
	return r;
}

class CUDAHoleFiller
{
public:
	CUDAHoleFiller() 
	{
		m_ImageHeight = 0;
		m_ImageWidth = 0;
		m_ImagePullPushLevels = 0;
		d_WorkingImages = NULL;
	}

	~CUDAHoleFiller() 
	{
		releaseResources();
	}

	void holeFill(float* d_depthMap, unsigned int width, unsigned height, unsigned int numLevels = 0, unsigned int laplaceIterations = 0) {

		resize(width, height, numLevels);

		//clear();


		//pull phase
		for (unsigned int i = 1; i < m_ImagePullPushLevels; i++) {
			const unsigned int inputWidth = m_WidthAtLevel[i-1];
			const unsigned int inputHeight = m_HeightAtLevel[i-1];
			const unsigned int outputWidth = m_WidthAtLevel[i];
			const unsigned int outputHeight = m_HeightAtLevel[i];
			if (i == 1) {
				pull(d_depthMap, d_WorkingImages[i], inputWidth, inputHeight, outputWidth, outputHeight);
			} else {
				pull(d_WorkingImages[i-1], d_WorkingImages[i], inputWidth, inputHeight, outputWidth, outputHeight);
			}
		}

		//push phase
		for (unsigned int i = m_ImagePullPushLevels-1; i > 0; i--) {
			unsigned int outputWidth = m_WidthAtLevel[i-1];
			unsigned int outputHeight = m_HeightAtLevel[i-1];
			const unsigned int inputWidth = m_WidthAtLevel[i];
			const unsigned int inputHeight = m_HeightAtLevel[i];
			if (i == 1) {
				push(d_WorkingImages[i], d_depthMap, inputWidth, inputHeight, outputWidth, outputHeight);
			} else {
				push(d_WorkingImages[i], d_WorkingImages[i-1], inputWidth, inputHeight, outputWidth, outputHeight);
			}
		}

		
		//laplace interpolation for smoother results
		if (laplaceIterations > 0) {
			cutilSafeCall(cudaMemcpy(d_WorkingImages[0], d_depthMap, m_ImageWidth*m_ImageHeight*sizeof(float), cudaMemcpyDeviceToDevice));
			for (unsigned int i = 0; i < laplaceIterations; i++) {
				//in this case the second parameter is just a constraint for depth values that already existed and should not be overwritten by the kernel
				laplaceInterpolate(d_depthMap, d_WorkingImages[0], m_ImageWidth, m_ImageHeight);
			}
		}
	}

private:
	//! sets all memory to zero (just debug)
	void clear(float c = 0.0f) {
		if (!d_WorkingImages) return;

		for (unsigned int i = 0; i < m_ImagePullPushLevels; i++) {
			//const unsigned int numPixels = m_WidthAtLevel[i]*m_HeightAtLevel[i];
			//float* zero = new float[numPixels];
			//for (unsigned int j = 0; j < numPixels; j++)	zero[j] = c;			
			//cutilSafeCall(cudaMemcpy(d_WorkingImages[i], (void*)zero, sizeof(float)*numPixels, cudaMemcpyHostToDevice));
			//delete[] zero;

			clearImage(d_WorkingImages[i], m_WidthAtLevel[i], m_HeightAtLevel[i], c);
		}
	}

	//! frees all the data
	void releaseResources() {
		if (d_WorkingImages) {
			for (unsigned int i = 0; i < m_ImagePullPushLevels; i++) {
				cutilSafeCall(cudaFree(d_WorkingImages[i]));
			}
			delete[] d_WorkingImages;
			d_WorkingImages = NULL;
		}
		m_ImageHeight = 0;
		m_ImageWidth = 0;
		m_ImagePullPushLevels = 0;
		m_WidthAtLevel.clear();
		m_HeightAtLevel.clear();
	}

	//! allocates GPU memory if necessary (can be safely called every frame)
	void resize(unsigned int width, unsigned int height, unsigned int numLevels) {
		if (numLevels == 0) {

			unsigned int numX = log2IntegerCeil(width);
			unsigned int numY = log2IntegerCeil(height);
			numLevels = std::max(numX, numY);
		}
		if (width == m_ImageWidth && height == m_ImageHeight && m_ImagePullPushLevels == numLevels)	return;
		releaseResources();
		

		m_ImageWidth = width;
		m_ImageHeight = height;
		m_ImagePullPushLevels = numLevels;

		unsigned int currWidth = m_ImageWidth;
		unsigned int currHeight = m_ImageHeight;

		m_WidthAtLevel.resize(m_ImagePullPushLevels+1);
		m_HeightAtLevel.resize(m_ImagePullPushLevels+1);
		for (unsigned int i = 0; i < m_ImagePullPushLevels+1; i++) {
			m_WidthAtLevel[i] = currWidth;
			m_HeightAtLevel[i] = currHeight;
			currWidth = (currWidth+1)/2;
			currHeight = (currHeight+1)/2;

		}

		d_WorkingImages = new float*[m_ImagePullPushLevels];
		for (unsigned int i = 0; i < m_ImagePullPushLevels; i++) {
			cutilSafeCall(cudaMalloc(&(d_WorkingImages[i]),m_WidthAtLevel[i]*m_HeightAtLevel[i]*sizeof(float)));
		}
	}


	unsigned int m_ImageWidth;
	unsigned int m_ImageHeight;
	std::vector<unsigned int> m_WidthAtLevel;
	std::vector<unsigned int> m_HeightAtLevel;

	unsigned int m_ImagePullPushLevels;

	//! device memory of the images
	float** d_WorkingImages;
};

