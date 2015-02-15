#include <cutil_inline.h>
#include <cutil_math.h>

#include "cuda_SimpleMatrixUtil.h"

#define T_PER_BLOCK 8
#define B_GROUND -1.0f
#define MINF __int_as_float(0xff800000)
#define I_DEPTH MINF

__global__ void pullKernel(float* d_input, float* d_output, unsigned int inputWidth, unsigned int inputHeight, unsigned int outputWidth, unsigned int outputHeight) 
{
	const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x < outputWidth && y < outputHeight)
	{	
		const uint2 idx = make_uint2(x,y);

		float sum = 0.0f;
		float weight = 0.0f;
		
		const uint2 inputIdx = 2*idx;

		if ((inputIdx.x+1) < inputWidth && (inputIdx.y+0) < inputHeight &&
			d_input[(inputIdx.x+1) + (inputIdx.y+0)*inputWidth] != I_DEPTH && d_input[(inputIdx.x+1) + (inputIdx.y+0)*inputWidth] != B_GROUND) {	
			sum += abs(d_input[(inputIdx.x+1) + (inputIdx.y+0)*inputWidth]);	weight += 1.0f;	
		}
		if ((inputIdx.x+0) < inputWidth && (inputIdx.y+0) < inputHeight &&
			d_input[(inputIdx.x+0) + (inputIdx.y+0)*inputWidth] != I_DEPTH && d_input[(inputIdx.x+0) + (inputIdx.y+0)*inputWidth] != B_GROUND) {
			sum += abs(d_input[(inputIdx.x+0) + (inputIdx.y+0)*inputWidth]);	weight += 1.0f;	
		}
		if ((inputIdx.x+0) < inputWidth && (inputIdx.y+1) < inputHeight &&
			d_input[(inputIdx.x+0) + (inputIdx.y+1)*inputWidth] != I_DEPTH && d_input[(inputIdx.x+0) + (inputIdx.y+1)*inputWidth] != B_GROUND) {	
			sum += abs(d_input[(inputIdx.x+0) + (inputIdx.y+1)*inputWidth]);	weight += 1.0f;	
		}
		if ((inputIdx.x+1) < inputWidth && (inputIdx.y+1) < inputHeight &&
			d_input[(inputIdx.x+1) + (inputIdx.y+1)*inputWidth] != I_DEPTH && d_input[(inputIdx.x+1) + (inputIdx.y+1)*inputWidth] != B_GROUND) {
			sum += abs(d_input[(inputIdx.x+1) + (inputIdx.y+1)*inputWidth]);	weight += 1.0f;	
		}

		if (weight != 0.0f) sum /= weight;
		else				sum = B_GROUND;	//set to background if there was no contour data

		d_output[idx.x + idx.y*outputWidth] = sum;
	}
}



extern "C" void pull(float* d_input, float* d_output, unsigned int inputWidth, unsigned int inputHeight, unsigned int outputWidth, unsigned int outputHeight) {
	const dim3 gridSize((outputWidth + T_PER_BLOCK - 1)/T_PER_BLOCK, (outputHeight + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	pullKernel<<<gridSize, blockSize>>>(d_input, d_output, inputWidth, inputHeight, outputWidth, outputHeight);

	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}


__global__ void pushKernel(float* d_input, float* d_output, unsigned int inputWidth, unsigned int inputHeight, unsigned int outputWidth, unsigned int outputHeight) 
{
	const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x < outputWidth && y < outputHeight && (d_output[x + y*outputWidth] == I_DEPTH || d_output[x + y*outputWidth] == B_GROUND))
	{	
		const uint2 idx = make_uint2(x,y);
				
		//smooth interpolate
		uint2 idx_lb = make_uint2(idx.x-1, idx.y-1);	idx_lb.x /= 2;	idx_lb.y /= 2;
		uint2 idx_lt = make_uint2(idx.x-1, idx.y+1);	idx_lt.x /= 2;	idx_lt.y /= 2;
		uint2 idx_rb = make_uint2(idx.x+1, idx.y-1);	idx_rb.x /= 2;	idx_rb.y /= 2;
		uint2 idx_rt = make_uint2(idx.x+1, idx.y+1);	idx_rt.x /= 2;	idx_rt.y /= 2;


		float lb = I_DEPTH;
		float lt = I_DEPTH;
		float rb = I_DEPTH;
		float rt = I_DEPTH;

		if (idx_lb.x < inputWidth && idx_lb.y < inputHeight) lb = d_input[idx_lb.x + idx_lb.y*inputWidth];
		if (idx_lt.x < inputWidth && idx_lt.y < inputHeight) lt = d_input[idx_lt.x + idx_lt.y*inputWidth];
		if (idx_rb.x < inputWidth && idx_rb.y < inputHeight) rb = d_input[idx_rb.x + idx_rb.y*inputWidth];
		if (idx_rt.x < inputWidth && idx_rt.y < inputHeight) rt = d_input[idx_rt.x + idx_rt.y*inputWidth];


		uint test_x = (idx.x+1) % 2;
		uint test_y = (idx.y+1) % 2;

		float sum_numerator = 0.0f;
		float sum_denominator = 0.0f;

		if (test_x == 0)	{		//we are at left 
			if (test_y == 0)	{	//we are at left bottom
				if (lb != I_DEPTH && lb != B_GROUND)	{sum_numerator += 9.0f*lb;	sum_denominator += 9.0f;}	
				if (lt != I_DEPTH && lt != B_GROUND)	{sum_numerator += 3.0f*lt;	sum_denominator += 3.0f;}
				if (rb != I_DEPTH && rb != B_GROUND)	{sum_numerator += 3.0f*rb;	sum_denominator += 3.0f;}
				if (rt != I_DEPTH && rt != B_GROUND)	{sum_numerator += 1.0f*rt;	sum_denominator += 1.0f;}
			} else {				//we are at left top
				if (lb != I_DEPTH && lb != B_GROUND)	{sum_numerator += 3.0f*lb;	sum_denominator += 3.0f;}
				if (lt != I_DEPTH && lt != B_GROUND)	{sum_numerator += 9.0f*lt;	sum_denominator += 9.0f;}
				if (rb != I_DEPTH && rb != B_GROUND)	{sum_numerator += 1.0f*rb;	sum_denominator += 1.0f;}
				if (rt != I_DEPTH && rt != B_GROUND)	{sum_numerator += 3.0f*rt;	sum_denominator += 3.0f;}
			}
		}	else {					//we are at right
			if (test_y == 0)	{	//we are at right bottom
				if (lb != I_DEPTH && lb != B_GROUND)	{sum_numerator += 3.0f*lb;	sum_denominator += 3.0f;}
				if (lt != I_DEPTH && lt != B_GROUND)	{sum_numerator += 1.0f*lt;	sum_denominator += 1.0f;}
				if (rb != I_DEPTH && rb != B_GROUND)	{sum_numerator += 9.0f*rb;	sum_denominator += 9.0f;}
				if (rt != I_DEPTH && rt != B_GROUND)	{sum_numerator += 3.0f*rt;	sum_denominator += 3.0f;}
			} else {				//we are at right top
				if (lb != I_DEPTH && lb != B_GROUND)	{sum_numerator += 1.0f*lb;	sum_denominator += 1.0f;}
				if (lt != I_DEPTH && lt != B_GROUND)	{sum_numerator += 3.0f*lt;	sum_denominator += 3.0f;}
				if (rb != I_DEPTH && rb != B_GROUND)	{sum_numerator += 3.0f*rb;	sum_denominator += 3.0f;}
				if (rt != I_DEPTH && rt != B_GROUND)	{sum_numerator += 9.0f*rt;	sum_denominator += 9.0f;}
			}
		}

		//only set if upper level had something to contribute AND current level is uninitialized interior
		//if (sum_denominator >= 9.0f) {
		if (sum_denominator > 0.0f) {
			d_output[idx.x + idx.y*outputWidth] = sum_numerator / sum_denominator;
		}
	}
}



extern "C" void push(float* d_input, float* d_output, unsigned int inputWidth, unsigned int inputHeight, unsigned int outputWidth, unsigned int outputHeight) {
	
	const dim3 gridSize((outputWidth + T_PER_BLOCK - 1)/T_PER_BLOCK, (outputHeight + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	pushKernel<<<gridSize, blockSize>>>(d_input, d_output, inputWidth, inputHeight, outputWidth, outputHeight);

	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}



__global__ void laplaceInterpolateKernel(float* d_input, float* d_constraint, unsigned int width, unsigned int height) {
	
	const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x < width && y < height && x > 0 && y > 0 && d_constraint[x + y*width] == I_DEPTH)
	{	
		float sum = 0.0f;
		float weight = 0.0f;

		const uint2 idx = make_uint2(x,y);
		if ((idx.x+1) < width && (idx.y+0) < height &&
			d_input[(idx.x+1) + (idx.y+0)*width] != I_DEPTH) {	sum += abs(d_input[(idx.x+1) + (idx.y+0)*width]);	weight += 1.0f;	}
		if ((idx.x-1) < width && (idx.y+0) < height &&
			d_input[(idx.x-1) + (idx.y+0)*width] != I_DEPTH) {	sum += abs(d_input[(idx.x-1) + (idx.y+0)*width]);	weight += 1.0f;	}
		if ((idx.x+0) < width && (idx.y+1) < height &&
			d_input[(idx.x+0) + (idx.y+1)*width] != I_DEPTH) {	sum += abs(d_input[(idx.x+0) + (idx.y+1)*width]);	weight += 1.0f;	}
		if ((idx.x+0) < width && (idx.y-1) < height &&
			d_input[(idx.x+0) + (idx.y-1)*width] != I_DEPTH) {	sum += abs(d_input[(idx.x+0) + (idx.y-1)*width]);	weight += 1.0f;	}


		if (weight != 0.0f) {
			d_input[idx.x + idx.y*width] = sum / weight;
		}
	}
}


extern "C" void laplaceInterpolate(float* d_input, float* d_constraint, unsigned int width, unsigned int height) {

	const dim3 gridSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	laplaceInterpolateKernel<<<gridSize, blockSize>>>(d_input, d_constraint, width, height);

	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}


__global__ void clearImageKernel(float* d_inout, unsigned int imageWidth, unsigned int imageHeight, float clearValue)  {

	const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x < imageWidth && y < imageHeight) 
	{
		d_inout[x + y*imageWidth] = clearValue;
	}


}


extern "C" void clearImage(float* d_inout, unsigned int imageWidth, unsigned int imageHeight, float clearValue) {

	const dim3 gridSize((imageWidth + T_PER_BLOCK - 1)/T_PER_BLOCK, (imageHeight + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	clearImageKernel<<<gridSize, blockSize>>>(d_inout, imageWidth, imageHeight, clearValue);

	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}