#ifndef _FREEIMAGEWRAPPER_H_
#define _FREEIMAGEWRAPPER_H_

#include "freeImageWrapperHelper.h"
#include "FreeImage.h"

namespace ml {

class FreeImageWrapper {
public:

	template<class T>
	static void loadImage(const std::string &filename, BaseImage<T>& resultImage) {
		if (util::getFileExtension(filename) == "mbinRGB" || util::getFileExtension(filename) == "mbindepth") {
			resultImage.loadFromBinaryMImage(filename);
			return;
		}

		FREE_IMAGE_FORMAT fif = FIF_UNKNOWN;
		FIBITMAP *dib(0);
		BYTE* bits;

		FreeImage_Initialise();
		fif = FreeImage_GetFileType(filename.c_str());
		dib = FreeImage_Load(fif, filename.c_str());
		if (!dib) throw MLIB_EXCEPTION("Could not load image: " + filename);
		FREE_IMAGE_TYPE fitype = FreeImage_GetImageType(dib);

		if (fitype != FIT_BITMAP && fitype != FIT_UINT16) throw MLIB_EXCEPTION("Unknown image format");

		bits = FreeImage_GetBits(dib);
		unsigned int width = FreeImage_GetWidth(dib);
		unsigned int height = FreeImage_GetHeight(dib);
		unsigned int nBits = FreeImage_GetBPP(dib);
		unsigned int pitch = FreeImage_GetPitch(dib);

		resultImage.allocateToSize(height, width);

		if (fitype == FIT_UINT16) {
			const BYTE* data = (BYTE*)bits;
			unsigned int bytesPerPixel = nBits/8;
			assert(bytesPerPixel == 2);
			#pragma omp parallel for
			for (int i = 0; i < (int)height; i++) {
				const BYTE* dataRowStart = data + (height-1-i)*pitch;
				for (int j = 0; j < (int)width; j++) {
					convertFromUSHORT(resultImage(i,j), (USHORT*)&dataRowStart[j*bytesPerPixel]);
				}
			}	
		} else if (fitype == FIT_BITMAP) {
			const BYTE* data = (BYTE*)bits;
			unsigned int bytesPerPixel = nBits/8;
			if (bytesPerPixel == 3) {
				#pragma omp parallel for
				for (int i = 0; i < (int)height; i++) {
					const BYTE* dataRowStart = data + (height-1-i)*pitch;
					for (int j = 0; j < (int)width; j++) {						
						convertFromBYTE3(resultImage(i,j), &dataRowStart[j*bytesPerPixel]);
					}
				}
			} else if (bytesPerPixel == 4) {
				#pragma omp parallel for
				for (int i = 0; i < (int)height; i++) {
					const BYTE* dataRowStart = data + (height-1-i)*pitch;
					for (int j = 0; j < (int)width; j++) {
						convertFromBYTE4(resultImage(i,j), &dataRowStart[j*bytesPerPixel]);
					}
				}
			} else {
				throw MLIB_EXCEPTION("Unknown image format");
			}
		}

		FreeImage_Unload(dib);

		std::cout << __FUNCTION__ << ":" << filename << " (width=" << width << ";height=" << height << "; " << resultImage.getNumChannels() << "; " << resultImage.getNumBytesPerChannel() <<  ")" << std::endl;
		FreeImage_DeInitialise();
	}



	template<class T>
	static void saveImage(const std::string &filename, const BaseImage<T>& image) {
		if (util::getFileExtension(filename) == "mbinRGB" || util::getFileExtension(filename) == "mbindepth") {
			image.saveAsBinaryMImage(filename);
			return;
		}

		FreeImage_Initialise();

		unsigned int width = image.getWidth();
		unsigned int height = image.getHeight();

		unsigned int bytesPerPixel = image.getNumChannels();


		assert(image.getNumChannels() == 3 || image.getNumChannels() == 4);
		//TODO do it for depth maps

		if (filename.length() > 4 && filename.find(".jpg") != std::string::npos ||
			filename.length() > 4 && filename.find(".png") != std::string::npos) {
			FIBITMAP *dib = FreeImage_Allocate(width, height, bytesPerPixel * 8);
			BYTE* bits = FreeImage_GetBits(dib);
			unsigned int pitch = FreeImage_GetPitch(dib);


			if (bytesPerPixel == 3) {
				#pragma omp parallel for
				for (int i = 0; i < (int)height; i++) {
					BYTE* bitsRowStart = bits + (height-1-i)*pitch;
					for (int j = 0; j < (int)width; j++) {
						vec3uc color;		convertToVEC3UC(color, image(i,j));
						bitsRowStart[j*bytesPerPixel + FI_RGBA_RED] =	(unsigned char)color.x;
						bitsRowStart[j*bytesPerPixel + FI_RGBA_GREEN] = (unsigned char)color.y;
						bitsRowStart[j*bytesPerPixel + FI_RGBA_BLUE] = (unsigned char)color.z;
					}
				}
			} else if (bytesPerPixel == 4) {
				assert(filename.find(".jpg") == std::string::npos);	//jpgs with transparencies don't work...

				#pragma omp parallel for
				for (int i = 0; i < (int)height; i++) {
					BYTE* bitsRowStart = bits + (height-1-i)*pitch;
					for (int j = 0; j < (int)width; j++) {
						vec4uc color;		convertToVEC4UC(color, image(i,j));
						bitsRowStart[j*bytesPerPixel + FI_RGBA_RED] =	(unsigned char)color.x;
						bitsRowStart[j*bytesPerPixel + FI_RGBA_GREEN] = (unsigned char)color.y;
						bitsRowStart[j*bytesPerPixel + FI_RGBA_BLUE] = (unsigned char)color.z;
						bitsRowStart[j*bytesPerPixel + FI_RGBA_ALPHA] = (unsigned char)color.w;
					}
				}
			} else {
				throw MLIB_EXCEPTION("Unknown image format");
			}


			if (filename.length() > 4 && filename.find(".jpg") != std::string::npos) {
				FreeImage_Save(FIF_JPEG, dib, filename.c_str());
			} else if (filename.length() > 4 && filename.find(".png") != std::string::npos) {
				FreeImage_Save(FIF_PNG, dib, filename.c_str());
			} else {
				assert(false);
			}
			FreeImage_Unload(dib);
		} else {
			throw MLIB_EXCEPTION("Unknown file format");
		}

		std::cout << __FUNCTION__ << ":" << filename << " (width=" << width << ";height=" << height << "; " << image.getNumChannels() << "; " << image.getNumBytesPerChannel() <<  ")" << std::endl;
		FreeImage_DeInitialise();
	}













	/*


	//////////////////////////////////////////////////
	/// UN-TEMPLETIZED IMAGE LOADERS (deprecated!) ///	//pitch is wrong...
	//////////////////////////////////////////////////
	
	static void loadColorImageRGBFromFile(const std::string &filename, ColorImageRGB &resultImage) {

		if (util::getFileExtension(filename) == "mbinRGB") {
			resultImage.loadFromBinaryMImage(filename);
			return;
		}

		FREE_IMAGE_FORMAT fif = FIF_UNKNOWN;
		FIBITMAP *dib(0);
		BYTE* bits;

		FreeImage_Initialise();
		fif = FreeImage_GetFileType(filename.c_str());
		dib = FreeImage_Load(fif, filename.c_str());
		if (!dib) throw MLIB_EXCEPTION("Could not load image: " + filename);
		FREE_IMAGE_TYPE fitype = FreeImage_GetImageType(dib);

		if (fitype != FIT_BITMAP && fitype != FIT_RGBAF && fitype != FIT_RGBF) throw MLIB_EXCEPTION("Unknown image format");

		bits = FreeImage_GetBits(dib);
		unsigned int width = FreeImage_GetWidth(dib);
		unsigned int height = FreeImage_GetHeight(dib);
		unsigned int nBits = FreeImage_GetBPP(dib);

		resultImage.allocateToSize(height, width);

		if (fitype == FIT_RGBAF) {
			float* data = (float*)bits;
			#pragma omp parallel for
			for (int i = 0; i < (int)height; i++) {
				for (int j = 0; j < (int)width; j++) {
					resultImage(i,j) = *((vec3f*)&data[4 * (i*width+j)]);
				}
			}
		} else if (fitype == FIT_RGBF) {
			float* data = (float*)bits;
			#pragma omp parallel for
			for (int i = 0; i < (int)height; i++) {
				for (int j = 0; j < (int)width; j++) {
					resultImage(i,j) = *((vec3f*)&data[3 * (i*width+j)]);
				}
			}
		}
		else if (fitype == FIT_BITMAP) {

			BYTE* data = (BYTE*)bits;
			#pragma omp parallel for
			for (int i = 0; i < (int)height; i++) {
				for (int j = 0; j < (int)width; j++) {

					float r = data[(nBits/8) * (i*width+j) + 0];
					float g = data[(nBits/8) * (i*width+j) + 1];
					float b = data[(nBits/8) * (i*width+j) + 2];

					resultImage(i,j) = vec3f(b/255.0f, g/255.0f, r/255.0f);
				}
			}
		}

		FreeImage_Unload(dib);

		std::cout << "Info: " __FUNCTION__ << " : image loaded (width=" << width << ";height=" << height << ")" << std::endl;

		FreeImage_DeInitialise();
	}

	static void saveColorImageRGBToFile(const std::string &filename, const ColorImageRGB& image) {

		if (getFileExtension(filename) == "mbinRGB") {
			image.saveAsBinaryMImage(filename);
			return;
		}

		FreeImage_Initialise();

		unsigned int width = image.getWidth();
		unsigned int height = image.getHeight();

		if (filename.length() > 4 && filename.find(".png") != std::string::npos)
		{
			FIBITMAP *dib = FreeImage_Allocate(width, height, 32);
			BYTE* bits = FreeImage_GetBits(dib);

			#pragma omp parallel for
			for (int i = 0; i < (int)height; i++) {
				for (int j = 0; j < (int)width; j++) {
					vec4f color = image(i,j);
					color *= 255.0f;
					bits[4*(i*width + j) + FI_RGBA_RED] =	(unsigned char)color.x;
					bits[4*(i*width + j) + FI_RGBA_GREEN] = (unsigned char)color.y;
					bits[4*(i*width + j) + FI_RGBA_BLUE] =	(unsigned char)color.z;
					bits[4*(i*width + j) + FI_RGBA_ALPHA] = (unsigned char)color.w;
				}
			}

			FreeImage_Save(FIF_PNG, dib, filename.c_str());

			FreeImage_Unload(dib);
		}
		else if (filename.length() > 4 && filename.find(".jpg") != std::string::npos)
		{
			FIBITMAP *dib = FreeImage_Allocate(width, height, 24);
			BYTE* bits = FreeImage_GetBits(dib);

			#pragma omp parallel for
			for (int i = 0; i < (int)height; i++) {
				for (int j = 0; j < (int)width; j++) {
					vec4f color = image(i,j);
					color *= 255.0f;
					bits[3*(i*width + j) + FI_RGBA_RED] =	(unsigned char)color.x;
					bits[3*(i*width + j) + FI_RGBA_GREEN] = (unsigned char)color.y;
					bits[3*(i*width + j) + FI_RGBA_BLUE] =	(unsigned char)color.z;
				}
			}

			FreeImage_Save(FIF_JPEG, dib, filename.c_str());

			FreeImage_Unload(dib);
		}
		else {
			throw MLIB_EXCEPTION("Unknown file format");
		}

		std::cout << "Info: " __FUNCTION__ << " : done (width=" << width << ";height=" << height << ")" << std::endl;

		FreeImage_DeInitialise();
	}

	static void loadDepthImageFromFile(const std::string &filename, DepthImage &resultImage) {

		if (util::getFileExtension(filename) == "mbindepth") {
			resultImage.loadFromBinaryMImage(filename);
			return;
		}

		FREE_IMAGE_FORMAT fif = FIF_UNKNOWN;
		FIBITMAP *dib(0);
		BYTE* bits;

		FreeImage_Initialise();
		fif = FreeImage_GetFileType(filename.c_str());
		dib = FreeImage_Load(fif, filename.c_str());
		if (!dib) throw MLIB_EXCEPTION("Could not load image: " + filename);
		FREE_IMAGE_TYPE fitype = FreeImage_GetImageType(dib);

		if (fitype != FIT_FLOAT && fitype != FIT_RGBAF && fitype != FIT_RGBF && fitype != FIT_UINT16) throw MLIB_EXCEPTION("Unknown image format");

		bits = FreeImage_GetBits(dib);
		unsigned int width = FreeImage_GetWidth(dib);
		unsigned int height = FreeImage_GetHeight(dib);

		resultImage.allocateToSize(height, width);

		if (fitype == FIT_UINT16) {
			unsigned short* data = (unsigned short*)bits;
#pragma omp parallel for
			for (int i = 0; i < (int)height; i++) {
				for (int j = 0; j < (int)width; j++) {
					resultImage(i,j) = (float)data[i*width+j];
					resultImage(i,j) /= 1000.0f;	//convert to meters; typically for kinect data, however not very efficient
				}
			}
		} else if (fitype == FIT_FLOAT) {
			float* data = (float*)bits;
			#pragma omp parallel for
			for (int i = 0; i < (int)height; i++) {
				for (int j = 0; j < (int)width; j++) {
					resultImage(i,j) = data[i*width+j];
				}
			}
		} else if (fitype == FIT_RGBAF) {
			float* data = (float*)bits;
			#pragma omp parallel for
			for (int i = 0; i < (int)height; i++) {
				for (int j = 0; j < (int)width; j++) {
					resultImage(i,j) = data[4 * (i*width+j)];
				}
			}
		} else if (fitype == FIT_RGBF) {
			float* data = (float*)bits;
			#pragma omp parallel for
			for (int i = 0; i < (int)height; i++) {
				for (int j = 0; j < (int)width; j++) {
					resultImage(i,j) = data[3 * (i*width+j)];
				}
			}
		} else if (fitype == FIT_UINT16) {
			unsigned short* data = (unsigned short*)bits;
			#pragma omp parallel for
			for (int i = 0; i < (int)height; i++) {
				for (int j = 0; j < (int)width; j++) {
					resultImage(i,j) = (float)data[i*width+j]/1000.0f;
				}
			}
		}


		FreeImage_Unload(dib);

		std::cout << "Info: " __FUNCTION__ << " : image loaded (width=" << width << ";height=" << height << ")" << std::endl;

		FreeImage_DeInitialise();
	}

	static void saveDepthImageToFile(const std::string &filename, const DepthImage& image) {

		if (getFileExtension(filename) == "mbindepth") {
			image.saveAsBinaryMImage(filename);
			return;
		}

		FreeImage_Initialise();

		unsigned int width = image.getWidth();
		unsigned int height = image.getHeight();

		FIBITMAP *dib = FreeImage_AllocateT(FIT_FLOAT, width, height, 32);
		BYTE* bits = FreeImage_GetBits(dib);

#pragma omp parallel for
		for (int i = 0; i < (int)height; i++) {
			for (int j = 0; j < (int)width; j++) {
				float* data = (float*)bits;
				data[i*width + j] = image(i,j);
			}
		}

		if (filename.length() > 4 && filename.find(".tif") != std::string::npos) {
			FreeImage_Save(FIF_TIFF, dib, filename.c_str());
		} else {
			throw MLIB_EXCEPTION("Unknown file format");
		}

		FreeImage_Unload(dib);

		std::cout << "Info: " __FUNCTION__ << " : done (width=" << width << ";height=" << height << ")" << std::endl;

		FreeImage_DeInitialise();
	}

	static void saveDepthImageToFilePNG(const std::string &filename, const DepthImage& image)
	{
		FreeImage_Initialise();

		unsigned int width = image.getWidth();
		unsigned int height = image.getHeight();

		FIBITMAP *dib = FreeImage_Allocate(width, height, 24);
		RGBQUAD color;
		
		for (int i = 0; i < (int)height; i++)
		{
			for (int j = 0; j < (int)width; j++)
			{
				if(image.isValidValue(i, j))
				{
					float v = image(i,j)/4.0f; // 4 meter is max
					math::clamp<float>(v);

					color.rgbRed = (BYTE) (255.0f*v);
					color.rgbGreen = (BYTE) (255.0f*v);
					color.rgbBlue = (BYTE) (255.0f*v);
				
					FreeImage_SetPixelColor(dib, width-j-1, height-i-1, &color);
				}
				else
				{
					color.rgbRed = 0;
					color.rgbGreen = 0;
					color.rgbBlue = 0;
				}
			}
		}

		FreeImage_Save(FIF_PNG, dib, filename.c_str());

		FreeImage_DeInitialise();
	}

	static void saveDepthImageToFilePNGFloat(const std::string &filename, const DepthImage& image)
	{
		FreeImage_Initialise();

		unsigned int width = image.getWidth();
		unsigned int height = image.getHeight();

		FIBITMAP *dib = FreeImage_AllocateT(FIT_FLOAT, width, height, 32);
		BYTE* bits = FreeImage_GetBits(dib);

		for (int i = 0; i < (int)height; i++)
		{
			for (int j = 0; j < (int)width; j++)
			{					
				float* data = (float*)bits;
				data[i*width + j] = image(i,j);
			}
		}

		FreeImage_Save(FIF_PNG, dib, filename.c_str());

		FreeImage_DeInitialise();
	}

	static void saveColorImageToFilePNG(const std::string &filename, const ColorImageRGB& image)
	{
		FreeImage_Initialise();

		unsigned int width = image.getWidth();
		unsigned int height = image.getHeight();

		FIBITMAP *dib = FreeImage_Allocate(width, height, 24);
		RGBQUAD color;

		for (int i = 0; i < (int)height; i++)
		{
			for (int j = 0; j < (int)width; j++)
			{
				vec3f v = image(i, j);
		
				color.rgbRed = (BYTE) (255.0f*v.x);
				color.rgbGreen = (BYTE) (255.0f*v.y);
				color.rgbBlue = (BYTE) (255.0f*v.z);

				FreeImage_SetPixelColor(dib, j, i, &color);
			}
		}

		FreeImage_Save(FIF_PNG, dib, filename.c_str());

		FreeImage_DeInitialise();
	}
	*/
private:

};

} // end namespace

#endif
