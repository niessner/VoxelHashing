#pragma once

#include "DSAPITypes.h"

/// @defgroup Image Conversion Functions
/// For all image conversion functions, the parameters have the following meaning:
/// sourceImage - pointer to the start of pixel data for the image to convert from
/// width - number of columns of pixels in the source and destination images
/// height - number of rows of pixels in the source and destination images
/// stride - number of bytes from the start of one row to the start of the next row in the source image
/// shift - when converting 12 or 16 bit data to 8 bit, indicates how many bits we should right shift the pixel data before writing it out
/// destImage[out] - the buffer to receive the converted image
/// leftImage[out] - when converting from an interleaved stereo format, the buffer to receive the converted left image
/// rightImage[out] - when converting from an interleaved stereo format, the buffer to receive the converted right image
/// @{

DS_DECL void DSConvertRLLuminance8ToLuminance8(const void* sourceImage, int width, int height, uint8_t* leftImage, uint8_t* rightImage);
DS_DECL void DSConvertRLLuminance8ToLuminance8(const void* sourceImage, int width, int height, int stride, uint8_t* leftImage, uint8_t* rightImage);

DS_DECL void DSConvertRLLuminance12ToLuminance8(const void* sourceImage, int width, int height, int shift, uint8_t* leftImage, uint8_t* rightImage);
DS_DECL void DSConvertRLLuminance12ToLuminance8(const void* sourceImage, int width, int height, int stride, int shift, uint8_t* leftImage, uint8_t* rightImage);
DS_DECL void DSConvertRLLuminance12ToLuminance16(const void* sourceImage, int width, int height, uint16_t* leftImage, uint16_t* rightImage);
DS_DECL void DSConvertRLLuminance12ToLuminance16(const void* sourceImage, int width, int height, int stride, uint16_t* leftImage, uint16_t* rightImage);

DS_DECL void DSConvertRLLuminance16ToLuminance8(const void* sourceImage, int width, int height, int shift, uint8_t* leftImage, uint8_t* rightImage);
DS_DECL void DSConvertRLLuminance16ToLuminance8(const void* sourceImage, int width, int height, int stride, int shift, uint8_t* leftImage, uint8_t* rightImage);
DS_DECL void DSConvertRLLuminance16ToLuminance16(const void* sourceImage, int width, int height, uint16_t* leftImage, uint16_t* rightImage);
DS_DECL void DSConvertRLLuminance16ToLuminance16(const void* sourceImage, int width, int height, int stride, uint16_t* leftImage, uint16_t* rightImage);

DS_DECL void DSConvertYUY2ToRGB8(const void* sourceImage, int width, int height, uint8_t* destImage);
DS_DECL void DSConvertYUY2ToBGRA8(const void* sourceImage, int width, int height, uint8_t* destImage);

DS_DECL void DSConvertRaw10ToRGB8(const void* sourceImage, int width, int height, uint8_t* destImage);
DS_DECL void DSConvertRaw10ToBGRA8(const void* sourceImage, int width, int height, uint8_t* destImage);

/// @}