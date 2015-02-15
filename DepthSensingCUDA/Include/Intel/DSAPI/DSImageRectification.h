#pragma once

#include "DSAPITypes.h"
#include "DSAPI/DSCalibRectParameters.h"

/// @defgroup Image Rectification Functions
/// Image rectification done in software. Given a set of calibration/rectification parameters, first build a table. Then re-use the table for each 
/// rectification call for as long as the calibration/rectification parameters are valid.

/// For each pixel in the destination image, compute pixel location in the source image from which sampling should be done and store this in table
/// where each location is encoded by 32 bits - 11 bits row, 5 bits row fraction, 11 bits column, 5 bits column fraction
DS_DECL void DSRectificationTable(const DSCalibIntrinsicsNonRectified& sourceIntrinsics, const double rotation[9], const DSCalibIntrinsicsRectified& destIntrinsics, uint32_t* table);

DS_DECL void DSRectifyRGB8ToRGB8(const uint32_t* table, const void* sourceImage, int sourceWidth, int destWidth, int destHeight, uint8_t* destImage);
DS_DECL void DSRectifyBGRA8ToBGRA8(const uint32_t* table, const void* sourceImage, int sourceWidth, int destWidth, int destHeight, uint8_t* destImage);

/// @}
