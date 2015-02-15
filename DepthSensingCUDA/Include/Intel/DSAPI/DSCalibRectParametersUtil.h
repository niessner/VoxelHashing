#pragma once

#include "DSCalibRectParameters.h"
#define _USE_MATH_DEFINES
#include <cmath>
#undef _USE_MATH_DEFINES
 
/// @defgroup Helper functions for calibration parameters
/// Inline helper functions that show how to use the calibration data that are obtained via the DSAPI interface.
/// @{

/// From z image to z camera (right-handed coordinate system).
/// zImage is assumed to contain [z row, z column, z depth].
/// Get zIntrinsics via DSAPI getCalibIntrinsicsZ.
inline void DSTransformFromZImageToZCamera(const DSCalibIntrinsicsRectified& zIntrinsics, const float zImage[3], float zCamera[3])
{
    zCamera[0] = zImage[2] * (zImage[0] - zIntrinsics.rpx) / zIntrinsics.rfx;
    zCamera[1] = zImage[2] * (zImage[1] - zIntrinsics.rpy) / zIntrinsics.rfy;
    zCamera[2] = zImage[2];
}

/// From z camera to rectified third camera
/// Get translation via DSAPI getCalibExtrinsicsZToRectThird.
inline void DSTransformFromZCameraToRectThirdCamera(const double translation[3], const float zCamera[3], float thirdCamera[3])
{
    thirdCamera[0] = static_cast<float>(zCamera[0] + translation[0]);
    thirdCamera[1] = static_cast<float>(zCamera[1] + translation[1]);
    thirdCamera[2] = static_cast<float>(zCamera[2] + translation[2]);
}

/// From z camera to non-rectified third camera
/// Get rotation and translation via DSAPI getCalibExtrinsicsZToNonRectThird.
inline void DSTransformFromZCameraToNonRectThirdCamera(const double rotation[9], const double translation[3], const float zCamera[3], float thirdCamera[3])
{
    thirdCamera[0] = static_cast<float>(rotation[0] * zCamera[0] + rotation[1] * zCamera[1] + rotation[2] * zCamera[2] + translation[0]);
    thirdCamera[1] = static_cast<float>(rotation[3] * zCamera[0] + rotation[4] * zCamera[1] + rotation[5] * zCamera[2] + translation[1]);
    thirdCamera[2] = static_cast<float>(rotation[6] * zCamera[0] + rotation[7] * zCamera[1] + rotation[8] * zCamera[2] + translation[2]);
}

/// From third camera to rectified third image
/// Get thirdIntrinsics via DSAPI getCalibIntrinsicsRectThird.
inline void DSTransformFromThirdCameraToRectThirdImage(const DSCalibIntrinsicsRectified& thirdIntrinsics, const float thirdCamera[3], float thirdImage[2])
{
    thirdImage[0] = thirdCamera[0] / thirdCamera[2];
    thirdImage[1] = thirdCamera[1] / thirdCamera[2];

    thirdImage[0] = thirdIntrinsics.rfx * thirdImage[0] + thirdIntrinsics.rpx;
    thirdImage[1] = thirdIntrinsics.rfy * thirdImage[1] + thirdIntrinsics.rpy;
}

/// From third camera to non-rectified third image
/// Get thirdIntrinsics via DSAPI getCalibIntrinsicsNonRectThird.
inline void DSTransformFromThirdCameraToNonRectThirdImage(const DSCalibIntrinsicsNonRectified& thirdIntrinsics, const float thirdCamera[3], float thirdImage[2])
{
    float t[2];
    t[0] = thirdCamera[0] / thirdCamera[2];
    t[1] = thirdCamera[1] / thirdCamera[2];

    const double* k = thirdIntrinsics.k;
    float r2 = t[0] * t[0] + t[1] * t[1];
    float r = static_cast<float>(1 + r2 * (k[0]  + r2 * (k[1] + r2 * k[4])));
    t[0] *= r;
    t[1] *= r;
    
    thirdImage[0] = static_cast<float>(t[0] + 2 * k[2] * t[0] * t[1] + k[3] * (r2 + 2 * t[0] * t[0]));
    thirdImage[1] = static_cast<float>(t[1] + 2 * k[3] * t[0] * t[1] + k[2] * (r2 + 2 * t[1] * t[1]));

    thirdImage[0] = thirdIntrinsics.fx * thirdImage[0] + thirdIntrinsics.px;
    thirdImage[1] = thirdIntrinsics.fy * thirdImage[1] + thirdIntrinsics.py;
}

/// From z image to rectified third image
/// Get zIntrinsics via DSAPI getCalibIntrinsicsZ.
/// Get translation via DSAPI getCalibExtrinsicsZToRectThird.
/// Get thirdIntrinsics via DSAPI getCalibIntrinsicsRectThird.
inline void DSTransformFromZImageToRectThirdImage(const DSCalibIntrinsicsRectified& zIntrinsics, const double translation[3],
    const DSCalibIntrinsicsRectified& thirdIntrinsics, const float zImage[3], float thirdImage[2])
{
    float zCamera[3];
    float thirdCamera[3];
    DSTransformFromZImageToZCamera(zIntrinsics, zImage, zCamera);
    DSTransformFromZCameraToRectThirdCamera(translation, zCamera, thirdCamera);
    DSTransformFromThirdCameraToRectThirdImage(thirdIntrinsics, thirdCamera, thirdImage);
}

/// From z image to non-rectified third image
/// Get zIntrinsics via DSAPI getCalibIntrinsicsZ.
/// Get rotation and translation via DSAPI getCalibExtrinsicsZToNonRectThird.
/// Get thirdIntrinsics via DSAPI getCalibIntrinsicsNonRectThird.
inline void DSTransformFromZImageToNonRectThirdImage(const DSCalibIntrinsicsRectified& zIntrinsics, const double rotation[9], const double translation[3],
    const DSCalibIntrinsicsNonRectified& thirdIntrinsics, const float zImage[3], float thirdImage[2])
{
    float zCamera[3];
    float thirdCamera[3];
    DSTransformFromZImageToZCamera(zIntrinsics, zImage, zCamera);
    DSTransformFromZCameraToNonRectThirdCamera(rotation, translation, zCamera, thirdCamera);
    DSTransformFromThirdCameraToNonRectThirdImage(thirdIntrinsics, thirdCamera, thirdImage);
}

/// From rect third image to non-rectified third image
/// Get thirdIntrinsicsRect via DSAPI getCalibIntrinsicsRectThird.
/// Get rotation via DSAPI getCalibExtrinsicsRectThirdToNonRectThird.
/// Get thirdIntrinsicsNonRect via DSAPI getCalibIntrinsicsNonRectThird.
inline void DSTransformFromRectThirdImageToNonRectThirdImage(const DSCalibIntrinsicsRectified& thirdIntrinsicsRect, const double rotation[9],
    const DSCalibIntrinsicsNonRectified& thirdIntrinsicsNonRect, const float rectImage[2], float nonRectImage[2])
{
    float rectCamera[3];
    rectCamera[0] = (rectImage[0] - thirdIntrinsicsRect.rpx) / thirdIntrinsicsRect.rfx;
    rectCamera[1] = (rectImage[1] - thirdIntrinsicsRect.rpy) / thirdIntrinsicsRect.rfy;
    rectCamera[2] = 1;

    float nonRectCamera[3];
    nonRectCamera[0] = static_cast<float>(rotation[0] * rectCamera[0] + rotation[1] * rectCamera[1] + rotation[2] * rectCamera[2]);
    nonRectCamera[1] = static_cast<float>(rotation[3] * rectCamera[0] + rotation[4] * rectCamera[1] + rotation[5] * rectCamera[2]);
    nonRectCamera[2] = static_cast<float>(rotation[6] * rectCamera[0] + rotation[7] * rectCamera[1] + rotation[8] * rectCamera[2]);

    DSTransformFromThirdCameraToNonRectThirdImage(thirdIntrinsicsNonRect, nonRectCamera, nonRectImage);
}

/// From z camera to world.
/// Get rotation and translation via DSAPI getCalibZToWorldTransform.
inline void DSTransformFromZCameraToWorld(const double rotation[9], const double translation[3], const float zCamera[3], float world[3])
{
    world[0] = static_cast<float>(rotation[0] * zCamera[0] + rotation[1] * zCamera[1] + rotation[2] * zCamera[2] + translation[0]);
    world[1] = static_cast<float>(rotation[3] * zCamera[0] + rotation[4] * zCamera[1] + rotation[5] * zCamera[2] + translation[1]);
    world[2] = static_cast<float>(rotation[6] * zCamera[0] + rotation[7] * zCamera[1] + rotation[8] * zCamera[2] + translation[2]);
}

/// Compute field of view angles in degrees from rectified intrinsics
/// Get intrinsics via DSAPI getCalibIntrinsicsZ, getCalibIntrinsicsRectLeftRight or getCalibIntrinsicsRectThird
inline void DSFieldOfViewsFromIntrinsicsRect(const DSCalibIntrinsicsRectified& intrinsics, float& horizontalFOV, float& verticalFOV)
{
    horizontalFOV = atan2(intrinsics.rpx + 0.5f, intrinsics.rfx) + atan2(intrinsics.rw - intrinsics.rpx - 0.5f, intrinsics.rfx);
    verticalFOV = atan2(intrinsics.rpy + 0.5f, intrinsics.rfy) + atan2(intrinsics.rh - intrinsics.rpy - 0.5f, intrinsics.rfy);

    // Convert to degrees
    horizontalFOV = horizontalFOV * 180.0f / static_cast<float>(math::PIf);
    verticalFOV = verticalFOV * 180.0f / static_cast<float>(math::PIf);
}

/// @}