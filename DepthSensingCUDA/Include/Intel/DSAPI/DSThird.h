#pragma once

#include "DSAPITypes.h"
#include "DSAPI/DSCalibRectParameters.h"

/// @class DSThird
/// Defines methods specific to an implementation that has a third camera.
class DSThird
{
public:
    /// Get number supported resolution modes for third (includes pairings with framerates)
    virtual int getThirdNumberOfResolutionModes(bool rectified) = 0;
    /// For each index = 0 to getLRZNumberOfResolutionModes() - 1 will get the third width, third height and framerate.
    virtual bool getThirdResolutionMode(bool rectified, int index, int& thirdWidth, int& thirdHeight, int& thirdFps, DSPixelFormat& thirdPixelFormat) = 0;
    /// Set third width, third height and framerate.
    virtual bool setThirdResolutionMode(bool rectified, int thirdWidth, int thirdHeight, int thirdFps, DSPixelFormat thirdPixelFormat) = 0;

    /// Get current frame rate.
    virtual uint32_t getThirdFramerate() = 0;

    /// Returns time stamp of the current third image data.
    /// The double is representing time in seconds since 00:00 Coordinated Universal Time (UTC), January 1, 1970.
    virtual double getThirdFrameTime() = 0;

    /// Gets the current frame number for third. Currently not really a frame number, but will change for consecutive frames.
    virtual int getThirdFrameNumber() = 0; 

    /// Returns the pixel format of the third imager.
    virtual DSPixelFormat getThirdPixelFormat() = 0;
    /// Returns true if pixel format is a native format for the third imager.
    virtual bool isThirdPixelFormatNative(DSPixelFormat pixelFormat) = 0;

    /// Returns true if rectification is enabled for the third imager.
    virtual bool isThirdRectificationEnabled() = 0;

    /// Enables the capture of images from the third imager.
    virtual bool enableThird(bool state) = 0;
    /// Returns true if capture of images from third imager is enabled.
    virtual bool isThirdEnabled() = 0;

    /// Returns a pointer to the image data from the third imager.
    virtual void* getThirdImage() = 0;

    /// Gets width of third image.
    virtual int thirdWidth() = 0;
    /// Gets height of third image.
    virtual int thirdHeight() = 0;

    /// Get calibration parameters for currently selected rectified mode
    virtual bool getCalibIntrinsicsRectThird(DSCalibIntrinsicsRectified& intrinsics) = 0;
    virtual bool getCalibExtrinsicsZToRectThird(double translation[3]) = 0;

    /// Get calibration parameters for currently selected non-rectified mode
    virtual bool getCalibIntrinsicsNonRectThird(DSCalibIntrinsicsNonRectified& intrinsics) = 0;
    virtual bool getCalibExtrinsicsZToNonRectThird(double rotation[9], double translation[3]) = 0;

    /// Get the rotation from the coordinate system of rectified third to the coordinate system 
    /// of non-rectified third, defined such that Xnonrect = rotation * Xrect
    virtual bool getCalibExtrinsicsRectThirdToNonRectThird(double rotation[9]) = 0;

protected:
    // Creation (and deletion) of an object of this
    // type is supported through the DSFactory functions.
    DSThird() {};
    DSThird(const DSThird& other) DS_DELETED_FUNCTION;
    DSThird(DSThird&& other) DS_DELETED_FUNCTION;
    DSThird& operator=(const DSThird& other) DS_DELETED_FUNCTION;
    DSThird& operator=(DSThird&& other) DS_DELETED_FUNCTION;
    virtual ~DSThird() {};
};
