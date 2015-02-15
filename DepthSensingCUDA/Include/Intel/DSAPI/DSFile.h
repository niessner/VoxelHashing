#pragma once

#include "DSAPITypes.h"

/// @class DSFile
/// Defines methods used only in file mode.
class DS_DECL DSFile
{
public:
    /// Sets the parameters necessary for providing input images from files.
    /// The path provided is the root image sequence directory and should contain by default
    /// Left, Right, Depth, Color subdirectories and a CameraParameters.ini file describing
    /// the parameters at the time of capture. File names of the images within the subdirectories
    /// should consist of a sequence number and a .pgm (8 bit images) or .psm (16 bit images)
    /// suffix. This is the default format output.
    /// @param path the root path for the image files
    /// @param firstImageNumber the number of the first image in the sequence (default of -1 will find this automatically)
    /// @param nrImages the total number of images to read in the sequence (default of -1 will read all available images)
    virtual void setReadFileModeParams(const char* path, int firstImageNumber = -1, int nrImages = -1) = 0;
    /// Gets the data path, first image, and number of images for reading images from files.
    /// @return the data path, first image, and number of images for reading images from files.
    /// @param path set to the root path for the image files
    /// @param[out] firstImage set to the first image number
    /// @param[out] nImages set to the number of images to read
    virtual const char* getReadFileModeParams(int& firstImageNumber, int& nrImages) = 0;

    /// Returns true if data from left imager is recorded
    virtual bool isLeftRecorded() = 0;
    /// Returns true if data from right imager is recorded
    virtual bool isRightRecorded() = 0;
    /// Returns true if depth data is recorded
    virtual bool isDepthRecorded() = 0;
    /// Returns true if data from third imager is recorded
    virtual bool isThirdRecorded() = 0;

    /// Returns true if the image data stored are rectified.
    virtual bool areRecordedImagesRectified() = 0;

    /// Sets file reading to loop back to the first image after the last image is reached.
    /// Default is no looping. nImages must be set for looping to work.
    virtual void enableReadFileModeLoop(bool state) = 0;
    /// Returns true if file mode looping is enabled.
    virtual bool isReadFileModeLoopEnabled() = 0;

    /// Set the next read file's sequence image number.
    virtual void setNextReadFileImageNumber(int imageNumber) = 0;
    /// Returns the sequence number of the image just read from file.
    virtual int getReadFileImageNumber() = 0;

protected:
    // Creation (and deletion) of an object of this
    // type is supported through the DSFactory functions.
    DSFile() {};
    DSFile(const DSFile& other) DS_DELETED_FUNCTION;
    DSFile(DSFile&& other) DS_DELETED_FUNCTION;
    DSFile& operator=(const DSFile& other) DS_DELETED_FUNCTION;
    DSFile& operator=(DSFile&& other) DS_DELETED_FUNCTION;
    virtual ~DSFile() {};
};
