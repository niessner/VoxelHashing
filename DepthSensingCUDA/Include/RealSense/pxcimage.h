/*******************************************************************************

INTEL CORPORATION PROPRIETARY INFORMATION
This software is supplied under the terms of a license agreement or nondisclosure
agreement with Intel Corporation and may not be copied or disclosed except in
accordance with the terms of that agreement
Copyright(c) 2011-2014 Intel Corporation. All Rights Reserved.

*******************************************************************************/
/** @file pxcimage.h
Defines the PXCImage interface, which manages image buffer access.
*/
#pragma once
#include "pxcbase.h"
#include "pxcaddref.h"
#pragma warning(push)
#pragma warning(disable:4201) /* nameless structs/unions */

/**
This class defines a standard interface for image buffer access.

The interface extends PXCAddRef. Use QueryInstance<PXCAddRef>(), or the helper
function AddRef() to access the PXCAddRef features.

The interface extends PXCMetadata. Use QueryInstance<PXCMetadata>() to access 
the PXCMetadata features.
*/
class PXCImage:public PXCBase {
public:
    PXC_CUID_OVERWRITE(0x24740F76);
    PXC_DEFINE_CONST(NUM_OF_PLANES, 4);
	PXC_DEFINE_CONST(METADATA_DEVICE_PROPERTIES,0x61516733);
	PXC_DEFINE_CONST(METADATA_DEVICE_PROJECTION,0x3546785a);

    /** 
    @enum PixelFormat
    Describes the image sample pixel format
    */
    enum PixelFormat {
        PIXEL_FORMAT_ANY=0,                     /* Unknown/undefined */

        /* STREAM_TYPE_COLOR */
        PIXEL_FORMAT_YUY2 = 0x00010000,         /* YUY2 image  */
        PIXEL_FORMAT_NV12,                      /* NV12 image */
        PIXEL_FORMAT_RGB32,                     /* BGRA layout on a little-endian machine */
        PIXEL_FORMAT_RGB24,                     /* BGR layout on a little-endian machine */
        PIXEL_FORMAT_Y8,                        /* 8-Bit Gray Image, or IR 8-bit */

        /* STREAM_TYPE_DEPTH */
        PIXEL_FORMAT_DEPTH = 0x00020000,        /* 16-bit unsigned integer with precision mm. */
        PIXEL_FORMAT_DEPTH_RAW,                 /* 16-bit unsigned integer with device specific precision (call device->QueryDepthUnit()) */
        PIXEL_FORMAT_DEPTH_F32,                 /* 32-bit float-point with precision mm. */

        /* STREAM_TYPE_IR */
        PIXEL_FORMAT_Y16 = 0x00040000,          /* 16-Bit Gray Image */
		PIXEL_FORMAT_Y8_IR_RELATIVE= 0x00080000    /* Relative IR Image */
    };

    /** 
    @brief Convert pixel format to a string representation
    @param[in]  format        pixel format.
    @return string presentation.
    */
    __inline static const pxcCHAR *PixelFormatToString(PixelFormat format) {
        switch (format) {
        case PIXEL_FORMAT_RGB24:        return (const pxcCHAR*)L"RGB24";
        case PIXEL_FORMAT_RGB32:        return (const pxcCHAR*)L"RGB32";
        case PIXEL_FORMAT_YUY2:         return (const pxcCHAR*)L"YUY2";
        case PIXEL_FORMAT_NV12:         return (const pxcCHAR*)L"NV12";
        case PIXEL_FORMAT_Y8:           return (const pxcCHAR*)L"Y8";
		case PIXEL_FORMAT_Y8_IR_RELATIVE:  return (const pxcCHAR*)L"Y8_REL";
        case PIXEL_FORMAT_Y16:          return (const pxcCHAR*)L"Y16";
        case PIXEL_FORMAT_DEPTH:        return (const pxcCHAR*)L"DEPTH";
        case PIXEL_FORMAT_DEPTH_F32:    return (const pxcCHAR*)L"DEPTH(FLOAT)";
        case PIXEL_FORMAT_DEPTH_RAW:    return (const pxcCHAR*)L"DEPTH(NATIVE)";
        }
        return (const pxcCHAR*)L"Unknown";
    }

    /** 
    @struct ImageInfo
    Describes the image sample detailed information.
    */
    struct ImageInfo {
        pxcI32      width;              /* width of the image in pixels */
        pxcI32      height;             /* height of the image in pixels */
        PixelFormat format;             /* image pixel format */
        pxcI32      reserved;
    };

    /** 
    @struct ImageData
    Describes the image storage details.
    */
    struct ImageData {
        PixelFormat     format;                     /* image pixel format */
        pxcI32          reserved[3];
        pxcI32          pitches[NUM_OF_PLANES];     /* image pitches */
        pxcBYTE*        planes[NUM_OF_PLANES];      /* image buffers */
    };

    /** 
    @enum Access
    Describes the image access mode.
    */
    enum Access {
        ACCESS_READ         = 1,                            /* read only access     */
        ACCESS_WRITE        = 2,                            /* write only access    */
        ACCESS_READ_WRITE   = ACCESS_READ | ACCESS_WRITE,   /* read write access    */
    };

    /** 
    @enum Option
    Describes the image options.
    */
    enum Option {
        OPTION_ANY          = 0,
    };

    /** 
    @brief Return the image sample information.
    @return the image sample information in the ImageInfo structure.
    */
    virtual ImageInfo PXCAPI QueryInfo(void)=0;


    virtual pxcI64 PXCAPI QueryTimeStamp(void)=0;

    /** 
    @brief Return the image stream type. The application should cast the
	returned type to PXCCapture::StreamType.
    @return the stream type.
    */
    virtual pxcEnum PXCAPI QueryStreamType(void)=0;

    /** 
    @brief Get the image option flags.
    @return the option flags.
    */
    virtual Option PXCAPI QueryOptions(void)=0;

    /** 
    @brief Set the sample time stamp.
    @param[in] ts        The time stamp value, in 100ns.
    */
    virtual void PXCAPI SetTimeStamp(pxcI64 ts)=0;

    /** 
    @brief Set the sample stream type.
    @param[in] streamType    The sample stream type.
    */
    virtual void PXCAPI SetStreamType(pxcEnum streamType)=0;

    /** 
    @brief Set the sample options. This function overrides any previously set options.
    @param[in] options      The image options.
    */
    virtual void PXCAPI SetOptions(Option options)=0;

    /**    
    @brief Copy image data from another image sample.
    @param[in] src_image        The image sample to copy data from.
    @return PXC_STATUS_NO_ERROR     Successful execution.
    */
    virtual pxcStatus PXCAPI CopyImage(PXCImage *src_image)=0;

    /** 
    @brief Copy image data to the specified external buffer.
    @param[in] data             The ImageData structure that describes the image buffer.
    @param[in] flags            Reserved.
    @return PXC_STATUS_NO_ERROR     Successful execution.
    */
    virtual pxcStatus PXCAPI ExportData(ImageData *data, pxcEnum flags)=0;
    __inline pxcStatus PXCAPI ExportData(ImageData *data) { 
        return ExportData(data, 0); 
    }

    /** 
    @brief Copy image data from the specified external buffer.
    @param[in] data             The ImageData structure that describes the image buffer.
    @param[in] flags            Reserved.
    @return PXC_STATUS_NO_ERROR     Successful execution.
    */
    virtual pxcStatus PXCAPI ImportData(ImageData *data, pxcEnum flags)=0;
    __inline pxcStatus PXCAPI ImportData(ImageData *data) { 
        return ImportData(data, 0); 
    }

    /** 
    @brief Lock to access the internal storage of a specified format. The function will perform format conversion if unmatched. 
    @param[in] access           The access mode.
    @param[in] format           The requested smaple format.
    @param[in] options          The option flags.
    @param[out] data            The sample data storage, to be returned.
    @return PXC_STATUS_NO_ERROR    Successful execution.
    */
    virtual pxcStatus PXCAPI AcquireAccess(Access access, PixelFormat format, Option options, ImageData *data)=0;

    /** 
    @brief Lock to access the internal storage of a specified format. The function will perform format conversion if unmatched. 
    @param[in] access           The access mode.
    @param[in] format           The requested smaple format.
    @param[out] data            The sample data storage, to be returned.
    @return PXC_STATUS_NO_ERROR    Successful execution.
    */
    __inline pxcStatus AcquireAccess(Access access, PixelFormat format, ImageData *data) { 
        return AcquireAccess(access, format, OPTION_ANY, data); 
    }

    /** 
    @brief Lock to access the internal storage of a specified format. 
    @param[in] access           The access mode.
    @param[out] data            The sample data storage, to be returned.
    @return PXC_STATUS_NO_ERROR    Successful execution.
    */
    __inline pxcStatus AcquireAccess(Access access, ImageData *data) { 
        return AcquireAccess(access, PIXEL_FORMAT_ANY, OPTION_ANY, data); 
    }

    /** 
    @brief Unlock the previously acquired buffer.
    @param[in] data             The sample data storage previously acquired.
    @return PXC_STATUS_NO_ERROR    Successful execution.
    */
    virtual pxcStatus   PXCAPI  ReleaseAccess(ImageData *data)=0;

    /** 
    @brief Increase a reference count of the sample.
    */
    __inline void AddRef(void) {
        QueryInstance<PXCAddRef>()->AddRef();
    }
};

/** 
A helper function for bitwise OR of two flags.
*/
__inline static PXCImage::Option operator | (PXCImage::Option a, PXCImage::Option b) {
    return (PXCImage::Option)((int)a|(int)b);
}

#pragma warning(pop)
