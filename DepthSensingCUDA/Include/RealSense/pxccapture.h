                                                                                                                                                                                                                            /*******************************************************************************

INTEL CORPORATION PROPRIETARY INFORMATION
This software is supplied under the terms of a license agreement or nondisclosure
agreement with Intel Corporation and may not be copied or disclosed except in
accordance with the terms of that agreement
Copyright(c) 2011-2014 Intel Corporation. All Rights Reserved.

*******************************************************************************/
/** @file pxccapture.h
    Defines the PXCCapture interface, which allows a program create and interact
    with video streams. 
 */
#pragma once
#include "pxcimage.h"
#include "pxcsyncpoint.h"
#pragma warning(push)
#pragma warning(disable:4351) /* new behavior of array initialization */
class PXCCaptureDeviceExt;
class PXCProjection;

/**
   This interface provides member functions to create instances of and query stream capture devices.
*/
class PXCCapture:public PXCBase {
public:
    PXC_CUID_OVERWRITE(0x83F72A50);
    PXC_DEFINE_CONST(STREAM_LIMIT,8);
    class Device;

    /** 
        @enum StreamType
        Bit-OR'ed values of stream types, physical or virtual streams.
    */
    enum StreamType {
        STREAM_TYPE_ANY   = 0,          /* Unknown/undefined type */
        STREAM_TYPE_COLOR = 0x0001,     /* the color stream type  */
        STREAM_TYPE_DEPTH = 0x0002,     /* the depth stream type  */
        STREAM_TYPE_IR    = 0x0004,     /* the infrared stream type */
        STREAM_TYPE_LEFT  = 0x0008,     /* the stereoscopic left intensity image */
        STREAM_TYPE_RIGHT = 0x0010,     /* the stereoscopic right intensity image */
    };

	/** 
		@brief Get the stream type string representation
		@param[in] type		The stream type
		@return The corresponding string representation.
	**/
    __inline static const pxcCHAR *StreamTypeToString(StreamType type) {
        switch (type) {
        case STREAM_TYPE_COLOR: return L"Color";
        case STREAM_TYPE_DEPTH: return L"Depth";
        case STREAM_TYPE_IR:    return L"IR";
        case STREAM_TYPE_LEFT:  return L"Left";
        case STREAM_TYPE_RIGHT: return L"Right";
        }
        return L"Unknown";
    }

	/** 
		@brief Get the stream type from an index number
		@param[in] index		The stream index
		@return The corresponding stream type.
	**/
    __inline static StreamType StreamTypeFromIndex(pxcI32 index) {
		if (index<0 || index>=STREAM_LIMIT) return STREAM_TYPE_ANY;
		return (StreamType)(1<<index);
    }

	/** 
		@brief Get the stream index number
		@param[in] StreamType	The stream type
		@return The stream index number.
	**/
    __inline static pxcI32 StreamTypeToIndex(StreamType type) {
		pxcI32 s=0;
		while (type>1) type=(StreamType)(type>>1), s++;
		return s;
    }

    /** 
        @enum DeviceModel
        Describes the device model
    */
    enum DeviceModel {
        DEVICE_MODEL_GENERIC    = 0x00000000,    /* a generic device or unknown device */
        DEVICE_MODEL_IVCAM      = 0x0020000E,    /* the Intel(R) RealSense(TM) 3D Camera */
		DEVICE_MODEL_DS4		= 0x0020000F,    /* the Intel(R) RealSense(TM) DS4 Camera */
    };

    /** 
        @enum DeviceOrientation
        Describes the device orientation
    */
    enum DeviceOrientation {
        DEVICE_ORIENTATION_ANY          = 0x0,  /* Unknown orientation */
        DEVICE_ORIENTATION_USER_FACING  = 0x1,  /* A user facing camera */
        DEVICE_ORIENTATION_WORLD_FACING = 0x2,  /* A world facing camera */
    };

    /** 
        Describe device details.
    */
    struct DeviceInfo {
        pxcCHAR             name[224];      /* device name */
        pxcCHAR             serial[32];     /* serial number */
        pxcCHAR             did[256];       /* device identifier or the device symbolic name */
        pxcI32              firmware[4];    /* firmware version: limit to four parts of numbers */
        PXCPointF32         location;       /* device location in mm from the left bottom of the display panel. */
        DeviceModel         model;          /* device model */
        DeviceOrientation   orientation;    /* device orientation */
        StreamType          streams;        /* bit-OR'ed value of device stream types. */
        pxcI32              didx;           /* device index */
        pxcI32              duid;           /* device unique identifier within the SDK session */
        pxcI32              reserved[13];

        /** 
            @brief Get the available stream numbers.
            @return the number of streams.
        */
        __inline pxcI32 QueryStreamNum(void) {
            pxcI32 nstreams=0;
            for (pxcI32 i=0,j=1;i<STREAM_LIMIT;i++,j<<=1)
                if (streams&j) nstreams++;
            return nstreams;
        }
    };

    /** 
        @brief    Return the number of devices.
        @return the number of available devices
    */
    virtual pxcI32 PXCAPI QueryDeviceNum(void)=0;

    /** 
        @brief Return the device information structure for a given device.
        @param[in]  didx                The zero-based device index.
        @param[out] dinfo               The pointer to the DeviceInfo structure, to be returned.
        @return PXC_STATUS_NO_ERROR            Successful execution.
        @return PXC_STATUS_ITEM_UNAVAILABLE    The specified index does not exist.
    */
    virtual pxcStatus PXCAPI QueryDeviceInfo(pxcI32 didx, DeviceInfo *dinfo)=0;

    /** 
        @brief    Activate the device and return the video device instance.
        @param[in] didx                        The zero-based device index
        @return The device instance.
    */
    virtual Device* PXCAPI CreateDevice(pxcI32 didx)=0;

    /** 
        @struct Sample
        The capture sample that contains multiple streams.
    */
    struct Sample {
        PXCImage *color;
        PXCImage *depth;
        PXCImage *ir;
        PXCImage *left;
        PXCImage *right;
        PXCImage *reserved[STREAM_LIMIT-5];

        /** 
            @brief Return the image element by StreamType.
            @param[in] type        The stream type.
            @return The image instance.
        */
        __inline PXCImage* &operator[](StreamType type) {
            if (type==STREAM_TYPE_COLOR)    return color;
            if (type==STREAM_TYPE_DEPTH)    return depth;
            if (type==STREAM_TYPE_IR)       return ir;
            if (type==STREAM_TYPE_LEFT)     return left;
            if (type==STREAM_TYPE_RIGHT)    return right;
            for (int i=sizeof(reserved)/sizeof(reserved[0])-1,j=(1<<(STREAM_LIMIT-1));i>=0;i--,j>>=1)
                if (type&j) return reserved[i];
            return reserved[sizeof(reserved)/sizeof(reserved[0])-1];
        }

        /** 
            @brief Release the sample elements if not NULL
        */
        __inline void ReleaseImages(void) {
            if (color) color->Release(), color=0;
            if (depth) depth->Release(), depth=0;
            if (ir) ir->Release(), ir=0;
            if (left) left->Release(), left=0;
            if (right) right->Release(), right=0;
            for (int i=0;i<sizeof(reserved)/sizeof(reserved[0]);i++)
                if (reserved[i]) reserved[i]->Release(), reserved[i]=0;
        }

        /** 
            @brief The constructor zeros the image instance array
        */
        __inline Sample(void):color(0),depth(0),ir(0),left(0),right(0),reserved() {
        }
    };

    /**
        This is the video device interface.
        Use the member functions to interface with the video capture device.
    */
    class Device : public PXCBase {
        friend class PXCCaptureDeviceExt;
    public:

        PXC_CUID_OVERWRITE(0x938401C4);

        /** 
            @enum PowerLineFrequency
            Describes the power line compensation filter values.
        */
        enum PowerLineFrequency {
            POWER_LINE_FREQUENCY_DISABLED       =   0,          /* Disabled power line frequency */
            POWER_LINE_FREQUENCY_50HZ           =   1,         /* 50HZ power line frequency */
            POWER_LINE_FREQUENCY_60HZ           =   2,         /* 60HZ power line frequency */
        };

        /**
            @enum MirrorMode
            Describes the mirroring options.
        */
        enum MirrorMode {
            MIRROR_MODE_DISABLED                =   0,          /* Disabled. The images are displayed as in a world facing camera.  */
            MIRROR_MODE_HORIZONTAL              =   1,          /* The images are horizontally mirrored as in a user facing camera. */
        };

        /**
            @enum IVCAMAccuracy
            Describes the IVCAM accuracy.
        */
        enum IVCAMAccuracy {
            IVCAM_ACCURACY_FINEST                =   1,         /* The finest accuracy: 9 patterns */
            IVCAM_ACCURACY_MEDIAN                =   2,         /* The median accuracy: 8 patterns (default) */
            IVCAM_ACCURACY_COARSE                =   3,         /* The coarse accuracy: 7 patterns */
        };

        /** 
            @enum Property
            Describes the device properties.
            Use the inline functions to access specific device properties.
        */
        enum Property {
            /* Color Stream Properties */
            PROPERTY_COLOR_EXPOSURE             =   1,           /* pxcI32        RW    The color stream exposure, in log base 2 seconds. */
            PROPERTY_COLOR_BRIGHTNESS           =   2,           /* pxcI32        RW    The color stream brightness from  -10,000 (pure black) to 10,000 (pure white). */
            PROPERTY_COLOR_CONTRAST             =   3,           /* pxcI32        RW    The color stream contrast, from 0 to 10,000. */
            PROPERTY_COLOR_SATURATION           =   4,           /* pxcI32        RW    The color stream saturation, from 0 to 10,000. */
            PROPERTY_COLOR_HUE                  =   5,           /* pxcI32        RW    The color stream hue, from -180,000 to 180,000 (representing -180 to 180 degrees.) */
            PROPERTY_COLOR_GAMMA                =   6,           /* pxcI32        RW    The color stream gamma, from 1 to 500. */
            PROPERTY_COLOR_WHITE_BALANCE        =   7,           /* pxcI32        RW    The color stream balance, as a color temperature in degrees Kelvin. */
            PROPERTY_COLOR_SHARPNESS            =   8,           /* pxcI32        RW    The color stream sharpness, from 0 to 100. */
            PROPERTY_COLOR_BACK_LIGHT_COMPENSATION  =   9,       /* pxcBool       RW    The color stream back light compensation. */
            PROPERTY_COLOR_GAIN                     =   10,      /* pxcI32        RW    The color stream gain adjustment, with negative values darker, positive values brighter, and zero as normal. */
            PROPERTY_COLOR_POWER_LINE_FREQUENCY     =   11,      /* pxcI32        RW    The power line frequency in Hz. */
            PROPERTY_COLOR_FOCAL_LENGTH_MM      =   12,          /* pxcF32         R    The color-sensor focal length in mm. */
            PROPERTY_COLOR_FIELD_OF_VIEW        =   1000,        /* PXCPointF32    R    The color-sensor horizontal and vertical field of view parameters, in degrees. */
            PROPERTY_COLOR_FOCAL_LENGTH         =   1006,        /* PXCPointF32    R    The color-sensor focal length in pixels. The parameters vary with the resolution setting. */
            PROPERTY_COLOR_PRINCIPAL_POINT      =   1008,        /* PXCPointF32    R    The color-sensor principal point in pixels. The parameters vary with the resolution setting. */

            /* Depth Stream Properties */
            PROPERTY_DEPTH_LOW_CONFIDENCE_VALUE =   201,         /* pxcU16         R    The special depth map value to indicate that the corresponding depth map pixel is of low-confidence. */
            PROPERTY_DEPTH_CONFIDENCE_THRESHOLD =   202,         /* pxcI16        RW    The confidence threshold that is used to floor the depth map values. The range is from  0 to 15. */
            PROPERTY_DEPTH_UNIT                 =   204,         /* pxcI32         R    The unit of depth values in micrometer if PIXEL_FORMAT_DEPTH_RAW */
            PROPERTY_DEPTH_FOCAL_LENGTH_MM      =   205,         /* pxcF32         R    The depth-sensor focal length in mm. */
            PROPERTY_DEPTH_FIELD_OF_VIEW        =   2000,        /* PXCPointF32    R    The depth-sensor horizontal and vertical field of view parameters, in degrees. */
            PROPERTY_DEPTH_SENSOR_RANGE         =   2002,        /* PXCRangeF32    R    The depth-sensor, sensing distance parameters, in millimeters. */
            PROPERTY_DEPTH_FOCAL_LENGTH         =   2006,        /* PXCPointF32    R    The depth-sensor focal length in pixels. The parameters vary with the resolution setting. */
            PROPERTY_DEPTH_PRINCIPAL_POINT      =   2008,        /* PXCPointF32    R    The depth-sensor principal point in pixels. The parameters vary with the resolution setting. */

            /* Device Properties */
            PROPERTY_DEVICE_ALLOW_PROFILE_CHANGE =   302,        /* pxcBool       RW    If true, allow resolution change and throw PXC_STATUS_STREAM_CONFIG_CHANGED */
            PROPERTY_DEVICE_MIRROR               =   304,        /* MirrorMode    RW    The mirroring options. */

            /* Misc. Properties */
            PROPERTY_PROJECTION_SERIALIZABLE    =   3003,        /* pxcU32         R    The meta data identifier of the Projection instance serialization data. */

            /* Device Specific Properties */
            PROPERTY_IVCAM_LASER_POWER           = 0x10000,      /* pxcI32        RW    The laser power value from 0 (minimum) to 16 (maximum). */
            PROPERTY_IVCAM_ACCURACY              = 0x10001,      /* IVCAMAccuracy RW    The IVCAM accuracy value. */
			PROPERTY_IVCAM_FILTER_OPTION         = 0x10003,      /* pxcI32        RW    The filter option (smoothing aggressiveness) ranged from 0 (close range) to 7 (far range). */
			PROPERTY_IVCAM_MOTION_RANGE_TRADE_OFF= 0x10004,      /* pxcI32        RW    This option specifies the motion and range trade off. The value ranged from 0 (short exposure, range, and better motion) to 100 (long exposure, range). */

            /* Customized properties */
            PROPERTY_CUSTOMIZED=0x04000000,                        /* CUSTOMIZED properties */
        };

        /** 
            @brief Return the device information             
            @param[in] pointer to the DeviceInfo structure, to be returned.
        */
        virtual void      PXCAPI QueryDeviceInfo(DeviceInfo *dinfo)=0;

        /** 
            @brief Return the instance of the PXCProjection interface. Must be called after initialization.
             @return the PXCProjection instance.
        */
        virtual PXCProjection* PXCAPI CreateProjection(void)=0;

        /** 
            @structure StreamProfile
            Describes the video stream configuration parameters
        */
        struct StreamProfile {
            PXCImage::ImageInfo     imageInfo;        /* resolution and color format */
            PXCRangeF32             frameRate;        /* frame rate range. Set max when configuring FPS */
            pxcI32                  reserved[6];
        };

        /** 
            @structure StreamProfileSet
            The set of StreamProfile that describes the configuration parameters of all streams.
        */
        struct StreamProfileSet {
            StreamProfile color;
            StreamProfile depth;
            StreamProfile ir;
            StreamProfile left;
            StreamProfile right;
            StreamProfile reserved[STREAM_LIMIT-5];

            /**
                @brief Access the configuration parameters by the stream type.
                @param[in] type        The stream type.
                @return The StreamProfile instance.
            */
            __inline StreamProfile &operator[](StreamType type) {
                if (type==STREAM_TYPE_COLOR) return color;
                if (type==STREAM_TYPE_DEPTH) return depth;
                if (type==STREAM_TYPE_IR)    return ir;
                if (type==STREAM_TYPE_LEFT)  return left;
                if (type==STREAM_TYPE_RIGHT) return right;
                for (int i=sizeof(reserved)/sizeof(reserved[0])-1,j=(1<<(STREAM_LIMIT-1));i>=0;i--,j>>=1)
                    if (type&j) return reserved[i];
                return reserved[sizeof(reserved)/sizeof(reserved[0])-1];
            }
        };

		/** 
			@struct PropertyInfo
			Describe the property value range and attributes.
		*/
        struct PropertyInfo {
			PXCRangeF32 range;			/* The value range */
			pxcF32 step;			/* The value step */
			pxcF32 defaultValue;	/* Teh default value */
			pxcBool automatic;		/* Boolean if this property supports automatic */
			pxcI32 reserved[11];
		};

        /** 
            @brief Return the number of valid stream configurations for the streams of interest.
            @param[in] scope            The bit-OR'ed value of stream types of interest.
            @return the number of valid profile combinations.
        */
        virtual pxcI32 PXCAPI QueryStreamProfileSetNum(StreamType scope)=0;

        /** 
            @brief Return the unique configuration parameters for the selected video streams (types).
            @param[in] scope            The bit-OR'ed value of stream types of interest.
            @param[in] index            Zero-based index to retrieve all valid profiles.
            @param[out] profiles        The stream profile set.
            @return PXC_STATUS_NO_ERROR successful execution.
            @return PXC_STATUS_ITEM_UNAVAILABLE index out of range.
        */
        virtual pxcStatus PXCAPI QueryStreamProfileSet(StreamType scope, pxcI32 index, StreamProfileSet *profiles)=0;

        /** 
            @brief Return the active stream configuration parameters (during streaming).
            @param[out] profiles        The stream profile set, to be returned. 
            @return PXC_STATUS_NO_ERROR successful execution.
        */
        __inline pxcStatus QueryStreamProfileSet(StreamProfileSet *profiles) {
            return QueryStreamProfileSet(STREAM_TYPE_ANY, WORKING_PROFILE, profiles);
        }

        /** 
            @brief Check if stream profile set is valid.
            @param[in] profiles         The stream profile set to check
            @return true     stream profile is valid.
            @return false    stream profile is invalid.
        */
        virtual pxcBool PXCAPI IsStreamProfileSetValid (StreamProfileSet *profiles)=0;

        /** 
            @brief Set the active profile for the all video streams. The application must configure all streams before streaming.
            @param[in] profiles            The stream profile set. 
             @return PXC_STATUS_NO_ERROR successful execution.
        */
        virtual pxcStatus PXCAPI SetStreamProfileSet(StreamProfileSet *profiles)=0;

        /** 
            @brief Read the selected streams asynchronously. The function returns immediately. The application must
            synchronize sync point to get the stream samples. The application can read more than a single stream using
            the scope parameter, provided that all streams have the same frame rate. Otherwise, the function will return error.
            @param[in] scope                The bit-OR'ed value of stream types of interest.
            @param[in] sample               The output sample.
            @param[in] sp                   The pointer to the sync point to be returned.
            @return PXC_STATUS_NO_ERROR             successful execution.
            @return PXC_STATUS_DEVICE_LOST          the device is disconnected.
            @return PXC_STATUS_PARAM_UNSUPPORTED    the streams are of different frame rates.
        */
        virtual pxcStatus PXCAPI ReadStreamsAsync(StreamType scope, Sample *sample, PXCSyncPoint **sp)=0;

        /** 
            @brief Read the all configured streams asynchronously. The function returns immediately. The application must
            synchronize sync point to get the stream samples. The configured streams must have the same frame rate or 
            the function will return error.
            @param[in] sample               The output sample.
            @param[in] sp                   The pointer to the SP to be returned.
            @return PXC_STATUS_NO_ERROR             successful execution.
            @return PXC_STATUS_DEVICE_LOST          the device is disconnected.
            @return PXC_STATUS_PARAM_UNSUPPORTED    the streams are of different frame rates.
        */
        __inline pxcStatus ReadStreamsAsync(Sample *sample, PXCSyncPoint **sp) {
            return ReadStreamsAsync(STREAM_TYPE_ANY, sample, sp);
        }

        /** 
            @brief Read the selected streams synchronously. The function blocks until all stream samples are ready.
            The application can read more than a single stream using the scope parameter, provided that all streams 
            have the same frame rate. Otherwise, the function will return error.
            @param[in] scope                The bit-OR'ed value of stream types of interest.
            @param[in] sample               The output sample.
            @return PXC_STATUS_NO_ERROR             successful execution.
            @return PXC_STATUS_DEVICE_LOST          the device is disconnected.
            @return PXC_STATUS_PARAM_UNSUPPORTED    the streams are of different frame rates.
        */
        __inline pxcStatus PXCAPI ReadStreams(StreamType scope, Sample *sample) {
            PXCSyncPoint *sp;
            pxcStatus sts=ReadStreamsAsync(scope, sample, &sp);
            if (sts<PXC_STATUS_NO_ERROR) return sts;
            sts = sp->Synchronize();
            sp->Release();
            return sts;
        }

	protected:

			/**
				@reserved
				Internal function. Do not use.
			*/
			virtual pxcStatus PXCAPI QueryProperty(Property /*label*/, pxcF32 * /*value*/)=0;

		
			/**
				@reserved
				Internal function. Do not use.
			*/
			virtual pxcStatus PXCAPI SetPropertyAuto(Property /*pty*/, pxcBool /*ifauto*/)=0;

			/**
				@reserved
				Internal function. Do not use.
			*/
			virtual pxcStatus PXCAPI SetProperty(Property /*pty*/, pxcF32 /*value*/)=0;

			/**
				@reserved
				Internal function. Do not use.
			*/
			virtual pxcStatus PXCAPI QueryPropertyInfo(Property /*label*/,PropertyInfo* /*propertyInfo*/ )=0;


    public:

        /** 
            @brief Get the color stream exposure value.
            @return The color stream exposure, in log base 2 seconds.
        */
        __inline pxcI32    QueryColorExposure(void) {
            pxcF32 value=0;
            QueryProperty(PROPERTY_COLOR_EXPOSURE,&value);
            return (pxcI32)value;
        }
		
		/** 
            @brief Get the color stream exposure property information.
            @return The color stream exposure property information
        */
        __inline PropertyInfo    QueryColorExposureInfo(void) {
			PropertyInfo value={};
            QueryPropertyInfo(PROPERTY_COLOR_EXPOSURE,&value);
            return value;
        }

        /** 
            @brief Set the color stream exposure value.
            @param[in] value    The color stream exposure value, in log base 2 seconds.
            @return PXC_STATUS_NO_ERROR             successful execution.
            @return PXC_STATUS_ITEM_UNAVAILABLE     the device property is not supported.
        */
        __inline pxcStatus SetColorExposure(pxcI32 value) {
            return SetProperty(PROPERTY_COLOR_EXPOSURE,(pxcF32)value);
        }

        /** 
            @brief Set the color stream exposure value.
             @param[in] auto1    True to enable auto exposure.
            @return PXC_STATUS_NO_ERROR             successful execution.
            @return PXC_STATUS_ITEM_UNAVAILABLE     the device property is not supported.
        */
        __inline pxcStatus SetColorAutoExposure(pxcBool auto1) {
            return SetPropertyAuto(PROPERTY_COLOR_EXPOSURE,auto1?1:0);
        }

        /** 
            @brief Get the color stream brightness value.
            @return The color stream brightness from  -10,000 (pure black) to 10,000 (pure white).
        */
        __inline pxcI32    QueryColorBrightness(void) {
            pxcF32 value=0;
            QueryProperty(PROPERTY_COLOR_BRIGHTNESS,&value);
            return (pxcI32)value;
        }
		
		/** 
            @brief Get the color stream brightness property information.
            @return The color stream brightness property information
        */
        __inline PropertyInfo    QueryColorBrightnessInfo(void) {
			PropertyInfo value={};
            QueryPropertyInfo(PROPERTY_COLOR_BRIGHTNESS,&value);
            return value;
        }

        /** 
            @brief Set the color stream brightness value.
            @param[in] value    The color stream brightness from  -10,000 (pure black) to 10,000 (pure white).
            @return PXC_STATUS_NO_ERROR             successful execution.
            @return PXC_STATUS_ITEM_UNAVAILABLE     the device property is not supported.
        */
        __inline pxcStatus SetColorBrightness(pxcI32 value) {
            return SetProperty(PROPERTY_COLOR_BRIGHTNESS,(pxcF32)value);
        }

        /** 
            @brief Get the color stream contrast value.
            @return The color stream contrast, from 0 to 10,000.
        */
        __inline pxcI32    QueryColorContrast(void) {
            pxcF32 value=0;
            QueryProperty(PROPERTY_COLOR_CONTRAST,&value);
            return (pxcI32)value;
        }
		
		/** 
            @brief Get the color stream contrast property information.
            @return The color stream contrast property information
        */
        __inline PropertyInfo    QueryColorContrastInfo(void) {
			PropertyInfo value={};
            QueryPropertyInfo(PROPERTY_COLOR_CONTRAST,&value);
            return value;
        }


        /** 
            @brief Set the color stream contrast value.
            @param[in] value    The color stream contrast, from 0 to 10,000.
            @return PXC_STATUS_NO_ERROR             successful execution.
            @return PXC_STATUS_ITEM_UNAVAILABLE     the device property is not supported.
        */
        __inline pxcStatus SetColorContrast(pxcI32 value) {
            return SetProperty(PROPERTY_COLOR_CONTRAST,(pxcF32)value);
        }

        /** 
            @brief Get the color stream saturation value.
            @return The color stream saturation, from 0 to 10,000.
        */
        __inline pxcI32    QueryColorSaturation(void) {
            pxcF32 value=0;
            QueryProperty(PROPERTY_COLOR_SATURATION,&value);
            return (pxcI32)value;
        }

		/** 
            @brief Get the color stream saturation property information.
            @return The color stream saturation property information
        */
        __inline PropertyInfo    QueryColorSaturationInfo(void) {
			PropertyInfo value={};
            QueryPropertyInfo(PROPERTY_COLOR_SATURATION,&value);
            return value;
        }

        /** 
            @brief Set the color stream saturation value.
            @param[in] value    The color stream saturation, from 0 to 10,000.
            @return PXC_STATUS_NO_ERROR             successful execution.
            @return PXC_STATUS_ITEM_UNAVAILABLE     the device property is not supported.
        */
        __inline pxcStatus SetColorSaturation(pxcI32 value) {
            return SetProperty(PROPERTY_COLOR_SATURATION,(pxcF32)value);
        }

        /** 
            @brief Get the color stream hue value.
            @return The color stream hue, from -180,000 to 180,000 (representing -180 to 180 degrees.)
        */
        __inline pxcI32    QueryColorHue(void) {
            pxcF32 value=0;
            QueryProperty(PROPERTY_COLOR_HUE,&value);
            return (pxcI32)value;
        }

		/** 
            @brief Get the color stream Hue property information.
            @return The color stream Hue property information
        */
        __inline PropertyInfo    QueryColorHueInfo(void) {
			PropertyInfo value={};
            QueryPropertyInfo(PROPERTY_COLOR_HUE,&value);
            return value;
        }

        /** 
            @brief Set the color stream hue value.
            @param[in] value    The color stream hue, from -180,000 to 180,000 (representing -180 to 180 degrees.)
            @return PXC_STATUS_NO_ERROR             successful execution.
            @return PXC_STATUS_ITEM_UNAVAILABLE     the device property is not supported.
        */
        __inline pxcStatus SetColorHue(pxcI32 value) {
            return SetProperty(PROPERTY_COLOR_HUE,(pxcF32)value);
        }

        /** 
            @brief Get the color stream gamma value.
            @return The color stream gamma, from 1 to 500.
        */
        __inline pxcI32    QueryColorGamma(void) {
            pxcF32 value=0;
            QueryProperty(PROPERTY_COLOR_GAMMA,&value);
            return (pxcI32)value;
        }

		
		/** 
            @brief Get the color stream gamma property information.
            @return The color stream gamma property information
        */
        __inline PropertyInfo    QueryColorGammaInfo(void) {
			PropertyInfo value={};
            QueryPropertyInfo(PROPERTY_COLOR_GAMMA,&value);
            return value;
        }

        /** 
            @brief Set the color stream gamma value.
            @param[in] value    The color stream gamma, from 1 to 500.
            @return PXC_STATUS_NO_ERROR             successful execution.
            @return PXC_STATUS_ITEM_UNAVAILABLE     the device property is not supported.
        */
        __inline pxcStatus SetColorGamma(pxcI32 value) {
            return SetProperty(PROPERTY_COLOR_GAMMA,(pxcF32)value);
        }

        /** 
            @brief Get the color stream white balance value.
            @return The color stream balance, as a color temperature in degrees Kelvin.
        */
        __inline pxcI32    QueryColorWhiteBalance(void) {
            pxcF32 value=0;
            QueryProperty(PROPERTY_COLOR_WHITE_BALANCE,&value);
            return (pxcI32)value;
        }

		/** 
            @brief Get the color stream  white balance property information.
            @return The color stream  white balance property information
        */
        __inline PropertyInfo    QueryColorWhiteBalanceInfo(void) {
			PropertyInfo value={};
            QueryPropertyInfo(PROPERTY_COLOR_WHITE_BALANCE,&value);
            return value;
        }

        /** 
            @brief Set the color stream white balance value.
            @param[in] value    The color stream balance, as a color temperature in degrees Kelvin.
            @return PXC_STATUS_NO_ERROR             successful execution.
            @return PXC_STATUS_ITEM_UNAVAILABLE     the device property is not supported.
        */
        __inline pxcStatus SetColorWhiteBalance(pxcI32 value) {
            return SetProperty(PROPERTY_COLOR_WHITE_BALANCE,(pxcF32)value);
        }

        /** 
            @brief Set the color stream auto white balance mode.
            @param[in] auto1    The flag if the auto is enabled or not.
            @return PXC_STATUS_NO_ERROR             successful execution.
            @return PXC_STATUS_ITEM_UNAVAILABLE     the device property is not supported.
        */
        __inline pxcStatus SetColorAutoWhiteBalance(pxcBool auto1) {
            return SetPropertyAuto(PROPERTY_COLOR_WHITE_BALANCE,auto1);
        }

        /** 
            @brief Get the color stream sharpness value.
            @return The color stream sharpness, from 0 to 100.
        */
        __inline pxcI32    QueryColorSharpness(void) {
            pxcF32 value=0;
            QueryProperty(PROPERTY_COLOR_SHARPNESS,&value);
            return (pxcI32)value;
        }

		/** 
            @brief Get the color stream Sharpness property information.
            @return The color stream  Sharpness property information
        */
        __inline PropertyInfo    QueryColorSharpnessInfo(void) {
			PropertyInfo value={};
            QueryPropertyInfo(PROPERTY_COLOR_SHARPNESS,&value);
            return value;
        }

        /** 
            @brief Set the color stream sharpness value.
            @param[in] value    The color stream sharpness, from 0 to 100.
            @return PXC_STATUS_NO_ERROR             successful execution.
            @return PXC_STATUS_ITEM_UNAVAILABLE     the device property is not supported.
        */
        __inline pxcStatus SetColorSharpness(pxcI32 value) {
            return SetProperty(PROPERTY_COLOR_SHARPNESS,(pxcF32)value);
        }

        /** 
            @brief Get the color stream back light compensation status.
            @return The color stream back light compensation status from 0 to 4.
        */
        __inline pxcI32 QueryColorBackLightCompensation(void) {
            pxcF32 value=0;
            QueryProperty(PROPERTY_COLOR_BACK_LIGHT_COMPENSATION,&value);
            return (pxcI32)value;
        }

		/** 
            @brief Get the color stream back light compensation property information.
            @return The color stream  back light compensation property information
        */
        __inline PropertyInfo    QueryColorBackLightCompensationInfo(void) {
			PropertyInfo value={};
            QueryPropertyInfo(PROPERTY_COLOR_BACK_LIGHT_COMPENSATION,&value);
            return value;
        }

        /** 
            @brief Set the color stream back light compensation status.
            @param[in] value    The color stream back light compensation from 0 to 4.
            @return PXC_STATUS_NO_ERROR             successful execution.
            @return PXC_STATUS_ITEM_UNAVAILABLE     the device property is not supported.
        */
        __inline pxcStatus SetColorBackLightCompensation(pxcI32 value) {
            return SetProperty(PROPERTY_COLOR_BACK_LIGHT_COMPENSATION,(pxcF32)value);
        }

        /** 
            @brief Get the color stream gain value.
            @return The color stream gain adjustment, with negative values darker, positive values brighter, and zero as normal.
        */
        __inline pxcI32    QueryColorGain(void) {
            pxcF32 value=0;
            QueryProperty(PROPERTY_COLOR_GAIN,&value);
            return (pxcI32)value;
        }

		/** 
            @brief Get the color stream gain property information.
            @return The color stream  gain property information
        */
        __inline PropertyInfo    QueryColorGainInfo(void) {
			PropertyInfo value={};
            QueryPropertyInfo(PROPERTY_COLOR_GAIN,&value);
            return value;
        }

        /** 
            @brief Set the color stream gain value.
            @param[in] value    The color stream gain adjustment, with negative values darker, positive values brighter, and zero as normal.
            @return PXC_STATUS_NO_ERROR             successful execution.
            @return PXC_STATUS_ITEM_UNAVAILABLE     the device property is not supported.
        */
        __inline pxcStatus SetColorGain(pxcI32 value) {
            return SetProperty(PROPERTY_COLOR_GAIN,(pxcF32)value);
        }

        /** 
            @brief Get the color stream power line frequency value.
            @return The power line frequency in Hz.
        */
        __inline PowerLineFrequency    QueryColorPowerLineFrequency(void) {
            pxcF32 value=0;
            QueryProperty(PROPERTY_COLOR_POWER_LINE_FREQUENCY,&value);
            return (PowerLineFrequency)(pxcI32)value;
        }


        /** 
            @brief Set the color stream power line frequency value.
            @param[in] value    The power line frequency in Hz.
            @return PXC_STATUS_NO_ERROR             successful execution.
            @return PXC_STATUS_ITEM_UNAVAILABLE     the device property is not supported.
        */
        __inline pxcStatus SetColorPowerLineFrequency(PowerLineFrequency value) {
            return SetProperty(PROPERTY_COLOR_POWER_LINE_FREQUENCY,(pxcF32)value);
        }

        /** 
            @brief Get the color stream field of view.
            @return The color-sensor horizontal and vertical field of view parameters, in degrees. 
        */
        __inline PXCPointF32 QueryColorFieldOfView(void) {
            PXCPointF32 value={};
            QueryProperty(PROPERTY_COLOR_FIELD_OF_VIEW,&value.x);
            QueryProperty((Property)(PROPERTY_COLOR_FIELD_OF_VIEW+1),&value.y);
            return value;
        }

        /** 
            @brief Get the color stream focal length.
            @return The color-sensor focal length in pixels. The parameters vary with the resolution setting.
        */
        __inline PXCPointF32 QueryColorFocalLength(void) {
            PXCPointF32 value={};
            QueryProperty(PROPERTY_COLOR_FOCAL_LENGTH,&value.x);
            QueryProperty((Property)(PROPERTY_COLOR_FOCAL_LENGTH+1),&value.y);
            return value;
        }

        /** 
            @brief Get the color stream focal length in mm.
            @return The color-sensor focal length in mm.
        */
        __inline pxcF32 QueryColorFocalLengthMM(void) {
            pxcF32 value=0;
            QueryProperty(PROPERTY_COLOR_FOCAL_LENGTH_MM,&value);
            return value;
        }

        /** 
            @brief Get the color stream principal point.
            @return The color-sensor principal point in pixels. The parameters vary with the resolution setting.
        */
        __inline PXCPointF32 QueryColorPrincipalPoint(void) {
            PXCPointF32 value={};
            QueryProperty(PROPERTY_COLOR_PRINCIPAL_POINT,&value.x);
            QueryProperty((Property)(PROPERTY_COLOR_PRINCIPAL_POINT+1),&value.y);
            return value;
        }

        /** 
            @brief Get the depth stream low confidence value.
            @return The special depth map value to indicate that the corresponding depth map pixel is of low-confidence.
        */
        __inline pxcU16 QueryDepthLowConfidenceValue(void) {
            pxcF32 value=0;
            QueryProperty(PROPERTY_DEPTH_LOW_CONFIDENCE_VALUE,&value);
            return (pxcU16)value;
        }

        /** 
            @brief Get the depth stream confidence threshold.
            @return The confidence threshold that is used to floor the depth map values. The range is from 0 to 15.
        */
        __inline pxcI16 QueryDepthConfidenceThreshold(void) {
            pxcF32 value=0;
            QueryProperty(PROPERTY_DEPTH_CONFIDENCE_THRESHOLD,&value);
            return (pxcI16)value;
        }

		 /** 
            @brief Get the depth stream confidence threshold information.
            @return The property information for the confidence threshold that is used to floor the depth map values. The range is from 0 to 15.
        */
        __inline PropertyInfo QueryDepthConfidenceThresholdInfo(void) {
			PropertyInfo value={};
			QueryPropertyInfo(PROPERTY_DEPTH_CONFIDENCE_THRESHOLD,&value);
			return value;
        }

        /** 
            @brief Set the depth stream confidence threshold.
            @param[in] value    The confidence threshold that is used to floor the depth map values. The range is from 0 to 15.
            @return PXC_STATUS_NO_ERROR             successful execution.
            @return PXC_STATUS_ITEM_UNAVAILABLE     the device property is not supported.
        */
        __inline pxcStatus SetDepthConfidenceThreshold(pxcI16 value) {
            return SetProperty(PROPERTY_DEPTH_CONFIDENCE_THRESHOLD, (pxcF32)value);
        }

        /** 
            @brief Get the depth stream unit value.
            @return The unit of depth values in micrometre.
        */
        __inline pxcF32 QueryDepthUnit(void) {
            pxcF32 value=0;
            QueryProperty(PROPERTY_DEPTH_UNIT,&value);
            return value;
        }
	
        /** 
            @brief Get the depth stream field of view.
            @return The depth-sensor horizontal and vertical field of view parameters, in degrees. 
        */
        __inline PXCPointF32 QueryDepthFieldOfView(void) {
            PXCPointF32 value={};
            QueryProperty(PROPERTY_DEPTH_FIELD_OF_VIEW,&value.x);
            QueryProperty((Property)(PROPERTY_DEPTH_FIELD_OF_VIEW+1),&value.y);
            return value;
        }

        /** 
            @brief Get the depth stream sensor range.
            @return The depth-sensor, sensing distance parameters, in millimeters.
        */
        __inline PXCRangeF32 QueryDepthSensorRange(void) {
            PXCRangeF32 value={};
            QueryProperty(PROPERTY_DEPTH_SENSOR_RANGE,&value.min);
            QueryProperty((Property)(PROPERTY_DEPTH_SENSOR_RANGE+1),&value.max);
            return value;
        }

        /** 
            @brief Get the depth stream focal length.
            @return The depth-sensor focal length in pixels. The parameters vary with the resolution setting.
        */
        __inline PXCPointF32 QueryDepthFocalLength(void) {
            PXCPointF32 value={};
            QueryProperty(PROPERTY_DEPTH_FOCAL_LENGTH,&value.x);
            QueryProperty((Property)(PROPERTY_DEPTH_FOCAL_LENGTH+1),&value.y);
            return value;
        }

        /** 
            @brief Get the depth stream focal length in mm.
            @return The depth-sensor focal length in mm.
        */
        __inline pxcF32 QueryDepthFocalLengthMM(void) {
            pxcF32 value=0;
            QueryProperty(PROPERTY_DEPTH_FOCAL_LENGTH_MM,&value);
            return value;
        }

        /** 
            @brief Get the depth stream principal point.
            @return The depth-sensor principal point in pixels. The parameters vary with the resolution setting.
        */
        __inline PXCPointF32 QueryDepthPrincipalPoint(void) {
            PXCPointF32 value={};
            QueryProperty(PROPERTY_DEPTH_PRINCIPAL_POINT,&value.x);
            QueryProperty((Property)(PROPERTY_DEPTH_PRINCIPAL_POINT+1),&value.y);
            return value;
        }

        /** 
            @brief Get the device allow profile change status.
            @return If true, allow resolution change and throw PXC_STATUS_STREAM_CONFIG_CHANGED.
        */
        __inline pxcBool QueryDeviceAllowProfileChange(void) {
            pxcF32 value=0;
            QueryProperty(PROPERTY_DEVICE_ALLOW_PROFILE_CHANGE,&value);
            return value!=0;
        }

		

        /** 
            @brief Set the device allow profile change status.
            @param[in] value    If true, allow resolution change and throw PXC_STATUS_STREAM_CONFIG_CHANGED.
            @return PXC_STATUS_NO_ERROR             successful execution.
            @return PXC_STATUS_ITEM_UNAVAILABLE     the device property is not supported.
        */
        __inline pxcStatus SetDeviceAllowProfileChange(pxcBool value) {
            return SetProperty(PROPERTY_DEVICE_ALLOW_PROFILE_CHANGE, (pxcF32)(value!=0));
        }

        /** 
            @brief Get the mirror mode.
             @return The mirror mode
        */
        __inline MirrorMode QueryMirrorMode(void) {
            pxcF32 value=0;
            QueryProperty(PROPERTY_DEVICE_MIRROR,&value);
            return (MirrorMode)(pxcI32)value;
        }

        /** 
            @brief Set the mirror mode.
            @param[in] value    The mirror mode
            @return PXC_STATUS_NO_ERROR             successful execution.
            @return PXC_STATUS_ITEM_UNAVAILABLE     the device property is not supported.
        */
        __inline pxcStatus SetMirrorMode(MirrorMode value) {
            return SetProperty(PROPERTY_DEVICE_MIRROR,(pxcF32)value);
        }

        /** 
            @brief Get the IVCAM laser power.
            @return The laser power value from 0 (minimum) to 16 (maximum).
        */
        __inline pxcI32 QueryIVCAMLaserPower(void) {
            pxcF32 value=0;
            QueryProperty(PROPERTY_IVCAM_LASER_POWER,&value);
            return (pxcI32)value;
        }

		
        /** 
            @brief Get the IVCAM laser power property information.
            @return The laser power proeprty information. 
        */
        __inline PropertyInfo QueryIVCAMLaserPowerInfo(void) {
			PropertyInfo value={};
			QueryPropertyInfo(PROPERTY_IVCAM_LASER_POWER,&value);
			return value;
        }

        /** 
            @brief Set the IVCAM laser power.
            @param[in] value    The laser power value from 0 (minimum) to 16 (maximum).
            @return PXC_STATUS_NO_ERROR             successful execution.
            @return PXC_STATUS_ITEM_UNAVAILABLE     the device property is not supported.
        */
        __inline pxcStatus SetIVCAMLaserPower(pxcI32 value) {
            return SetProperty(PROPERTY_IVCAM_LASER_POWER,(pxcF32)value);
        }
        
        /** 
            @brief Get the IVCAM accuracy.
            @return The accuracy value
        */
        __inline IVCAMAccuracy QueryIVCAMAccuracy(void) {
            pxcF32 value=0;
            QueryProperty(PROPERTY_IVCAM_ACCURACY,&value);
            return (IVCAMAccuracy)(pxcI32)value;
        }

        /** 
            @brief Set the IVCAM accuracy.
            @param[in] value    The accuracy value
            @return PXC_STATUS_NO_ERROR             successful execution.
            @return PXC_STATUS_ITEM_UNAVAILABLE     the device property is not supported.
        */
        __inline pxcStatus SetIVCAMAccuracy(IVCAMAccuracy value) {
            return SetProperty(PROPERTY_IVCAM_ACCURACY,(pxcF32)value);
        }

		/** 
			@brief Get the IVCAM filter option (smoothing aggressiveness) ranged from 0 (close range) to 7 (far range).
			@return The filter option value.
		*/
		__inline pxcI32 QueryIVCAMFilterOption(void) {
			pxcF32 value=0;
			pxcStatus sts=QueryProperty(PROPERTY_IVCAM_FILTER_OPTION,&value);
			if (sts<PXC_STATUS_NO_ERROR) QueryProperty((Property)(PROPERTY_CUSTOMIZED+6), &value);
			return (pxcI32)value;
		}

		 /** 
            @brief Get the IVCAM Filter Option property information.
            @return The IVCAM Filter Option property information. 
        */
        __inline PropertyInfo QueryIVCAMFilterOptionInfo(void) {
			PropertyInfo value={};
			pxcStatus sts=QueryPropertyInfo(PROPERTY_IVCAM_FILTER_OPTION,&value);
			if (sts<PXC_STATUS_NO_ERROR) QueryPropertyInfo((Property)(PROPERTY_CUSTOMIZED+6), &value);
			return value;
        }

		/** 
			@brief Set the IVCAM filter option (smoothing aggressiveness) ranged from 0 (close range) to 7 (far range).
			@param[in] value	The filter option value
			@return PXC_STATUS_NO_ERROR			successful execution.
			@return PXC_STATUS_ITEM_UNAVAILABLE the device property is not supported.
		*/
		__inline pxcStatus SetIVCAMFilterOption(pxcI32 value) {
			pxcStatus sts=SetProperty(PROPERTY_IVCAM_FILTER_OPTION, (pxcF32)value);
			if (sts<PXC_STATUS_NO_ERROR) sts=SetProperty((Property)(PROPERTY_CUSTOMIZED+6), (pxcF32)value);
			return sts;
		}

		/** 
			@brief Get the IVCAM motion range trade off option, ranged from 0 (short range, better motion) to 100 (far range, long exposure).
			@return The motion range trade option.
		*/
		__inline pxcI32 QueryIVCAMMotionRangeTradeOff(void) {
			pxcF32 value=0;
			pxcStatus sts=QueryProperty(PROPERTY_IVCAM_MOTION_RANGE_TRADE_OFF,&value);
			if (sts<PXC_STATUS_NO_ERROR) QueryProperty((Property)(PROPERTY_CUSTOMIZED+4), &value);
			return (pxcI32)value;
		}

		
		 /** 
            @brief Get the IVCAM Filter Option property information.
            @return The IVCAM Filter Option property information. 
        */
        __inline PropertyInfo QueryIVCAMMotionRangeTradeOffInfo(void) {
			PropertyInfo value={};
			pxcStatus sts=QueryPropertyInfo(PROPERTY_IVCAM_MOTION_RANGE_TRADE_OFF,&value);
			if (sts<PXC_STATUS_NO_ERROR) QueryPropertyInfo((Property)(PROPERTY_CUSTOMIZED+4), &value);
			return value;
        }

		/** 
			@brief Set the IVCAM motion range trade off option, ranged from 0 (short range, better motion) to 100 (far range, long exposure).
			@param[in] value		The motion range trade option.
			@return PXC_STATUS_NO_ERROR			successful execution.
			@return PXC_STATUS_ITEM_UNAVAILABLE the device property is not supported.
		*/
		__inline pxcStatus SetIVCAMMotionRangeTradeOff(pxcI32 value) {
			pxcStatus sts=SetProperty(PROPERTY_IVCAM_MOTION_RANGE_TRADE_OFF, (pxcF32)value);
			if (sts<PXC_STATUS_NO_ERROR) sts=SetProperty((Property)(PROPERTY_CUSTOMIZED+4), (pxcF32)value);
			return sts;
		}
    };
};

/** 
    A helper function for bitwise OR of two flags.
*/
__inline static PXCCapture::StreamType operator|(PXCCapture::StreamType a, PXCCapture::StreamType b) {
    return (PXCCapture::StreamType)((int)a | (int)b);
}

/** 
    A helper function for traversing the stream types: ++stream_type
*/
__inline static PXCCapture::StreamType& operator++(PXCCapture::StreamType &a) {
	a=(PXCCapture::StreamType)((a<<1)&((1<<(PXCCapture::STREAM_LIMIT))-1));
	return a;
}
 
/** 
    A helper function for traversing the stream types: stream_type++
*/
__inline static PXCCapture::StreamType operator++(PXCCapture::StreamType &a, int) {
	return ++a;
}
#pragma warning(pop)
