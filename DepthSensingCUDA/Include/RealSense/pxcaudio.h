/*******************************************************************************

INTEL CORPORATION PROPRIETARY INFORMATION
This software is supplied under the terms of a license agreement or nondisclosure
agreement with Intel Corporation and may not be copied or disclosed except in
accordance with the terms of that agreement
Copyright(c) 2011-2014 Intel Corporation. All Rights Reserved.

*******************************************************************************/
/** @file pxcaudio.h
    Defines the PXCAudio interface for managing audio buffer access.
 */
#pragma once
#include "pxcaddref.h"
#pragma warning(push)
#pragma warning(disable:4201) /* nameless structs/unions */

/** 
    The PXCAudio interface manages the audio buffer access.

    The interface extends PXCAddRef. Use QueryInstance<PXCAddRef>(), or the helper
    function AddRef() to access the PXCAddRef features.

    The interface extends PXCMetadata. Use QueryInstance<PXCMetadata>() to access 
    the PXCMetadata features.
*/
class PXCAudio: public PXCBase {
public:
    PXC_CUID_OVERWRITE(0x395A39C8);
    
    /** 
        @enum AudioFormat
        Describes the audio sample format
    */
    enum AudioFormat {
        AUDIO_FORMAT_PCM        = PXC_UID(16,'P','C','M'),  /* 16-bit PCM            */
        AUDIO_FORMAT_IEEE_FLOAT = PXC_UID(32,'F','L','T'),  /* 32-bit float point    */
    };

    /** 
        @brief Get the audio format string representation
        @param[in]    format        The Audio format enumerator.
        @return the string representation of the audio format.
    */
    __inline static const pxcCHAR *AudioFormatToString(AudioFormat format) {
        switch (format) {
        case AUDIO_FORMAT_PCM: return (const pxcCHAR*)L"PCM";
        case AUDIO_FORMAT_IEEE_FLOAT: return (const pxcCHAR*)L"Float";
        }
        return (const pxcCHAR*)L"Unknown";
    }

    /** 
        @brief Return the audio sample size.
        @return the sample size in bytes
    */
    __inline static pxcI32 AudioFormatToSize(AudioFormat format) {
        return (format&0xff)>>3;
    }

    /** 
        @enum ChannelMask
        Describes the channels of the audio source.
    */
    enum ChannelMask {
        CHANNEL_MASK_FRONT_LEFT     =0x00000001,    /* The source is at front left      */
        CHANNEL_MASK_FRONT_RIGHT    =0x00000002,    /* The source is at front right     */
        CHANNEL_MASK_FRONT_CENTER   =0x00000004,    /* The source is at front center    */
        CHANNEL_MASK_LOW_FREQUENCY  =0x00000008,    /* The source is for low frequency  */
        CHANNEL_MASK_BACK_LEFT      =0x00000010,    /* The source is from back left     */
        CHANNEL_MASK_BACK_RIGHT     =0x00000020,    /* The source is from back right    */
        CHANNEL_MASK_SIDE_LEFT      =0x00000200,    /* The source is from side left     */
        CHANNEL_MASK_SIDE_RIGHT     =0x00000400,    /* The source is from side right    */
    };

    /** 
        @structure AudioInfo
        Describes the audio sample details.
    */
    struct AudioInfo {
        pxcI32      bufferSize;     /* buffer size in number samples */
        AudioFormat format;         /* sample format */
        pxcI32      sampleRate;     /* samples per second */
        pxcI32      nchannels;      /* number of channels */
        ChannelMask channelMask;    /* channel mask */
        pxcI32      reserved[3];
    };

    /** 
        @structure AudioData
        Describes the audio storage details.
    */
    struct AudioData {
        AudioFormat     format;     /* sample format */
        pxcI32          dataSize;   /* sample data size in number of samples */
        pxcBYTE*        dataPtr;    /* the sample buffer */
    };

    /** 
        @enum Access
        Describes the audio sample access mode.
    */
    enum Access {
        ACCESS_READ         = 1,                            /* read only access */
        ACCESS_WRITE        = 2,                            /* write only access */
        ACCESS_READ_WRITE   = ACCESS_READ | ACCESS_WRITE,   /* read write access */
    };

    /** 
        @enum Option
        Describes the audio options.
    */
    enum Option {
        OPTION_ANY = 0,                /* unknown/undefined */
    };

    /** 
        @brief Return the audio sample information.
        @return the audio sample information in the AudioInfo structure.
    */
    virtual AudioInfo PXCAPI QueryInfo(void)=0;

    /** 
        @brief Return the audio sample time stamp.
        @return the time stamp, in 100ns.
    */
    virtual pxcI64 PXCAPI QueryTimeStamp(void)=0;

    /** 
        @brief Return the audio sample option flags.
        @return the option flags.
    */
    virtual Option PXCAPI QueryOptions(void)=0;

    /** 
        @brief Set the sample time stamp.
        @param[in] ts           The time stamp value, in 100ns.
    */
    virtual void PXCAPI SetTimeStamp(pxcI64 ts)=0;

    /** 
        @brief Set the sample options.
        @param[in] options      The option flags.
    */
    virtual void PXCAPI SetOptions(Option options)=0;

    /** 
        @brief Copy data from another audio sample.
        @param[in] src_audio    The audio sample to copy data from.
        @return PXC_STATUS_NO_ERROR    Successful execution.
    */
    virtual pxcStatus PXCAPI CopyAudio(PXCAudio *src_audio)=0;

    /** 
        @brief Lock to access the internal storage of a specified format. The function will perform format conversion if unmatched. 
        @param[in] access       The access mode.
        @param[in] format       The requested smaple format.
        @param[in] options      The option flags.
        @param[out] data        The sample data storage, to be returned.
        @return PXC_STATUS_NO_ERROR    Successful execution.
    */
    virtual pxcStatus PXCAPI AcquireAccess(Access access, AudioFormat format, pxcEnum options, AudioData *data)=0;

    /** 
        @brief Lock to access the internal storage of a specified format. The function will perform format conversion if unmatched. 
        @param[in]  access      The access mode.
        @param[in]  format      The requested smaple format.
        @param[out] data        The sample data storage, to be returned.
        @return PXC_STATUS_NO_ERROR    Successful execution.
    */
    __inline pxcStatus AcquireAccess(Access access, AudioFormat format, AudioData *data) { return AcquireAccess(access, format, 0, data); }

    /** 
        @brief Lock to access the internal storage of a specified format. 
        @param[in]  access      The access mode.
        @param[out] data        The sample data storage, to be returned.
        @return PXC_STATUS_NO_ERROR    Successful execution.
    */
    __inline pxcStatus AcquireAccess(Access access, AudioData *data) { return AcquireAccess(access, (AudioFormat)0, 0, data); }

    /** 
        @brief Unlock the previously acquired buffer.
        @param[in] data         The sample data storage previously acquired.
        @return PXC_STATUS_NO_ERROR    Successful execution.
    */
    virtual pxcStatus PXCAPI ReleaseAccess(AudioData *data)=0;

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
__inline static PXCAudio::ChannelMask operator | (PXCAudio::ChannelMask a, PXCAudio::ChannelMask b) {
    return (PXCAudio::ChannelMask)((int)a|(int)b);
}

/** 
    A helper function for bitwise OR of two flags.
*/
__inline static PXCAudio::Option operator | (PXCAudio::Option a, PXCAudio::Option b) {
    return (PXCAudio::Option)((int)a|(int)b);
}
#pragma warning(pop)