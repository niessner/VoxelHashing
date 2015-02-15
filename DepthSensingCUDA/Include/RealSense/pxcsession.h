/*******************************************************************************

INTEL CORPORATION PROPRIETARY INFORMATION
This software is supplied under the terms of a license agreement or nondisclosure
agreement with Intel Corporation and may not be copied or disclosed except in
accordance with the terms of that agreement
Copyright(c) 2011-2014 Intel Corporation. All Rights Reserved.

*******************************************************************************/
/** @file pxcsession.h
    Defines the PXCSession interface, which maintains the SDK context.  The
    application can query and create instances of I/O and algorithm module
    implementations.
*/
#pragma once
#include "pxcbase.h"
#include "pxcversion.h"
#include "pxcimage.h"
#include "pxcaudio.h"

class PXCSenseManager;
class PXCCaptureManager;
class PXCAudioSource;
class PXCSession;
extern "C" PXCSession* PXCAPI PXCSession_Create(void);

/**
    This class defines a standard interface for maintaining the SDK context.
    The application can query and create instances of I/O and algorithm module
    implementations.

    The interface extends PXCMetadata. Use QueryInstance<PXCMetadata>() to access 
    the PXCMetadata features.
*/
class PXCSession:public PXCBase {
public:
    PXC_CUID_OVERWRITE(PXC_UID('S','E','S',' '));

    /** 
        @structure ImplVersion
        Describe the video streams requested by a module implementation.
    */
    struct ImplVersion {
        pxcI16  major;    /* The major version number */
        pxcI16  minor;    /* The minor version number */
    };

    /** 
        @enum ImplGroup
        The SDK group I/O and algorithm modules into groups and subgroups.
        This is the enumerator for algorithm groups.
    */
    enum ImplGroup {
        IMPL_GROUP_ANY                  =    0,             /* Undefine group */
        IMPL_GROUP_OBJECT_RECOGNITION   =    0x00000001,    /* Object recognition algorithms */
        IMPL_GROUP_SPEECH_RECOGNITION   =    0x00000002,    /* Speech recognition algorithms */
        IMPL_GROUP_SENSOR               =    0x00000004,    /* I/O modules */
        IMPL_GROUP_CORE                 =    0x80000000,    /* Core SDK modules */
        IMPL_GROUP_USER                 =    0x40000000,    /* User defined algorithms */
    };

    /**
        @enum ImplSubgroup
        The SDK group I/O and algorithm modules into groups and subgroups.
        This is the enumerator for algorithm subgroups.
    */
    enum ImplSubgroup {
        IMPL_SUBGROUP_ANY                   = 0,            /* Undefined subgroup */

        /* object recognition building blocks */
        IMPL_SUBGROUP_FACE_ANALYSIS         = 0x00000001,    /* face analysis subgroup */
        IMPL_SUBGROUP_GESTURE_RECOGNITION   = 0x00000010,    /* gesture recognition subgroup */
        IMPL_SUBGROUP_SEGMENTATION          = 0x00000020,    /* segmentation subgroup */
        IMPL_SUBGROUP_PULSE_ESTIMATION      = 0x00000040,    /* pulse estimation subgroup */
        IMPL_SUBGROUP_EMOTION_RECOGNITION   = 0x00000080,    /* emotion recognition subgroup */
		IMPL_SUBGROUP_OBJECT_TRACKING		= 0x00000100,    /* object detection subgroup */
        IMPL_SUBGROUP_3DSEG                 = 0x00000200,    /* user segmentation subgroup */
        IMPL_SUBGROUP_3DSCAN                = 0x00000400,    /* mesh capture subgroup */

        /* sensor building blocks */
        IMPL_SUBGROUP_AUDIO_CAPTURE         = 0x00000001,    /* audio capture subgroup */
        IMPL_SUBGROUP_VIDEO_CAPTURE         = 0x00000002,    /* video capture subgroup */

        /* speech recognition building blocks */
        IMPL_SUBGROUP_SPEECH_RECOGNITION     = 0x00000001,    /* speech recognition subgroup */
        IMPL_SUBGROUP_SPEECH_SYNTHESIS       = 0x00000002,    /* speech synthesis subgroup */
    };

    /** 
        @structure ImplDesc
        The module descriptor lists details about the module implementation.
    */
    struct ImplDesc {
        ImplGroup           group;              /* implementation group */
        ImplSubgroup        subgroup;           /* implementation sub-group */
        pxcUID              algorithm;          /* algorithm identifier */      
        pxcUID              iuid;               /* implementation unique id */
        ImplVersion         version;            /* implementation version */
        pxcI32              reserved2;
        pxcI32              merit;              /* implementation merit */
        pxcI32              vendor;             /* vendor */
        pxcUID              cuids[4];           /* interfaces supported by implementation */
        pxcCHAR             friendlyName[256];  /* friendly name */
        pxcI32              reserved[12];
    };

    /** 
        @brief Return the SDK version.
        @return the SDK version.
    */
    virtual ImplVersion PXCAPI QueryVersion(void)=0;

    /** 
        @brief Search a module implementation.
        @param[in]    templat           The template for the module search. Zero field values match any.
        @param[in]    idx               The zero-based index to retrieve multiple matches.
        @param[out]   desc              The matched module descritpor, to be returned.
        @return PXC_STATUS_NO_ERROR            Successful execution.
        @return PXC_STATUS_ITEM_UNAVAILABLE    No matched module implementation.
    */
    virtual pxcStatus PXCAPI QueryImpl(ImplDesc *templat, pxcI32 idx, ImplDesc *desc)=0;

    /** 
        @brief Create an instance of the specified module.
        @param[in]    desc              The module descriptor.
        @param[in]    iuid              Optional module implementation identifier.
        @param[in]    cuid              Optional interface identifier.
        @param[out]   instance          The created instance, to be returned.
        @return PXC_STATUS_NO_ERROR            Successful execution.
        @return PXC_STATUS_ITEM_UNAVAILABLE    No matched module implementation.
    */
    virtual pxcStatus PXCAPI CreateImpl(ImplDesc *desc, pxcUID iuid, pxcUID cuid, void **instance)=0;
    
    /** 
        @brief Create an instance of the specified module.
        @param[in]    desc              The module descriptor.
        @param[in]    cuid              Optional interface identifier.
        @param[out]   instance          The created instance, to be returned.
        @return PXC_STATUS_NO_ERROR            Successful execution.
        @return PXC_STATUS_ITEM_UNAVAILABLE    No matched module implementation.
    */
    pxcStatus __inline CreateImpl(ImplDesc *desc, pxcUID cuid, void **instance) {
        return CreateImpl(desc,0,cuid,instance); 
    }

    /** 
        @brief Create an instance of the specified module.
        @param[in]   cuid               Optional interface identifier.
        @param[out]  instance           The created instance, to be returned.
        @return PXC_STATUS_NO_ERROR            Successful execution.
        @return PXC_STATUS_ITEM_UNAVAILABLE    No matched module implementation.
    */
    pxcStatus __inline CreateImpl(pxcUID cuid, void **instance) { 
        return CreateImpl(0, cuid,instance); 
    }

    /** 
        @brief Create an instance of the specified module of the specified type.
        @param[in]   desc               The module descriptor.
        @param[in]   iuid               Optional module implementation identifier.
        @param[out]  instance           The created instance, to be returned.
        @return PXC_STATUS_NO_ERROR            Successful execution.
        @return PXC_STATUS_ITEM_UNAVAILABLE    No matched module implementation.
    */
    template <class T> pxcStatus __inline CreateImpl(ImplDesc *desc, pxcUID iuid, T **instance) {
        return CreateImpl(desc,iuid,T::CUID,(void**)instance); 
    }

    /** 
        @brief Create an instance of the specified module of the specified type.
        @param[in]   desc               The module descriptor.
        @param[out]  instance           The created instance, to be returned.
        @return PXC_STATUS_NO_ERROR            Successful execution.
        @return PXC_STATUS_ITEM_UNAVAILABLE    No matched module implementation.
    */
    template <class T> pxcStatus __inline CreateImpl(ImplDesc *desc, T **instance) {
        return CreateImpl(desc,T::CUID,(void**)instance); 
    }

    /** 
        @brief Create an instance of the specified module of the specified type.
        @param[in]   iuid               Optional module implementation identifier.
        @param[out]  instance           The created instance, to be returned.
        @return PXC_STATUS_NO_ERROR            Successful execution.
        @return PXC_STATUS_ITEM_UNAVAILABLE    No matched module implementation.
    */
    template <class T> pxcStatus __inline CreateImpl(pxcUID iuid, T **instance) {
        return CreateImpl(0,iuid,T::CUID,(void**)instance); 
    }

    /** 
        @brief Create an instance of the specified module of the specified type.
        @param[out]  instance           The created instance, to be returned.
        @return PXC_STATUS_NO_ERROR            Successful execution.
        @return PXC_STATUS_ITEM_UNAVAILABLE    No matched module implementation.
    */
    template <class T> pxcStatus __inline CreateImpl(T **instance) { 
        return CreateImpl(T::CUID,(void**)instance); 
    }

    /** 
        @brief Create an instance of the PXCSenseManager interface.
        @return The PXCSenseManager instance.
    */
    __inline PXCSenseManager *CreateSenseManager(void) {
        PXCSenseManager *sm=0;
        CreateImpl(0, PXC_UID('P','P','U','T'), 0, (void**)&sm); 
        return sm;
    }

    /** 
        @brief Create an instance of the PXCCaptureManager interface.
        @return The PXCCaptureManager instance.
    */
    __inline PXCCaptureManager* CreateCaptureManager(void) {
        PXCCaptureManager *cm=0;
        CreateImpl(0, PXC_UID('C','P','U','T'), 0, (void**)&cm);
        return cm;
    }

    /** 
		@brief Create an instance of the PXCAudioSource interface.
		@return The PXCAudioSource instance.
    */
    __inline PXCAudioSource *CreateAudioSource(void) {
		PXCAudioSource *am=0;
		CreateImpl(0, PXC_UID('A','D','S','R'), 0, (void**)&am);
		return am;
	}

    /** 
        @brief Return the module descriptor
        @param[in]  module          The module instance
        @param[out] desc            The module descriptor, to be returned.
        @return PXC_STATUS_NO_ERROR         Successful execution.
        @return PXC_STATUS_ITEM_UNAVAILABLE Failed to identify the module instance.
    */
    virtual pxcStatus PXCAPI QueryModuleDesc(PXCBase *module, PXCSession::ImplDesc *desc) = 0;

    /** 
        @brief Create an instance of the PXCImage interface with data. The application must
        maintain the life cycle of the image data for the PXCImage instance.
        @param[in]  info            The format and resolution of the image.
        @param[in]  data            Optional image data.
        @return The PXCImage instance.
    */
    virtual PXCImage* PXCAPI CreateImage(PXCImage::ImageInfo *info, PXCImage::ImageData *data)=0;

    /** 
        @brief Create an instance of the PXCImage interface.
        @param[in]  info            The format and resolution of the image.
        @return The PXCImage instance.
    */
    __inline PXCImage* CreateImage(PXCImage::ImageInfo *info) {
        return CreateImage(info, 0);
    }

    /** 
        @brief Create an instance of the PXCAudio interface with data. The application must
        maintain the life cycle of the audio data for the PXCAudio instance.
        @param[in]  info            The audio channel information.
        @param[in]  data            Optional audio data.
        @return The PXCAudio instance.
    */
    virtual PXCAudio* PXCAPI CreateAudio(PXCAudio::AudioInfo *info, PXCAudio::AudioData *data)=0;

    /** 
        @brief Create an instance of the PXCAudio interface.
        @param[in]  info            The audio channel information.
        @return The PXCAudio instance.
    */
    __inline PXCAudio* CreateAudio(PXCAudio::AudioInfo *info) {
        return CreateAudio(info, 0);
    }

    /** 
        @brief Load the module from a file.
        @param[in]  moduleName      The module file name.
        @return PXC_STATUS_NO_ERROR    Successful execution.
    */
    virtual pxcStatus PXCAPI LoadImplFromFile(pxcCHAR *moduleName)=0;

    /** 
        @brief Unload the specified module.
        @param[in]  moduleName      The module file name.
        @return PXC_STATUS_NO_ERROR    Successful execution.
    */
    virtual pxcStatus PXCAPI UnloadImplFromFile(pxcCHAR *moduleName)=0;

    /** 
        @brief Create an instance of the PXCSession interface.
        @return The PXCSession instance
    */
    __inline static PXCSession* CreateInstance(void) {
        return PXCSession_Create();
    }
};

/** 
    A helper function for bitwise OR of two flags.
*/
__inline static PXCSession::ImplGroup operator|(PXCSession::ImplGroup a, PXCSession::ImplGroup b) {
    return (PXCSession::ImplGroup)((int)a | (int)b);
}

/** 
    A helper function for bitwise OR of two flags.
*/
__inline static PXCSession::ImplSubgroup operator|(PXCSession::ImplSubgroup a, PXCSession::ImplSubgroup b) {
    return (PXCSession::ImplSubgroup)((int)a | (int)b);
}
