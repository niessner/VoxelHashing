/*******************************************************************************

INTEL CORPORATION PROPRIETARY INFORMATION
This software is supplied under the terms of a license agreement or nondisclosure
agreement with Intel Corporation and may not be copied or disclosed except in
accordance with the terms of that agreement
Copyright(c) 2013-2014 Intel Corporation. All Rights Reserved.

*******************************************************************************/
#pragma once
#include <string.h>
#include "pxccapture.h"
#include "pxcvideomodule.h"

/** 
    The CaptureManager interface provides the following features:
    (1) Locate an I/O device that meets all module input needs.
    (2) Record any streaming data to a file and playback from the file.
*/
class PXCCaptureManager: public PXCBase {
public:
    PXC_CUID_OVERWRITE(0xD8912345);

    /** 
        This is the PXCCaptureManager callback interface.
    */
    class Handler {
    public:
        /**
            @brief    The CaptureManager callbacks this function when creating a device instance.
            @param[in]    mdesc     The I/O module descriptor.
            @param[in]    device    The device instance.
            @return The CaptureManager aborts the device match if the status is an error.
        */
        virtual pxcStatus PXCAPI OnCreateDevice(PXCSession::ImplDesc* /*mdesc*/, PXCCapture::Device* /*device*/)  {
            return PXC_STATUS_NO_ERROR; 
        }

        /**
            @brief    The CaptureManager callbacks this function when configuring the device streams.
            @param[in]    device    The device instance.
            @param[in]    types     The bit-OR'ed value of all streams.
            @return The CaptureManager aborts the device match if the status is an error.
        */
        virtual pxcStatus PXCAPI OnSetupStreams(PXCCapture::Device* /*device*/, PXCCapture::StreamType /*types*/) {
            return PXC_STATUS_NO_ERROR; 
        }

        /**
            @brief    The CaptureManager callbacks this function when the current device failed to 
            meet the algorithm needs. If the function returns any error, the CaptureManager performs
            the current device match again, allowing to try multiple configurations on the same device.
            @param[in]    device    The device instance.
            @return The CaptureManager repeats the match on the same device if the status code is any 
            error, or go onto the next device if the status code is no error.
        */
        virtual pxcStatus PXCAPI OnNextDevice(PXCCapture::Device*)  { return PXC_STATUS_NO_ERROR; }
    };

    /**
        @brief    The function adds the specified DeviceInfo to the DeviceInfo filter list.
        @param[in] dinfo    The DeviceInfo structure to be added to the filter list, or NULL to clean up the filter list.
    */
    virtual void PXCAPI FilterByDeviceInfo(PXCCapture::DeviceInfo *dinfo)=0;

    /**
        @brief    The function adds the specified device information to the DeviceInfo filter list.
        @param[in] name     The optional device friendly name.
        @param[in] did      The optional device symbolic name.
        @param[in] didx     The optional device index.
    */
    void __inline FilterByDeviceInfo(pxcCHAR *name, pxcCHAR *did, pxcI32 didx) {
        PXCCapture::DeviceInfo dinfo;
        memset(&dinfo,0,sizeof(dinfo));
        if (name) wcscpy_s<sizeof(dinfo.name)/sizeof(pxcCHAR)>(dinfo.name,name);
        if (did) wcscpy_s<sizeof(dinfo.did)/sizeof(pxcCHAR)>(dinfo.did,did);
        dinfo.didx=didx;
        FilterByDeviceInfo(&dinfo);
    }

    /**
        @brief    The function adds the specified StreamProfile to the StreamProfile filter list.
        @param[in] dinfo    The stream configuration to be added to the filter list, or NULL to clean up the filter list.
    */
    virtual void PXCAPI FilterByStreamProfiles(PXCCapture::Device::StreamProfileSet *profiles)=0;

    /**
        @brief    The function adds the specified StreamProfile to the StreamProfile filter list.
        @param[in] type     The stream type.
        @param[in] width    The optional image width.
        @param[in] height   The optional image height.
        @param[in] fps      The optional frame rate.
    */
    void __inline FilterByStreamProfiles(PXCCapture::StreamType type, pxcI32 width, pxcI32 height, pxcF32 fps) {
        PXCCapture::Device::StreamProfileSet profiles={};
        profiles[type].imageInfo.width=width;
        profiles[type].imageInfo.height=height;
        profiles[type].frameRate.min=profiles[type].frameRate.max=fps;
        FilterByStreamProfiles(&profiles);
    }

    /**
        @brief    Add the module input needs to the CaptureManager device search. The application must call
        this function for all modules before the LocalStreams function, where the CaptureManager performs
        the device match.
        @param[in]  mid         The module identifier. The application can use any unique value to later identify the module.
        @param[in]  inputs      The module input descriptor.
        @return PXC_STATUS_NO_ERROR        Successful execution.
    */
    virtual pxcStatus PXCAPI RequestStreams(pxcUID mid, PXCVideoModule::DataDesc *inputs)=0;

    /**
        @brief    The function locates an I/O device that meets any module input needs previously specified
        by the RequestStreams function. The device and its streams are configured upon a successful return.
        @param[in]  handler     The optional handler instance for callbacks during the device match.
        @return PXC_STATUS_NO_ERROR        Successful execution.
    */
    virtual pxcStatus PXCAPI LocateStreams(Handler *handler)=0;

    /**
        @brief    The function locates an I/O device that meets any module input needs previously specified
        by the RequestStreams function. The device and its streams are configured upon a successful return.
        @return PXC_STATUS_NO_ERROR        Successful execution.
    */
    pxcStatus __inline LocateStreams(void) { 
        return LocateStreams(0); 
    }

    /**
        @brief    Close the streams.
    */
    virtual void PXCAPI CloseStreams(void)=0;

    /**
        @brief    Return the capture instance.
        @return the capture instance.
    */
    virtual PXCCapture* PXCAPI QueryCapture(void)=0;

    /**
        @brief    Return the device instance.
        @return the device instance.
    */
    virtual PXCCapture::Device* PXCAPI QueryDevice(void)=0;

    /**
        @brief    Return the stream resolution of the specified stream type.
        @param[in]  type        The stream type.
        @return the stream resolution.
    */
    virtual PXCSizeI32 PXCAPI QueryImageSize(PXCCapture::StreamType type)=0;

    /**
        @brief    Read the image samples for a specified module.
        @param[in]  mid         The module identifier.
        @param[out] sample      The captured sample, to be returned.
        @param[out] sp          The SP, to be returned.
        @return PXC_STATUS_NO_ERROR        Successful execution.
    */
    virtual pxcStatus PXCAPI ReadModuleStreamsAsync(pxcUID mid, PXCCapture::Sample *sample, PXCSyncPoint **sp)=0;

    /**
        @brief    Setup file recording or playback.
        @param[in]  file        The file name.
        @param[in]  record      If true, the file is opened for recording. Otherwise, the file is opened for playback.
        @return PXC_STATUS_NO_ERROR        Successful execution.
    */
    virtual pxcStatus PXCAPI SetFileName(const pxcCHAR *file, pxcBool record)=0;

    /**
        @brief    Set up to record or playback certain stream types.
        @param[in]  types       The bit-OR'ed stream types.
    */
    virtual void PXCAPI SetMask(PXCCapture::StreamType types)=0;

    /**
        @brief    Pause/Resume recording or playing back.
        @param[in]  pause       True for pause and false for resume.
    */
    virtual void PXCAPI SetPause(pxcBool pause)=0;

    /**
        @brief    Set the realtime playback mode.
        @param[in]  realtime    True to playback in real time, or false to playback as fast as possible.
    */
    virtual void PXCAPI SetRealtime(pxcBool realtime)=0;

    /**
        @brief    Reset the playback position by the frame index.
        @param[in]  frame       The frame index.
    */
    virtual void PXCAPI SetFrameByIndex(pxcI32 frame)=0;

    /**
        @brief    Return the current playback position in frame index.
        @return The frame index.
    */
    virtual pxcI32 PXCAPI QueryFrameIndex(void)=0;

    /**
        @brief    Reset the playback position by the nearest time stamp.
        @param[in]  ts          The time stamp, in 100ns.
    */
    virtual void PXCAPI SetFrameByTimeStamp(pxcI64 ts)=0;

    /**
        @brief    Return the current playback frame time stamp.
        @return The time stamp, in 100ns.
    */
    virtual pxcI64 PXCAPI QueryFrameTimeStamp(void)=0;

    /**
        @brief    Return the frame number in the recorded file.
        @return The number of frames.
    */
    virtual pxcI32 PXCAPI QueryNumberOfFrames(void)=0;
};
