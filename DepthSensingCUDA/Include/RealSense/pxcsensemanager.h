/*******************************************************************************

INTEL CORPORATION PROPRIETARY INFORMATION
This software is supplied under the terms of a license agreement or nondisclosure
agreement with Intel Corporation and may not be copied or disclosed except in
accordance with the terms of that agreement
Copyright(c) 2013-2014 Intel Corporation. All Rights Reserved.

*******************************************************************************/
#pragma once
#include <string.h>
#include "pxccapturemanager.h"
#include "pxcsession.h"
#include "pxcfacemodule.h"
#include "pxcemotion.h"
#include "pxctracker.h"
#include "pxchandmodule.h"
#include "pxctouchlesscontroller.h"
#include "pxc3dseg.h"
#include "pxc3dscan.h"

/**
    This is the main interface for the SDK pipeline.
    Control the pipeline execution with this interface.
*/
class PXCSenseManager: public PXCBase {
public:

    PXC_CUID_OVERWRITE(0xD8954321);
    PXC_DEFINE_CONST(TIMEOUT_INFINITE,-1);

    /**
        The PXCSenseManager callback instance.
    */
    class Handler {
    public:

        /**
            @brief The SenseManager calls back this function when there is a device connection or
            disconnection. During initialization, the SenseManager callbacks this function when 
            openning or closing any capture devices.
            @param[in] device           The video device instance.
            @param[in] connected        The device connection status.
            @return The return status is ignored during the PXCSenseManager initialization. During
            streaming, the SenseManager aborts the execution pipeline if the status is an error.
        */
        virtual pxcStatus PXCAPI OnConnect(PXCCapture::Device* /*device*/, pxcBool /*connected*/) { 
            return PXC_STATUS_DEVICE_FAILED; 
        }

        /**
            @brief The SenseManager calls back this function during initialization after each device 
            configuration is set.
            @param[in] mid          The module identifier. Usually this is the interface identifier, or PXCCapture::CUID+n for raw video streams. 
            @param[in] module       The module instance, or NULL for raw video streams.
            @return The SenseManager aborts the execution pipeline if the status is an error.
        */
        virtual pxcStatus PXCAPI OnModuleSetProfile(pxcUID /*mid*/, PXCBase* /*module*/) { 
            return PXC_STATUS_NO_ERROR; 
        }

        /**
            @brief The SenseManager calls back this function after a module completed processing the frame data.
            @param[in] mid          The module identifier. Usually this is the interface identifier. 
            @param[in] module       The module instance.
            @return The SenseManager aborts the execution pipeline if the status is an error.
        */
        virtual pxcStatus PXCAPI OnModuleProcessedFrame(pxcUID /*mid*/, PXCBase* /*module*/, PXCCapture::Sample* /*sample*/) { 
            return PXC_STATUS_NO_ERROR; 
        }

        /**
            @brief The SenseManager calls back this function when raw video streams (explicitly requested) are available.
            @param[in] mid          The module identifier. Usually this is the interface identifier. 
            @param[in] sample       The sample from capture device
            @return The SenseManager aborts the execution pipeline if the status is an error.
        */
        virtual pxcStatus PXCAPI OnNewSample(pxcUID /*mid*/, PXCCapture::Sample* /*sample*/) {
            return PXC_STATUS_NO_ERROR; 
        }
    };

    /**
        @brief    Return the PXCSession instance. Internally managed. Do not release the instance.
        The session instance is managed internally by the SenseManager. Do not release the session instance.
        @return The PXCSession instance.
    */
    virtual PXCSession* PXCAPI QuerySession(void)=0;

    /**
        @brief    Return the PXCCaptureManager instance. Internally managed. Do not release the instance.
        The instance is managed internally by the SenseManager. Do not release the instance.
        @return The PXCCaptureManager instance.
    */
    virtual PXCCaptureManager* PXCAPI QueryCaptureManager(void)=0;

    /**
        @brief    Return the captured sample for the specified module or explicitly/impl requested streams. 
        For modules, use mid=module interface identifier. 
        For explictly requested streams via multiple calls to EnableStream(s), use mid=PXCCapture::CUID+0,1,2... 
        The captured sample is managed internally by the SenseManager. Do not release the instance.
        @param[in] mid        The module identifier. Usually this is the interface identifier, or PXCCapture::CUID+n for raw video streams.
        @return The sample instance, or NULL if the captured sample is not available.
    */
    virtual PXCCapture::Sample* PXCAPI QuerySample(pxcUID mid)=0;

    /**
    @brief    Return available captured sample, explicitly or implicitly requested.
    The captured sample is managed internally by the SenseManager. Do not release the sample.
    @return The sample instance, or NULL if the captured sample is not available.
    */
    __inline PXCCapture::Sample* QuerySample(void) {
        return QuerySample(0);
    }

    /**
        @brief    Return the captured sample for the face module.
        The captured sample is managed internally by the SenseManager. Do not release the sample.
        @return The sample instance, or NULL if the captured sample is not available.
    */
    __inline PXCCapture::Sample* QueryFaceSample(void) {
        return QuerySample(PXCFaceModule::CUID);
    }

    /**
        @brief    Return the captured sample for the hand module.
        The captured sample is managed internally by the SenseManager. Do not release the sample.
        @return The sample instance, or NULL if the captured sample is not available.
    */
    __inline PXCCapture::Sample* QueryHandSample(void) {
        return QuerySample(PXCHandModule::CUID);
    }

    /**
        @brief    Return the captured sample for the emotion module.
        The captured sample is managed internally by the SenseManager. Do not release the sample.
        @return The sample instance, or NULL if the captured sample is not available.
    */
    __inline PXCCapture::Sample* QueryEmotionSample(void) {
        return QuerySample(PXCEmotion::CUID);
    }

    /**
		@brief	  Return the captured sample for the object tracker module.
		The captured sample is managed internally by the SenseManager. Do not release the sample.
		@return The sample instance, or NULL if the captured sample is not available.
	*/
	__inline const PXCCapture::Sample* QueryTrackerSample(void) {
		return QuerySample(PXCTracker::CUID);
	}


    /**
        @brief    Return the module instance. Between AcquireFrame/ReleaseFrame, the function returns
        NULL if the specified module hasn't completed processing the current frame of image data.
        The instance is managed internally by the SenseManager. Do not release the instance.
        @param[in] mid        The module identifier. Usually this is the interface identifier.
        @return The module instance.
    */
    virtual PXCBase* PXCAPI QueryModule(pxcUID mid)=0;

    /**
        @brief    Return the Face module instance. Between AcquireFrame/ReleaseFrame, the function returns
        NULL if the specified module hasn't completed processing the current frame of image data.
        The instance is managed internally by the SenseManager. Do not release the instance.
        @return The module instance.
    */
    __inline PXCFaceModule* QueryFace(void) { 
        PXCBase *instance=QueryModule(PXCFaceModule::CUID);
        return instance?instance->QueryInstance<PXCFaceModule>():0; 
    }

    /**
        @brief    Return the emotion module instance. Between AcquireFrame/ReleaseFrame, the function returns
        NULL if the specified module hasn't completed processing the current frame of image data.
        The instance is managed internally by the SenseManager. Do not release the instance.
        @return The module instance.
    */
    __inline PXCEmotion* QueryEmotion(void) { 
        PXCBase *instance=QueryModule(PXCEmotion::CUID);
        return instance?instance->QueryInstance<PXCEmotion>():0; 
    }

    /**
		@brief	Return the tracker module instance. Between AcquireFrame/ReleaseFrame, the function returns
		NULL if the specified module hasn't completed processing the current frame of image data.
		The instance is managed internally by the SenseManager. Do not release the instance.
		@return The module instance.
	*/
	__inline PXCTracker* QueryTracker(void) { 
		PXCBase *instance=QueryModule(PXCTracker::CUID);
		return instance?instance->QueryInstance<PXCTracker>():0; 
	}


    /**
        @brief    Return the hand module instance. Between AcquireFrame/ReleaseFrame, the function returns
        NULL if the specified module hasn't completed processing the current frame of image data.
        The instance is managed internally by the SenseManager. Do not release the instance.
        @return The module instance.
    */
    __inline PXCHandModule* QueryHand(void) { 
        PXCBase *instance=QueryModule(PXCHandModule::CUID);
        return instance?instance->QueryInstance<PXCHandModule>():0;
    }

    /**
        @brief    Return the Touchless module instance. Between AcquireFrame/ReleaseFrame, the function returns
        NULL if the specified module hasn't completed processing the current frame of image data.
        The instance is managed internally by the SenseManager. Do not release the instance.
        @return The module instance.
    */
    __inline PXCTouchlessController* QueryTouchlessController(void) { 
        PXCBase *instance=QueryModule(PXCTouchlessController::CUID);
        return instance?instance->QueryInstance<PXCTouchlessController>():0;
    }

    /**
        @brief    Return the 3D Segmentation module instance. Between AcquireFrame/ReleaseFrame, the function returns
        NULL if the specified module hasn't completed processing the current frame of image data.
        The instance is managed internally by the SenseManager. Do not release the instance.
        @return The module instance.
    */
    __inline PXC3DSeg* Query3DSeg(void) { 
        PXCBase *instance=QueryModule(PXC3DSeg::CUID);
        return instance?instance->QueryInstance<PXC3DSeg>():0; 
    }
    
    /**
        @brief    Return the Mesh Capture module instance. Between AcquireFrame/ReleaseFrame, the function returns
        NULL if the specified module hasn't completed processing the current frame of image data.
        The instance is managed internally by the SenseManager. Do not release the instance.
        @return The module instance.
    */
    __inline PXC3DScan* Query3DScan(void) { 
        PXCBase *instance=QueryModule(PXC3DScan::CUID);
        return instance?instance->QueryInstance<PXC3DScan>():0; 
    }
    
    /**
        @brief    Initialize the SenseManager pipeline for streaming with callbacks. The application must 
        enable raw streams or algorithm modules before this function.
        @param[in] handler          Optional callback instance. 
        @return PXC_STATUS_NO_ERROR        Successful execution.
    */
    virtual pxcStatus PXCAPI Init(Handler *handler)=0;

    /**
        @brief    Initialize the SenseManager pipeline for streaming. The application must enable raw 
        streams or algorithm modules before this function.
        @return PXC_STATUS_NO_ERROR        Successful execution.
    */
    __inline pxcStatus Init(void) { 
        return Init(0); 
    }

    /**
        @brief    Stream frames from the capture module to the algorithm modules. The application must 
        initialize the pipeline before calling this function. If blocking, the function blocks until
        the streaming stops (upon any capture device error or any callback function returns any error.
        If non-blocking, the function returns immediately while running streaming in a thread.
        AcquireFrame/ReleaseFrame are not compatible with StreamFrames. Run the SenseManager in the pulling
        mode with AcquireFrame/ReleaseFrame, or the callback mode with StreamFrames.
        @param[in]    blocking      The blocking status.
        @return PXC_STATUS_NO_ERROR        Successful execution.
    */
    virtual pxcStatus PXCAPI StreamFrames(pxcBool blocking)=0;

    /**
        @brief    Stream frames from the capture module to the algorithm modules. The application must 
        initialize the pipeline before calling this function. If blocking, the function blocks until
        the streaming stops (upon any capture device error or any callback function returns any error.
        If non-blocking, the function returns immediately while running streaming in a thread.
        @param[in]    blocking      The blocking status.
        @return PXC_STATUS_NO_ERROR        Successful execution.
    */
    virtual pxcBool PXCAPI IsConnected(void)=0;

    /**
        @brief    This function starts streaming and waits until certain events occur. If ifall=true, 
        the function blocks until all samples are ready and the modules completed processing the samples.
        If ifall=false, the function blocks until any of the mentioned is ready. The SenseManager 
        pipeline pauses at this point for the application to retrieve the processed module data, until 
        the application calls ReleaseFrame.
        AcquireFrame/ReleaseFrame are not compatible with StreamFrames. Run the SenseManager in the pulling
        mode with AcquireFrame/ReleaseFrame, or the callback mode with StreamFrames.
        @param[in]    ifall                If true, wait for all modules to complete processing the data.
        @param[in]    timeout                The time out value in milliseconds.
        @return PXC_STATUS_NO_ERROR        Successful execution.
    */
    virtual pxcStatus PXCAPI AcquireFrame(pxcBool ifall, pxcI32 timeout)=0;

    /**
        @brief    This function starts streaming and waits until certain events occur. If ifall=true, 
        the function blocks until all samples are ready and the modules completed processing the samples.
        If ifall=false, the function blocks until any of the mentioned is ready. The SenseManager 
        pipeline pauses at this point for the application to retrieve the processed module data, until 
        the application calls ReleaseFrame.
        AcquireFrame/ReleaseFrame are not compatible with StreamFrames. Run the SenseManager in the pulling
        mode with AcquireFrame/ReleaseFrame, or the callback mode with StreamFrames.
        @param[in]    ifall         If true, wait for all modules to complete processing the data.
        @return PXC_STATUS_NO_ERROR        Successful execution.
    */
    __inline pxcStatus AcquireFrame(pxcBool ifall=true) { 
        return AcquireFrame(ifall, (pxcI32)TIMEOUT_INFINITE); 
    }

    /**
        @reserved DO NOT USE THIS FUNCTION.
        @brief    This function discards any internally cached sample from the capture device (and restart
        new streaming.) Use this function together after file playback repositioning to avoid any caching 
        effects. Use also this function in the snapshot mode to get the most recent sample without any 
        streaming caching effect.
    */
    virtual void PXCAPI FlushFrame(void)=0;

    /**
        @brief    This function resumes streaming after AcquireFrame.
        AcquireFrame/ReleaseFrame are not compatible with StreamFrames. Run the SenseManager in the pulling
        mode with AcquireFrame/ReleaseFrame, or the callback mode with StreamFrames.
    */
    virtual void PXCAPI ReleaseFrame(void)=0;

    /**
        @brief    This function closes the execution pipeline.
    */
    virtual void PXCAPI Close(void)=0;

    /**
        @brief    Explicitly request to stream the specified raw streams. If specified more than a stream, 
        SenseManager will synchronize these streams. If called multiple times, the function treats each
        stream request as independent (unaligned.) The stream identifier is PXCCapture::CUID+n.
        @param[in] sdesc            The stream descriptor.
        @return PXC_STATUS_NO_ERROR        Successful execution.
    */
    virtual pxcStatus PXCAPI EnableStreams(PXCVideoModule::DataDesc *sdesc)=0;

    /**
        @brief    Explicitly request to stream the specified raw stream. If specified more than one stream, 
        SenseManager will synchronize these streams. If called multiple times, the function treats each
        stream request as independent (unaligned). The stream identifier is PXCCapture::CUID+n.
        @param[in] type             The stream type.
        @param[in] width            Optional width.
        @param[in] height           Optional height.
        @param[in] fps              Optional frame rate.
        @return PXC_STATUS_NO_ERROR        Successful execution.
    */
    __inline pxcStatus EnableStream(PXCCapture::StreamType type, pxcI32 width=0, pxcI32 height=0, pxcF32 fps=0) {
        PXCVideoModule::DataDesc ddesc={};
        ddesc.deviceInfo.streams = type;
        PXCVideoModule::StreamDesc& sdesc=ddesc.streams[type];
        sdesc.sizeMin.width=sdesc.sizeMax.width=width;
        sdesc.sizeMin.height=sdesc.sizeMax.height=height;
        sdesc.frameRate.min=fps;
        sdesc.frameRate.max=fps;
        return EnableStreams(&ddesc);
    }

    /**
        @brief    Enable a module in the pipeline.
        @param[in] mid              The module identifier. This is usually the interface identifier.
        @param[in] mdesc            The module descriptor.
        @return PXC_STATUS_NO_ERROR        Successful execution.
    */
    virtual pxcStatus PXCAPI EnableModule(pxcUID mid, PXCSession::ImplDesc *mdesc)=0;

    /**
        @brief    Enable the face module in the pipeline.
        @param[in] name             The optional module name.
        @return PXC_STATUS_NO_ERROR        Successful execution.
    */
    __inline pxcStatus EnableFace(pxcCHAR *name=0) {
        PXCSession::ImplDesc mdesc;
        memset(&mdesc,0,sizeof(mdesc));
        mdesc.cuids[0]=PXCFaceModule::CUID;
        if (name) wcscpy_s<sizeof(mdesc.friendlyName)/sizeof(pxcCHAR)>(mdesc.friendlyName, name);
        return EnableModule(PXCFaceModule::CUID,&mdesc);
    }

    /**
        @brief    Enable the emotion module in the pipeline.
        @return PXC_STATUS_NO_ERROR        Successful execution.
    */
    __inline pxcStatus EnableEmotion(void) {
        PXCSession::ImplDesc mdesc;
        memset(&mdesc,0,sizeof(mdesc));
        mdesc.cuids[0]=PXCEmotion::CUID;
        return EnableModule(PXCEmotion::CUID,&mdesc);
    }

    /**
		@brief	Enable the Tracker module in the pipeline.
		@return PXC_STATUS_NO_ERROR		Successful execution.
	*/
	__inline pxcStatus EnableTracker(void) {
		PXCSession::ImplDesc mdesc;
		memset(&mdesc,0,sizeof(mdesc));
		mdesc.cuids[0]=PXCTracker::CUID;
		return EnableModule(PXCTracker::CUID,&mdesc);
	}


    /**
        @brief    Enable the hand module in the pipeline.
        @param[in] name        The optional module name.
        @return PXC_STATUS_NO_ERROR        Successful execution.
    */
    __inline pxcStatus EnableHand(pxcCHAR *name=0) {
        PXCSession::ImplDesc mdesc;
        memset(&mdesc,0,sizeof(mdesc));
        mdesc.cuids[0]=PXCHandModule::CUID;
        if (name) wcscpy_s<sizeof(mdesc.friendlyName)/sizeof(pxcCHAR)>(mdesc.friendlyName, name);
        return EnableModule(PXCHandModule::CUID,&mdesc);
    }

    /**
        @brief    Enable the touchless controller module in the pipeline.
        @return PXC_STATUS_NO_ERROR        Successful execution.
    */
    __inline pxcStatus EnableTouchlessController(void){
        PXCSession::ImplDesc mdesc;
        memset(&mdesc,0,sizeof(mdesc));
        mdesc.cuids[0]=PXCTouchlessController::CUID;
        return EnableModule(PXCTouchlessController::CUID,&mdesc);
    }

    /**
        @brief    Enable the 3D Segmentation module in the pipeline.
        @return PXC_STATUS_NO_ERROR        Successful execution.
    */
    __inline pxcStatus Enable3DSeg(pxcCHAR *name = NULL) {
        PXCSession::ImplDesc mdesc;
        memset(&mdesc,0,sizeof(mdesc));
        mdesc.cuids[0]=PXC3DSeg::CUID;
        if (name) wcscpy_s<sizeof(mdesc.friendlyName)/sizeof(pxcCHAR)>(mdesc.friendlyName,name);
        return EnableModule(PXC3DSeg::CUID,&mdesc);
    }
    

    /**
        @brief    Enable the Mesh Capture module in the pipeline.
        @return PXC_STATUS_NO_ERROR        Successful execution.
    */
    __inline pxcStatus Enable3DScan(pxcCHAR *name = NULL) {
        PXCSession::ImplDesc mdesc;
        memset(&mdesc,0,sizeof(mdesc));
        mdesc.cuids[0]=PXC3DScan::CUID;
        if (name) wcscpy_s<sizeof(mdesc.friendlyName)/sizeof(pxcCHAR)>(mdesc.friendlyName,name);
        return EnableModule(PXC3DScan::CUID,&mdesc);
    }

    /**
        @brief    Pause/Resume the execution of the specified module.
        @param[in] mid          The module identifier. This is usually the interface identifier.
        @param[in] pause        If true, pause the module. Otherwise, resume the module.
    */
    virtual void PXCAPI PauseModule(pxcUID mid, pxcBool pause)=0;

    /**
        @brief    Pause/Resume the execution of the face module.
        @param[in] pause        If true, pause the module. Otherwise, resume the module.
    */
    __inline void PauseFace(pxcBool pause) { 
        PauseModule(PXCFaceModule::CUID,pause); 
    }

    /**
        @brief    Pause/Resume the execution of the emotion module.
        @param[in] pause        If true, pause the module. Otherwise, resume the module.
    */
    __inline void PauseEmotion(pxcBool pause) {
        PauseModule(PXCEmotion::CUID,pause); 
    }

    /**
		@brief	Pause/Resume the execution of the Tracker module.
		@param[in] pause	If true, pause the module. Otherwise, resume the module.
	*/
    __inline void PauseTracker(pxcBool pause) {
		PauseModule(PXCTracker::CUID,pause); 
	}


    /**
        @brief    Pause/Resume the execution of the hand module.
        @param[in] pause        If true, pause the module. Otherwise, resume the module.
    */
    __inline void PauseHand(pxcBool pause) {
        PauseModule(PXCHandModule::CUID,pause); 
    }

    /**
        @brief    Pause/Resume the execution of the touchless controller module.
        @param[in] pause        If true, pause the module. Otherwise, resume the module.
    */
    __inline void PauseTouchlessController(pxcBool pause) {
        PauseModule(PXCTouchlessController::CUID,pause); 
    }

    /**
        @brief    Pause/Resume the execution of the 3D Segmentation module.
        @param[in] pause        If true, pause the module. Otherwise, resume the module.
    */
    __inline void Pause3DSeg(pxcBool pause) {
        PauseModule(PXC3DSeg::CUID,pause);
    }

    /**
        @brief    Create an instance of the PXCSenseManager interface.
        @return The PXCSenseManager instance.
    */
    __inline static PXCSenseManager* CreateInstance(void) {
        PXCSession *session=PXCSession_Create();
        if (!session) return 0;
        PXCSenseManager *sm=session->CreateSenseManager();
        session->Release();
        return sm;
    }
};
