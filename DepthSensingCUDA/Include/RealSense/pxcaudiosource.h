/*******************************************************************************

INTEL CORPORATION PROPRIETARY INFORMATION
This software is supplied under the terms of a license agreement or nondisclosure
agreement with Intel Corporation and may not be copied or disclosed except in
accordance with the terms of that agreement
Copyright(c) 2013-2014 Intel Corporation. All Rights Reserved.

*******************************************************************************/
#pragma once
#include "pxcbase.h"

/**
	The audio source interface manages the audio devices.
*/
class PXCAudioSource: public PXCBase {
public:
    PXC_CUID_OVERWRITE(0xD8419523);

	/**
		@struct DeviceInfo
		This structure describes the audio device information.
	*/
	struct DeviceInfo {
        pxcCHAR	name[256];	/* device name */
		pxcCHAR	did[256];   /* device identifier or the device symbolic name */
		pxcI32  reserved[16];
	};

	/**
		@brief Scan the available audio devices.
	*/
    virtual void PXCAPI ScanDevices(void)=0;

	/**
		@brief Get the number of available audio devices previously scanned.
		@return the number of audio devices.
	*/
    virtual pxcI32 PXCAPI QueryDeviceNum(void)=0;

	/**
		@brief Enumerate the audio devices previously scanned.
		@param[in]  didx		The zero-based index to enumerate all devices.
		@param[out] dinfo		The DeviceInfo structure to return the device information. 
		@return PXC_STATUS_NO_ERROR				Successful execution.
		@return PXC_STATUS_ITEM_UNAVAILABLE		No more devices.	
	*/
    virtual pxcStatus PXCAPI QueryDeviceInfo(pxcI32 didx, DeviceInfo *dinfo)=0;

	/**
		@brief Get the currnet working device
		@param[out] dinfo		The working device info
		@return PXC_STATUS_NO_ERROR				Successful execution.
	*/
    __inline pxcStatus QueryDeviceInfo(DeviceInfo *dinfo) {
		return QueryDeviceInfo(WORKING_PROFILE, dinfo);
	}

	/**
		@brief Set the audio device for subsequent module processing.
		@param[in] dinfo		The audio source
		@return PXC_STATUS_NO_ERROR				Successful execution.
	*/
    virtual pxcStatus PXCAPI SetDevice(DeviceInfo *dinfo)=0;

	/**
		@brief Get the audio device volume
		@return the volume from 0 (min) to 1 (max).
	*/
	virtual pxcF32 PXCAPI QueryVolume(void)=0;

	/**
		@brief Set the audio device volume
		@param volume    The audio volume from 0 (min) to 1 (max).
		@return PXC_STATUS_NO_ERROR Successful execution.
	*/
	virtual pxcStatus PXCAPI SetVolume(pxcF32 volume)=0;
};
