/*******************************************************************************

INTEL CORPORATION PROPRIETARY INFORMATION
This software is supplied under the terms of a license agreement or nondisclosure
agreement with Intel Corporation and may not be copied or disclosed except in
accordance with the terms of that agreement
Copyright(c) 2013-2014 Intel Corporation. All Rights Reserved.

*******************************************************************************/
#pragma once
#include "pxcbase.h"


class PXCHandConfiguration;
class PXCHandData;

/**
	@Class PXCHandModule 
	The main interface for the hand-module's classes.
	Use this interface to get access to the hand-module's configuration and output data
*/
class PXCHandModule : public PXCBase 
{
public:

	PXC_CUID_OVERWRITE(PXC_UID('H','A','N','N'));

	/** 
	Create a new instance of the hand-module's active configuration.
	@return a pointer to the configuration instance 
	@see PXCHandConfiguration
	*/
	virtual PXCHandConfiguration* PXCAPI CreateActiveConfiguration() = 0;

	/** 
	Create a new instance of the hand-module's output data 
	@return a pointer to the output-data instance
	@see PXCHandData
	*/
	virtual PXCHandData* PXCAPI CreateOutput() = 0;
};
