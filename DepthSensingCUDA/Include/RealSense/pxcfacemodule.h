/*******************************************************************************

INTEL CORPORATION PROPRIETARY INFORMATION
This software is supplied under the terms of a license agreement or nondisclosure
agreement with Intel Corporation and may not be copied or disclosed except in
accordance with the terms of that agreement
Copyright(c) 2011-2014 Intel Corporation. All Rights Reserved.

*******************************************************************************/
#pragma once
#include "pxcbase.h"

class PXCFaceConfiguration;
class PXCFaceData;
class PXCFaceModule : public PXCBase 
{
public:
	PXC_CUID_OVERWRITE(PXC_UID('F','A','3','D'));
	
	/* create a new copy of active configuration */
	virtual PXCFaceConfiguration* PXCAPI CreateActiveConfiguration() = 0;

	/* create a placeholder for output */
	virtual PXCFaceData* PXCAPI CreateOutput() = 0;
};
