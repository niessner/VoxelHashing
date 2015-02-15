/*******************************************************************************
INTEL CORPORATION PROPRIETARY INFORMATION
This software is supplied under the terms of a license agreement or nondisclosure
agreement with Intel Corporation and may not be copied or disclosed except in
accordance with the terms of that agreement
Copyright(c) 2014 Intel Corporation. All Rights Reserved.
*******************************************************************************/
/// @file pxc3dseg.h
/// User Segmentation video module interface

#ifndef PXC3DSEG_H
#define PXC3DSEG_H
#include "pxccapture.h"

class PXC3DSeg : public PXCBase 
{
public:
    /// Allocate and return a copy of the module's most recent segmented image
    /// The returned object's Release method can be used to deallocate it
    virtual PXCImage* PXCAPI AcquireSegmentedImage(void)=0;

    PXC_CUID_OVERWRITE(PXC_UID('S','G','I','1'));
};
#endif


