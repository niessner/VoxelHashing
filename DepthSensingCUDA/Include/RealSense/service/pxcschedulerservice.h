/*******************************************************************************

INTEL CORPORATION PROPRIETARY INFORMATION
This software is supplied under the terms of a license agreement or nondisclosure
agreement with Intel Corporation and may not be copied or disclosed except in
accordance with the terms of that agreement
Copyright(c) 2011-2014 Intel Corporation. All Rights Reserved.

*******************************************************************************/
#pragma once
#include "service/pxcsessionservice.h"
#include "pxcsyncpoint.h"

/* scoreboarding and syncpoint services */
class PXCSchedulerService:public PXCBase {
public:
    PXC_CUID_OVERWRITE(PXC_UID('S','C','H','2'));

    /* scoreboarding services */
    class Callback:public PXCBase {
    public:
        PXC_CUID_OVERWRITE(PXC_UID('C','L','L','B'));
        virtual void  PXCAPI   Run(pxcStatus sts)=0;
        virtual const pxcCHAR* PXCAPI QueryCallbackName() { return 0; }
    };
    virtual pxcStatus PXCAPI RequestInputs(pxcI32 ninput, void** inputs, Callback *cb)=0;
    virtual pxcStatus PXCAPI MarkOutputs(pxcI32 noutput, void** outputs, pxcStatus sts)=0;

    /* create sync point */
    virtual pxcStatus PXCAPI CreateSyncPoint(pxcI32 noutput, void** outputs, PXCSyncPoint **sp)=0;
};
