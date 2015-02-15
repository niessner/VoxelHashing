/*******************************************************************************

INTEL CORPORATION PROPRIETARY INFORMATION
This software is supplied under the terms of a license agreement or nondisclosure
agreement with Intel Corporation and may not be copied or disclosed except in
accordance with the terms of that agreement
Copyright(c) 2014 Intel Corporation. All Rights Reserved.

*******************************************************************************/
#pragma once
#include "pxcsyncpoint.h"

/* service for PXCScheduler::PXCSyncPoint interface */
class PXCSyncPointService:public PXCBase {
public:
    PXC_CUID_OVERWRITE(PXC_UID('S','Y','N','1'));

    virtual pxcStatus PXCAPI SignalSyncPoint(pxcStatus status)=0;
};
