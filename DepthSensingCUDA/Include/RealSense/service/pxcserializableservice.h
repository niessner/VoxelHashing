/*******************************************************************************

INTEL CORPORATION PROPRIETARY INFORMATION
This software is supplied under the terms of a license agreement or nondisclosure
agreement with Intel Corporation and may not be copied or disclosed except in
accordance with the terms of that agreement
Copyright(c) 2012 Intel Corporation. All Rights Reserved.

*******************************************************************************/
#pragma once
#include "pxcsession.h"

class PXCSerializableService: public PXCBase {
public:
    PXC_CUID_OVERWRITE(PXC_UID('S','L','Z','S'));

    struct ProfileInfo {
        PXCSession::ImplDesc    implDesc;
        pxcI32                  dataSize;
        pxcI32                  reserved[7];
    };

    virtual pxcStatus PXCAPI QueryProfile(ProfileInfo *pinfo, pxcBYTE *data)=0;
    virtual pxcStatus PXCAPI SetProfile(ProfileInfo *pinfo, pxcBYTE *data)=0;
};
