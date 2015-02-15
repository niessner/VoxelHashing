/*******************************************************************************

INTEL CORPORATION PROPRIETARY INFORMATION
This software is supplied under the terms of a license agreement or nondisclosure
agreement with Intel Corporation and may not be copied or disclosed except in
accordance with the terms of that agreement
Copyright(c) 2013-2014 Intel Corporation. All Rights Reserved.

*******************************************************************************/
#pragma once
#include "pxcpowerstate.h"

class PXCPowerStateServiceClient :public PXCBase {
public:
    PXC_CUID_OVERWRITE(PXC_UID('P','W','M','C'));

    /* Register module with the Power Manager, specifying desired device & stream to rule power settings on */
    /* Upon registering, he unique ID will be assigned to registered module */
    /* Unique ID should be used by client to identify itself */
    virtual pxcStatus RegisterModule( pxcI32 deviceId, pxcI32 streamId, PXCSession::ImplGroup group, PXCSession::ImplSubgroup subGroup, pxcI32* pUniqueId ) = 0;

    /* Register module with the Power Manager, specifying desired device & stream to rule power settings on, and set initial state */
    virtual pxcStatus RegisterModule( pxcI32 deviceId, pxcI32 streamId, PXCPowerState::State initialState, PXCSession::ImplGroup group, PXCSession::ImplSubgroup subGroup, pxcI32* uniqueId ) = 0;

    /* Unregister module from certain device & stream. All further requests for this device from this module will be ignored */
    virtual pxcStatus UnregisterModule( pxcI32 uniqueId ) = 0;

    /* Request state for stream on device, module may call QueryState to test if the state was actually set */
    virtual pxcStatus SetState(  pxcI32 uniqueId , PXCPowerState::State state ) = 0;

    /* Query power state on stream on device */
    virtual pxcStatus QueryState( pxcI32 uniqueId , PXCPowerState::State* state) = 0;

};

