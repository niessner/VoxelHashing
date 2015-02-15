/*******************************************************************************

INTEL CORPORATION PROPRIETARY INFORMATION
This software is supplied under the terms of a license agreement or nondisclosure
agreement with Intel Corporation and may not be copied or disclosed except in
accordance with the terms of that agreement
Copyright(c) 2011-2013 Intel Corporation. All Rights Reserved.

*******************************************************************************/
/** @file pxcpowerstate.h
    Defines the PXCPowerState interface,  which exposes controls for
    detecting and changing the power state of the runtimes. 
  */
#pragma once
#include "pxcsession.h"

/**
   This interface manages the SDK implementation power state.  Any SDK I/O
   or algorithm module implementation that are power aware exposes this
   interface.  Programs may use the QueryInstance function to bind to this
   interface from any module instance.
 */
class PXCPowerState: public PXCBase {
public:
    PXC_CUID_OVERWRITE(PXC_UID('P','W','M','G'));

    enum State 
    {
        STATE_PERFORMANCE,         /* full feature set/best algorithm */
        STATE_BATTERY,   
    };

    /* Query current power state of the device, returns maximal used state */
    virtual PXCPowerState::State PXCAPI QueryState()=0;

    /* Try to set power state of all used devices, all streams, application should call 
       QueryStream to check if the desired state was set */
    virtual pxcStatus PXCAPI SetState(State state)=0;
};










