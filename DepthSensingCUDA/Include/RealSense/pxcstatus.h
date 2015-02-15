/*******************************************************************************

INTEL CORPORATION PROPRIETARY INFORMATION
This software is supplied under the terms of a license agreement or nondisclosure
agreement with Intel Corporation and may not be copied or disclosed except in
accordance with the terms of that agreement
Copyright(c) 2011-2013 Intel Corporation. All Rights Reserved.

*******************************************************************************/
/** @file pxcstatus.h
    Defines the return status codes used by SDK interfaces.
  */
#pragma once

/**
   This enumeration defines various return codes that SDK interfaces
   use.  Negative values indicate errors, a zero value indicates success,
   and positive values indicate warnings.
 */
enum pxcStatus {
    /* success */
    PXC_STATUS_NO_ERROR=0,                      /** Indicates the operation succeeded without any warning */

    /* errors */
    PXC_STATUS_FEATURE_UNSUPPORTED=     -1,     /** Unsupported feature */
    PXC_STATUS_PARAM_UNSUPPORTED=       -2,     /** Unsupported parameter(s) */
    PXC_STATUS_ITEM_UNAVAILABLE=        -3,     /** Item not found/not available */

    PXC_STATUS_HANDLE_INVALID=          -101,   /** Invalid session, algorithm instance, or pointer */
    PXC_STATUS_ALLOC_FAILED=            -102,   /** Memory allocation failure */

    PXC_STATUS_DEVICE_FAILED=           -201,   /** device failed due to malfunctioning */
    PXC_STATUS_DEVICE_LOST=             -202,   /** device failed due to unplug or unavailability */
    PXC_STATUS_DEVICE_BUSY=             -203,   /** device busy */

    PXC_STATUS_EXEC_ABORTED=            -301,   /** Execution aborted due to errors in upstream components */
    PXC_STATUS_EXEC_INPROGRESS=         -302,   /** Asynchronous operation is in execution */
    PXC_STATUS_EXEC_TIMEOUT=            -303,   /** Operation time out */

    PXC_STATUS_FILE_WRITE_FAILED=       -401,   /** Failure in open file in WRITE mode */
    PXC_STATUS_FILE_READ_FAILED=        -402,   /** Failure in open file in READ mode */
    PXC_STATUS_FILE_CLOSE_FAILED=       -403,   /** Failure in close a file handle */
      
    PXC_STATUS_DATA_UNAVAILABLE=         -501,   /** Data not available for MW model or processing */
    
    PXC_STATUS_DATA_NOT_INITIALIZED=    -502,   /** Data failed to initialize */
    PXC_STATUS_INIT_FAILED=             -503,   /** Module failure during initialization */
    

    PXC_STATUS_STREAM_CONFIG_CHANGED=   -601,   /** Configuration for the stream has changed */

    PXC_STATUS_POWER_UID_ALREADY_REGISTERED = -701,
    PXC_STATUS_POWER_UID_NOT_REGISTERED =     -702,
    PXC_STATUS_POWER_ILLEGAL_STATE =          -703,
    PXC_STATUS_POWER_PROVIDER_NOT_EXISTS =    -704,
    PXC_STATUS_CAPTURE_CONFIG_ALREADY_SET=      -801, /** parameter cannot be changed since configuration for capturing has been already set */

    /* warnings */
    PXC_STATUS_TIME_GAP=                101,    /** time gap in time stamps */
    PXC_STATUS_PARAM_INPLACE=           102,    /** the same parameters already defined */
    PXC_STATUS_DATA_NOT_CHANGED    =        103,        /** Data not changed (no new data available)*/
    PXC_STATUS_PROCESS_FAILED=          104   /** Module failure during processing */
};
