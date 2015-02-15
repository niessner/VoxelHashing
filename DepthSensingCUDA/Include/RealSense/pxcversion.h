/*******************************************************************************

INTEL CORPORATION PROPRIETARY INFORMATION
This software is supplied under the terms of a license agreement or nondisclosure
agreement with Intel Corporation and may not be copied or disclosed except in
accordance with the terms of that agreement
Copyright(c) 2012 Intel Corporation. All Rights Reserved.

*******************************************************************************/
#pragma once

#define PXC_VERSION_MAJOR    2
#define PXC_VERSION_MINOR    0
#define PXC_VERSION_BUILD    1
#define PXC_VERSION_REVISION 22222

#define RSSDK_REG_DEV           TEXT("Software\\Intel\\RSSDK")
#define RSSDK_REG_DEV32         TEXT("Software\\Wow6432Node\\Intel\\RSSDK")

#define RSSDK_REG_RUNTIME       TEXT("Software\\Intel\\RSSDK\\v2")
#define RSSDK_REG_RUNTIME32     TEXT("Software\\Wow6432Node\\Intel\\RSSDK\\v2")

#define RSSDK_REG_DISPATCH      RSSDK_REG_RUNTIME   TEXT("\\Dispatch")
#define RSSDK_REG_DISPATCH32    RSSDK_REG_RUNTIME32 TEXT("\\Dispatch")

#define RSDCM_REG_DEV       TEXT("Software\\Intel\\RSDCM")
#define RSDCM_REG_DEV32     TEXT("Software\\Wow6432Node\\Intel\\RSDCM")
