/*******************************************************************************

INTEL CORPORATION PROPRIETARY INFORMATION
This software is supplied under the terms of a license agreement or nondisclosure
agreement with Intel Corporation and may not be copied or disclosed except in
accordance with the terms of that agreement
Copyright(c) 2011-2014 Intel Corporation. All Rights Reserved.

*******************************************************************************/
/** @file pxcmetadata.h
    Defines the PXCMetadata interface, which is used for managing
    metadata storage.
 */
#pragma once
#include "pxcbase.h"

/**
    This interface manages metadata storage.  The PXCSession, PXCImage
    and PXCAudio implementations expose the PXCMetadata interface.
 */
class PXCMetadata:public PXCBase {
public:
    PXC_CUID_OVERWRITE(0x62398423);

    /**
        @brief The function returns a unique identifier for the meta data storage.
        @return the unique identifier.
    */
    virtual pxcUID PXCAPI QueryUID(void)=0;

    /**
        @brief The function retrieves the identifiers of all available meta data.
        @param[in] idx          The zero-based index to retrieve all identifiers.
        @return the metadata identifier, or zero if not available.
    */
    virtual pxcUID PXCAPI QueryMetadata(pxcI32 idx)=0;

    /**
        @brief The function detaches the specified metadata.
        @param[in] id           The metadata identifier.
        @return PXC_STATUS_NO_ERROR                Successful execution.
        @return PXC_STATUS_ITEM_UNAVAILABLE        The metadata is not found.
    */
    virtual pxcStatus PXCAPI DetachMetadata(pxcUID id)=0;

    /**
        @brief The function attaches the specified metadata.
        @param[in] id           The metadata identifier.
        @param[in] buffer       The metadata buffer.
        @param[in] size         The metadata buffer size, in bytes.
        @return PXC_STATUS_NO_ERROR                Successful execution.
    */
    virtual pxcStatus PXCAPI AttachBuffer(pxcUID id, pxcBYTE *buffer, pxcI32 size)=0;

    /**
        @brief The function returns the specified metadata buffer size.
        @param[in] id           The metadata identifier.
        @return the metadata buffer size, or zero if the metadata is not available.
    */
    virtual pxcI32 PXCAPI QueryBufferSize(pxcUID id)=0;

    /**
        @brief The function retrieves the specified metadata.
        @param[in] id           The metadata identifier.
        @param[in] buffer       The buffer pointer to retrieve the metadata.
        @param[in] size         The buffer size in bytes.
        @return PXC_STATUS_NO_ERROR         Successful execution.
    */
    virtual pxcStatus PXCAPI QueryBuffer(pxcUID id, pxcBYTE *buffer, pxcI32 size)=0;

    /**
        @brief The function attaches an instance of a serializeable interface to be metadata storage.
        @param[in] id           The metadata identifier.
        @param[in] instance     The serializable instance.
        @return PXC_STATUS_NO_ERROR         Successful execution.
    */
    virtual pxcStatus PXCAPI AttachSerializable(pxcUID id, PXCBase *instance)=0;

    /**
        @brief The function creates an instance of a serializeable interface from the metadata storage.
        @param[in] id           The metadata identifier.
        @param[in] cuid         The interface identifier.
        @param[out] instance    The serializable instance, to be returned.
        @return PXC_STATUS_NO_ERROR         Successful execution.
    */
    virtual pxcStatus PXCAPI CreateSerializable(pxcUID id, pxcUID cuid, void **instance)=0;

    /**
        @brief The function creates an instance of a serializeable interface from the metadata storage.
        @param[in] id           The metadata identifier.
        @param[out] instance    The serializable instance, to be returned.
        @return PXC_STATUS_NO_ERROR         Successful execution.
    */
    template <class T> pxcStatus __inline CreateSerializable(pxcUID id, T **instance) {
        return CreateSerializable(id, T::CUID, (void**)instance); 
    }
};
