/*******************************************************************************

INTEL CORPORATION PROPRIETARY INFORMATION
This software is supplied under the terms of a license agreement or nondisclosure
agreement with Intel Corporation and may not be copied or disclosed except in
accordance with the terms of that agreement
Copyright(c) 2011-2014 Intel Corporation. All Rights Reserved.

*******************************************************************************/
#ifndef __PXCADDREF_H__
#define __PXCADDREF_H__
#include "pxcbase.h"

///////////////////////////////////////////////////////////////////////////////////////

/**
The interface adds a reference count to the supported object.
*/
class PXCAddRef {
public:
    PXC_CUID_OVERWRITE(PXC_UID('B','A','S','S'));

    /** 
        @brief Increase the reference counter of the underlying object.
        @return The increased reference counter value.
    */
    virtual pxcI32 PXCAPI AddRef(void) = 0;

private:
    /* Prohibit using delete operator */
    void operator delete(void* pthis);
};

///////////////////////////////////////////////////////////////////////////////////////

/**
This is the base implementation of the PXCAddRef interface.
*/
template <class T>
class PXCAddRefImpl:public T, public PXCAddRef {
public:

    PXCAddRefImpl()
    {
        m_refCount = 1;
    }

    virtual pxcI32 PXCAPI AddRef(void)
    {
        return InterlockedIncrement((volatile long*)&m_refCount);
    }

    virtual void   PXCAPI Release(void)
    {
        if (!InterlockedDecrement((volatile long*)&m_refCount))
            ::delete this;
    }

    virtual void* PXCAPI QueryInstance(pxcUID cuid)
    {
        return (cuid == PXCAddRef::CUID) ? (PXCAddRef*)this : T::QueryInstance(cuid);
    }

    void operator delete(void* pthis) { ((PXCBase*)pthis)->Release(); }

protected:

    __declspec(align(32)) pxcI32  m_refCount;

};

#endif
