/*******************************************************************************

INTEL CORPORATION PROPRIETARY INFORMATION
This software is supplied under the terms of a license agreement or nondisclosure
agreement with Intel Corporation and may not be copied or disclosed except in
accordance with the terms of that agreement
Copyright(c) 2011-2014 Intel Corporation. All Rights Reserved.

*******************************************************************************/
/** @file pxcdefs.h
    Defines various primitive types used throughout the SDK as well as some
    common structures.  All primitive types start with the prefix ``pxc''.
    Integral primitive types follow that with either an `I' or `U' to indicate
    signed or unsigned and then an integer indicating the number of bits used
    by that type.  For instance, pxcI32 aliases a 32-bit integral type.

    Finally, this class also defines some helper macros to construct some of
    these types as well as other general tasks.
 */
#pragma once

//TODO
//#if defined (WIN32) || defined(WIN64) || defined(WINAPI_FAMILY)
//#include <wchar.h>
typedef wchar_t pxcCHAR;
//#else
//typedef char pxcCHAR;
//#endif

/** A signed 16-bit integer */
typedef short               pxcI16;
/** This type is used as a unique identifier */
typedef int                 pxcUID;
/** A signed 32-bit integer */
typedef int                 pxcI32;
/** An signed 64-bit integer */
typedef __int64             pxcI64;
/** An unsigned 16-bit integer */
typedef unsigned short int  pxcU16;
/** A byte (an unsigned 8-bit integer) */
typedef unsigned char       pxcBYTE;
/** A common type for enumeration values */
typedef int                 pxcEnum;
/** A boolean value, zero indicates false */
typedef int                 pxcBool;
/** A single-precision (32-bit) floating point type */
typedef float               pxcF32;
/** A double-precision (64-bit) floating point type */
typedef double              pxcF64;

/** A rectangle type defined with pxcI32 values */
struct PXCRectI32 {
    pxcI32 x, y, w, h;
};

/** A type representing an two dimensional extent defined with pxcI32 values */
struct PXCSizeI32 {
    pxcI32 width, height;
};

/** A type representing a two dimensional point defined with pxcF32 values */
struct PXCPointF32 {
    pxcF32 x, y;
};

/** A type representing a two dimensional point defined with pxcI32 values */
struct PXCPointI32 {
    pxcI32 x, y;
};

/** A type representing a range defined with pxcF32 values */
struct PXCRangeF32 {
    pxcF32 min, max;
};

/** A type representing a three-dimensional point defined with pxcF32 values */
struct PXCPoint3DF32 {
    pxcF32 x, y, z;
};

/** A type representing a four-dimensional point or vector, for instance quaternion */
struct PXCPoint4DF32 {
    pxcF32 x, y, z, w;
};

/** This macro constructs a pxcUID given four byte values.  The arguments will
    be evaluated exactly once, cast to unsigned int and shifted into one of the
    byte positions.  Hence, arguments must not hold values larger than a byte.
    The result is a pxcUID. */
#define PXC_UID(X1,X2,X3,X4) ((pxcUID)(((unsigned int)(X4)<<24)+((unsigned int)(X3)<<16)+((unsigned int)(X2)<<8)+(unsigned int)(X1)))

/** This macro defines a constant symbol of an integral type within the scope
    of a class or struct.  The type of the constant is that of a symbol in an
    anonymous enumeration (with unspecified width), hence an unspecified
    integral compile-time constant. */
#define PXC_DEFINE_CONST(Y,X) enum {Y=X}

/** This macro defines a UID in the context of a class deriving from PXCBase. */
#define PXC_DEFINE_UID(Y,X1,X2,X3,X4) enum {Y=PXC_UID(X1,X2,X3,X4)}