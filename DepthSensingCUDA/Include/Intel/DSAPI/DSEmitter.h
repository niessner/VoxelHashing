#pragma once

#include "DSAPITypes.h"

/// @class DSEmitter
/// Defines methods specific to an implementation that has an emitter.
class DSEmitter
{
public:
    /// Turn emitter on or off.
    virtual bool enableEmitter(bool enable) = 0;
    /// Returns true if emitter is on, else false.
    virtual bool isEmitterEnabled() = 0;

protected:
    // Creation (and deletion) of an object of this
    // type is supported through the DSFactory functions.
    DSEmitter() {};
    DSEmitter(const DSEmitter& other) DS_DELETED_FUNCTION;
    DSEmitter(DSEmitter&& other) DS_DELETED_FUNCTION;
    DSEmitter& operator=(const DSEmitter& other) DS_DELETED_FUNCTION;
    DSEmitter& operator=(DSEmitter&& other) DS_DELETED_FUNCTION;
    virtual ~DSEmitter() {};
};
