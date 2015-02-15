#pragma once

#include "DSAPITypes.h"

class DSAPI;

/// Check to see if DS hardware is available.
/// @return true if hardware is likely to be available
DS_DECL bool DSIsHardwareLikelyAvailable();

/// Creates an instance of an implementation of the DSAPI interface
/// @param p the requested runtime platform (DS4 device, DS5 device, or playback from a recording)
/// @return new DSAPI instance of the requested platform type
/// @see DSDestroy
DS_DECL DSAPI* DSCreate(DSPlatform p);

/// Destroys an instance of the DSAPI interface previously returned by DSCreate(...)
/// @param ds the DSAPI instance to destroy
/// @see DSCreate
DS_DECL void DSDestroy(DSAPI* ds);
