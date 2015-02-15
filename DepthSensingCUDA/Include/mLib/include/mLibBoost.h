#ifndef MLIBBOOST_H_
#define MLIBBOOST_H_

// Get rid of warning due to ambiguity in pre-XP vs XP and later Windows versions
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0501
#endif

//
// external boost headers
//
#include <boost/assign.hpp>

// TODO: Find appropriate set of convenience headers to include here

//
// ext-boost headers
//
#include "ext-boost/serialization.h"
#include "ext-boost/options.h"

#endif  // MLIBBOOST_H_
