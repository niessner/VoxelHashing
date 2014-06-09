#pragma once

/***********************************************************************************/
/* Global App state for camera tracking: reads and stores all tracking parameters  */
/***********************************************************************************/

#include "DXUT.h"

#include "stdafx.h"

#include <vector>

#define X_GLOBAL_CAMERA_APP_STATE_FIELDS \
	X(unsigned int, s_maxLevels) \
	X(std::vector<unsigned int>, s_blockSizeNormalize) \
	X(std::vector<unsigned int>, s_numBucketsNormalize) \
	X(std::vector<unsigned int>, s_localWindowSize) \
	X(std::vector<unsigned int>, s_maxOuterIter) \
	X(std::vector<unsigned int>, s_maxInnerIter) \
	X(std::vector<float>, s_distThres) \
	X(std::vector<float>, s_normalThres) \
	X(std::vector<float>, s_angleTransThres) \
	X(std::vector<float>, s_distTransThres) \
	X(std::vector<float>, s_residualEarlyOut)

#ifndef VAR_NAME
#define VAR_NAME(x) #x
#endif


class GlobalCameraTrackingState
{	
	public:
#define X(type, name) type name;
		X_GLOBAL_CAMERA_APP_STATE_FIELDS
#undef X

		//! sets the parameter file and reads
		void readMembers(const ParameterFile& parameterFile) {
				s_ParameterFile = parameterFile;
				readMembers();
		}

		//! reads all the members from the given parameter file (could be called for reloading)
		void readMembers() {
#define X(type, name) \
	if (!s_ParameterFile.readParameter(std::string(#name), name)) {MLIB_WARNING(std::string(#name).append(" ").append("uninitialized"));	name = type();}
			X_GLOBAL_CAMERA_APP_STATE_FIELDS
#undef X
		}

		//! prints all members
		void print() {
#define X(type, name) \
	std::cout << #name " = " << name << std::endl;
			X_GLOBAL_CAMERA_APP_STATE_FIELDS
#undef X
		}

		static GlobalCameraTrackingState& getInstance() {
			static GlobalCameraTrackingState s;
			return s;
		}	
private:
	ParameterFile s_ParameterFile;
};
