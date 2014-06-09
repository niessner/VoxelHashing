#pragma once

/************************************************************************/
/* Might be used to determine whether camera tracking is lost or not    */
/************************************************************************/

#include "GlobalCameraTrackingState.h"
#include "matrix4x4.h"

#include <iostream>
#include <vector>
#include <list>
#include <cassert>
#include <fstream>

struct LinearSystemConfidence {
	LinearSystemConfidence() {
		reset();
	}

	void reset() {
		sumRegError = 0.0f;
		sumRegWeight = 0.0f;
		numCorr = 0;
		matrixCondition = 0.0f;
		trackingLostTresh = false;
	}

	void print() const {
		std::cout << 
			"relRegError \t" << sumRegError / sumRegWeight << "\n" <<
			"sumRegError:\t" << sumRegError << "\n" << 
			"sumRegWeight:\t" << sumRegWeight << "\n" <<
			"numCorrespon:\t" << numCorr <<  "\n" << 
			"matrixCondit:\t" << matrixCondition << "\n\n";
	}

	bool isTrackingLost() const {
		const float			threshMatrixCondition = 150.0f;
		const float			threshSumRegError = 2000.0f;

		//const float			threshMatrixCondition = 200.0f;
		//const float			threshSumRegError = 4000.0f;
		const float			threshRelError = 1.5f;

		if (sumRegError > threshSumRegError)				return true;
		if (matrixCondition > threshMatrixCondition)		return true;
		if (sumRegError / sumRegWeight > threshRelError)	return true;
		if (trackingLostTresh)	return true;
		return false;
	}

	float sumRegError;
	float sumRegWeight;
	unsigned int numCorr;
	float matrixCondition;
	bool trackingLostTresh;
};

struct ICPLostState {
	ICPLostState() {
		m_TrackingWasLost = false;
		m_DeltaSinceTrackingLost.setIdentity();
	}
	bool	m_TrackingWasLost;
	mat4f	m_DeltaSinceTrackingLost;
};

class ICPErrorLog
{
public:
	ICPErrorLog() {
		m_LogData = NULL;
		clearLog();
	}
	~ICPErrorLog() {
		 SAFE_DELETE_ARRAY(m_LogData);
	}

	void clearLog() {
		SAFE_DELETE_ARRAY(m_LogData);
		m_LogData = new std::vector<std::vector<LinearSystemConfidence>>[GlobalCameraTrackingState::getInstance().s_maxLevels];
		for (unsigned int i = 0; i < GlobalCameraTrackingState::getInstance().s_maxLevels; i++) {
			m_LogData[i].clear();
		}
	}

	//! adds the confidence of the current 
	void addCurrentICPIteration(const LinearSystemConfidence& conf, unsigned int level = 0) {
		assert(m_LogData[level].size() > 0);
		m_LogData[level][m_LogData[level].size()-1].push_back(conf);

	}
	void newICPFrame(unsigned int level = 0) {
		assert(level < GlobalCameraTrackingState::getInstance().s_maxLevels);
		m_LogData[level].push_back(std::vector<LinearSystemConfidence>());
	}


	const std::vector<LinearSystemConfidence>& getLastConfFrame(unsigned int level = 0) const {
		assert(m_LogData[level].size() > 0);
		return m_LogData[level][m_LogData[level].size()-1];
	}
	const LinearSystemConfidence& getLastConf(unsigned int level = 0) const {
		const std::vector<LinearSystemConfidence>& lastConfFrame = getLastConfFrame(level);
		assert(lastConfFrame.size() > 0);
		return lastConfFrame[lastConfFrame.size()-1];
	}

	bool isCurrentTrackingLost() const {
		//return getLastConfFrame(0)[frame].isTrackingLost();
		return getLastConf(0).isTrackingLost();
	}

	void printErrorLastFrame() const {
		std::cout.precision(5);
		for (unsigned int l = 0; l < GlobalCameraTrackingState::getInstance().s_maxLevels; l++) {
			std::cout << "level " << l << ":\n";
			for (unsigned int i = 0; i < getLastConfFrame(l).size(); i++) {
				std::cout << getLastConfFrame()[i].sumRegError << "\t";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}

	//! writes the complete log file of one level to file
	void writeLogToFile(const std::string& filename, unsigned int level = 0) {
		std::ofstream out(filename);
		assert(out.is_open());

		for (unsigned int i = 0; i < m_LogData[level].size(); i++) {
			std::vector<LinearSystemConfidence> &currFrame = m_LogData[level][i];
			for (unsigned int j = 0; j < currFrame.size(); j++) {
				float relErr = (currFrame[j].sumRegError/currFrame[j].numCorr)*10000.0f;
				out << relErr << "\t" << currFrame[j].matrixCondition << "\t" << currFrame[j].numCorr << "\t" << currFrame[j].sumRegError << "\n";
			}
			out << "\n";
		}
		out.close();
	}

	ICPLostState& getICPLostState() {
		return m_LostState;
	}
private:	
	//! log data array for each level
	std::vector<std::vector<LinearSystemConfidence>>* m_LogData;
	ICPLostState	m_LostState;
};


