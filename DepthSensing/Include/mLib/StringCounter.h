#pragma once

#ifndef _STRING_COUNTER_H_
#define _STRING_COUNTER_H_

#include <string>
#include <sstream>
#include <algorithm>

class StringCounter {
public:
	StringCounter(const std::string& base, const std::string fileEnding, unsigned int numCountDigits = 0, unsigned int initValue = 0) {
		m_Base = base;
		if (fileEnding[0] != '.') {
			m_FileEnding = ".";
			m_FileEnding.append(fileEnding);
		} else {
			m_FileEnding = fileEnding;
		}
		m_NumCountDigits = numCountDigits;
		m_InitValue = initValue;
		resetCounter();
	}

	~StringCounter() {
	}

	std::string getNext() {
		std::string curr = getCurrent();
		m_Current++;
		return curr;
	}

	std::string getCurrent() {
		std::stringstream ss;
		ss << m_Base;
		for (unsigned int i = std::max(1u, (unsigned int)std::ceilf(std::log10f((float)m_Current+1))); i < m_NumCountDigits; i++) ss << "0";
		ss << m_Current;
		ss << m_FileEnding;
		return ss.str();
	}

	void resetCounter() {
		m_Current = m_InitValue;
	}
private:
	std::string		m_Base;
	std::string		m_FileEnding;
	unsigned int	m_NumCountDigits;
	unsigned int	m_Current;
	unsigned int	m_InitValue;
};

#endif