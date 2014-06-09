#pragma once

#ifndef _STRING_UTIL_H_
#define _STRING_UTIL_H_


#include <algorithm>

namespace ml {

namespace StringUtil {


	//! converts all chars of a string to lowercase
	__forceinline void toLower(std::string &str) {
		for (size_t i = 0; i < str.length(); i++) {
			if (str[i] <= 'Z' &&  str[i] >= 'A') {
				str[i] -= ('Z'-'z');
			}
		}
	}

	//! removes all characters from a string
	__forceinline void removeChar(std::string &str, const char c) {
		 str.erase (std::remove(str.begin(), str.end(), c), str.end());
	}
	__forceinline std::string removeChar(const std::string &strInput, const char c) {
		std::string str(strInput);
		str.erase (std::remove(str.begin(), str.end(), c), str.end());
		return str;
	}


	//////////////////////
	// native functions //
	//////////////////////
	__forceinline int convertStringToINT(const std::string& s) {
		return atoi(s.c_str());
	}
	__forceinline unsigned int convertStringToUINT(const std::string& s) {
		return (unsigned int)convertStringToINT(s);
	}
	__forceinline double convertStringToDOUBLE(const std::string& s) {
		return atof(s.c_str());
	}
	__forceinline float convertStringToFLOAT(const std::string& s) {
		return (float)convertStringToDOUBLE(s);
	}
	__forceinline char convertStringToCHAR(const std::string& s) {
		return s[0];
	}
	template<class U> inline point2d<U> convertStringToPoint2D(const std::string& s) {
		point3d<U> ret;
		std::stringstream ss(removeChar(s,'f'));
		ss >> ret.x >> ret.y;
		return ret;
	}
	template<class U> inline point3d<U> convertStringToPoint3D(const std::string& s) {
		point3d<U> ret;
		std::stringstream ss(removeChar(s,'f'));
		ss >> ret.x >> ret.y >> ret.z;
		return ret;
	}
	template<class U> inline point4d<U> convertStringToPoint4D(const std::string& s) {
		point4d<U> ret;
		std::stringstream ss(removeChar(s,'f'));
		ss >> ret.x >> ret.y >> ret.z >> ret.w;
		return ret;
	}
	__forceinline bool convertStringToBOOL(const std::string& s) {
		if (s == "false" || s == "0")	return false;
		else return true;		
	}

	////////////////////////
	// template overloads //
	////////////////////////

	template<class T>	__forceinline void convertStringTo(const std::string& s, T& res);

	template<>	__forceinline void convertStringTo<int>(const std::string& s, int& res) {
		res = convertStringToINT(s);
	}
	template<>	__forceinline void convertStringTo<unsigned int>(const std::string& s, unsigned int& res) {
		res = convertStringToUINT(s);
	}
	template<>	__forceinline void convertStringTo<double>(const std::string& s, double& res) {
		res = convertStringToDOUBLE(s);
	}
	template<>	__forceinline void convertStringTo<float>(const std::string& s, float& res) {
		res = convertStringToFLOAT(s);
	}
	template<>	__forceinline void convertStringTo<std::string>(const std::string& s, std::string& res) {
		res = s;
	}
	template<>	__forceinline void convertStringTo<char>(const std::string& s, char& res) {
		res = convertStringToCHAR(s);
	}
	template<class U> __forceinline void convertStringTo(const std::string& s, point2d<U>& res) {
		std::stringstream ss(removeChar(s,'f'));
		ss >> res.x >> res.y;
	}
	template<class U> __forceinline void convertStringTo(const std::string& s, point3d<U>& res) {
		std::stringstream ss(removeChar(s,'f'));
		ss >> res.x >> res.y >> res.z;
	}
	template<class U> __forceinline void convertStringTo(const std::string& s, point4d<U>& res) {
		std::stringstream ss(removeChar(s,'f'));
		ss >> res.x >> res.y >> res.z >> res.w;
	}
	template<> __forceinline void convertStringTo<bool>(const std::string& s, bool& res) {
		res = convertStringToBOOL(s);
	}
}

}

#endif