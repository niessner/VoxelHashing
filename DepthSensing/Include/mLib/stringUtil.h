#ifndef CORE_UTIL_STRINGUTIL_H_
#define CORE_UTIL_STRINGUTIL_H_

#include <string>
#include <vector>

namespace ml
{

namespace util
{
	//TODO TEST
	inline bool startsWith(const std::string& str, const std::string& startCandidate) {
		if (str.length() < startCandidate.length()) { return false; }
		for (size_t i = 0; i < startCandidate.length(); i++) {
			if (str[i] != startCandidate[i]) { return false; }
		}
		return true;
	}

	//TODO TEST
	inline bool endsWith(const std::string& str, const std::string& endCandidate) {
		if (str.length() < endCandidate.length()) { return false; }
		for (size_t i = 0; i < endCandidate.length(); i++) {
			if (str[str.length() - endCandidate.length() + i] != endCandidate[i]) { return false; }
		}
		return true;
	}

	//TODO TEST
	inline bool exactMatchAtOffset(const std::string& str, const std::string& find, size_t offset) {
		size_t MatchLength = 0;
		for (size_t i = 0; i + offset < str.length() && i < find.length(); i++) {
			if (str[i + offset] == find[i]) {
				MatchLength++;
				if (MatchLength == find.length()) { return true; }
			}
		}
		return false;
	}

	//TODO TEST
	inline std::string replace(const std::string& str, const std::string& find, const std::string& replace) {
		std::string result;
		for (size_t i = 0; i < str.length(); i++) {
			if (exactMatchAtOffset(str, find, i)) {
				result += replace;
				i += find.length() - 1;
			} else { result += str[i]; }
		}
		return result;
	}

    //TODO TEST
    inline std::string remove(const std::string& str, const std::string& find) {
        return replace(str, find, "");
    }

	inline std::string replace(const std::string& str, char find, char replace) {
		return util::replace(str, std::string(1, find), std::string(1, replace));
	}


	//TODO TEST
	inline std::vector<std::string> split(const std::string& str, const std::string& separator, bool pushEmptyStrings = false) {
		MLIB_ASSERT_STR(separator.length() >= 1, "empty seperator");
		std::vector<std::string> result;
		std::string entry;
		for (size_t i = 0; i < str.length(); i++) {
			bool isSeperator = true;
			for (size_t testIndex = 0; testIndex < separator.length() && i + testIndex < str.length() && isSeperator; testIndex++) {
				if (str[i + testIndex] != separator[testIndex]) {
					isSeperator = false;
				}
			}
			if (isSeperator) {
				if (entry.length() > 0 || pushEmptyStrings) {
					result.push_back(entry);
					entry.clear();
				}
				i += separator.size() - 1;
			} else {
				entry.push_back(str[i]);
			}
		}
		if (entry.length() > 0) { result.push_back(entry); }
		return result;
	}

	inline std::vector<std::string> split(const std::string& str, const char separator, bool pushEmptyStrings = false) {
		return split(str, std::string(1, separator), pushEmptyStrings);
	}


	//! converts all chars of a string to lowercase (returns the result)
	inline std::string toLower(const std::string& str) {
		std::string res(str);
		for (size_t i = 0; i < res.length(); i++) {
			if (res[i] <= 'Z' &&  res[i] >= 'A') {
				res[i] -= ('Z' - 'z');
			}
		}
		return res;
	}
	//! converts all chars of a string to uppercase (returns the result)
	inline std::string toUpper(const std::string& str) {
		std::string res(str);
		for (size_t i = 0; i < res.length(); i++) {
			if (res[i] <= 'z' &&  res[i] >= 'a') {
				res[i] += ('Z' - 'z');
			}
		}
		return res;
	}

	//! removes all characters from a string
	inline std::string removeChar(const std::string& strInput, const char c) {
		std::string str(strInput);
		str.erase(std::remove(str.begin(), str.end(), c), str.end());
		return str;
	}

	//! gets the file extension (ignoring case)
	inline std::string getFileExtension(const std::string& filename) {
		std::string extension = filename.substr(filename.find_last_of(".") + 1);
		for (unsigned int i = 0; i < extension.size(); i++) {
			extension[i] = tolower(extension[i]);
		}
		return extension;
	}

	//! returns substring from beginning of str up to before last occurrence of delim
	inline std::string substrBeforeLast(const std::string& str, const std::string& delim) {
		std::string trimmed = str;
		return trimmed.erase(str.find_last_of(delim));
	}

	//! returns filename with extension removed
	inline std::string dropExtension(const std::string& filename) {
		return substrBeforeLast(filename, ".");
	}

	//! trims any whitespace on right of str and returns
	inline std::string rtrim(const std::string& str) {
		std::string trimmed = str;
		return trimmed.erase(str.find_last_not_of(" \n\r\t") + 1);
	}

	inline std::string fileNameFromPath(const std::string &path)
	{
		return ml::util::split(ml::util::replace(path, '\\', '/'), '/').back();
	}

	inline std::string directoryFromPath(const std::string &path)
	{
		return replace(path, fileNameFromPath(path), "");
	}

	inline void makeDirectory(const std::string &directory)
	{
		const std::string dir = replace(directory,'\\', '/');
		const std::vector<std::string> dirParts = split(dir, '/');
		std::string soFar = "";
		for (const std::string& part : dirParts) {
			soFar += part + "/";
			CreateDirectoryA(soFar.c_str(), nullptr);
		}
	}

}  // namespace util

}  // namespace ml

#endif  // CORE_UTIL_STRINGUTIL_H__
