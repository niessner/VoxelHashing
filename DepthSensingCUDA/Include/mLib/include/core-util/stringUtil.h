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
    inline bool contains(const std::string& str, const std::string& find) {
        for (size_t i = 0; i < str.length(); i++)
        {
            if (exactMatchAtOffset(str, find, i))
                return true;
        }
        return false;
    }

    //TODO TEST
    inline std::string zeroPad(UINT value, UINT totalLength) {
        std::string result = std::to_string(value);
        while (result.size() < totalLength)
            result = "0" + result;
        return result;
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

    template <class T>
    inline const T& randomElement(const std::vector<T>& v) {
        MLIB_ASSERT_STR(v.size() > 0, "empty vector in randomElement");
        return v[ randomInteger(0, (long)v.size() - 1) ];
    }


	//TODO TEST
	inline std::vector<std::string> split(const std::string& str, const std::string& separator, bool pushEmptyStrings = false) {
		MLIB_ASSERT_STR(separator.length() >= 1, "empty seperator");
		std::vector<std::string> result;

        if (str.size() == 0)
        {
            result.push_back("");
            return result;
        }

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

    //! splits string about the first instance of delim
    inline std::pair<std::string, std::string> splitOnFirst(const std::string& str, const std::string& delim) {
        std::pair<std::string, std::string> result;
        auto firstIndex = str.find_first_of(delim);
        result.first = str.substr(0, firstIndex);
        result.second = str.substr(firstIndex + delim.size());
        return result;
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

	//! returns the integer of the last suffix
	inline unsigned int getNumericSuffix(const std::string& str) {
		std::string suffix;
		unsigned int i = 0;
		while (i < str.length()) {
			char curr = str[str.length()-1-i];
			if (curr >= '0' && curr <= '9') {
				suffix = curr + suffix;
			} else {
				break;
			}
			i++;
		}
		if (suffix.length() > 0) {
			return std::atoi(suffix.c_str());
		} else {
			return (unsigned int)(-1);
		}
	}

	inline std::string getBaseBeforeNumericSuffix(const std::string& str) {
		std::string base = str;
		unsigned int i = 0;
		while (i < str.length()) {
			char curr = str[str.length()-1-i];
			if (curr >= '0' && curr <= '9') {
				base.pop_back();
			} else {
				break;
			}
			i++;
		}
		return base;
	}

}  // namespace util

template<class T>
inline std::ostream& operator<<(std::ostream& s, const std::vector<T>& v) {
	s << "vector size " << v.size() << "\n";
	for (size_t i = 0; i < v.size(); i++) {
		s << '\t' << v[i];
		if (i != v.size() - 1) s << '\n';
	}
	return s;
}


template<class T>
inline std::ostream& operator<<(std::ostream& s, const std::list<T>& l) {
	s << "list size " << l.size() << "\n";  
	for (auto iter = l.begin(); iter != l.end(); iter++) {
		s << '\t' << *iter;
		if (iter != --l.end()) s << '\n';
	}
	return s;
}

}  // namespace ml

#endif  // CORE_UTIL_STRINGUTIL_H__
