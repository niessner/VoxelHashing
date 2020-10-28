#include "sensordatazhou.h"
#include "stdafx.h"


void read_directory(const std::string& name, std::vector<std::string>& v)
{
	std::string pattern(name);
	pattern.append("\\*");
	WIN32_FIND_DATAA data;
	HANDLE hFind;
	hFind = FindFirstFileA(pattern.c_str(), &data);

	if (hFind != INVALID_HANDLE_VALUE) {
		do {
			v.push_back(std::string((char*)data.cFileName));
		} while (FindNextFileA(hFind, &data) != 0);
		FindClose(hFind);
	}
}