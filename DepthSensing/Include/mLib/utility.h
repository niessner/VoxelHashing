
#ifndef CORE_UTIL_UTILITY_H_
#define CORE_UTIL_UTILITY_H_

#include <string>
#include <algorithm>
#include <vector>

namespace ml
{

namespace math
{
	const double PI = 3.1415926535897932384626433832795028842;
	const float PIf = 3.14159265358979323846f;

	inline float degreesToRadians(float x) {
		return x * (PIf / 180.0f);
	}

	inline float radiansToDegrees(float x) {
		return x * (180.0f / PIf);
	}

	inline double degreesToRadians(double x) {
		return x * (PI / 180.0);
	}

	inline double radiansToDegrees(double x) {
		return x * (180.0 / PI);
	}

	template<class T>
	inline bool floatEqual(const T& v0, const T& v1, T eps = (T)0.000001) {
		return (std::abs(v0 - v1) <= eps);
	}


	template<class T>
	inline T linearMap(T s1, T e1, T s2, T e2, T start) {
		return ((start - s1) * (e2 - s2) / (e1 - s1) + s2);
	}

	template<class T, class U>
	inline T lerp(T left, T right, U s) {
		return static_cast<T>(left + s * (right - left));
	}

	inline int mod(int x, UINT M) {
		if (x >= 0) { return (x % M); }
		else { return ((x + (x / static_cast<int>(M) + 2) * static_cast<int>(M)) % M); }
	}

	template <class T> inline T square(T x) {
		return x * x;
	}

	template <class T> inline T min(T A, T B) {
		if (A < B) { return A; }
		else { return B; }
	}

	template <class T> inline T min(T A, T B, T C) {
		if (A < B && A < C) { return A; }
		else if (B < C) { return B; }
		else { return C; }
	}

	template <class T> inline T max(T A, T B) {
		if (A > B) { return A; }
		else { return B; }
	}

	template <class T> inline T max(T A, T B, T C) {
		if (A > B && A > C) { return A; }
		else if (B > C) { return B; }
		else { return C; }
	}

    template <class T> inline unsigned int maxIndex(T A, T B, T C) {
        if (A > B && A > C) { return 0; }
        else if (B > C) { return 1; }
        else { return 2; }
    }

	//! returns the clamped value between min and max
	template<class T>
	inline T clamp(T x, T pMin = (T)0.0, T pMax = (T)0.0) {
		if (x < pMin) { return pMin; }
		if (x > pMax) { return pMax; }
		return x;
	}

	template<class T>
	inline T abs(T x) {
		if (x < 0) { return -x; }
		return x;
	}

	template<class T>
	inline T round(const T& f) {
		return (f > (T)0.0) ? floor(f + (T)0.5) : ceil(f - (T)0.5);
	}

	template<class T>
	inline bool isPower2(const T& x) {
		return (x & (x - 1)) == 0;
	}

	template<class T>
	inline T nextLargeestPow2(T x) {
		x |= (x >> 1);
		x |= (x >> 2);
		x |= (x >> 4);
		x |= (x >> 8);
		x |= (x >> 16);
		return (x + 1);
	}

	template<class T>
	inline T log2Integer(T x) {
		T r = 0;
		while (x >>= 1) { r++; }
		return r;
	}


	//! non-zero 32-bit integer value to compute the log base 10 of
	template<class T>
	inline T log10Integer(T x) {
		T r;  // result goes here

		const unsigned int PowersOf10[] = {
			1, 10, 100, 1000, 10000, 100000,
			1000000, 10000000, 100000000, 1000000000
		};

		T t = (log2Integer(x) + 1) * 1233 >> 12; // (use a lg2 method from above)
		r = t - (x < PowersOf10[t]);
		return r;
	}

	//! returns -1 if negative, 0 if 0, +1 if positive
	template <typename T>
	inline int sign(T val) {
		return (T(0) < val) - (val < T(0));
	}

	//! returns -1 if negative; +1 otherwise (includes 0)
	template <typename T>
	inline int sgn(T val) {
		return val < 0 ? -1 : 1;
	}

	//! solves a^2 + bx + c = 0
	template<typename T>
	inline void quadraticFormula(T a, T b, T c, T& x0, T& x1) {
		T tmp = (T) - 0.5 * (b + (T)sgn(b) * sqrt(b * b - (T)4 * a * c));
		x0 = tmp / a;
		x1 = c / tmp;
	}

}  // namespace math

namespace util
{
    //
    // iterator helpers
    //
    template<class container, class assignFunction>
    void fill(container &c, assignFunction func) {
        int i = 0;
        for(auto &x : c) {
            x = func(i++);
        }
    }


	//
	// hashing
	//
	UINT32 hash32(const BYTE* start, UINT length);
	UINT64 hash64(const BYTE* start, UINT length);

	template <class T> inline UINT32 hash32(const T& obj) {
		return hash32((const BYTE*)&obj, sizeof(T));
	}

	template <class T> inline UINT64 hash64(const T& obj) {
		return hash64((const BYTE*)&obj, sizeof(T));
	}

	//
	// casting
	//
	inline UINT castBoolToUINT(bool b) {
		if (b) { return 1; }
		else { return 0; }
	}

	template <class T> inline BYTE boundToByte(T value) {
		if (value < 0) { value = 0; }
		if (value > 255) { value = 255; }
		return BYTE(value);
	}

	//
	// file utility
	//
	//bool fileExists(const std::string& filename);
	inline bool fileExists(const std::string &filename)
	{
		std::ifstream file(filename);
		return (!file.fail());
	}
	//! returns the file size in bytes
	//UINT64 getFileSize(const std::string& filename);
	inline UINT64 getFileSize(const std::string &filename)
	{
		BOOL success;
		WIN32_FILE_ATTRIBUTE_DATA fileInfo;
		success = GetFileAttributesExA(filename.c_str(), GetFileExInfoStandard, (void*)&fileInfo);
		MLIB_ASSERT_STR(success && fileInfo.nFileSizeHigh == 0, std::string("GetFileAttributesEx failed on ") + filename);
		//return fileInfo.nFileSizeLow + fileInfo.nFileSizeHigh;
		LARGE_INTEGER size;
		size.HighPart = fileInfo.nFileSizeHigh;
		size.LowPart = fileInfo.nFileSizeLow;
		return size.QuadPart;
	}
	void copyFile(const std::string& sourceFile, const std::string& destFile);

	//
	// There are OS-specific functions
	//
	void messageBox(const char* string);
	void messageBox(const std::string& S);
	void copyStringToClipboard(const std::string& S);
	std::string loadStringFromClipboard();
	int runCommand(const std::string& executablePath, const std::string& commandLine, bool Blocking);
	std::string workingDirectory();

	inline bool directoryExists(const std::string& directory) {

		DWORD ftyp = GetFileAttributesA(directory.c_str());
		if (ftyp == INVALID_FILE_ATTRIBUTES)
			return false;  //something is wrong with your path!

		if (ftyp & FILE_ATTRIBUTE_DIRECTORY)
			return true;   // this is a directory!

		return false;    // this is not a directory!
	}

	//
	// Returns the next line in the given file
	//

	std::string getNextLine(std::ifstream& file);
	std::vector<BYTE> getFileData(const std::string& filename);

	//
	// Returns the set of all lines in the given file
	//
	std::vector<std::string> getFileLines(std::ifstream& file, UINT minLineLength = 0);
	std::vector<std::string> getFileLines(const std::string& filename, UINT minLineLength = 0);

	//! Save lines to file
	void saveLinesToFile(const std::vector<std::string>& lines, const std::string& filename);

	//
	// FILE wrappers
	//
	inline FILE* checkedFOpen(const char* filename, const char* mode) {
		FILE* file = fopen(filename, mode);
		MLIB_ASSERT_STR(file != NULL && !ferror(file), std::string("Failed to open file: ") + std::string(filename));
		return file;
	}

	inline void checkedFRead(void* dest, UINT64 elementSize, UINT64 elementCount, FILE* file) {
		UINT64 elementsRead = fread(dest, elementSize, elementCount, file);
		MLIB_ASSERT_STR(!ferror(file) && elementsRead == elementCount, "fread failed");
	}

	inline void checkedFWrite(const void* Src, UINT64 elementSize, UINT64 elementCount, FILE* file) {
		UINT64 elementsWritten = fwrite(Src, elementSize, elementCount, file);
		MLIB_ASSERT_STR(!ferror(file) && elementsWritten == elementCount, "fwrite failed");
	}

	inline void checkedFSeek(UINT offset, FILE* file) {
		int result = fseek(file, offset, SEEK_SET);
		MLIB_ASSERT_STR(!ferror(file) && result == 0, "fseek failed");
	}

    template<class T, class U>
    void insert(T &vec, const U &iterable)
    {
        vec.insert(iterable.begin(), iterable.end());
    }

    template<class T, class U>
    void push_back(T &vec, const U &iterable)
    {
        for(const auto &e : iterable)
            vec.push_back(e);
    }

    template<class T>
    bool contains(const std::vector<T> &vec, const T &element)
    {
        for(const T &e : vec)
            if(e == element)
                return true;
        return false;
    }

    //Usage: auto mappedVector = map(v, [](int a) { return a * 2.0; });
    template<class mapFunction, class T>
    auto map(const std::vector<T> &v, mapFunction function) -> std::vector<decltype(function(std::declval<T>()))>
    {
        size_t size = v.size();
        std::vector<decltype(function(std::declval<T>()))> result(size);
        for(size_t i = 0; i < size; i++) result[i] = function(v[i]);
        return result;
    }

	//! uses the <, >  and = operator of the key type
	template<typename Iterator, typename T>
	inline Iterator binarySearch(Iterator& begin, Iterator& end, const T& key) {
		while (begin < end) {
			Iterator middle = begin + (std::distance(begin, end) / 2);

			if (*middle == key) {   // in that case we exactly found the value
				return middle;
			} else if (*middle > key) {
				end = middle;
			} else {
				begin = middle + 1;
			}
		}

		// if the value is not found return the lower interval
		if (begin < end)    { return begin; }
		else                { return end; }
	}
}  // namespace utility

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
	for (std::list<T>::const_iterator iter = l.begin(); iter != l.end(); iter++) {
		s << '\t' << *iter;
		if (iter != --l.end()) s << '\n';
	}
	return s;
}

}  // namespace ml


#endif  // CORE_UTIL_UTILITY_H_
