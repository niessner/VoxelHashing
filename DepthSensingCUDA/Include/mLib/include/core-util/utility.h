
#ifndef CORE_UTIL_UTILITY_H_
#define CORE_UTIL_UTILITY_H_

#include <string>
#include <algorithm>
#include <vector>
#include <sstream>

namespace ml
{

namespace math
{
	static const double PI = 3.1415926535897932384626433832795028842;
	static const float PIf = 3.14159265358979323846f;

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

	inline int mod(int x, size_t M) {
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

    template <class T> inline T max(T A, T B, T C, T D) {
        return max(max(A, B), max(C, D));
    }

    template <class T> inline unsigned int maxIndex(T A, T B, T C) {
        if (A > B && A > C) { return 0; }
        else if (B > C) { return 1; }
        else { return 2; }
    }

	//! returns the clamped value between min and max
	template<class T>
	inline T clamp(T x, T pMin, T pMax) {
		if (x < pMin) { return pMin; }
		if (x > pMax) { return pMax; }
		return x;
	}

    template<class T>
    inline long int floor(T x)
    {
        return (long int)std::floor(x);
    }

    template<class T>
    inline long int ceil(T x)
    {
        return (long int)std::ceil(x);
    }

	template<class T>
	inline T abs(T x) {
		if (x < 0) { return -x; }
		return x;
	}

	template<class T>
	inline int round(const T& f) {
		return (f > (T)0.0) ? (int)floor(f + (T)0.5) : (int)ceil(f - (T)0.5);
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

	//! computes sin(phi) and cos(phi)
	template<typename T>
	inline void sincos(T phi, T& sinPhi, T& cosPhi) {
		sinPhi = std::sin(phi);
		cosPhi = std::cos(phi);
	}
	//! computes sin(phi) and cos(phi)
	template<typename T>
	inline void sincos(T phi, T* sinPhi, T* cosPhi) {
		sincos(phi, *sinPhi, *cosPhi);
	}

	//! counts the number of bits in an unsigned integer 
	inline unsigned int numberOfSetBits(unsigned int i) {
		i = i - ((i >> 1) & 0x55555555);
		i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
		return (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
	}

	//! counts the number of bits in an integer 
	inline int numberOfSetBits(int i) {
		i = i - ((i >> 1) & 0x55555555);
		i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
		return (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
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
	bool fileExists(const std::string& filename);
	//! returns the file size in bytes
	UINT64 getFileSize(const std::string& filename);
	void copyFile(const std::string& sourceFile, const std::string& destFile);

	//
	// There are OS-specific functions
	//
	void messageBox(const char* string);
	void messageBox(const std::string& S);
	void copyStringToClipboard(const std::string& S);
	std::string loadStringFromClipboard();
	int runCommand(const std::string& executablePath, const std::string& commandLine, bool Blocking);
	void makeDirectory(const std::string& directory);
	bool directoryExists(const std::string& directory);
	std::string workingDirectory();
       
    inline void runSystemCommand(const std::string &s)
    {
        system(s.c_str());
    }

	//
	// Returns the next line in the given file
	//
    std::string directoryFromPath(const std::string &path);
    std::string fileNameFromPath(const std::string &path);
    std::string removeExtensions(const std::string &path);
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
		MLIB_ASSERT_STR(file != nullptr && !ferror(file), std::string("Failed to open file: ") + std::string(filename));
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

    template<class T>
    int indexOf(const std::vector<T> &vec, const T &element)
    {
        for (int index = 0; index < vec.size(); index++)
            if (vec[index] == element)
                return index;
        return -1;
    }

    //
    // String encoding
    //
    inline std::string encodeBytes(const unsigned char *data, const size_t byteCount)
    {
        std::ostringstream os;
        
        char hexDigits[16] = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f' };

        for (size_t byteIndex = 0; byteIndex < byteCount; byteIndex++)
        {
            unsigned char byte = data[byteIndex];
            os << hexDigits[byte & 0x0f];
            os << hexDigits[(byte & 0xf0) >> 4];
        }

        return os.str();
    }

    template<class T>
    inline std::string encodeBytes(const T &data)
    {
        return encodeBytes((unsigned char *)&data, sizeof(data));
    }

    inline void decodeBytes(const std::string &str, unsigned char *data)
    {
        auto digitToValue = [](char c) {
            if (c >= '0' && c <= '9')
                return (int)c - (int)'0';
            else
                return (int)c - (int)'a' + 10;
        };
        
        for (size_t byteIndex = 0; byteIndex < str.size() / 2; byteIndex++)
        {
            unsigned char c0 = str[byteIndex * 2 + 0];
            unsigned char c1 = str[byteIndex * 2 + 1];

            data[byteIndex] = digitToValue(c0) + (digitToValue(c1) << 4);
        }
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

    //Usage: float result = minValue(v, [](vec3f x) { return x.length(); });
    template<class mapFunction, class T>
    auto minValue(const std::vector<T> &collection, mapFunction function) -> decltype(function(std::declval<T>()))
    {
        auto result = function(*(collection.begin()));
        for (const auto &element : collection)
        {
            auto value = function(element);
            if (value < result)
                result = value;
        }
        return result;
    }

    //Usage: float result = minValue(v, [](vec3f x) { return x.length(); });
    template<class mapFunction, class T>
    auto maxValue(const std::vector<T> &collection, mapFunction function) -> decltype(function(std::declval<T>()))
    {
        auto result = function(*(collection.begin()));
        for (const auto &element : collection)
        {
            auto value = function(element);
            if (value > result)
                result = value;
        }
        return result;
    }

    template<class T>
    void removeSwap(std::vector<T> &collection, size_t index)
    {
        std::swap(collection[index], collection.back());
        collection.pop_back();
    }

    template<class T>
    int findFirstIndex(const std::vector<T> &collection, const T &value)
    {
        int index = 0;
        for (const auto &element : collection)
        {
            if (element == value)
                return index;
            index++;
        }
        return -1;
    }

    //Usage: size_t result = minValue(v, [](const vec3f &x) { return (x.length() == 0.0f); });
    template<class T, class selectFunction>
    int findFirstIndex(const std::vector<T> &collection, selectFunction function)
    {
        size_t index = 0;
        for (const auto &element : collection)
        {
            if (function(element))
                return index;
        }
        return -1;
    }

    //Usage: float result = minValue(v, [](vec3f x) { return x.length(); });
    template<class mapFunction, class T>
    size_t minIndex(const std::vector<T> &collection, mapFunction function)
        //float zz(std::vector<int> &collection)
    {
        auto minValue = function(*(collection.begin()));
        size_t minIndex = 0, curIndex = 0;
        for (const auto &element : collection)
        {
            auto value = function(element);
            if (value < minValue)
            {
                minValue = value;
                minIndex = curIndex;
            }
            curIndex++;
        }
        return minIndex;
    }

    //Usage: auto filteredVector = filter(v, [](int a) { return a > 10; });
    template<class filterFunction, class T>
    auto filter(const std::vector<T> &v, filterFunction function) -> std::vector<T>
    {
        std::vector<T> result;
        for(const T &e : v)
            if(function(e))
                result.push_back(e);
        return result;
    }

	//! uses the <, >  and = operator of the key type
	template<typename Iterator, typename T>
	inline Iterator binarySearch(Iterator begin, Iterator end, const T& key) {
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



	template<class Matrix, class FloatType>
	unsigned int rank(Matrix mat, unsigned int dimension, FloatType eps = (FloatType)0.00001) 
	{
		const unsigned int n = dimension;

		for (unsigned int k = 0; k < n; k++) { //loop over columns
			for (unsigned int i = k+1; i < n; i++) { //loop over rows (to zero a specific column)

				if (std::abs(mat(k,k)) <= eps) {
					//search for a non-zero element
					for (unsigned int j = k+1; j < n; j++) {
						if (std::abs(mat(j,k) > eps)) {
							//swap the column
							for (unsigned int l = 0; l < n; l++) {
								std::swap(mat(k,l),mat(j,l));
							}
							break;
						}
					}
				}
				if (std::abs(mat(k,k)) > eps) {
					FloatType s = mat(i,k) / mat(k,k);
					for (unsigned int j = k; j < n; j++) {
						mat(i,j) = mat(i,j) - s*mat(k,j);
					}
				}
			}
		}
		unsigned int r = 0;
		for (unsigned int i = 0; i < n; i++) {
			for (unsigned int j = 0; j < n; j++) {
				if (std::abs(mat(i,j)) > eps) {
					r++;
					break;
				}
			}
		}
		return r;
	}


	template<class T>
	std::vector<T*> toVecPtr(std::vector<T>& v) {
		std::vector<T*> res; res.reserve(v.size());
		for (auto& e : v) {
			res.push_back(&e);
		}
		return res;
	}

	template<class T>
	std::vector<const T*> toVecPtr(const std::vector<T>& v) {
		std::vector<const T*> res; res.reserve(v.size());
		for (const auto& e : v) {
			res.push_back(&e);
		}
		return res;
	}

	template<class T>
	std::vector<const T*> toVecConstPtr(const std::vector<T>& v) {
		std::vector<const T*> res; res.reserve(v.size());
		for (const auto& e : v) {
			res.push_back(&e);
		}
		return res;
	}


}  // namespace utility

}  // namespace ml


#endif  // CORE_UTIL_UTILITY_H_
