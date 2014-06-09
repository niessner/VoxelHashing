
#ifndef CORE_BASE_COMMON_H_
#define CORE_BASE_COMMON_H_

#ifdef _WIN32

#ifndef _SCL_SECURE_NO_WARNINGS
#define _SCL_SECURE_NO_WARNINGS
#endif

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

// Disable iterator debugging if _IDL0 set
#ifdef _IDL0
#if _SECURE_SCL
#undef _SECURE_SCL
#define _SECURE_SCL 0
#endif

#ifndef _ITERATOR_DEBUG_LEVEL
#define _ITERATOR_DEBUG_LEVEL 0
#endif
#endif

#define MLIB_OPENMP

#define DEBUG_BREAK __debugbreak()

#endif

#ifdef LINUX
#define DEBUG_BREAK assert(false)
#endif

#include <cstring>
#include <exception>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <functional>
#include <algorithm>
#include <fstream>
#include <memory>
#include <thread>
#include <mutex>
#include <unordered_map>
#include <numeric>
#include <type_traits>
#include <array>
#include <unordered_set>

namespace ml
{

#ifndef NOMINMAX
#define NOMINMAX
#endif

#if defined (LINUX)
#define __FUNCTION__ __func__
#ifndef __LINE__
#define __LINE__
#endif
#endif

#ifndef _MSC_VER
#define NOEXCEPT noexcept
#else
#define NOEXCEPT
#endif

class MLibException : public std::exception {
public:
	MLibException(const std::string& what) : std::exception() {
		m_msg = what;
	}
	MLibException(const char* what) : std::exception() {
		m_msg = std::string(what);
	}
	const char* what() const NOEXCEPT {
		return m_msg.c_str();
	}
private:
	std::string m_msg;
};


#define FUNCTION_LINE_STRING (std::string(__FUNCTION__) + ":" + std::to_string(__LINE__))
//#define FUNCTION_LINE_STRING (std::string(__FUNCTION__))

#define MLIB_EXCEPTION(s) ml::MLibException(std::string(__FUNCTION__).append(": ").append(s).c_str())

#ifdef MLIB_ERROR_CHECK


#define MLIB_WARNING(s) ml::warningFunctionMLIB(std::string(FUNCTION_LINE_STRING) + std::string() + ": " + std::string(s))
#define MLIB_ERROR(s) ml::errorFunctionMLIB(std::string(FUNCTION_LINE_STRING) + ": " + std::string(s))
#define MLIB_ASSERT_STR(b,s) { if(!(b)) ml::assertFunctionMLIB(b, std::string(FUNCTION_LINE_STRING) + ": " + std::string(s)); }
#define MLIB_ASSERT(b) { if(!(b)) ml::assertFunctionMLIB(b, FUNCTION_LINE_STRING); }

void warningFunctionMLIB(const std::string &description);
void errorFunctionMLIB(const std::string &description);
void assertFunctionMLIB(bool statement, const std::string &description);


#else

#define MLIB_WARNING(s)
#define MLIB_ERROR(s)
#define MLIB_ASSERT_STR(b,s)
#define MLIB_ASSERT(b)

#endif

#ifndef SAFE_DELETE
#define SAFE_DELETE(p)       { if (p) { delete (p);     (p)=NULL; } }
#endif

#ifndef SAFE_DELETE_ARRAY
#define SAFE_DELETE_ARRAY(p) { if (p) { delete[] (p);   (p)=NULL; } }
#endif

#ifndef SAFE_FREE
#define SAFE_FREE(p) { if (p) { free (p);   (p)=NULL; } }
#endif

#ifndef SAFE_RELEASE
#define SAFE_RELEASE(p) { if (p) { p->Release();   (p)=NULL; } }
#endif

//#ifndef V_RETURN
//#define V_RETURN(hr) { if (FAILED(hr)) { return (hr); } }
//#endif

//#ifndef E_RETURN
//#define E_RETURN(hr) { if(FAILED(hr)) { Console::log() << #hr << " " << hr << std::endl; } }
//#endif

#ifndef D3D_VALIDATE
#define D3D_VALIDATE(statement) { HRESULT hr = statement;  if(FAILED(hr)) { MLIB_ERROR(#statement); } }
#endif

}  // namespace ml


#ifndef UINT
typedef unsigned int UINT;
#endif

#ifndef UCHAR
typedef unsigned char UCHAR;
#endif

#ifndef INT64
#ifdef WIN32
typedef __int64 INT64;
#else
typedef int64_t INT64;
#endif
#endif

#ifndef UINT32
#ifdef WIN32
typedef unsigned __int32 UINT32;
#else
typedef uint32_t UINT32;
#endif
#endif

#ifndef UINT64
#ifdef WIN32
typedef unsigned __int64 UINT64;
#else
typedef uint64_t UINT64;
#endif
#endif

#ifndef FLOAT
typedef float FLOAT;
#endif

#ifndef DOUBLE
typedef double DOUBLE;
#endif

#ifndef BYTE
typedef unsigned char BYTE;
#endif

#ifndef USHORT
typedef unsigned short USHORT;
#endif

#endif  // CORE_BASE_COMMON_H_
