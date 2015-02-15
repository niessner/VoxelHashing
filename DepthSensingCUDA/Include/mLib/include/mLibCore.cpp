
#ifdef _WIN32
#include <windows.h>
#endif

#ifdef LINUX
#define _POSIX_SOURCE
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/time.h>
#include <dirent.h>
#endif

//
// core-base source files
//
#include "../src/core-base/common.cpp"
#include "../src/core-base/console.cpp"

//
// core-math source files
//
#include "../src/core-math/rng.cpp"
#include "../src/core-math/triangleIntersection.cpp"

//
// core-util source files
//
#include "../src/core-util/utility.cpp"
#include "../src/core-util/directory.cpp"
#include "../src/core-util/timer.cpp"
#include "../src/core-util/pipe.cpp"
#include "../src/core-util/UIConnection.cpp"

//
// core-multithreading source files
//
#include "../src/core-multithreading/threadPool.cpp"
#include "../src/core-multithreading/workerThread.cpp"

//
// core-graphics source files
//
#include "../src/core-graphics/RGBColor.cpp"

//
// core-mesh source files
//
#include "../src/core-mesh/meshShapes.cpp"
#include "../src/core-mesh/meshUtil.cpp"
