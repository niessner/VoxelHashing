#ifndef MLIBCORE_H_
#define MLIBCORE_H_

#ifndef _WIN32
#define LINUX
#endif

#ifdef MLIB_SOCKETS
#ifdef _WIN32
#   define VC_EXTRALEAN
#   define WIN32_LEAN_AND_MEAN
#   include <WinSock2.h>
#   include <Windows.h>
#   include <Ws2tcpip.h>
#   pragma comment(lib, "ws2_32.lib")
#endif
#endif

//
// core-base headers
//
#include "core-base/common.h"
#include "core-base/console.h"
#include "core-base/grid2.h"
#include "core-base/grid3.h"

//
// core-util headers (these are required by matrices)
//
#include "core-util/utility.h"
#include "core-util/stringUtil.h"
#include "core-util/flagSet.h"
#include "core-util/binaryDataCompressor.h"
#include "core-util/binaryDataBuffer.h"
#include "core-util/binaryDataSerialize.h"
#include "core-util/binaryDataStream.h"

//
// core-math headers
//
#include "core-math/nr_templates.h"
#include "core-math/point1d.h"
#include "core-math/point2d.h"
#include "core-math/point3d.h"
#include "core-math/point4d.h"
#include "core-math/point6d.h"
#include "core-math/matrix2x2.h"
#include "core-math/matrix3x3.h"
#include "core-math/matrix4x4.h"
#include "core-math/quaternion.h"
#include "core-math/mathVector.h"
#include "core-math/sparseMatrix.h"
#include "core-math/denseMatrix.h"
#include "core-math/linearSolver.h"
#include "core-math/eigenSolver.h"
#include "core-math/rng.h"
#include "core-math/kMeansClustering.h"
#include "core-math/sampling.h"
#include "core-math/mathUtil.h"

//
// core-util headers
//
#include "core-util/stringUtilConvert.h"
#include "core-util/directory.h"
#include "core-util/timer.h"
#include "core-util/nearestNeighborSearch.h"
#include "core-util/commandLineReader.h"
#include "core-util/parameterFile.h"
#include "core-util/keycodes.h"
#include "core-util/pipe.h"
#include "core-util/UIConnection.h"
#include "core-util/sparseGrid3.h"
#include "core-base/binaryGrid3.h"
#include "core-base/distanceField3.h"
#include "core-util/baseImage.h"

//
// core-multithreading headers
//
#include "core-multithreading/taskList.h"
#include "core-multithreading/workerThread.h"
#include "core-multithreading/threadPool.h"

//
// core-graphics headers
//
#include "core-graphics/RGBColor.h"
#include "core-graphics/colorUtils.h"
#include "core-graphics/ray.h"
#include "core-graphics/camera.h"
#include "core-graphics/cameraTrackball.h"
#include "core-graphics/lineSegment2.h"
#include "core-graphics/lineSegment3.h"
#include "core-graphics/line2.h"
#include "core-graphics/plane.h"
#include "core-graphics/triangle.h"
#include "core-graphics/polygon.h"
#include "core-graphics/intersection.h"
#include "core-graphics/boundingBox2.h"
#include "core-graphics/boundingBox3.h"
#include "core-graphics/orientedBoundingBox3.h"
#include "core-graphics/dist.h"
#include "core-util/colorGradient.h"

//
// core-mesh headers
//
#include "core-mesh/material.h"
#include "core-mesh/meshData.h"
#include "core-mesh/plyHeader.h"
#include "core-mesh/meshIO.h"
#include "core-mesh/pointCloud.h"
#include "core-mesh/pointCloudIO.h"

#include "core-mesh/triMesh.h"
#include "core-mesh/triMeshSampler.h"

#include "core-mesh/triMeshAccelerator.h"
#include "core-mesh/triMeshRayAccelerator.h"
#include "core-mesh/triMeshCollisionAccelerator.h"
#include "core-mesh/triMeshAcceleratorBruteForce.h"
#include "core-mesh/triMeshAcceleratorBVH.h"

#include "core-mesh/meshShapes.h"
#include "core-mesh/meshUtil.h"


#endif  // MLIBCORE_H_
