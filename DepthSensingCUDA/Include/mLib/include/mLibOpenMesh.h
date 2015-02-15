
#pragma once
#ifndef INCLUDE_MLIBOPENMESH_H_
#define INCLUDE_MLIBOPENMESH_H_

//// If you link statically against OpenMesh, you have to add
//// the define OM_STATIC_BUILD to your application. This will
//// ensure that readers and writers get initialized correctly.
//#ifndef OM_STATIC_BUILD
//#define OM_STATIC_BUILD
//#endif

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <OpenMesh/Core/Utils/Property.hh>

#include <OpenMesh/Tools/Decimater/DecimaterT.hh>
#include <OpenMesh/Tools/Decimater/ModQuadricT.hh>
#include <OpenMesh/Tools/Decimater/ModProgMeshT.hh>
#include <OpenMesh/Tools/Decimater/BaseDecimaterT.hh>

//
// ext-openmesh headers
//
#include "ext-openmesh/loader.h"
#include "ext-openmesh/triMesh.h"

#endif  // INCLUDE_MLIBOPENMESH_H_
