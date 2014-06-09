#pragma once

/************************************************************************/
/* Executes marching cubes for all chunks                               */
/************************************************************************/

#include <D3D11.h>
#include "DXUT.h"
#include "DX11Utils.h"
#include "DX11MarchingCubesHashSDF.h"
#include "DX11SceneRepChunkGrid.h"

#include "stdafx.h"

class DX11MarchingCubesChunkGrid
{
	public:		
		static HRESULT extractIsoSurface(ID3D11DeviceContext* context, DX11SceneRepChunkGrid& chunkGrid, DX11SceneRepHashSDF& hash, const vec3f& camPos, float radius, const std::string &filename, const mat4f *transform = NULL);


	
	private:

};
