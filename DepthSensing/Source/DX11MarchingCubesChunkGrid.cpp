#include "stdafx.h"

#include "DX11MarchingCubesChunkGrid.h"

HRESULT DX11MarchingCubesChunkGrid::extractIsoSurface( ID3D11DeviceContext* context, DX11SceneRepChunkGrid& chunkGrid, DX11SceneRepHashSDF& hash, const vec3f& camPos, float radius, const std::string &filename, const mat4f *transform /*= NULL*/ )
{
	HRESULT hr = S_OK;

	vec3i minGridPos = chunkGrid.getMinGridPos();
	vec3i maxGridPos = chunkGrid.getMaxGridPos();

	DX11MarchingCubesHashSDF::clearMeshBuffer();

	V_RETURN(chunkGrid.StreamOutToCPUAll(context, hash));

	for(int x = minGridPos.x; x<maxGridPos.x; x+=1)
	{
		for(int y = minGridPos.y; y<maxGridPos.y; y+=1)
		{
			for(int z = minGridPos.z; z<maxGridPos.z; z+=1)
			{
				vec3i chunk(x, y, z);
				if(chunkGrid.containsSDFBlocksChunk(chunk))
				{
					std::cout << "Marching Cubes on chunk (" << x << ", " << y << ", " << z << ") " << std::endl;

					V_RETURN(chunkGrid.StreamInToGPUChunkNeighborhood(context, hash, chunk, 1));

					//V_RETURN(DX11MarchingCubesHashSDF::extractIsoSurface(context, hash.GetHashSRV(), hash.GetSDFBlocksSDFSRV(), hash.GetSDFBlocksRGBWSRV(), hash.MapAndGetConstantBuffer(context), hash.GetHashNumBuckets(), hash.GetHashBucketSize()));

					vec3f& chunkCenter = chunkGrid.getWorldPosChunk(chunk);
					vec3f& voxelExtends = chunkGrid.getVoxelExtends();
					float virtualVoxelSize = chunkGrid.getVirtualVoxelSize();

					vec3f minCorner = chunkCenter-voxelExtends/2.0f-vec3f(virtualVoxelSize, virtualVoxelSize, virtualVoxelSize)*SDF_BLOCK_SIZE;
					vec3f maxCorner = chunkCenter+voxelExtends/2.0f+vec3f(virtualVoxelSize, virtualVoxelSize, virtualVoxelSize)*SDF_BLOCK_SIZE;

					V_RETURN(DX11MarchingCubesHashSDF::extractIsoSurface(context, hash.GetHashSRV(), hash.GetSDFBlocksSDFSRV(), hash.GetSDFBlocksRGBWSRV(), hash.MapAndGetConstantBuffer(context), hash.GetHashNumBuckets(), hash.GetHashBucketSize(), minCorner, maxCorner, true));

					V_RETURN(chunkGrid.StreamOutToCPUAll(context, hash));
				}
			}
		}
	}

	DX11MarchingCubesHashSDF::saveMesh(filename, transform);

	unsigned int nStreamedBlock;
	V_RETURN(chunkGrid.StreamInToGPUAll(context, hash, camPos, radius, true, nStreamedBlock));

	return hr;
}
