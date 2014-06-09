#pragma once

#include "stdafx.h"

#include "GlobalAppState.h"

#include <vector>

struct SDFBlock
{
	int data[2*SDF_BLOCK_SIZE*SDF_BLOCK_SIZE*SDF_BLOCK_SIZE];
};

class SDFDesc
{
	public:

		bool operator<(const SDFDesc& other) const
		{
			if(pos.x == other.pos.x)
			{
				if (pos.y == other.pos.y)
				{
					return pos.z < other.pos.z;
				}

				return pos.y < other.pos.y;
			}
			return pos.x < other.pos.x;
		}

		bool operator==(const SDFDesc& other) const
		{
			return pos.x == other.pos.x && pos.y == other.pos.y && pos.z == other.pos.z;
		}


		vec3i pos;
		int ptr;
};

class ChunkDesc
{
	public:
	
		ChunkDesc(unsigned int initialChunkListSize)
		{
			m_SDFBlocks = std::vector<SDFBlock>(); m_SDFBlocks.reserve(initialChunkListSize);
			m_ChunkDesc = std::vector<SDFDesc>(); m_ChunkDesc.reserve(initialChunkListSize);
		}

		void addSDFBlock(const SDFDesc& desc, const SDFBlock& data)
		{
			m_ChunkDesc.push_back(desc);
			m_SDFBlocks.push_back(data);
		}

		unsigned int getNElements()
		{
			return (unsigned int)m_SDFBlocks.size();
		}

		SDFDesc& getSDFBlockDesk(unsigned int i)
		{
			return m_ChunkDesc[i];
		}

		SDFBlock& getSDFBlock(unsigned int i)
		{
			return m_SDFBlocks[i];
		}

		void clear()
		{
			m_ChunkDesc.clear();
			m_SDFBlocks.clear();
		}

		bool isStreamedOut()
		{
			return m_SDFBlocks.size() > 0;
		}

		std::vector<SDFDesc>& getSDFBlockDescs()
		{
			return m_ChunkDesc;
		}

		std::vector<SDFBlock>& getSDFBlocks()
		{
			return m_SDFBlocks;
		}



	//private:

		std::vector<SDFBlock> m_SDFBlocks;
		std::vector<SDFDesc> m_ChunkDesc;
};

struct Voxel
{
	float sdf;
	vec3i color;
	unsigned int weight;
};

namespace VoxelUtilHelper
{
	Voxel getVoxel(const SDFBlock& sdfBlock, unsigned int id);
	
	void setVoxel(SDFBlock& sdfBlock, unsigned int id, const Voxel& voxel);
			
	vec3i SDFBlockToVirtualVoxelPos(vec3i sdfBlock);

	vec3i worldToVirtualVoxelPos(vec3f pos);
		
	vec3i virtualVoxelPosToSDFBlock(vec3i virtualVoxelPos);

	vec3f virtualVoxelPosToWorld(vec3i pos);

	vec3f SDFBlockToWorld(vec3i sdfBlock);

	vec3f computeSDFBlockCenter(vec3i sdfBlock);

	vec3i worldToSDFBlock(vec3f worldPos);
};
