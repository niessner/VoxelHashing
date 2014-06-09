#ifndef _VOXEL_UTIL_H_
#define _VOXEL_UTIL_H_


static const int g_WeightHit = 10;
static const int g_WeightStarve = 1;
static const int g_WeightMax = 254;

static const float	cullRange = 0.05f;	//5 cm
static const int	g_MinRenderWeight = 9;
static const int	g_ImmortalWeight = 29;
static const float	gaussBlendSigma = 10.0f;

#define MINF asfloat(0xff800000)

#ifndef FLT_EPSILON
#define FLT_EPSILON 1.192092896e-07f
#endif

///////////////////////////////////
// Basic helper functions /////////
///////////////////////////////////

float gauss(float sigma, float x) {
	return exp(-(x*x)/(2.0f*sigma*sigma));
}

struct VoxelData
{
	float3 position;
	float3 color;
	int weight;
};

float3 virtualVoxelPosToWorld(in int3 id) {
	return float3(id)*g_VirtualVoxelSize;
}

int3 getVirtualVoxelPos(in uint id) {
	return int3(
		g_VoxelHash[4*id+0],
		g_VoxelHash[4*id+1],
		g_VoxelHash[4*id+2]);
}

float3 getWorldVoxelPos(in uint id) {
	return virtualVoxelPosToWorld(getVirtualVoxelPos(id));
}

VoxelData getVoxelData(in uint id)
{
	VoxelData voxel;
	voxel.position = getWorldVoxelPos(id);

	int last = g_VoxelHash[4*id+3];
	voxel.weight = last & 0x000000ff;
	last >>= 0x8;
	voxel.color.x = (float)(last & 0x000000ff);
	last >>= 0x8;
	voxel.color.y = (float)(last & 0x000000ff);
	last >>= 0x8;
	voxel.color.z = (float)(last & 0x000000ff);
	voxel.color /= 255.0f;
	return voxel;
}

VoxelData getVoxelDataFromTexture(in uint id)
{
	VoxelData voxel;
	uint2 coord = uint2(id % g_ImageWidth, id / g_ImageWidth);
	voxel.color = prevColor[coord].xyz;
	voxel.position = prevPos[coord].xyz;
	voxel.weight = (int)(prevPos[coord].w + 0.5f);
	return voxel;
}

#endif
