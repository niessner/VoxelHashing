#define PINF asfloat(0x7f800000)
#define MINF asfloat(0xff800000)

uint linearizeIndex(uint3 idx)
{
	return idx.x+idx.y*g_gridDimensions.x+idx.z*g_gridDimensions.x*g_gridDimensions.y;
}

struct Voxel
{
	float sdf;
    uint3 color;
    uint weight;
};

void setVoxel(RWBuffer<int> buffer, uint id, in Voxel voxel)
{
	buffer[2*id+0] = asint(voxel.sdf);
	int last = 0;
	last |= voxel.color.z & 0x000000ff;
	last <<= 8;
	last |= voxel.color.y & 0x000000ff;
	last <<= 8;
	last |= voxel.color.x & 0x000000ff;
	last <<= 8;
	last |= voxel.weight & 0x000000ff;
	buffer[2*id+1] = last;
}

Voxel getVoxel(RWBuffer<int> buffer, uint id)
{
	Voxel voxel;
	voxel.sdf = asfloat(buffer[2*id+0]);
	int last = buffer[2*id+1];
	voxel.weight = last & 0x000000ff;
	last >>= 0x8;
	voxel.color.x = last & 0x000000ff;
	last >>= 0x8;
	voxel.color.y = last & 0x000000ff;
	last >>= 0x8;
	voxel.color.z = last & 0x000000ff;
	return voxel;
}

Voxel getVoxel(Buffer<int> buffer, uint id)
{
	Voxel voxel;
	voxel.sdf = asfloat(buffer[2*id+0]);
	int last = buffer[2*id+1];
	voxel.weight = last & 0x000000ff;
	last >>= 0x8;
	voxel.color.x = last & 0x000000ff;
	last >>= 0x8;
	voxel.color.y = last & 0x000000ff;
	last >>= 0x8;
	voxel.color.z = last & 0x000000ff;
	return voxel;
}

float3 posWorldToVoxelFloat(float3 pos)
{
	return (pos-g_gridPosition)/g_voxelExtends;
}

int3 posWorldToVoxel(float3 pos)
{
	return (int3)((pos-g_gridPosition)/g_voxelExtends);
}

float3 voxelToPosWorld(int3 id)
{
	return g_gridPosition+id*g_voxelExtends;
}

float3 computeSamplePositions(int3 id)
{
	return voxelToPosWorld(id);
}

float3 getVoxelCenter(int3 id)
{
	return g_gridPosition+id*g_voxelExtends+g_voxelExtends/2.0f;
}

bool isValidId(int3 id)
{
	return ((id.x >= 0 && id.x < g_gridDimensions.x-1) &&
			(id.y >= 0 && id.y < g_gridDimensions.y-1) &&
			(id.z >= 0 && id.z < g_gridDimensions.z-1));
}

// Port of the SIGGRAPH ray-box intersection test
// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm
// from the CUDA SDK volume render sample
bool intersectRayBox(float3 o, float3 d, float3 boxmin, float3 boxmax, float tMinOff, out float tnear, out float tfar)
{
    // compute intersection of ray with all six bbox planes
    float3 invR = float3(1.0f, 1.0f, 1.0f)/d; // problem d might have zero components
    float3 tbot = invR * (boxmin - o);
    float3 ttop = invR * (boxmax - o);

    // re-order intersections to find smallest and largest on each axis
    float3 tmin = min(ttop, tbot);
    float3 tmax = max(ttop, tbot);

    // find the largest tmin and the smallest tmax
    float largest_tmin = max(max(tmin.x, tmin.y), max(tmin.x, tmin.z));
    float smallest_tmax = min(min(tmax.x, tmax.y), min(tmax.x, tmax.z));

    tnear = largest_tmin;
    tfar = smallest_tmax;
	
	tnear = max(tMinOff, tnear);

	return smallest_tmax >= largest_tmin;
}

// Implementation of the SIGGRAPH ray-box intersection test
// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm
bool intersectRayBoxSafe(float3 o, float3 d, float3 minCorner, float3 maxCorner, float tMinOff, out float tNear, out float tFar)
{
	tNear = MINF; tFar = PINF; // Initialize

	[unroll]
	for(uint i = 0; i<3; i++) // for all slabs
	{
		if(d[i] != 0.0f) // ray is parallel to plane
		{						
			// if the ray is NOT parallel to the plane
			float t1 = (minCorner[i] - o[i]) / d[i]; // compute the intersection distance to the planes
			float t2 = (maxCorner[i] - o[i]) / d[i];

			if(t1 > t2) { float tmp = t1; t1 = t2; t2 = tmp; } // by definition t1 is first intersection

			if(t1 > tNear) tNear = t1; // we want largest tNear
			if(t2 < tFar) tFar = t2; // we want smallest tFar
			
			if(tNear > tFar) return false; // box is missed
			if(tFar < 0.0f) return false; // box is behind ray
		}
	}

	tNear = max(tMinOff, tNear);

	return tNear <= tFar;
}

float findIntersectionLinear(float tNear, float tFar, float dNear, float dFar)
{
	return tNear+(dNear/(dNear-dFar))*(tFar-tNear);
}
