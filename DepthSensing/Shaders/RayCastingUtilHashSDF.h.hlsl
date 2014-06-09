#ifndef _RAYCASTING_UTIL_HASH_SDF_H_
#define _RAYCASTING_UTIL_HASH_SDF_H_

#define PINF asfloat(0x7f800000)
#define MINF asfloat(0xff800000)
#define INTMAX 0x7fffffff


// Takes a world space position pos -> returns false if not all samples were available
bool trilinearInterpolation(float3 pos, out float dist, out float3 color)
{
	dist = 0.0f;
	color = float3(0.0f, 0.0f, 0.0f);

	// Find for the 000th voxel
	float3 vVoxPosF = worldToVirtualVoxelPosFloat(pos);
	int3 vVoxPos0 = floor(vVoxPosF);
	int3 sdfBlock0 = virtualVoxelPosToSDFBlock(vVoxPos0);
	float3 weight = vVoxPosF - vVoxPos0; // frac(vVoxPosF);
	int ptr0 = getHashEntryForSDFBlockPos(g_Hash, sdfBlock0).ptr; 
	if(ptr0 == FREE_ENTRY) 
		return false; 

	// Find for other voxels
	float x = weight.x, y = weight.y, z = weight.z;
	float x0 = 1 - x,   y0 = 1 - y,   z0 = 1 - z;
	const int3 shifts[8] = {int3(0,0,0), int3(1,0,0), int3(0,1,0), int3(0,0,1), int3(1,1,0), int3(0,1,1), int3(1,0,1), int3(1,1,1)};
	const float ws[8] =    {x0*y0*z0,        x*y0*z0,     x0*y*z0,     x0*y0*z,      x*y*z0,      x0*y*z,      x*y0*z,      x*y*z};
	float maxw = ws[0];
	int maxIdx = 0;
	Voxel voxels[8];

	[allow_uav_condition] for (int i = 0; i < 8; i++){ 
		int3 vVoxPos = vVoxPos0 + shifts[i];
		int3 sdfBlock = virtualVoxelPosToSDFBlock(vVoxPos);
		int ptr = any(sdfBlock - sdfBlock0) ? (getHashEntryForSDFBlockPos(g_Hash, sdfBlock).ptr) : ptr0; 
		if (ptr == FREE_ENTRY)
			return false;
		int	linearMemoryIndex = ptr+virtualVoxelPosToLocalSDFBlockIndex(vVoxPos); 
		voxels[i] = getVoxel(g_SDFBlocksSDF, g_SDFBlocksRGBW, linearMemoryIndex);
		if (voxels[i].weight == 0)
			return false;
	}
	[loop] for (i = 0; i < 8; i++){
		dist += ws[i] * voxels[i].sdf;
		if(maxw < ws[i])
			maxw = ws[i], maxIdx = i;
	}
	color = voxels[maxIdx].color;
	return true;
}


/* // check for ptr for FREE_ENTRY !!!
bool trilinearInterpolationComplexStub(int linearMemoryIndex[8], float3 weight, out float dist, out float3 color)
{
	Voxel v000 = getVoxel(g_SDFBlocks, linearMemoryIndex[0]);
	Voxel v100 = getVoxel(g_SDFBlocks, linearMemoryIndex[1]);
	Voxel v010 = getVoxel(g_SDFBlocks, linearMemoryIndex[2]);
	Voxel v001 = getVoxel(g_SDFBlocks, linearMemoryIndex[3]);
	Voxel v110 = getVoxel(g_SDFBlocks, linearMemoryIndex[4]);
	Voxel v011 = getVoxel(g_SDFBlocks, linearMemoryIndex[5]);
	Voxel v101 = getVoxel(g_SDFBlocks, linearMemoryIndex[6]);
	Voxel v111 = getVoxel(g_SDFBlocks, linearMemoryIndex[7]);
	
	if(v000.weight == 0 || v100.weight == 0 || v010.weight == 0 || v001.weight == 0 || v110.weight == 0 || v011.weight == 0 || v101.weight == 0 || v111.weight == 0) return false;

	float tmp0 = lerp(lerp(v000.sdf, v100.sdf, weight.x), lerp(v010.sdf, v110.sdf, weight.x), weight.y);
	float tmp1 = lerp(lerp(v001.sdf, v101.sdf, weight.x), lerp(v011.sdf, v111.sdf, weight.x), weight.y);
	
	dist = lerp(tmp0, tmp1, weight.z);
	color = v000.color;
	
	return true;
}
 
// Takes a world space position pos -> returns false if not all samples were available
bool trilinearInterpolationComplex(float3 pos, out float dist, out float3 color)
{
	const float oSet = g_VirtualVoxelSize;
	const float3 posDual = pos-float3(oSet/2.0f, oSet/2.0f, oSet/2.0f);

	int3 localSDFBlockPos = delinearizeVoxelIndex(worldToLocalSDFBlockIndex(posDual));
	int linearMemoryIndex[8];

	if(localSDFBlockPos.x >= 0 && localSDFBlockPos.y >= 0 && localSDFBlockPos.z >= 0 && localSDFBlockPos.x < SDF_BLOCK_SIZE-1 && localSDFBlockPos.y < SDF_BLOCK_SIZE-1 && localSDFBlockPos.z < SDF_BLOCK_SIZE-1)
	{
		int3 virtualVoxelPos = worldToVirtualVoxelPos(posDual+float3(0.0f, 0.0f, 0.0f)); int ptr = getHashEntryForSDFBlockPos(g_Hash, virtualVoxelPosToSDFBlock(virtualVoxelPos)).ptr;
		
		linearMemoryIndex[0] = ptr+virtualVoxelPosToLocalSDFBlockIndex(virtualVoxelPos);

		linearMemoryIndex[1] = ptr+worldToLocalSDFBlockIndex(posDual+float3(oSet, 0.0f, 0.0f));
		linearMemoryIndex[2] = ptr+worldToLocalSDFBlockIndex(posDual+float3(0.0f, oSet, 0.0f));
		linearMemoryIndex[3] = ptr+worldToLocalSDFBlockIndex(posDual+float3(0.0f, 0.0f, oSet));
		linearMemoryIndex[4] = ptr+worldToLocalSDFBlockIndex(posDual+float3(oSet, oSet, 0.0f));
		linearMemoryIndex[5] = ptr+worldToLocalSDFBlockIndex(posDual+float3(0.0f, oSet, oSet));
		linearMemoryIndex[6] = ptr+worldToLocalSDFBlockIndex(posDual+float3(oSet, 0.0f, oSet));
		linearMemoryIndex[7] = ptr+worldToLocalSDFBlockIndex(posDual+float3(oSet, oSet, oSet));
	}
	else
	{
		int3 virtualVoxelPos = worldToVirtualVoxelPos(posDual+float3(0.0f, 0.0f, 0.0f)); linearMemoryIndex[0] = getHashEntryForSDFBlockPos(g_Hash, virtualVoxelPosToSDFBlock(virtualVoxelPos)).ptr+virtualVoxelPosToLocalSDFBlockIndex(virtualVoxelPos);
			 virtualVoxelPos = worldToVirtualVoxelPos(posDual+float3(oSet, 0.0f, 0.0f)); linearMemoryIndex[1] = getHashEntryForSDFBlockPos(g_Hash, virtualVoxelPosToSDFBlock(virtualVoxelPos)).ptr+virtualVoxelPosToLocalSDFBlockIndex(virtualVoxelPos);
			 virtualVoxelPos = worldToVirtualVoxelPos(posDual+float3(0.0f, oSet, 0.0f)); linearMemoryIndex[2] = getHashEntryForSDFBlockPos(g_Hash, virtualVoxelPosToSDFBlock(virtualVoxelPos)).ptr+virtualVoxelPosToLocalSDFBlockIndex(virtualVoxelPos);
			 virtualVoxelPos = worldToVirtualVoxelPos(posDual+float3(0.0f, 0.0f, oSet)); linearMemoryIndex[3] = getHashEntryForSDFBlockPos(g_Hash, virtualVoxelPosToSDFBlock(virtualVoxelPos)).ptr+virtualVoxelPosToLocalSDFBlockIndex(virtualVoxelPos);
			 virtualVoxelPos = worldToVirtualVoxelPos(posDual+float3(oSet, oSet, 0.0f)); linearMemoryIndex[4] = getHashEntryForSDFBlockPos(g_Hash, virtualVoxelPosToSDFBlock(virtualVoxelPos)).ptr+virtualVoxelPosToLocalSDFBlockIndex(virtualVoxelPos);
			 virtualVoxelPos = worldToVirtualVoxelPos(posDual+float3(0.0f, oSet, oSet)); linearMemoryIndex[5] = getHashEntryForSDFBlockPos(g_Hash, virtualVoxelPosToSDFBlock(virtualVoxelPos)).ptr+virtualVoxelPosToLocalSDFBlockIndex(virtualVoxelPos);
			 virtualVoxelPos = worldToVirtualVoxelPos(posDual+float3(oSet, 0.0f, oSet)); linearMemoryIndex[6] = getHashEntryForSDFBlockPos(g_Hash, virtualVoxelPosToSDFBlock(virtualVoxelPos)).ptr+virtualVoxelPosToLocalSDFBlockIndex(virtualVoxelPos);
			 virtualVoxelPos = worldToVirtualVoxelPos(posDual+float3(oSet, oSet, oSet)); linearMemoryIndex[7] = getHashEntryForSDFBlockPos(g_Hash, virtualVoxelPosToSDFBlock(virtualVoxelPos)).ptr+virtualVoxelPosToLocalSDFBlockIndex(virtualVoxelPos);
	}
	
	// Interpolate
	float3 weight = frac(worldToVirtualVoxelPosFloat(pos));
	return trilinearInterpolationComplexStub(linearMemoryIndex, weight, dist, color);
}*/

// Takes a world space position pos -> returns false if not all samples were available
bool trilinearInterpolationSimpleFastFast(float3 pos, out float dist, out float3 color)
{
	const float oSet = g_VirtualVoxelSize;
	const float3 posDual = pos-float3(oSet/2.0f, oSet/2.0f, oSet/2.0f);
	float3 weight = frac(worldToVirtualVoxelPosFloat(pos));

	dist = 0.0f;
	int3 virtualVoxelPos = worldToVirtualVoxelPos(posDual+float3(0.0f, 0.0f, 0.0f)); int ptr = getHashEntryForSDFBlockPos(g_Hash, virtualVoxelPosToSDFBlock(virtualVoxelPos)).ptr; if(ptr == FREE_ENTRY) return false; int	linearMemoryIndex = ptr+virtualVoxelPosToLocalSDFBlockIndex(virtualVoxelPos); Voxel	v = getVoxel(g_SDFBlocksSDF, g_SDFBlocksRGBW, linearMemoryIndex); if(v.weight == 0) return false; dist+= (1.0f-weight.x)*(1.0f-weight.y)*(1.0f-weight.z)*v.sdf;
		 virtualVoxelPos = worldToVirtualVoxelPos(posDual+float3(oSet, 0.0f, 0.0f));	 ptr = getHashEntryForSDFBlockPos(g_Hash, virtualVoxelPosToSDFBlock(virtualVoxelPos)).ptr; if(ptr == FREE_ENTRY) return false;		linearMemoryIndex = ptr+virtualVoxelPosToLocalSDFBlockIndex(virtualVoxelPos);		v = getVoxel(g_SDFBlocksSDF, g_SDFBlocksRGBW, linearMemoryIndex); if(v.weight == 0) return false; dist+=	   weight.x *(1.0f-weight.y)*(1.0f-weight.z)*v.sdf;
		 virtualVoxelPos = worldToVirtualVoxelPos(posDual+float3(0.0f, oSet, 0.0f));	 ptr = getHashEntryForSDFBlockPos(g_Hash, virtualVoxelPosToSDFBlock(virtualVoxelPos)).ptr; if(ptr == FREE_ENTRY) return false;		linearMemoryIndex = ptr+virtualVoxelPosToLocalSDFBlockIndex(virtualVoxelPos);		v = getVoxel(g_SDFBlocksSDF, g_SDFBlocksRGBW, linearMemoryIndex); if(v.weight == 0) return false; dist+= (1.0f-weight.x)*	   weight.y *(1.0f-weight.z)*v.sdf;
		 virtualVoxelPos = worldToVirtualVoxelPos(posDual+float3(0.0f, 0.0f, oSet));	 ptr = getHashEntryForSDFBlockPos(g_Hash, virtualVoxelPosToSDFBlock(virtualVoxelPos)).ptr; if(ptr == FREE_ENTRY) return false;		linearMemoryIndex = ptr+virtualVoxelPosToLocalSDFBlockIndex(virtualVoxelPos);		v = getVoxel(g_SDFBlocksSDF, g_SDFBlocksRGBW, linearMemoryIndex); if(v.weight == 0) return false; dist+= (1.0f-weight.x)*(1.0f-weight.y)*	   weight.z *v.sdf;
		 virtualVoxelPos = worldToVirtualVoxelPos(posDual+float3(oSet, oSet, 0.0f));	 ptr = getHashEntryForSDFBlockPos(g_Hash, virtualVoxelPosToSDFBlock(virtualVoxelPos)).ptr; if(ptr == FREE_ENTRY) return false;		linearMemoryIndex = ptr+virtualVoxelPosToLocalSDFBlockIndex(virtualVoxelPos);		v = getVoxel(g_SDFBlocksSDF, g_SDFBlocksRGBW, linearMemoryIndex); if(v.weight == 0) return false; dist+=	   weight.x *	   weight.y *(1.0f-weight.z)*v.sdf;
		 virtualVoxelPos = worldToVirtualVoxelPos(posDual+float3(0.0f, oSet, oSet));	 ptr = getHashEntryForSDFBlockPos(g_Hash, virtualVoxelPosToSDFBlock(virtualVoxelPos)).ptr; if(ptr == FREE_ENTRY) return false;		linearMemoryIndex = ptr+virtualVoxelPosToLocalSDFBlockIndex(virtualVoxelPos);		v = getVoxel(g_SDFBlocksSDF, g_SDFBlocksRGBW, linearMemoryIndex); if(v.weight == 0) return false; dist+= (1.0f-weight.x)*	   weight.y *	   weight.z *v.sdf;
		 virtualVoxelPos = worldToVirtualVoxelPos(posDual+float3(oSet, 0.0f, oSet));	 ptr = getHashEntryForSDFBlockPos(g_Hash, virtualVoxelPosToSDFBlock(virtualVoxelPos)).ptr; if(ptr == FREE_ENTRY) return false;		linearMemoryIndex = ptr+virtualVoxelPosToLocalSDFBlockIndex(virtualVoxelPos);		v = getVoxel(g_SDFBlocksSDF, g_SDFBlocksRGBW, linearMemoryIndex); if(v.weight == 0) return false; dist+=	   weight.x *(1.0f-weight.y)*	   weight.z *v.sdf;
		 virtualVoxelPos = worldToVirtualVoxelPos(posDual+float3(oSet, oSet, oSet));	 ptr = getHashEntryForSDFBlockPos(g_Hash, virtualVoxelPosToSDFBlock(virtualVoxelPos)).ptr; if(ptr == FREE_ENTRY) return false;		linearMemoryIndex = ptr+virtualVoxelPosToLocalSDFBlockIndex(virtualVoxelPos);		v = getVoxel(g_SDFBlocksSDF, g_SDFBlocksRGBW, linearMemoryIndex); if(v.weight == 0) return false; dist+=	   weight.x *	   weight.y *	   weight.z *v.sdf;
		 																																																																															
	color = v.color;
	
	return true;
}

// Takes a world space position pos -> returns false if not all samples were available
/*bool trilinearInterpolationSimpleFast(float3 pos, out float dist, out float3 color)
{
	const float oSet = g_VirtualVoxelSize;
	const float3 posDual = pos-float3(oSet/2.0f, oSet/2.0f, oSet/2.0f);

	int3 virtualVoxelPos = worldToVirtualVoxelPos(posDual+float3(0.0f, 0.0f, 0.0f)); int ptr = getHashEntryForSDFBlockPos(g_Hash, virtualVoxelPosToSDFBlock(virtualVoxelPos)).ptr; if(ptr == FREE_ENTRY) return false; Voxel v000 = getVoxel(g_SDFBlocksSDF, g_SDFBlocksRGBW, ptr+virtualVoxelPosToLocalSDFBlockIndex(virtualVoxelPos));
		 virtualVoxelPos = worldToVirtualVoxelPos(posDual+float3(oSet, 0.0f, 0.0f));	 ptr = getHashEntryForSDFBlockPos(g_Hash, virtualVoxelPosToSDFBlock(virtualVoxelPos)).ptr; if(ptr == FREE_ENTRY) return false; Voxel v100 = getVoxel(g_SDFBlocksSDF, g_SDFBlocksRGBW, ptr+virtualVoxelPosToLocalSDFBlockIndex(virtualVoxelPos));
		 virtualVoxelPos = worldToVirtualVoxelPos(posDual+float3(0.0f, oSet, 0.0f));	 ptr = getHashEntryForSDFBlockPos(g_Hash, virtualVoxelPosToSDFBlock(virtualVoxelPos)).ptr; if(ptr == FREE_ENTRY) return false; Voxel v010 = getVoxel(g_SDFBlocksSDF, g_SDFBlocksRGBW, ptr+virtualVoxelPosToLocalSDFBlockIndex(virtualVoxelPos));
		 virtualVoxelPos = worldToVirtualVoxelPos(posDual+float3(0.0f, 0.0f, oSet));	 ptr = getHashEntryForSDFBlockPos(g_Hash, virtualVoxelPosToSDFBlock(virtualVoxelPos)).ptr; if(ptr == FREE_ENTRY) return false; Voxel v001 = getVoxel(g_SDFBlocksSDF, g_SDFBlocksRGBW, ptr+virtualVoxelPosToLocalSDFBlockIndex(virtualVoxelPos));
		 virtualVoxelPos = worldToVirtualVoxelPos(posDual+float3(oSet, oSet, 0.0f));	 ptr = getHashEntryForSDFBlockPos(g_Hash, virtualVoxelPosToSDFBlock(virtualVoxelPos)).ptr; if(ptr == FREE_ENTRY) return false; Voxel v110 = getVoxel(g_SDFBlocksSDF, g_SDFBlocksRGBW, ptr+virtualVoxelPosToLocalSDFBlockIndex(virtualVoxelPos));
		 virtualVoxelPos = worldToVirtualVoxelPos(posDual+float3(0.0f, oSet, oSet));	 ptr = getHashEntryForSDFBlockPos(g_Hash, virtualVoxelPosToSDFBlock(virtualVoxelPos)).ptr; if(ptr == FREE_ENTRY) return false; Voxel v011 = getVoxel(g_SDFBlocksSDF, g_SDFBlocksRGBW, ptr+virtualVoxelPosToLocalSDFBlockIndex(virtualVoxelPos));
		 virtualVoxelPos = worldToVirtualVoxelPos(posDual+float3(oSet, 0.0f, oSet));     ptr = getHashEntryForSDFBlockPos(g_Hash, virtualVoxelPosToSDFBlock(virtualVoxelPos)).ptr; if(ptr == FREE_ENTRY) return false; Voxel v101 = getVoxel(g_SDFBlocksSDF, g_SDFBlocksRGBW, ptr+virtualVoxelPosToLocalSDFBlockIndex(virtualVoxelPos));
		 virtualVoxelPos = worldToVirtualVoxelPos(posDual+float3(oSet, oSet, oSet));     ptr = getHashEntryForSDFBlockPos(g_Hash, virtualVoxelPosToSDFBlock(virtualVoxelPos)).ptr; if(ptr == FREE_ENTRY) return false; Voxel v111 = getVoxel(g_SDFBlocksSDF, g_SDFBlocksRGBW, ptr+virtualVoxelPosToLocalSDFBlockIndex(virtualVoxelPos));
	
	if(v000.weight == 0 || v100.weight == 0 || v010.weight == 0 || v001.weight == 0 || v110.weight == 0 || v011.weight == 0 || v101.weight == 0 || v111.weight == 0) return false;

	float3 weight = frac(worldToVirtualVoxelPosFloat(pos));
	float tmp0 = lerp(lerp(v000.sdf, v100.sdf, weight.x), lerp(v010.sdf, v110.sdf, weight.x), weight.y);
	float tmp1 = lerp(lerp(v001.sdf, v101.sdf, weight.x), lerp(v011.sdf, v111.sdf, weight.x), weight.y);
	
	dist = lerp(tmp0, tmp1, weight.z);
	color = v000.color;
	
	return true;
}

// Takes a world space position pos -> returns false if not all samples were available
bool trilinearInterpolationSimple(float3 pos, out float dist, out float3 color)
{
	const float oSet = g_VirtualVoxelSize;
	const float3 posDual = pos-float3(oSet/2.0f, oSet/2.0f, oSet/2.0f);
	 
	float3 posSample = posDual+float3(0.0f, 0.0f, 0.0f);   int ptr = getHashEntry(g_Hash, posSample).ptr; if(ptr == FREE_ENTRY) return false; Voxel v000 = getVoxel(g_SDFBlocksSDF, g_SDFBlocksRGBW, ptr+worldToLocalSDFBlockIndex(posSample));
		   posSample = posDual+float3(oSet, 0.0f, 0.0f); 	   ptr = getHashEntry(g_Hash, posSample).ptr; if(ptr == FREE_ENTRY) return false; Voxel v100 = getVoxel(g_SDFBlocksSDF, g_SDFBlocksRGBW, ptr+worldToLocalSDFBlockIndex(posSample));
		   posSample = posDual+float3(0.0f, oSet, 0.0f); 	   ptr = getHashEntry(g_Hash, posSample).ptr; if(ptr == FREE_ENTRY) return false; Voxel v010 = getVoxel(g_SDFBlocksSDF, g_SDFBlocksRGBW, ptr+worldToLocalSDFBlockIndex(posSample));
		   posSample = posDual+float3(0.0f, 0.0f, oSet); 	   ptr = getHashEntry(g_Hash, posSample).ptr; if(ptr == FREE_ENTRY) return false; Voxel v001 = getVoxel(g_SDFBlocksSDF, g_SDFBlocksRGBW, ptr+worldToLocalSDFBlockIndex(posSample));
		   posSample = posDual+float3(oSet, oSet, 0.0f); 	   ptr = getHashEntry(g_Hash, posSample).ptr; if(ptr == FREE_ENTRY) return false; Voxel v110 = getVoxel(g_SDFBlocksSDF, g_SDFBlocksRGBW, ptr+worldToLocalSDFBlockIndex(posSample));
		   posSample = posDual+float3(0.0f, oSet, oSet); 	   ptr = getHashEntry(g_Hash, posSample).ptr; if(ptr == FREE_ENTRY) return false; Voxel v011 = getVoxel(g_SDFBlocksSDF, g_SDFBlocksRGBW, ptr+worldToLocalSDFBlockIndex(posSample));
		   posSample = posDual+float3(oSet, 0.0f, oSet); 	   ptr = getHashEntry(g_Hash, posSample).ptr; if(ptr == FREE_ENTRY) return false; Voxel v101 = getVoxel(g_SDFBlocksSDF, g_SDFBlocksRGBW, ptr+worldToLocalSDFBlockIndex(posSample));
		   posSample = posDual+float3(oSet, oSet, oSet); 	   ptr = getHashEntry(g_Hash, posSample).ptr; if(ptr == FREE_ENTRY) return false; Voxel v111 = getVoxel(g_SDFBlocksSDF, g_SDFBlocksRGBW, ptr+worldToLocalSDFBlockIndex(posSample));
	
	if(v000.weight == 0 || v100.weight == 0 || v010.weight == 0 || v001.weight == 0 || v110.weight == 0 || v011.weight == 0 || v101.weight == 0 || v111.weight == 0) return false;

	float3 weight = frac(worldToVirtualVoxelPosFloat(pos));
	float tmp0 = lerp(lerp(v000.sdf, v100.sdf, weight.x), lerp(v010.sdf, v110.sdf, weight.x), weight.y);
	float tmp1 = lerp(lerp(v001.sdf, v101.sdf, weight.x), lerp(v011.sdf, v111.sdf, weight.x), weight.y);
	
	dist = lerp(tmp0, tmp1, weight.z);
	color = v111.color;
	
	return true;
}*/

float findIntersectionLinear(float tNear, float tFar, float dNear, float dFar)
{
	return tNear+(dNear/(dNear-dFar))*(tFar-tNear);
}

static const unsigned int nIterationsBisection = 3;

// d0 near, d1 far
bool findIntersectionBisection(float3 worldCamPos, float3 worldDir, float d0, float r0, float d1, float r1, out float alpha)
{
	float a = r0; float aDist = d0;
	float b = r1; float bDist = d1;
	float c = 0.0f;
	
	[unroll(3)]
	for(uint i = 0; i<nIterationsBisection; i++)
	{
		c = findIntersectionLinear(a, b, aDist, bDist);
		
		float cDist; float3 color;
		if(!trilinearInterpolationSimpleFastFast(worldCamPos+c*worldDir, cDist, color)) return false;

		if(aDist*cDist > 0.0) { a = c; aDist = cDist; }
		else { b = c; bDist = cDist; }
	}

	alpha = c;

	return true;
}

// Takes a world space position pos -> returns false if not all samples were available
bool nearestNeighbor(float3 pos, out float dist, out float3 color)
{
	Voxel voxel = getVoxel(g_SDFBlocksSDF, g_SDFBlocksRGBW, getHashEntry(g_Hash, pos).ptr+worldToLocalSDFBlockIndex(pos));

	dist = voxel.sdf;
	color = voxel.color;

	return (voxel.weight > 0);
}

struct Sample
{
	float sdf;
	float alpha;
	uint weight;
	//uint3 color;
};

float computeAlphaOut(float3 o, float3 d, float3 boxmin, float3 boxmax)
{
	float3 tbot = (boxmin-o)/d;
	float3 ttop = (boxmax-o)/d;
	
	float3 tmax = max(ttop, tbot);
	return min(min(tmax.x, tmax.y), min(tmax.x, tmax.z));
}

/*
// Port of the SIGGRAPH ray-box intersection test
// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm
// from the CUDA SDK volume render sample
bool intersectRayBox(float3 o, float3 d, float3 boxmin, float3 boxmax, float tMinOff, out float tnear, out float tfar, bool useOffset)
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
	
	if(useOffset) tnear = max(tMinOff, tnear);

	return smallest_tmax >= largest_tmin;
}

// Implementation of the SIGGRAPH ray-box intersection test
// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm
bool intersectRayBoxSafe(float3 o, float3 d, float3 minCorner, float3 maxCorner, float tMinOff, out float tNear, out float tFar, bool useOffset)
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

	if(useOffset) tNear = max(tMinOff, tNear);

	return tNear <= tFar;
}*/

#endif