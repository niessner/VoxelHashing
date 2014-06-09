/*bool traverseFineGrid(HashEntry entry, float3 worldCamPos, float3 worldDir, float3 camDir, float rayCurrent, float rayEnd, int3 dTid)
{
	if(rayCurrent >= rayEnd) { return false; }

	int3 idCurrentVoxel = worldToLocalSDFBlockIndex(worldCamPos+rayCurrent*worldDir);
	int3 idEnd = worldToLocalSDFBlockIndex(worldCamPos+rayEnd*worldDir);
	
	float3 step = sign(worldDir);
	float3 boundaryPos = virtualVoxelPosToWorld(SDFBlockToVirtualVoxelPos(entry.pos)+idCurrentVoxel+(int3)clamp(step, 0.0, 1.0f))-g_VirtualVoxelSize/2.0f; // corner of fine grid
	float3 tMax = (boundaryPos-worldCamPos)/worldDir;
	float3 tDelta = (step*g_VirtualVoxelSize)/worldDir;
	int3 idBound = idEnd+step;
	idCurrentVoxel-=step; // hack !!! it is not correct !!! here more problematic than before
	
	[unroll]
	for(int c = 0; c < 3; c++)
	{
		if(worldDir[c] == 0.0f) { tMax[c] = PINF; tDelta[c] = PINF; }
	}

	unsigned int maxIter = 0;
	[loop]
	while(true && maxIter < g_MaxLoopIterCount)
	{
		Voxel voxel = getVoxel(g_SDFBlocks, entry.ptr+linearizeVoxelPos(idCurrentVoxel));

		if(voxel.weight)// > 0 && voxel.sdf < 0.0f)
		{
			const float3 currentCamPos = rayCurrent*camDir;
			const float truncation = g_Truncation+g_TruncScale*currentCamPos.z;

			if(abs(voxel.sdf) < 0.5f*truncation)
			{
				const float depthToRayLength = 1.0f/camDir.z; // scale factor to convert from depth to ray length
				float3 worldCurrentVoxel = virtualVoxelPosToWorld(SDFBlockToVirtualVoxelPos(entry.pos)+idCurrentVoxel);
				g_output[dTid.xy] = distance(worldCamPos, worldCurrentVoxel)/depthToRayLength;
				g_outputColor[dTid.xy] = float4(voxel.color/255.0f, 1.0f);
				return true;
			}
		}
		
		// Traverse voxel grid
		if(tMax.x < tMax.y && tMax.x < tMax.z)
		{
			idCurrentVoxel.x += step.x;
			if(idCurrentVoxel.x == idBound.x) return false;
			tMax.x += tDelta.x;
		}
		else if(tMax.z < tMax.y)
		{
			idCurrentVoxel.z += step.z;
			if(idCurrentVoxel.z == idBound.z) return false;
			tMax.z += tDelta.z;
		}
		else
		{
			idCurrentVoxel.y += step.y;
			if(idCurrentVoxel.y == idBound.y) return false;
			tMax.y += tDelta.y;
		}

		maxIter++;
	}

	return false;
}

void traverseCoarseGrid(float3 worldCamPos, float3 worldDir, float3 camDir, int3 dTid)
{
	const float depthToRayLength = 1.0f/camDir.z; // scale factor to convert from depth to ray length
	int2 screenPos = cameraToKinectScreenInt(camDir);
	
	float rayCurrent = depthToRayLength*max(g_SensorDepthWorldMin, kinectProjZToCamera(g_RayIntervalMin[screenPos])); // Convert depth to raylength
	float rayEnd = depthToRayLength*min(g_SensorDepthWorldMax, kinectProjZToCamera(g_RayIntervalMax[screenPos])); // Convert depth to raylength

	if(rayCurrent >= rayEnd) { return; }

	int3 idCurrentVoxel = worldToSDFBlock(worldCamPos+rayCurrent*worldDir);
	int3 idEnd = worldToSDFBlock(worldCamPos+rayEnd*worldDir);
	
	float3 step = sign(worldDir);
	float3 boundaryPos = SDFBlockToWorld(idCurrentVoxel+(int3)clamp(step, 0.0, 1.0f))-g_VirtualVoxelSize/2.0f; // corner of coarse grid
	float3 tMax = (boundaryPos-worldCamPos)/worldDir;
	float3 tDelta = (step*SDF_BLOCK_SIZE*g_VirtualVoxelSize)/worldDir;
	int3 idBound = idEnd+step;
	idCurrentVoxel-=step; // hack !!! it is not correct !!!

	[unroll]
	for(int c = 0; c < 3; c++)
	{
		if(worldDir[c] == 0.0f) { tMax[c] = PINF; tDelta[c] = PINF; }
	}

	unsigned int maxIter = 0;
	[loop]
	while(true && maxIter < g_MaxLoopIterCount)
	{
		HashEntry entry =  getHashEntryForSDFBlockPos(g_Hash, idCurrentVoxel);

		if(entry.ptr != FREE_ENTRY)
		{
			float3 worldCurrentVoxel = SDFBlockToWorld(idCurrentVoxel);
			g_output[dTid.xy] = distance(worldCamPos, worldCurrentVoxel)/depthToRayLength;
			return;

			float3 minCorner = SDFBlockToWorld(entry.pos)-g_VirtualVoxelSize/2.0;
			float3 maxCorner = minCorner+SDF_BLOCK_SIZE*g_VirtualVoxelSize;

			float tNear; float tFar;
			bool intersects = intersectRayBoxSafe(worldCamPos, worldDir, minCorner, maxCorner, 0.0f, tNear, tFar, false);
			if(intersects) // should be always true
			{
				if(traverseFineGrid(entry, worldCamPos, worldDir, camDir, tNear, tFar, dTid)) return;
			}
		}
		
		// Traverse voxel grid
		if(tMax.x < tMax.y && tMax.x < tMax.z)
		{
			idCurrentVoxel.x += step.x;
			if(idCurrentVoxel.x == idBound.x) return;
			tMax.x += tDelta.x;
		}
		else if(tMax.z < tMax.y)
		{
			idCurrentVoxel.z += step.z;
			if(idCurrentVoxel.z == idBound.z) return;
			tMax.z += tDelta.z;
		}
		else
		{
			idCurrentVoxel.y += step.y;
			if(idCurrentVoxel.y == idBound.y) return;
			tMax.y += tDelta.y;
		}

		maxIter++;
	}
}*/
