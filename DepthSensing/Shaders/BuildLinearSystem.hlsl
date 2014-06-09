cbuffer cbConsts : register(b0)
{
	int g_imageWidth;
	int g_imageHeigth;
	int2 dummy0;
	float4x4 g_deltaTransform; // dx style transformation matrix !!!
	float3 g_mean;
	float g_meanStDevInv;
};

Texture2D<float4> input : register(t0);
Texture2D<float4> target : register(t1);
Texture2D<float4> targetNormals : register(t2);

RWBuffer<float> output : register(u0);

/////////////////////////////////////////////////////
// Defines
/////////////////////////////////////////////////////

// should be set by application
#ifndef LOCALWINDOWSIZE
#define LOCALWINDOWSIZE 12
#endif

#ifndef groupthreads
#define groupthreads 64
#endif

#ifndef ARRAYSIZE
#define ARRAYSIZE 30
#endif

#define MINF asfloat(0xff800000)

#pragma warning(disable:3203)
#pragma warning(disable:3206)
#pragma warning(disable:2554)
#pragma warning(disable:3556)


/////////////////////////////////////////////////////
// Shared Memory
/////////////////////////////////////////////////////

struct ScanElement
{
	float data[ARRAYSIZE];
};

groupshared ScanElement bucket[groupthreads];

/////////////////////////////////////////////////////
// Helper Functions
/////////////////////////////////////////////////////

void addToLocalScanElement(uint inpGTid, uint resGTid)
{
	[unrool]
	for(uint i = 0; i<ARRAYSIZE; i++)
	{
		bucket[resGTid].data[i] += bucket[inpGTid].data[i];
	}
}

void CopyToResultScanElement(uint GTid, uint GID)
{
	[unrool]
	for(uint i = 0; i<ARRAYSIZE; i++)
	{
		output[ARRAYSIZE*GID+i] = bucket[GTid].data[i];
	}
}

void SetZeroScanElement(uint GTid)
{
	[unrool]
	for(uint i = 0; i<ARRAYSIZE; i++)
	{
		bucket[GTid].data[i] = 0.0f;
	}
}

/////////////////////////////////////////////////////
// Linearized System Matrix
/////////////////////////////////////////////////////

// Matrix Struct
struct Float1x6
{
	float data[6];
};

// Arguments: q moving point, n normal target
Float1x6 buildRowSystemMatrixPlane(float3 q, float3 n, float w)
{
	Float1x6 row;
	row.data[0] = n.x*q.y-n.y*q.x;
	row.data[1] = n.z*q.x-n.x*q.z;
	row.data[2] = n.y*q.z-n.z*q.y;

	row.data[3] = -n.x;
	row.data[4] = -n.y;
	row.data[5] = -n.z;

	return row;
}

// Arguments: p target point, q moving point, n normal target
float buildRowRHSPlane(float3 p, float3 q, float3 n, float w)
{
	return n.x*(q.x-p.x)+n.y*(q.y-p.y)+n.z*(q.z-p.z);
}

// Arguments: p target point, q moving point, n normal target
void addToLocalSystem(float3 p, float3 q, float3 n, float weight, uint GTid)
{
	const Float1x6 row = buildRowSystemMatrixPlane(q, n, weight);
	const float b = buildRowRHSPlane(p, q, n, weight);
	uint linRowStart = 0;

	[unrool]
	for(uint i = 0; i<6; i++)
	{
		[unrool]
		for(uint j = i; j<6; j++)
		{
			bucket[GTid].data[linRowStart+j-i] += weight*row.data[i]*row.data[j];
		}

		linRowStart += 6-i;

		bucket[GTid].data[21+i] += weight*row.data[i]*b;
	}

	const float dN = dot(p-q, n);
	//bucket[GTid].data[27] += weight*dN*dN; // has to be rescaled on the CPU if used, because of the variance normalization !!!
	bucket[GTid].data[27] += weight*dN*dN;		//residual
	bucket[GTid].data[28] += weight;			//corr weight
	bucket[GTid].data[29] += 1.0f;				//corr number
}

/////////////////////////////////////////////////////
// Scan
/////////////////////////////////////////////////////

void warpReduce(int GTid) // See Optimizing Parallel Reduction in CUDA by Mark Harris
{
	addToLocalScanElement(GTid + 32, GTid);
	addToLocalScanElement(GTid + 16, GTid);
	addToLocalScanElement(GTid + 8 , GTid);
	addToLocalScanElement(GTid + 4 , GTid);
	addToLocalScanElement(GTid + 2 , GTid);
	addToLocalScanElement(GTid + 1 , GTid);
}

[numthreads(groupthreads, 1, 1)]
void scanScanElementsCS(uint3 DTid : SV_DispatchThreadID, uint3 GTid : SV_GroupThreadID, uint3 GID: SV_GroupID)
{
	// Set system to zero
	SetZeroScanElement(GTid.x);
	
	//Locally sum small window
	[unrool]
	for(uint i = 0; i<LOCALWINDOWSIZE; i++)
	{
		const int index1D = LOCALWINDOWSIZE*DTid.x+i;
		const uint2 index = uint2(index1D%g_imageWidth, index1D/g_imageWidth);

		if(index.x < g_imageWidth && index.y < g_imageHeigth)
		{
			if(target[index].x != MINF && input[index].x != MINF && targetNormals[index].x != MINF)
			{
				const float3 inputT = g_meanStDevInv*(mul(input[index], g_deltaTransform).xyz-g_mean);
				const float3 targetT = g_meanStDevInv*(target[index]-g_mean);
				const float weight = targetNormals[index].w;

				// Compute Linearized System
				addToLocalSystem(targetT, inputT, targetNormals[index].xyz, weight, GTid.x);
			}
		}
	}
	
	GroupMemoryBarrierWithGroupSync();

	// Up sweep 2D
    [unroll]
	for(unsigned int stride = groupthreads/2; stride > 32; stride >>= 1)
	{
		if(GTid.x < stride) addToLocalScanElement(GTid.x+stride/2, GTid.x);

		GroupMemoryBarrierWithGroupSync();
	}

	if(GTid.x < 32) warpReduce(GTid.x);

	// Copy to output texture
	if(GTid.x == 0) CopyToResultScanElement(0, GID.x);
}
