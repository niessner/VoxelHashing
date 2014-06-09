Texture2D<float4> g_inputTex : register( t0 );

struct float8
{
	float3 mean;
	float3 meanSquared;
	float nValidCorrs;
	float pad;
};

StructuredBuffer<float8> g_input : register( t1 );
RWStructuredBuffer<float8> g_result : register( u0 );

cbuffer cbCS : register( b0 )
{
	uint g_imageWidth;
	uint g_numElements;
	uint pad0;
	uint pad1;
};

// should be set by application
#ifndef groupthreads
#define groupthreads 1
#endif

// should be set by the application
#ifndef BUCKET_SIZE
#define BUCKET_SIZE 1
#endif

groupshared float8 bucket[groupthreads];

#define MINF asfloat(0xff800000)

void CSScan(uint3 DTid, uint GI, uint Idx, uint GID, int enableTex)  
{
	bucket[GI].mean = float3(0.0f, 0.0f, 0.0f);
	bucket[GI].meanSquared = float3(0.0f, 0.0f, 0.0f);
	bucket[GI].nValidCorrs = 0.0f;
	
	if(enableTex == 1)
	{
		if(Idx < g_numElements)
		{
			uint2 index = uint2(Idx%g_imageWidth, Idx/g_imageWidth);
			float3 p = g_inputTex[index].xyz;
			if(p.x != MINF)
			{
				bucket[GI].mean += p;
				bucket[GI].meanSquared += p*p;
				bucket[GI].nValidCorrs += 1.0f;
			}
		}
	}
	else
	{
		bucket[GI] = g_input[Idx];
	}

    // Up sweep
    [unroll]
    for(uint stride = 2; stride <= groupthreads; stride <<= 1)
    {
       GroupMemoryBarrierWithGroupSync();
        
       if((GI & (stride - 1)) == (stride - 1))
       {
			bucket[GI].mean += bucket[GI - stride/2].mean;
			bucket[GI].meanSquared += bucket[GI - stride/2].meanSquared;
			bucket[GI].nValidCorrs += bucket[GI - stride/2].nValidCorrs;
       }
    }

	if(GI == groupthreads-1)
	{
		g_result[GID].mean = bucket[GI].mean;
		g_result[GID].meanSquared = bucket[GI].meanSquared;
		g_result[GID].nValidCorrs = bucket[GI].nValidCorrs;
	}
}

// scan in each bucket
[numthreads(groupthreads, 1, 1 )]
void CSScanInBucket(uint3 DTid : SV_DispatchThreadID, uint3 GTid : SV_GroupThreadID, uint GI: SV_GroupIndex, uint GID: SV_GroupID)
{
	CSScan(DTid, GI, DTid.x, GID, 1);
}

// record and scan the sum of each bucket
[numthreads( groupthreads, 1, 1 )]
void CSScanBucketResult( uint3 DTid : SV_DispatchThreadID, uint3 GTid : SV_GroupThreadID, uint GI: SV_GroupIndex, uint GID: SV_GroupID)
{
	CSScan(GTid.x, GTid.x, DTid.x, GID, 0);
}
