Buffer<int>			g_VoxelHash : register(t0);
Texture2D<float>	prevDepth	: register(t1);
Texture2D<float4>	prevColor	: register(t2);
Texture2D<float4>	prevPos		: register(t3);
   
cbuffer cbConstant : register(b0)
{
	float		g_VirtualVoxelSize; // in meter
	float		g_SplatSize;		// multiplier for voxel size
	uint		g_ImageWidth;
	uint		g_ImageHeight;
	float4x4	g_ViewMat;
};

#include "VoxelUtil.h.hlsl"
#include "KinectCameraUtil.h.hlsl"

static const float4 offsets[4] = 
{
    float4( 0.5, -0.5, 0.0, 0.0),
    float4(-0.5, -0.5, 0.0, 0.0),
	
	float4( 0.5,  0.5, 0.0, 0.0),
    float4(-0.5,  0.5, 0.0, 0.0)
 
};

struct PS_INPUT
{
    float4 position		: SV_POSITION;
    float4 color		: COLOR;
	float4 worldPos		: POS;
	float4 voxelCenter  : CENTER;
};

struct PS_OUT_COL_POS {
	float4 color;
	float4 pos;
};


PS_INPUT VS_SINGLE_POINT(uint primID : SV_VertexID) {

	PS_INPUT output;

	VoxelData voxel = getVoxelData(primID);

	if (voxel.weight > g_MinRenderWeight) {	//found a valid voxel
		
		float4 viewPos = mul(float4(voxel.position.xyz, 1.0f), g_ViewMat);
		float4 scale = float4(g_VirtualVoxelSize, g_VirtualVoxelSize, 0.0, 0.0);
		float4 corner = viewPos;

		output.position = float4(cameraToKinectProj(corner.xyz), 1.0f);
		output.color = float4(voxel.color, 1.0f);
		output.worldPos = corner;
		output.voxelCenter = viewPos;

	} else { //did not find a valid voxel
		output.position = 1.0f;
		output.color = -1.0f;
		output.worldPos = 1.0f;
		output.voxelCenter = 1.0f;
	}

	return output;
}

static const uint mapID[6] = {0,1,2, 1,3,2};


//PS_INPUT VS_SPRITE(uint vID : SV_VertexID, uint primID : SV_InstanceID) { 
//	uint c = vID;
PS_INPUT VS_SPRITE(uint vID : SV_VertexID) {
	uint primID = vID / 6;
	uint c = mapID[vID % 6];

	PS_INPUT output;

	VoxelData voxel = getVoxelData(primID);

	if (voxel.weight > g_MinRenderWeight) {	//found a valid voxel
		
		float4 viewPos = mul(float4(voxel.position.xyz, 1.0f), g_ViewMat);
		float4 scale = float4(g_VirtualVoxelSize, g_VirtualVoxelSize, 0.0, 0.0);
		float4 corner = viewPos + offsets[c]*scale * g_SplatSize;

		output.position = float4(cameraToKinectProj(corner.xyz), 1.0f);
		output.color = float4(voxel.color, 1.0f);
		output.worldPos = corner;
		output.voxelCenter = viewPos;

	} else { //did not find a valid voxel
		output.position = 1.0f;
		output.color = -1.0f;
		output.worldPos = 1.0f;
		output.voxelCenter = 1.0f;
	}

	return output;
}


//PS_OUT_COL_POS PS_SPRITE_Blend(PS_INPUT input) : SV_Target
//{
//	//float prevZ = kinectProjToCameraZ(prevDepth[cameraToKinectScreen(input.voxelCenter.xyz).xy]);
//	float prevZ = kinectProjToCameraZ(prevDepth[input.position.xy]);
//	if (abs(prevZ - input.worldPos.z) >= cullRange) discard;

//	PS_OUT_COL_POS output;
//	float dist = distance(input.voxelCenter, input.worldPos) / g_VirtualVoxelSize;	//distance in virtual voxel space
//	if (dist > 0.5f * g_SplatSize) discard;
//	//float weight = 1.0f;
//	float weight = gauss(gaussBlendSigma * g_SplatSize, dist);
//    output.color = float4(input.color.xyz, weight);
//	output.pos = float4(input.worldPos.xyz, weight);
//	return output;
//}




struct GS_INPUT
{
};


GS_INPUT VS()
{
    GS_INPUT output = (GS_INPUT)0;
 
    return output;
}



[maxvertexcount(4)]
void GS(point GS_INPUT points[1], uint primID : SV_PrimitiveID, inout TriangleStream<PS_INPUT> triStream)
{
	VoxelData voxel = getVoxelData(primID);

	if (voxel.weight > g_MinRenderWeight) {
		PS_INPUT output;

		float4 viewPos = mul(float4(voxel.position.xyz, 1.0f), g_ViewMat);
		float4 scale = float4(g_VirtualVoxelSize, g_VirtualVoxelSize, 0.0, 0.0);

		[unroll]
		for (uint c = 0; c < 4; ++c) {
			float4 corner = viewPos+offsets[c]*scale * g_SplatSize;
			output.position = float4(cameraToKinectProj(corner.xyz), 1.0f);
			output.color = float4(voxel.color, 1.0f);
			output.worldPos = corner;
			output.voxelCenter = viewPos;
			triStream.Append(output);
		}
	}
}


[maxvertexcount(4)]
void GSBlend(point GS_INPUT points[1], uint primID : SV_PrimitiveID, inout TriangleStream<PS_INPUT> triStream)
{
	VoxelData voxel = getVoxelData(primID);

	if(voxel.weight > g_MinRenderWeight)
	{ 
		PS_INPUT output;

		float4 viewPos = mul(float4(voxel.position.xyz, 1.0f), g_ViewMat);
		float4 scale = float4(g_VirtualVoxelSize, g_VirtualVoxelSize, 0.0, 0.0);
		 
		//CULL RANGE HERE
		//float prevZ = kinectProjToCameraZ(prevDepth[cameraToKinectScreen(viewPos.xyz).xy]);
		//if (abs(prevZ - viewPos.z) < cullRange) {
			[unroll]
			for(uint c = 0; c < 4; ++c)
			{
				float4 corner = viewPos+offsets[c]*scale * g_SplatSize;
     
				//output.position = mul(corner, projMat);
				output.position = float4(cameraToKinectProj(corner.xyz), 1.0f);
				//output.color = float4(1.0f, 1.0f, 1.0f, 1.0f);
				output.color = float4(voxel.color, 1.0f);
				output.worldPos = corner;
				output.voxelCenter = viewPos;
				triStream.Append(output);
			}
		//}
	}
}




[maxvertexcount(4)]
void GSReSplat(point GS_INPUT points[1], uint primID : SV_PrimitiveID, inout TriangleStream<PS_INPUT> triStream)
{
	VoxelData voxel = getVoxelDataFromTexture(primID);

	if (voxel.weight == 1)
	{
		PS_INPUT output;

		float4 viewPos = float4(voxel.position.xyz, 1.0f);
		//float4 viewPos = mul(voxel.position, g_ViewMat);

		float4 scale = float4(g_VirtualVoxelSize, g_VirtualVoxelSize, 0.0, 0.0);

		[unroll]
		for (uint c = 0; c < 4; ++c)
		{
			float4 corner = viewPos+offsets[c]*scale * g_SplatSize;
     
			//output.position = mul(corner, projMat);
			output.position = float4(cameraToKinectProj(corner.xyz), 1.0f);
			//output.color = float4(1.0f, 1.0f, 1.0f, 1.0f);
			output.color = float4(voxel.color, 1.0f);
			output.worldPos = corner;
			output.voxelCenter = viewPos;
			triStream.Append(output);
		}
	}
}



[maxvertexcount(4)]
void GSReSplatBlend(point GS_INPUT points[1], uint primID : SV_PrimitiveID, inout TriangleStream<PS_INPUT> triStream)
{
	VoxelData voxel = getVoxelDataFromTexture(primID);

	if(voxel.weight == 1)
	{
		PS_INPUT output;

		float4 viewPos = float4(voxel.position.xyz, 1.0f);
		//float4 viewPos = mul(voxel.position, g_ViewMat);

		float4 scale = float4(g_VirtualVoxelSize, g_VirtualVoxelSize, 0.0, 0.0);
		//CULL RANGE HERE
		//float prevZ = kinectProjToCameraZ(prevDepth[cameraToKinectScreen(viewPos.xyz).xy]);
		//if (abs(prevZ - viewPos.z) < cullRange) {
			[unroll]
			for(uint c = 0; c < 4; ++c)
			{
				float4 corner = viewPos+offsets[c]*scale * g_SplatSize;
     
				//output.position = mul(corner, projMat);
				output.position = float4(cameraToKinectProj(corner.xyz), 1.0f);
				//output.color = float4(1.0f, 1.0f, 1.0f, 1.0f);
				output.color = float4(voxel.color, 1.0f);
				output.worldPos = corner;
				output.voxelCenter = viewPos;
				triStream.Append(output);
			}
		//}
	}
}




//--------------------------------------------------------------------------------------
// Pixel shaders
//--------------------------------------------------------------------------------------
 

void PS(PS_INPUT input) 
{
	float dist = distance(input.voxelCenter, input.worldPos) / g_VirtualVoxelSize;	//distance in virtual voxel space
	if (dist > 0.5f * g_SplatSize) discard;
    //return float4(input.color);
}
 
PS_OUT_COL_POS PSBlend(PS_INPUT input) : SV_Target
{
	PS_OUT_COL_POS output;
	
	float prevZ = kinectProjToCameraZ(prevDepth[input.position.xy]);
	if (abs(prevZ - input.worldPos.z) >= cullRange) discard;

	float dist = distance(input.voxelCenter, input.worldPos) / g_VirtualVoxelSize;	//distance in virtual voxel space
	if (dist > 0.5f * g_SplatSize) discard;
	//float weight = 1.0f;
	float weight = gauss(gaussBlendSigma * g_SplatSize, dist);
    output.color = float4(input.color.xyz, weight);
	output.pos = float4(input.worldPos.xyz, weight);
	return output;
}
 






//--------------------------------------------------------------------------------------
// Normalize shader
//--------------------------------------------------------------------------------------
 
sampler g_PointSampler : register (s10);

Texture2D<float4>	colors : register(t10);
Texture2D<float4>	positions : register (t11);


struct VS_OUTPUT_QUAD {
	float4 vPosition	: SV_POSITION;
	float2 vTexcoord	: TEXCOORD;
};


PS_OUT_COL_POS PSNorm(VS_OUTPUT_QUAD input) : SV_Target
{
	PS_OUT_COL_POS output;
	output.color =	colors.Sample(g_PointSampler, input.vTexcoord);
	if (output.color.w != 0.0f)	output.color /= output.color.w;

	output.pos =	positions.Sample(g_PointSampler, input.vTexcoord);
	if (output.pos.w != 0.0f)	output.pos /=	output.pos.w;
	else output.pos = float4(MINF, MINF, MINF, MINF);

	return output;
}