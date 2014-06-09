Texture2D<float4> inputPositions : register (t10);
Texture2D<float4> inputNormals : register (t11);
Texture2D<float4> inputColors : register (t12);
Texture2D<float> inputSSAOMap : register (t13);

#include "GlobalAppStateShaderBuffer.h.hlsl"

sampler g_PointSampler : register (s10);

cbuffer cbPerFrame : register( b0 )
{
	uint g_useMaterial; 
	uint g_useSSAO;
	uint dummy1;
	uint dummy2;
};

cbuffer cbPerFrame : register( b10 )
{
	float m_WidthoverNextPowOfTwo;
	float m_HeightoverNextPowOfTwo;
	float g_Scale;
	uint dummy10;
};

struct VS_INPUT
{
    float3 vPosition		: POSITION;
	float2 vTexcoord		: TEXCOORD;
};

struct VS_OUTPUT
{
	float4 vPosition		: SV_POSITION;
	float2 vTexcoord		: TEXCOORD;
};

#define MINF asfloat(0xff800000)

float4 PhongPS(VS_OUTPUT Input) : SV_TARGET
{
	float3 position = inputPositions.Sample(g_PointSampler, Input.vTexcoord).xyz;
	float3 normal = inputNormals.Sample(g_PointSampler, Input.vTexcoord).xyz;
	float3 color = inputColors.Sample(g_PointSampler, Input.vTexcoord).xyz;

	if(position.x != MINF && color.x != MINF && normal.x != MINF)
	{
		//float4 material= float4(1.0f, 1.0f, 1.0f, 1.0f);
		float4 material = materialDiffuse;

		if(g_useMaterial == 1)
		{
			material = float4(color, 1.0f);
		}

		float4 lightAmbientMod = lightAmbient;
		if(g_useSSAO == 1)
		{
			lightAmbientMod*=inputSSAOMap.Sample(g_PointSampler, Input.vTexcoord);
		}
			
		//float3 lightDir = float3(1.0f, 0.0f, 1.0f);
		//lightDir = normalize(lightDir);

		float3 lightDir;
		// point light
		if (g_LightDir.w > 0) {
			lightDir = position.xyz-g_LightDir.xyz;
		} else {	//directional light
			lightDir = g_LightDir.xyz;
		}
		lightDir = normalize(lightDir);
		
		const float3 eyeDir = normalize(position);
		const float3 R = normalize(reflect(-lightDir, -normal)); // should'nt the normal be flipped after it is computed
	
		float4 res =  lightAmbientMod * material // Ambient
					+ lightDiffuse * material * max(dot(-normal, lightDir), 0.0) // Diffuse
					+ lightSpecular * materialSpecular * pow(max(dot(R, eyeDir), 0.0f), materialShininess); // Specular

		return float4(res.xyz, 1.0f);
	}	
	else
	{
		//return float4(1.0f, 1.0f, 1.0f, 1.0f); // change this back to float4(0.0f, 0.0f, 0.0f, 0.0f); at some point.
		return float4(0.0f, 0.0f, 0.0f, 0.0f); // change this back to float4(0.0f, 0.0f, 0.0f, 0.0f); at some point.
	}
}
