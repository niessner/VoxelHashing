#ifndef _BASEIMAGE_HELPER_H_
#define _BASEIMAGE_HELPER_H_

namespace ml {

class BaseImageHelper {
public:
	
	template<class T, class S> 
	__forceinline static void convertBaseImagePixel(T& out, const S& in);

	template<> __forceinline static void convertBaseImagePixel<vec3f, vec3uc>(vec3f& out, const vec3uc& in) {
		out.x = in.x / 255.0f;
		out.y = in.y / 255.0f;
		out.z = in.z / 255.0f;
	}

	template<> __forceinline static void convertBaseImagePixel<vec3uc, vec3f>(vec3uc& out, const vec3f& in) {
		out.x = (unsigned char)(in.x * 255);
		out.y = (unsigned char)(in.y * 255);
		out.z = (unsigned char)(in.z * 255);
	}

	template<> __forceinline static void convertBaseImagePixel<vec4f, vec4uc>(vec4f& out, const vec4uc& in) {
		out.x = in.x / 255.0f;
		out.y = in.y / 255.0f;
		out.z = in.z / 255.0f;
		out.w = in.w / 255.0f;
	}

	template<> __forceinline static void convertBaseImagePixel<vec4uc, vec4f>(vec4uc& out, const vec4f& in) {
		out.x = (unsigned char)(in.x * 255);
		out.y = (unsigned char)(in.y * 255);
		out.z = (unsigned char)(in.z * 255);
		out.w = (unsigned char)(in.w * 255);
	}

	template<> __forceinline static void convertBaseImagePixel<vec3f, float>(vec3f& out, const float& in) {
		out = convertDepthToRGB(in);
	}

	template<> __forceinline static void convertBaseImagePixel<vec3uc, float>(vec3uc& out, const float& in) {
		vec3f tmp = convertDepthToRGB(in);
		convertBaseImagePixel(out, tmp);
	}
	template<> __forceinline static void convertBaseImagePixel<vec4f, float>(vec4f& out, const float& in) {
		out = convertDepthToRGB(in);
		out.w = 0.0f;
	}

	template<> __forceinline static void convertBaseImagePixel<vec4uc, float>(vec4uc& out, const float& in) {
		vec4f tmp = convertDepthToRGB(in);
		convertBaseImagePixel(out, tmp);
	}



	template<> __forceinline static void convertBaseImagePixel<vec3uc, vec4uc>(vec3uc& out, const vec4uc& in) {
		out.x = in.x;
		out.y = in.y;
		out.z = in.z;
	}





	__forceinline static vec3f convertHSVtoRGB(const vec3f& hsv) {
		float H = hsv[0];
		float S = hsv[1];
		float V = hsv[2];

		float hd = H/60.0f;
		unsigned int h = (unsigned int)hd;
		float f = hd-h;

		float p = V*(1.0f-S);
		float q = V*(1.0f-S*f);
		float t = V*(1.0f-S*(1.0f-f));

		if(h == 0 || h == 6)
		{
			return vec3f(V, t, p);
		}
		else if(h == 1)
		{
			return vec3f(q, V, p);
		}
		else if(h == 2)
		{
			return vec3f(p, V, t);
		}
		else if(h == 3)
		{
			return vec3f(p, q, V);
		}
		else if(h == 4)
		{
			return vec3f(t, p, V);
		}
		else
		{
			return vec3f(V, p, q);
		}
	}

	__forceinline static vec3f convertDepthToRGB(float depth, float depthMin = 0.0f, float depthMax = 1.0f) {
		float depthZeroOne = (depth - depthMin)/(depthMax - depthMin);
		float x = 1.0f-depthZeroOne;
		if (x < 0.0f)	x = 0.0f;
		if (x > 1.0f)	x = 1.0f;
		return convertHSVtoRGB(vec3f(240.0f*x, 1.0f, 0.5f));
	}

};

} // namespace ml

#endif

