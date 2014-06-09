
#ifndef _FREEIMAGEWRAPPER_HELPER_H_
#define _FREEIMAGEWRAPPER_HELPER_H_

namespace ml {

////////////////////////////////////////
// Conversions for free image warper ///
////////////////////////////////////////


//////////////////////
// Data Read Helper //
//////////////////////

//BYTE3
template<class T>	__forceinline void convertFromBYTE3(T& output, const BYTE* input) {
	throw MLIB_EXCEPTION("Invalid Data Conversion");
	//static_assert(false, "Function should never be called");
}
template<>	__forceinline void convertFromBYTE3<vec3d>(vec3d& output, const BYTE* input) {
	output.z = input[0]/255.0;	
	output.y = input[1]/255.0;	
	output.x = input[2]/255.0;
}
template<>	__forceinline void convertFromBYTE3<vec4d>(vec4d& output, const BYTE* input) {
	output.z = input[0]/255.0;
	output.y = input[1]/255.0;
	output.x = input[2]/255.0;
	output.w = 1.0;
}
template<>	__forceinline void convertFromBYTE3<vec3f>(vec3f& output, const BYTE* input) {
	output.z = input[0]/255.0f;	
	output.y = input[1]/255.0f;	
	output.x = input[2]/255.0f;
}
template<>	__forceinline void convertFromBYTE3<vec4f>(vec4f& output, const BYTE* input) {
	output.z = input[0]/255.0f;
	output.y = input[1]/255.0f;
	output.x = input[2]/255.0f;
	output.w = 1.0f;
}
template<>	__forceinline void convertFromBYTE3<vec3i>(vec3i& output, const BYTE* input) {
	output.z = input[0];	
	output.y = input[1];	
	output.x = input[2];
}
template<>	__forceinline void convertFromBYTE3<vec4i>(vec4i& output, const BYTE* input) {
	output.z = input[0];	
	output.y = input[1];	
	output.x = input[2];
	output.w = 255;
}
template<>	__forceinline void convertFromBYTE3<vec3ui>(vec3ui& output, const BYTE* input) {
	output.z = input[0];	
	output.y = input[1];	
	output.x = input[2];
}
template<>	__forceinline void convertFromBYTE3<vec4ui>(vec4ui& output, const BYTE* input) {
	output.z = input[0];	
	output.y = input[1];	
	output.x = input[2];
	output.w = 255;
}
template<>	__forceinline void convertFromBYTE3<vec3uc>(vec3uc& output, const BYTE* input) {
	output.z = input[0];	
	output.y = input[1];	
	output.x = input[2];
}
template<>	__forceinline void convertFromBYTE3<vec4uc>(vec4uc& output, const BYTE* input) {
	output.z = input[0];	
	output.y = input[1];	
	output.x = input[2];
	output.w = 255;
}


//BYTE4
template<class T>	__forceinline void convertFromBYTE4(T& output, const BYTE* input) {
	throw MLIB_EXCEPTION("Invalid Data Conversion");
	//static_assert(false, "Function should never be called");
}
template<>	__forceinline void convertFromBYTE4<vec3d>(vec3d& output, const BYTE* input) {
	output.z = input[0]/255.0;	
	output.y = input[1]/255.0;	
	output.x = input[2]/255.0;
}
template<>	__forceinline void convertFromBYTE4<vec4d>(vec4d& output, const BYTE* input) {
	output.z = input[0]/255.0;
	output.y = input[1]/255.0;
	output.x = input[2]/255.0;
	output.w = input[3]/255.0;
}
template<>	__forceinline void convertFromBYTE4<vec3f>(vec3f& output, const BYTE* input) {
	output.z = input[0]/255.0f;	
	output.y = input[1]/255.0f;	
	output.x = input[2]/255.0f;
}
template<>	__forceinline void convertFromBYTE4<vec4f>(vec4f& output, const BYTE* input) {
	output.z = input[0]/255.0f;
	output.y = input[1]/255.0f;
	output.x = input[2]/255.0f;
	output.w = input[3]/255.0f;
}
template<>	__forceinline void convertFromBYTE4<vec3i>(vec3i& output, const BYTE* input) {
	output.z = input[0];	
	output.y = input[1];	
	output.x = input[2];
}
template<>	__forceinline void convertFromBYTE4<vec4i>(vec4i& output, const BYTE* input) {
	output.z = input[0];	
	output.y = input[1];	
	output.x = input[2];
	output.w = input[3];
}
template<>	__forceinline void convertFromBYTE4<vec3ui>(vec3ui& output, const BYTE* input) {
	output.z = input[0];	
	output.y = input[1];	
	output.x = input[2];
}
template<>	__forceinline void convertFromBYTE4<vec4ui>(vec4ui& output, const BYTE* input) {
	output.z = input[0];	
	output.y = input[1];	
	output.x = input[2];
	output.w = input[3];
}
template<>	__forceinline void convertFromBYTE4<vec3uc>(vec3uc& output, const BYTE* input) {
	output.z = input[0];	
	output.y = input[1];	
	output.x = input[2];
}
template<>	__forceinline void convertFromBYTE4<vec4uc>(vec4uc& output, const BYTE* input) {
	output.z = input[0];	
	output.y = input[1];	
	output.x = input[2];
	output.w = input[3];
}


//USHORT
template<class T>	__forceinline void convertFromUSHORT(T& output, const unsigned short* input) {
	throw MLIB_EXCEPTION("Invalid Data Conversion");
	//static_assert(false, "Function should never be called");
}
template<>	__forceinline void convertFromUSHORT<unsigned short>(unsigned short& output, const unsigned short* input) {
	output = *input;
}
template<>	__forceinline void convertFromUSHORT<float>(float& output, const unsigned short* input) {
	output = (float)*input;
	output /= 1000.0f;
}
template<>	__forceinline void convertFromUSHORT<double>(double& output, const unsigned short* input) {
	output = (double)*input;
	output /= 1000.0;
}




///////////////////////
// DATA WRITE HELPER //
///////////////////////

//VEC3UC
template<class T>	__forceinline void convertToVEC3UC(vec3uc& output, const T& input) {
	throw MLIB_EXCEPTION("Invalid Data Conversion");
	//static_assert(false, "Function should never be called");
}
template<>	__forceinline void convertToVEC3UC<vec3d>(vec3uc& output, const vec3d& input) {
	output.x = (unsigned char)(input[0]*255.0);	
	output.y = (unsigned char)(input[1]*255.0);	
	output.z = (unsigned char)(input[2]*255.0);
}
template<>	__forceinline void convertToVEC3UC<vec4d>(vec3uc& output, const vec4d& input) {
	output.x = (unsigned char)(input[0]*255.0);	
	output.y = (unsigned char)(input[1]*255.0);	
	output.z = (unsigned char)(input[2]*255.0);
}
template<>	__forceinline void convertToVEC3UC<vec3f>(vec3uc& output, const vec3f& input) {
	output.x = (unsigned char)(input[0]*255.0f);
	output.y = (unsigned char)(input[1]*255.0f);
	output.z = (unsigned char)(input[2]*255.0f);
}
template<>	__forceinline void convertToVEC3UC<vec4f>(vec3uc& output, const vec4f& input) {
	output.x = (unsigned char)(input[0]*255.0f);	
	output.y = (unsigned char)(input[1]*255.0f);	
	output.z = (unsigned char)(input[2]*255.0f);
}
template<>	__forceinline void convertToVEC3UC<vec3i>(vec3uc& output, const vec3i& input) {
	output.x = (unsigned char)input[0];	
	output.y = (unsigned char)input[1];	
	output.z = (unsigned char)input[2];
}
template<>	__forceinline void convertToVEC3UC<vec4i>(vec3uc& output, const vec4i& input) {
	output.x = (unsigned char)input[0];	
	output.y = (unsigned char)input[1];	
	output.z = (unsigned char)input[2];
}
template<>	__forceinline void convertToVEC3UC<vec3ui>(vec3uc& output, const vec3ui& input) {
	output.x = (unsigned char)input[0];	
	output.y = (unsigned char)input[1];	
	output.z = (unsigned char)input[2];
}
template<>	__forceinline void convertToVEC3UC<vec4ui>(vec3uc& output, const vec4ui& input) {
	output.x = (unsigned char)input[0];	
	output.y = (unsigned char)input[1];	
	output.z = (unsigned char)input[2];
}
template<>	__forceinline void convertToVEC3UC<vec3uc>(vec3uc& output, const vec3uc& input) {
	output.x = input[0];	
	output.y = input[1];	
	output.z = input[2];
}
template<>	__forceinline void convertToVEC3UC<vec4uc>(vec3uc& output, const vec4uc& input) {
	output.x = input[0];	
	output.y = input[1];	
	output.z = input[2];
}



//VEC4UC
template<class T>	__forceinline void convertToVEC4UC(vec4uc& output, const T& input) {
	throw MLIB_EXCEPTION("Invalid Data Conversion");
	//static_assert(false, "Function should never be called");
}
template<>	__forceinline void convertToVEC4UC<vec3d>(vec4uc& output, const vec3d& input) {
	output.x = (unsigned char)(input[0]*255.0);	
	output.y = (unsigned char)(input[1]*255.0);	
	output.z = (unsigned char)(input[2]*255.0);
	output.w = 255;
}
template<>	__forceinline void convertToVEC4UC<vec4d>(vec4uc& output, const vec4d& input) {
	output.x = (unsigned char)(input[0]*255.0);	
	output.y = (unsigned char)(input[1]*255.0);	
	output.z = (unsigned char)(input[2]*255.0);
	output.w = (unsigned char)(input[3]*255.0);
}
template<>	__forceinline void convertToVEC4UC<vec3f>(vec4uc& output, const vec3f& input) {
	output.x = (unsigned char)(input[0]*255.0f);
	output.y = (unsigned char)(input[1]*255.0f);
	output.z = (unsigned char)(input[2]*255.0f);
	output.w = 255;
}
template<>	__forceinline void convertToVEC4UC<vec4f>(vec4uc& output, const vec4f& input) {
	output.x = (unsigned char)(input[0]*255.0);
	output.y = (unsigned char)(input[1]*255.0);
	output.z = (unsigned char)(input[2]*255.0);
	output.w = (unsigned char)(input[3]*255.0f);
}
template<>	__forceinline void convertToVEC4UC<vec3i>(vec4uc& output, const vec3i& input) {
	output.x = (unsigned char)input[0];	
	output.y = (unsigned char)input[1];	
	output.z = (unsigned char)input[2];
	output.w = 255;
}
template<>	__forceinline void convertToVEC4UC<vec4i>(vec4uc& output, const vec4i& input) {
	output.x = (unsigned char)input[0];	
	output.y = (unsigned char)input[1];	
	output.z = (unsigned char)input[2];
	output.w = (unsigned char)input[3];
}
template<>	__forceinline void convertToVEC4UC<vec3ui>(vec4uc& output, const vec3ui& input) {
	output.x = (unsigned char)input[0];	
	output.y = (unsigned char)input[1];	
	output.z = (unsigned char)input[2];
	output.w = 255;
}
template<>	__forceinline void convertToVEC4UC<vec4ui>(vec4uc& output, const vec4ui& input) {
	output.x = (unsigned char)input[0];	
	output.y = (unsigned char)input[1];	
	output.z = (unsigned char)input[2];
	output.w = (unsigned char)input[3];
}
template<>	__forceinline void convertToVEC4UC<vec3uc>(vec4uc& output, const vec3uc& input) {
	output.x = input[0];	
	output.y = input[1];	
	output.z = input[2];
	output.w = 255;
}
template<>	__forceinline void convertToVEC4UC<vec4uc>(vec4uc& output, const vec4uc& input) {
	output.x = input[0];	
	output.y = input[1];	
	output.z = input[2];
	output.w = input[3];
}

} // namespace

#endif
