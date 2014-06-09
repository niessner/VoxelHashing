#ifndef _UTIL_H_
#define _UTIL_H_

bool isValid(float4 p)
{
	return (p.x != MINF); // && (p.y != MINF) && (p.z != MINF) && (p.w != MINF);
}

bool isValidCol(float4 p)
{
	return p.w != 0.0f;
}

#endif
