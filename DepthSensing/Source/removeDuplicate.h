#pragma  once

#ifndef REMOVE_DUPLICATE_VERTICES_H
#define REMOVE_DUPLICATE_VERTICES_H

/************************************************************************/
/* Helper functions for marching cubes to remove duplicate vertices     */
/************************************************************************/

#include "stdafx.h"

#include <vector>
#include <map>
#include <algorithm>

struct Vertex
{
	vec3f p;
	vec3f c;
};

struct Triangle
{
	unsigned int Idx0;
	unsigned int Idx1;
	unsigned int Idx2;
};

static bool Vertexless(const Vertex& v0, const Vertex& v1)
{
	if (v0.p[0] < v1.p[0]) return true;
	if (v0.p[0] > v1.p[0]) return false;
	if (v0.p[1] < v1.p[1]) return true;
	if (v0.p[1] > v1.p[1]) return false;
	if (v0.p[2] < v1.p[2]) return true;
	
	return false;
}

static Triangle sort(const Triangle& t)
{
	Triangle r = t;

	if(t.Idx0 < t.Idx1 && t.Idx0 < t.Idx2)
	{
		r.Idx0 = t.Idx0;

		if(t.Idx1 < t.Idx2)
		{
			r.Idx1 = t.Idx1;
			r.Idx2 = t.Idx2;
		}
		else
		{
			r.Idx1 = t.Idx2;
			r.Idx2 = t.Idx1;
		}

		return r;
	}
	
	if(t.Idx1 < t.Idx0 && t.Idx1 < t.Idx2)
	{
		r.Idx0 = t.Idx1;

		if(t.Idx0 < t.Idx2)
		{
			r.Idx1 = t.Idx0;
			r.Idx2= t.Idx2;
		}
		else
		{
			r.Idx1 = t.Idx2;
			r.Idx2 = t.Idx0;
		}

		return r;
	}

	if(t.Idx2 < t.Idx0 && t.Idx2 < t.Idx1)
	{
		r.Idx0 = t.Idx2;

		if(t.Idx0 < t.Idx1)
		{
			r.Idx1 = t.Idx0;
			r.Idx2 = t.Idx1;
		}
		else
		{
			r.Idx1 = t.Idx0;
			r.Idx2 = t.Idx1;
		}

		return r;
	}

	return r;
}


static bool Triangleless(const Triangle& t0, const Triangle& t1)
{
	Triangle t0Sorted = sort(t0);
	Triangle t1Sorted = sort(t1);

	if (t0Sorted.Idx0 < t1Sorted.Idx0) return true;
	if (t0Sorted.Idx0 > t1Sorted.Idx0) return false;
	if (t0Sorted.Idx1 < t1Sorted.Idx1) return true;
	if (t0Sorted.Idx1 > t1Sorted.Idx1) return false;
	if (t0Sorted.Idx2 < t1Sorted.Idx2) return true;
	
	return false;
}

namespace RemoveDuplicate
{
	unsigned int removeDuplicateVertices(std::vector<Vertex>& verts, std::vector<unsigned int>& tris);
	unsigned int removeDuplicateTriangles(std::vector<unsigned int>& tris);
};

#endif // REMOVE_DUPLICATE_VERTICES_H
