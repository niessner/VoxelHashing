#include "stdafx.h"

#include "removeDuplicate.h"

unsigned int RemoveDuplicate::removeDuplicateVertices(std::vector<Vertex>& verts, std::vector<unsigned int>& tris)
{
	int numV = (int)verts.size();
	//int numT = (int)tris.size();

	std::map<Vertex, int, bool(*)(const Vertex&, const Vertex&)> pts(Vertexless);
	
	std::vector<unsigned int> vertexLookUp;	vertexLookUp.resize(numV);
	std::vector<Vertex> new_verts; new_verts.reserve(numV);

	int cnt = 0;
	for(int i1 = 0; i1 < numV; i1++)
	{
		Vertex pt = verts[i1];

		std::map<Vertex, int, bool(*)(const Vertex&, const Vertex&)>::iterator it = pts.find(pt);

		if(it != pts.end())
		{
			vertexLookUp[i1] = it->second;
		}
		else
		{
			pts.insert(std::make_pair(pt, cnt));
			new_verts.push_back(pt);
			vertexLookUp[i1] = cnt;
			cnt++;
		}
	}

	// Update triangles
	for(std::vector<unsigned int>::iterator it = tris.begin(); it != tris.end(); it++)
	{
		*it = vertexLookUp[*it];
	}

	std::cerr << "Removed " << numV-cnt << " duplicate vertices of " << numV << " vertices" << std::endl;
	verts = std::vector<Vertex>(new_verts.begin(), new_verts.end());

	return cnt;
}

unsigned int RemoveDuplicate::removeDuplicateTriangles(std::vector<unsigned int>& tris)
{
	int numT = (int)tris.size();

	std::map<Triangle, int, bool(*)(const Triangle&, const Triangle&)> pts(Triangleless);
	
	std::vector<unsigned int> trisNew; trisNew.reserve(numT);

	int cnt = 0;
	for(int i1 = 0; i1 < numT/3; i1++)
	{
		Triangle tri;
		tri.Idx0 = tris[3*i1+0];
		tri.Idx1 = tris[3*i1+1];
		tri.Idx2 = tris[3*i1+2];

		std::map<Triangle, int, bool(*)(const Triangle&, const Triangle&)>::iterator it = pts.find(tri);

		if(it == pts.end())
		{
			pts.insert(std::make_pair(tri, cnt));
			trisNew.push_back(tri.Idx0);
			trisNew.push_back(tri.Idx1);
			trisNew.push_back(tri.Idx2);
			cnt++;
		}
	}

	tris = std::vector<unsigned int>(trisNew.begin(), trisNew.end());

	std::cerr << "Removed " << numT/3-cnt << " duplicate triangles of " << numT/3 << " triangles" << std::endl;

	return cnt;
}
