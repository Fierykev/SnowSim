#pragma once

#include <vector>
#include "Helper.h"

#include "ObjectFileLoader.h"

class TetGenObj
{
public:
	TetGenObj(const char *filename);
	~TetGenObj();
	double distance(float3 pFace, int tet) const;
	float3 Ddistance(int tet) const;

	void Render();

	void* GetTets() const
	{
		return pT;
	}

	void* GetVerts() const
	{
		return pV;
	}

	float friction = .001f;
	void* AABB;

private:
	void* pV, *pF, *pT; // Needed because CUDA compiler issues;

	ObjLoader obj;
	std::vector<double> distances;
	void computeDistances();
};