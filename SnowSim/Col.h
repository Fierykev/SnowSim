#pragma once

#include <set>
#include "Helper.h"
#include "TetGenObj.h"

struct Collision
{
	int body1; // indices into the list of rigid body instances
	int body2;

	int collidingVertex; // index into body1's vertex list
	int collidingTet; // index into body2's tetrahedra list

	// constructed so that only one collision between a vertex and a rigid body will be kept (in case the vertex straddles multiple tets)
	bool operator<(const Collision &other) const
	{
		if (body1 < other.body1)
			return true;
		if (body1 > other.body1)
			return false;
		if (body2 < other.body2)
			return true;
		if (body2 > other.body2)
			return false;
		return (collidingVertex < other.collidingVertex);
	}
};

extern std::set<Collision> ComputeTetHit(
	const std::vector<TetGenObj*> tetGenVec,
	float3 pos);