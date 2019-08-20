#include "Col.h"
#include "CollisionDetection.h"

std::set<Collision> ComputeTetHit(
	const std::vector<TetGenObj*> tetGenVec,
	float3 pos)
{
	std::set<Collision> col;
	auto p =
		Eigen::Vector3d(pos.x, pos.y, pos.z);
	collisionDetection(
		p,
		tetGenVec,
		col);

	return col;
}