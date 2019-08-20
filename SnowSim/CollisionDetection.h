#pragma once

#include <vector>
#include <set>
#include <Eigen/Core>

#include "TetGenObj.h"
#include "Col.h"

struct AABBNode;

bool vertInTet(const Eigen::Vector3d &p, const Eigen::Vector3d &q1, const Eigen::Vector3d &q2, const Eigen::Vector3d &q3, const Eigen::Vector3d &q4);
void collisionDetection(const std::vector<TetGenObj *> instances, std::set<Collision> &collisions);

void collisionDetection(
	Eigen::Vector3d point,
	const std::vector<TetGenObj *> instances,
	std::set<Collision> &collisions);

AABBNode *buildAABB(const TetGenObj * instance);