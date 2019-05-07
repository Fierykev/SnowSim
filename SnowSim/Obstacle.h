#pragma once

#include "Helper.h"

struct Obstacle
{
	float3 pos;
	float3 vel;
	float friction;

	float3 misc;

	int type;
};

typedef bool (*hitF)(const Obstacle& obs, const float3& point);
typedef float3(*normalF)(const Obstacle& obs, const float3& point);

__device__
static bool planeHit(const Obstacle& obs, const float3& point)
{
	return dot(point - obs.pos, obs.misc) <= 0.f;
}

__device__
static float3 planeNormal(const Obstacle& obs, const float3& point)
{
	return obs.misc;
}

__device__
static hitF hitFunctions[1] =
{
	planeHit
};

__device__
static normalF normalFunctions[1] =
{
	planeNormal
};

__device__
static bool computeHit(const Obstacle& obs, const float3& point)
{
	return hitFunctions[obs.type](obs, point);
}

__device__
static float3 computeNormal(const Obstacle& obs, const float3& point)
{
	return normalFunctions[obs.type](obs, point);
}