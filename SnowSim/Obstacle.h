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
static bool planeHitFlip(const Obstacle& obs, const float3& point)
{
	return dot(point - obs.pos, -1.f * obs.misc) <= 0.f;
}

__device__
static bool sphereHit(const Obstacle& obs, const float3& point)
{
	return length(point - obs.pos) <= obs.misc.x;
}

__device__
static float3 planeNormal(const Obstacle& obs, const float3& point)
{
	return obs.misc;
}

__device__
static float3 sphereNormal(const Obstacle& obs, const float3& point)
{
	return normalize(point - obs.pos);
}

__device__
static float3 planeNormalFlip(const Obstacle& obs, const float3& point)
{
	return -1.f * obs.misc;
}

__device__
static hitF hitFunctions[] =
{
	planeHit,
	sphereHit,
	planeHitFlip,
};

__device__
static normalF normalFunctions[] =
{
	planeNormal,
	sphereNormal,
	planeNormalFlip,
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