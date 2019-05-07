#pragma once

#include "Helper.h"

struct GridCell
{
	float mass = 0.f;
	float3 velocity =
		make_float3(0.f, 0.f, 0.f);
	float3 force =
		make_float3(0.f, 0.f, 0.f);
	float3 deltaV =
		make_float3(0.f, 0.f, 0.f);
};