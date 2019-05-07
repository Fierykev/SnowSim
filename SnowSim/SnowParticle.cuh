#pragma once

#include "Helper.h"
#include "Mat.cuh"

struct SnowParticle
{
	float3 position;
	float mass;
	float3 velocity =
		make_float3(0.f, 0.f, 0.f);
	float volume =
		1E-9;
	float3x3 plasticity;
	float3x3 elasticity;

	Mat material;
};

struct SnowParticleExternalData
{
	float3x3 sigma;
};