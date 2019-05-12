#pragma once

#include "Helper.h"
#include "Mat.cuh"

struct SnowParticle
{
	__host__
	SnowParticle()
	{

	}

	__device__
	SnowParticle(
		float HARDENING_COEFF,
		float POISSON_RATIO,
		float CRIT_COMP,
		float CRIT_STRETCH)
	{
		material.setup(
			HARDENING_COEFF,
			POISSON_RATIO,
			CRIT_COMP,
			CRIT_STRETCH);
	}

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