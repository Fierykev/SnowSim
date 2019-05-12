#pragma once

#include "Helper.h"

struct Mat
{
	float mu;
	float xi;
	float lambda;

	float compressionRatio;
	float stretchRatio;

	void __host__ __device__ setup(
		float HARDENING_COEFF,
		float POISSON_RATIO,
		float CRIT_COMP,
		float CRIT_STRETCH)
	{
		xi = HARDENING_COEFF;
		poisson(1.4e5, POISSON_RATIO);
		critStrain(CRIT_COMP, CRIT_STRETCH);
	}

	__host__ __device__
	void poisson(float E, float v)
	{
		lambda =
			(E * v) /
			((1.f + v) * (1.f - 2.f * v));
		mu =
			E /
			(2.f * (1.f + v));
	}

	__host__ __device__
	void critStrain(
		float thetaC,
		float thetaS)
	{
		compressionRatio = 1.f - thetaC;
		stretchRatio = 1.f + thetaS;
	}
};