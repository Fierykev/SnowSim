#pragma once

#include "Helper.h"

struct Mat
{
	float mu;
	float xi;
	float lambda;

	float compressionRatio;
	float stretchRatio;

	// YOUNGS MODULUS
	#define E0 1.4e5 // default modulus
	#define MIN_E0 4.8e4
	#define MAX_E0 1.4e5

	// CRITICAL COMPRESSION
	#define MIN_THETA_C 1.9e-2
	#define MAX_THETA_C 2.5e-2

	// CRITICAL STRETCH
	#define MIN_THETA_S 5e-3
	#define MAX_THETA_S 7.5e-3

	#define POISSONS_RATIO 0.2

	__host__ __device__ Mat()
	{
		// TODO: fix
		YoungsPoissons(E0, POISSONS_RATIO);
		xi = 10;
		CriticalStrains(MAX_THETA_C, MAX_THETA_S);
	}

	__host__ __device__
	void YoungsPoissons(float E, float v)
	{
		lambda =
			(E * v) /
			((1.f + v) * (1.f - 2.f * v));
		mu =
			E /
			(2.f * (1.f + v));
	}

	__host__ __device__
	void CriticalStrains(
		float thetaC,
		float thetaS)
	{
		compressionRatio = 1.f - thetaC;
		stretchRatio = 1.f + thetaS;
	}
};