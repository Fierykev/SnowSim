#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include "helper_math.h"

#include "float3x3.h"

#define cudaError(ans) { cudaAssert((ans), __FILE__, __LINE__); }

#define curandError(ans) if((ans) != CURAND_STATUS_SUCCESS) cudaAssert((cudaErrorAssert), __FILE__, __LINE__);

inline void cudaAssert(
	cudaError_t code,
	const char* file,
	int line,
	bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(
			stderr,
			"Cuda Assert: %s %s %d\n",
			cudaGetErrorString(code),
			file,
			line);

		if (abort)
		{
			exit(code);
		}
	}
}

inline bool operator== (const float2 &a, const float2 &b)
{
	return a.x == b.x && a.y == b.y;
}

inline bool operator== (const float3 &a, const float3 &b)
{
	return a.x == b.x && a.y == b.y && a.z == a.z;
}

inline bool operator== (const float4 &a, const float4 &b)
{
	return a.x == b.x && a.y == b.y && a.z == a.z && a.w == b.w;
}

struct Vec3Compare
{
	bool operator() (const float3 &a, const float3 &b)
	{
		if (a.x < b.x)
			return true;
		else if (a.x > b.x)
			return false;
		// must be equal check y value

		if (a.y < b.y)
			return true;
		else if (a.y > b.y)
			return false;
		// must be equal check z value
		if (a.z < b.z)
			return true;

		return false;
	}
};