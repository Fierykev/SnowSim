#pragma once

#include "Helper.h"

#include "svd.h"

struct float3x3
{
	float d[9];

	__host__ __device__ __forceinline__
	float3x3(float diagonal = 1.f)
	{
		d[0] = d[4] = d[8] = diagonal;
		d[1] = d[2] = d[3] = d[5] = d[6] = d[7] = 0.f;
	}

	__host__ __device__  __forceinline__
	float3x3& operator = (const float3x3 &val)
	{
#pragma unroll
		for (uint i = 0; i < 9; i++)
		{
			d[i] = val.d[i];
		}

		return *this;
	}

	__host__ __device__ __forceinline__
	float det()
	{
		return d[0] * (d[4] * d[8] - d[7] * d[5]) -
			d[3] * (d[1] * d[8] - d[7] * d[2]) +
			d[6] * (d[1] * d[5] - d[4] * d[2]);
	}

	__host__ __device__ __forceinline__
	float3x3 operator -(const float3x3 &val) const
	{
		float3x3 tmp = *this;
		
#pragma unroll
		for (uint i = 0; i < 9; i++)
		{
			tmp.d[i] -= val.d[i];
		}

		return tmp;
	}

	__host__ __device__ __forceinline__
	float3x3 operator +(const float3x3 &val) const
	{
		float3x3 tmp = *this;
		
#pragma unroll
		for (uint i = 0; i < 9; i++)
		{
			tmp.d[i] += val.d[i];
		}

		return tmp;
	}

	__host__ __device__ __forceinline__
	float3x3 operator *(const float &val) const
	{
		float3x3 tmp = *this;

#pragma unroll
		for (uint i = 0; i < 9; i++)
		{
			tmp.d[i] *= val;
		}

		return tmp;
	}

	__host__ __device__ __forceinline__
	float3 operator *(const float3 &val) const
	{
		float3 tmp;
		tmp.x = d[0] * val.x + d[3] * val.y + d[6] * val.z;
		tmp.y = d[1] * val.x + d[4] * val.y + d[7] * val.z;
		tmp.z = d[2] * val.x + d[5] * val.y + d[8] * val.z;

		return tmp;
	}

	__host__ __device__
	float3x3 polarDecomp()
	{
		float3x3 out;
		float u[9], s[9], v[9];

		svd(
			d[0], d[1], d[2],
			d[3], d[4], d[5],
			d[6], d[7], d[8],

			u[0], u[1], u[2],
			u[3], u[4], u[5],
			u[6], u[7], u[8],

			s[0], s[1], s[2],
			s[3], s[4], s[5],
			s[6], s[7], s[8],

			v[0], v[1], v[2],
			v[3], v[4], v[5],
			v[6], v[7], v[8]);

		multAtB(
			u[0], u[1], u[2],
			u[3], u[4], u[5],
			u[6], u[7], u[8],

			v[0], v[1], v[2],
			v[3], v[4], v[5],
			v[6], v[7], v[8],
			
			out.d[0], out.d[1], out.d[2],
			out.d[3], out.d[4], out.d[5],
			out.d[6], out.d[7], out.d[8]);

		return out;
	}

	__host__ __device__
	float3x3 multABt(float3x3 val)
	{
		float3x3 out;

		multAtB(
			d[0], d[1], d[2],
			d[3], d[4], d[5],
			d[6], d[7], d[8],

			val.d[0], val.d[1], val.d[2],
			val.d[3], val.d[4], val.d[5],
			val.d[6], val.d[7], val.d[8],

			out.d[0], out.d[1], out.d[2],
			out.d[3], out.d[4], out.d[5],
			out.d[6], out.d[7], out.d[8]);

		return out;
	}
};

__host__ __device__ __forceinline__
float3x3 operator* (float a, const float3x3& b)
{
	return b * a;
}