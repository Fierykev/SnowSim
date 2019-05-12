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
	float3x3& operator =(const float3x3 &val)
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

	__device__
		void svdDecomp(
			float3x3& u,
			float3x3& s,
			float3x3& v)
	{
		/*
		mat3 u_, s_, v_;

		mat3 a = mat3(d[0], d[1], d[2],
			d[3], d[4], d[5],
			d[6], d[7], d[8]);
		computeSVD(
			a, u_, s_, v_);

		for (int i = 0; i < 9; i++)
		{
			u.d[i] = u_.data[i];
			v.d[i] = v_.data[i];
			s.d[i] = s_.data[i];
		}*/
		
		svd(
			d[0], d[3], d[6],
			d[1], d[4], d[7],
			d[2], d[5], d[8],

			u.d[0], u.d[3], u.d[6],
			u.d[1], u.d[4], u.d[7],
			u.d[2], u.d[5], u.d[8],

			s.d[0], s.d[3], s.d[6],
			s.d[1], s.d[4], s.d[7],
			s.d[2], s.d[5], s.d[8],

			v.d[0], v.d[3], v.d[6],
			v.d[1], v.d[4], v.d[7],
			v.d[2], v.d[5], v.d[8]);
	}

	__device__
	float3x3 polarDecomp()
	{
		float3x3 u, s, v;

		svdDecomp(u, s, v);
		return u.multABt(v);
	}

	__host__ __device__
	float3x3 multABt(float3x3 val)
	{
		float3x3 out;
		out.d[0] = d[0] * val.d[0] + d[3] * val.d[3] + d[6] * val.d[6];
		out.d[1] = d[1] * val.d[0] + d[4] * val.d[3] + d[7] * val.d[6];
		out.d[2] = d[2] * val.d[0] + d[5] * val.d[3] + d[8] * val.d[6];
		out.d[3] = d[0] * val.d[1] + d[3] * val.d[4] + d[6] * val.d[7];
		out.d[4] = d[1] * val.d[1] + d[4] * val.d[4] + d[7] * val.d[7];
		out.d[5] = d[2] * val.d[1] + d[5] * val.d[4] + d[8] * val.d[7];
		out.d[6] = d[0] * val.d[2] + d[3] * val.d[5] + d[6] * val.d[8];
		out.d[7] = d[1] * val.d[2] + d[4] * val.d[5] + d[7] * val.d[8];
		out.d[8] = d[2] * val.d[2] + d[5] * val.d[5] + d[8] * val.d[8];

		return out;
	}

	__host__ __device__
	float3x3 multABCt(const float3x3& valB, const float3x3& valC)
	{
		float3x3 out;
		out.d[0] = d[0] * valC.d[0] * valB.d[0] + d[3] * valC.d[3] * valB.d[4] + d[6] * valC.d[6] * valB.d[8];
		out.d[1] = d[1] * valC.d[0] * valB.d[0] + d[4] * valC.d[3] * valB.d[4] + d[7] * valC.d[6] * valB.d[8];
		out.d[2] = d[2] * valC.d[0] * valB.d[0] + d[5] * valC.d[3] * valB.d[4] + d[8] * valC.d[6] * valB.d[8];
		out.d[3] = d[0] * valC.d[1] * valB.d[0] + d[3] * valC.d[4] * valB.d[4] + d[6] * valC.d[7] * valB.d[8];
		out.d[4] = d[1] * valC.d[1] * valB.d[0] + d[4] * valC.d[4] * valB.d[4] + d[7] * valC.d[7] * valB.d[8];
		out.d[5] = d[2] * valC.d[1] * valB.d[0] + d[5] * valC.d[4] * valB.d[4] + d[8] * valC.d[7] * valB.d[8];
		out.d[6] = d[0] * valC.d[2] * valB.d[0] + d[3] * valC.d[5] * valB.d[4] + d[6] * valC.d[8] * valB.d[8];
		out.d[7] = d[1] * valC.d[2] * valB.d[0] + d[4] * valC.d[5] * valB.d[4] + d[7] * valC.d[8] * valB.d[8];
		out.d[8] = d[2] * valC.d[2] * valB.d[0] + d[5] * valC.d[5] * valB.d[4] + d[8] * valC.d[8] * valB.d[8];

		return out;
	}

	__host__ __device__ __forceinline__
	static float3x3 outerProduct(
		const float3& a,
		const float3& b)
	{
		float3x3 out;
		{
			out.d[0] = a.x * b.x;
			out.d[1] = a.y * b.x;
			out.d[2] = a.z * b.x;
			out.d[3] = a.x * b.y;
			out.d[4] = a.y * b.y;
			out.d[5] = a.z * b.y;
			out.d[6] = a.x * b.z;
			out.d[7] = a.y * b.z;
			out.d[8] = a.z * b.z;
		}

		return out;
	}

	__host__ __device__
	float3x3 transpose()
	{
		float3x3 out;
		out.d[0] = d[0];
		out.d[1] = d[3];
		out.d[2] = d[6];
		out.d[3] = d[1];
		out.d[4] = d[4];
		out.d[5] = d[7];
		out.d[6] = d[2];
		out.d[7] = d[5];
		out.d[8] = d[8];

		return out;
	}

	__host__ __device__ __forceinline__
	float3x3 operator *(const float3x3 &val) const
	{
		float3x3 out;
		out.d[0] = d[0] * val.d[0] + d[3] * val.d[1] + d[6] * val.d[2];
		out.d[1] = d[1] * val.d[0] + d[4] * val.d[1] + d[7] * val.d[2];
		out.d[2] = d[2] * val.d[0] + d[5] * val.d[1] + d[8] * val.d[2];
		out.d[3] = d[0] * val.d[3] + d[3] * val.d[4] + d[6] * val.d[5];
		out.d[4] = d[1] * val.d[3] + d[4] * val.d[4] + d[7] * val.d[5];
		out.d[5] = d[2] * val.d[3] + d[5] * val.d[4] + d[8] * val.d[5];
		out.d[6] = d[0] * val.d[6] + d[3] * val.d[7] + d[6] * val.d[8];
		out.d[7] = d[1] * val.d[6] + d[4] * val.d[7] + d[7] * val.d[8];
		out.d[8] = d[2] * val.d[6] + d[5] * val.d[7] + d[8] * val.d[8];

		return out;
	}

	__host__ __device__ __forceinline__
	void print()
	{
		printf("\n%10f %10f %10f\n"\
			"%10f %10f %10f\n"\
			"%10f %10f %10f\n",
			d[0], d[3], d[6],
			d[1], d[4], d[7],
			d[2], d[5], d[8]);
	}
};

__host__ __device__ __forceinline__
float3x3 operator* (float a, const float3x3& b)
{
	return b * a;
}