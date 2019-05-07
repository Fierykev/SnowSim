#pragma once

#include "Helper.h"

struct GridInfo
{
	float3 position;
	unsigned int width, height, depth;
	float scale;

	__device__
	static uint3 GetRelativePos(uint index, uint3 dim)
	{
		uint3 ret;

		ret.z =
			index / (dim.x * dim.y);

		index -=
			ret.z * dim.x * dim.y;

		ret.y =
			index / dim.x;
		ret.x =
			index % dim.x;

		return ret;
	}

	__device__
	float3 GetGrid(uint index)
	{
		uint3 ret;

		ret.z =
			index / (width * height);

		index -=
			ret.z * width * height;

		ret.y =
			index / width;
		ret.x =
			index % width;

		return position +
			make_float3(ret.x, ret.y, ret.z)
			* scale;
	}

	__device__
	uint GetIndex(float3 pos)
	{
		float3 index = pos - position;
		index = index / scale;

		return uint(index.x) +
			uint(index.y) * width +
			uint(index.z) * width * height;
	}

	__device__
	uint GetIndex(uint3 cell)
	{
		return cell.x +
			cell.y * width +
			cell.z * width * height;
	}
	__device__
	static uint GetIndex(uint3 cell, uint3 dim)
	{
		return cell.x +
			cell.y * dim.x +
			cell.z * dim.x * dim.y;
	}

	__device__
	uint3 GetCellPos(float3 pos)
	{
		float3 index = pos - position;
		index = index / scale;

		return make_uint3(
			index.x,
			index.y,
			index.z);
	}

	__device__
	float3 GetCellPosF(float3 pos)
	{
		float3 index = pos - position;
		index = index / scale;

		return index;
	}

	__device__
	bool InsideGrid(float3 pos)
	{
		float3 endPos = position +
			make_float3(width, height, depth) * scale;

		return position.x <= pos.x &&
			pos.x <= endPos.x &&
			position.y <= pos.y&&
			pos.y <= endPos.y &&
			position.z <= pos.z &&
			pos.z <= endPos.z;
	}

	__device__
	bool InsideGrid(int3 cell)
	{
		float3 endPos =
			make_float3(width, height, depth);

		return 0 <= cell.x &&
			cell.x <= endPos.x &&
			0 <= cell.y&&
			cell.y <= endPos.y &&
			0 <= cell.z &&
			cell.z <= endPos.z;
	}
};

template <typename T>
class Grid
{
public:
	Grid() :
		gridInfo{ 0, 0, 0, 0 }
	{

	}

	Grid(
		float3 center,
		unsigned int width,
		unsigned int height,
		unsigned int depth,
		float scale)
	{
		Resize(
			center,
			width,
			height,
			depth,
			scale);
	}

	~Grid()
	{
		cudaFree(grid);
	}

	T* Data()
	{
		return grid;
	}

	void Resize(
		float3 center,
		unsigned int width,
		unsigned int height,
		unsigned int depth,
		float scale)
	{
		gridInfo.width = width;
		gridInfo.height = height;
		gridInfo.depth = depth;
		gridInfo.scale = scale;

		// Recompute bottom left.
		{
			gridInfo.position =
				center -
				make_float3(gridInfo.width, gridInfo.height, gridInfo.depth) / 2.f *
				gridInfo.scale;
		}

		cudaError(
			cudaFree(grid));

		cudaError(
			cudaMalloc(
				(void**)&grid,
				gridInfo.width * gridInfo.height * gridInfo.depth * sizeof(T)));
	}

	float3 GetPosition()
	{
		return gridInfo.position;
	}

	unsigned int GetWidth()
	{
		return gridInfo.width;
	}

	unsigned int GetHeight()
	{
		return gridInfo.height;
	}

	unsigned int GetDepth()
	{
		return gridInfo.depth;
	}

	float GetScale()
	{
		return gridInfo.scale;
	}

	const GridInfo* GetGridInfo()
	{
		return &gridInfo;
	}

private:
	GridInfo gridInfo;
	T* grid = nullptr;
};