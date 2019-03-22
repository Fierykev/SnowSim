#pragma once

#include "Helper.h"

struct GridInfo
{
	float3 position;
	unsigned int width, height, depth;
	float scale;
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

	const T* Data()
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

	unsigned int GetScale()
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