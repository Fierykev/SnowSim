#pragma once

#include <string>
#include <cassert>

#include "SnowParticle.h"
#include "Grid.h"
#include "ObjectFileLoader.h"

class SnowModel
{
public:
	enum DisplayType
	{
		NONE,
		MODEL,
		VOXELS,
		PARTICLES
	};

	SnowModel();
	SnowModel(const char* filename);

	void Load(const char* filename);

	void Voxelize(
		Grid<SnowParticle>* grid,
		DisplayType display = NONE);

	void RenderVoxels(
		Grid<SnowParticle>* grid,
		bool* occupied);

private:
	ObjLoader obj;

	static const unsigned int numThreads = 1024;
	static_assert(
		numThreads % 2 == 0,
		"Number of threads must be a multiple of 2.");
};