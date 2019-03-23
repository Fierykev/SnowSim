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
		NONE = 0x0,
		MODEL = 0x1,
		VOXELS = 0x1 << 1,
		PARTICLES = 0x1 << 2
	};

	SnowModel();
	SnowModel(const char* filename);

	void Load(const char* filename);

	void Voxelize(
		Grid<SnowParticle>* grid,
		short display = NONE);

	void RenderVoxels(
		Grid<SnowParticle>* grid,
		bool* occupied);

private:
	ObjLoader obj;

	static const unsigned int numThreads = 64;
	static_assert(
		numThreads % 2 == 0,
		"Number of threads must be a multiple of 2.");
};