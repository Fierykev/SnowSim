#pragma once

#include <string>
#include <cassert>

#include "SnowParticle.cuh"
#include "GridCell.cuh"
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

	void SampleParticles(
		Grid<GridCell>* grid,
		SnowParticle* particle,
		uint numParticles,
		short display);

	void RenderVoxels(
		Grid<GridCell>* grid,
		bool* occupied);

	void RenderParticles(
		SnowParticle* particle,
		uint numParticles);

private:
	ObjLoader obj;

	static const unsigned int numThreads = 32;
	static_assert(
		numThreads % 2 == 0,
		"Number of threads must be a multiple of 2.");
};