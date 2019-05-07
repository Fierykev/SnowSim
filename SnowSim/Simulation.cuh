#pragma once

#include "SnowParticle.cuh"
#include "GridCell.cuh"
#include "Grid.h"

class Simulation
{
public:
	void SetupSim(
		Grid<GridCell>* grid,
		SnowParticle* particleList,
		uint numParticles);

	void StepSim(float deltaT);

	void Draw();

private:
	Grid<GridCell>* grid;
	SnowParticle* particles;
	uint numParticles;
};