#pragma once

#include "SnowParticle.cuh"
#include "Obstacle.h"
#include "GridCell.cuh"
#include "Grid.h"

class Simulation
{
public:
	void SetupSim(
		Grid<GridCell>* grid,
		SnowParticle* particleList,
		uint numParticles,
		Obstacle* obstacles,
		uint numObstacles);

	void StepSim(
		float deltaT,
		uint frame);

	void Draw();

private:
	Grid<GridCell>* grid;
	SnowParticle* particles;
	uint numParticles;
	Obstacle* obstacles;
	uint numObstacles;
};