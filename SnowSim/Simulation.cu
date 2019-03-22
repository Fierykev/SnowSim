#include <cuda_runtime_api.h>
#include "SnowParticle.h"

#define NUM_THREADS 1024

__global__
void InitMass(
	SnowParticle* particles,
	int numParticles,
	int gridDimX,
	int gridDimY)
{
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y;

	int id = (blockIdx.y * blockDim.y) * gridDimX + px;

	// Bounds check.
	if (numParticles <= id)
	{
		return;
	}

	 const SnowParticle& particle =
		 particles[id];


}

void InitVolume(
	SnowParticle* particleList,
	int numParticles,
	int gridDimX,
	int gridDimY)
{
	size_t numVoxels =
		gridDimX * gridDimY;

	float* voxelMass;
	{
		cudaError(
			cudaMalloc(
				(void**)voxelMass,
				numVoxels * sizeof(float)));
		cudaError(
			cudaMemset(
				voxelMass,
				0,
				numVoxels * sizeof(float)));
	}

	const dim3 blocks(
		(numParticles + NUM_THREADS - 1)
		/ NUM_THREADS, 64, 1);
	const dim3 threads(
		NUM_THREADS / 64, 64, 1);

	{/*
		InitMass<<<blocks, threads>>>(
			,
			numParticles,
			gridDimX,
			gridDimY);*/
	}
}

void StepSimulation()
{

}