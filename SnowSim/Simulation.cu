#include <cuda_runtime_api.h>
#include <iostream>

#include "Obstacle.h"
#include "Simulation.cuh"

#define NUM_THREADS 1024

#define GRAVITY make_float3(0.f, -9.8f, 0.f)

__device__
float sign(float a)
{
	float val = a > 0;
	return val - (a < 0);
}

__device__
void atomicAdd(float3* address, float3 val)
{
	atomicAdd(&address->x, val.x);
	atomicAdd(&address->y, val.y);
	atomicAdd(&address->z, val.z);
}

__device__
float3 sign(float3 a)
{
	float3 val;
	
	return make_float3(
		sign(a.x),
		sign(a.y),
		sign(a.z));
}

__host__ __device__
float bSplineFalloff(float d)
{
	float _d =
		((0 <= _d && _d < 1) * (.5*_d*_d*_d - _d * _d + 2.f / 3.f) +
		(1 <= _d && _d < 2) * (-1.f / 6.f*_d*_d*_d + _d * _d - 2 * _d + 4.f / 3.f));

	return _d;
}
/*
__device__
float bSplineFalloff(float w)
{
	float a =
		(0.f <= w && w < 1.f) *
		(.5f * w * w * w -
			w * w +
			2.f / 3.f);
	float b =
		(1.f <= w && w < 2.f) *
		(-1.f / 6.f * w * w * w +
			w * w -
			2.f * w +
			4.f / 3.f);

	return a + b;
}*/

__device__
float bSplineGradFalloff(float w)
{
	float a =
		(0.f <= w && w < 1.f) *
		(1.5f * w * w - 2.f * w);
	float b =
		(1.f <= w && w < 2.f) *
		(-.5f * w * w + 2.f * w - 2.f);

	return a + b;
}

__device__
void computeWeightAndGrad(
	const float3& val,
	float& weight,
	float3& weightGrad)
{
	// TODO: rename shit
	const float3 sdx = sign(val);
	const float3 adx = fabs(val);
	const float3 N =
		make_float3(
			bSplineFalloff(adx.x),
			bSplineFalloff(adx.y),
			bSplineFalloff(adx.z));

	weight =
		N.x *
		N.y *
		N.z;

	const float3 Nx =
		sdx *
		make_float3(
			bSplineGradFalloff(adx.x),
			bSplineGradFalloff(adx.y),
			bSplineGradFalloff(adx.z));
	weightGrad.x = Nx.x * N.y * N.z;
	weightGrad.y = N.x * Nx.y * N.z;
	weightGrad.z = N.x * N.y * Nx.z;
}

__global__
void InitMass(
	SnowParticle* particles,
	float* voxelMass,
	GridInfo gridInfo,
	uint numParticles)
{
	int id = 
		blockIdx.y * gridDim.x * blockDim.x +
		blockIdx.x * blockDim.x + threadIdx.x;

	// Bounds check.
	if (numParticles <= id)
	{
		return;
	}

	const SnowParticle& particle =
		particles[id];

	float3 cellIndexF =
		gridInfo.GetCellPosF(
			particles[id].position);

	uint3 relPos =
		GridInfo::GetRelativePos(
			threadIdx.y,
			make_uint3(4, 4, 4));

	int3 cell =
		make_int3(
			cellIndexF.x - 1,
			cellIndexF.y - 1,
			cellIndexF.z - 1);
	cell += make_int3(
		relPos.x,
		relPos.y,
		relPos.z);

	// Check within grid.
	if (gridInfo.InsideGrid(cell))
	{
		float3 delta =
			fabs(make_float3(
				cell.x,
				cell.y,
				cell.z)
				- cellIndexF);

		// 1D b-spline falloff.
		float weight =
			bSplineFalloff(delta.x) *
			bSplineFalloff(delta.y) *
			bSplineFalloff(delta.z);

		uint3 cellU =
			make_uint3(
				cell.x,
				cell.y,
				cell.z);

		atomicAdd(
			&voxelMass[GridInfo::GetIndex(
				cellU, make_uint3(
					gridInfo.width + 1,
					gridInfo.height + 1,
					gridInfo.depth + 1))],
			particle.mass * weight);
	}
}

__global__
void InitDensity(
	SnowParticle* particles,
	float* voxelMass,
	GridInfo gridInfo,
	uint numParticles)
{
	int id =
		blockIdx.y * gridDim.x * blockDim.x +
		blockIdx.x * blockDim.x + threadIdx.x;

	// Bounds check.
	if (numParticles <= id)
	{
		return;
	}

	SnowParticle& particle =
		particles[id];

	float3 cellIndexF =
		gridInfo.GetCellPosF(
			particles[id].position);

	uint3 relPos =
		GridInfo::GetRelativePos(
			threadIdx.y,
			make_uint3(4, 4, 4));

	int3 cell =
		make_int3(
			cellIndexF.x - 1,
			cellIndexF.y - 1,
			cellIndexF.z - 1);
	cell += make_int3(
		relPos.x,
		relPos.y,
		relPos.z);

	// Check within grid.
	if (gridInfo.InsideGrid(cell))
	{
		float3 delta =
			fabs(make_float3(
				cell.x,
				cell.y,
				cell.z)
				- cellIndexF);

		// 1D b-spline falloff.
		float weight =
			bSplineFalloff(delta.x) *
			bSplineFalloff(delta.y) *
			bSplineFalloff(delta.z);

		uint3 cellU =
			make_uint3(
				cell.x,
				cell.y,
				cell.z);

		atomicAdd(
			&particle.volume,
				voxelMass[GridInfo::GetIndex(
					cellU, make_uint3(
						gridInfo.width + 1,
						gridInfo.height + 1,
						gridInfo.depth + 1))] *
				weight /
				(gridInfo.scale * gridInfo.scale * gridInfo.scale));
	}
}

__global__
void InitVolume(
	SnowParticle* particles,
	GridInfo gridInfo,
	uint numParticles)
{
	uint id = blockIdx.x * blockDim.x + threadIdx.x;

	// Bounds check.
	if (numParticles <= id)
	{
		return;
	}

	SnowParticle& particle =
		particles[id];
	
	particle.volume =
		particle.mass / particle.volume;
}

// FINISHED
void Simulation::SetupSim(
	Grid<GridCell>* grid,
	SnowParticle* particleList,
	uint numParticles)
{
	{
		this->grid = grid;
		this->particles = particleList;
		this->numParticles = numParticles;
	}

	GridInfo gridInfo =
		*grid->GetGridInfo();

	size_t numNodes =
		(gridInfo.width + 1) *
		(gridInfo.height + 1) *
		(gridInfo.depth + 1);

	float* voxelMass;
	{
		cudaError(
			cudaMalloc(
				&voxelMass,
				numNodes * sizeof(float)));
		cudaError(
			cudaMemset(
				voxelMass,
				0,
				numNodes * sizeof(float)));
	}

	const dim3 threads(
		NUM_THREADS / 64,
		64,
		1);
	const dim3 blocks(
		(numParticles + NUM_THREADS - 1) / NUM_THREADS,
		64,
		1);

	const dim3 threads_2(
		NUM_THREADS,
		1,
		1);
	const dim3 blocks_2(
		(numParticles + NUM_THREADS - 1) / NUM_THREADS,
		1,
		1);

	{
		InitMass<<<blocks, threads>>>(
			particleList,
			voxelMass,
			gridInfo,
			numParticles);

		InitDensity<<<blocks, threads>>>(
			particleList,
			voxelMass,
			gridInfo,
			numParticles);

		InitVolume<<<blocks_2, threads_2>>> (
			particleList,
			gridInfo,
			numParticles);
	}

	{
		cudaError(
			cudaFree(voxelMass));
	}
}

__global__
void UpdateObstacles(
	Obstacle* obstacles,
	const uint colliderNum,
	float deltaT)
{
	uint id = blockIdx.x * blockDim.x + threadIdx.x;

	if (colliderNum <= id)
	{
		return;
	}

	Obstacle& obstacle =
		obstacles[id];

	obstacle.pos +=
		obstacle.vel *
		deltaT;
}

__global__
void SolveSystem(
	SnowParticle* particles,
	SnowParticleExternalData* externalData,
	uint numParticles)
{
	uint id = blockIdx.x * blockDim.x + threadIdx.x;

	if (numParticles <= id)
	{
		return;
	}

	const SnowParticle& particle =
		particles[id];

	float3x3& plasticity =
		particles[id].plasticity;
	float3x3& elasticity =
		particles[id].elasticity;

	float detP =
		plasticity.det();
	float detE =
		elasticity.det();

	float3x3 pD =
		elasticity.polarDecomp();

	const Mat& material =
		particle.material;

	float muComp = material.mu *
		expf(material.xi * (1.f - detP));
	float lambdaComp = material.lambda *
		expf(material.xi * (1.f - detP));

	externalData[id].sigma =
		(2.f * muComp * (elasticity - pD).multABt(elasticity) +
		float3x3(detE * lambdaComp * (detE - 1.f))) * -particle.volume;
}

__global__
void ComputeSim(
	SnowParticle* particles,
	SnowParticleExternalData* externalData,
	GridCell* gridCell,
	GridInfo gridInfo,
	uint numParticles)
{
	int id =
		blockIdx.y * gridDim.x * blockDim.x +
		blockIdx.x * blockDim.x + threadIdx.x;

	// Bounds check.
	if (numParticles <= id)
	{
		return;
	}

	const SnowParticle& particle =
		particles[id];

	float3 cellIndexF =
		gridInfo.GetCellPosF(
			particles[id].position);

	uint3 relPos =
		GridInfo::GetRelativePos(
			threadIdx.y,
			make_uint3(4, 4, 4));

	int3 cell =
		make_int3(
			cellIndexF.x - 1,
			cellIndexF.y - 1,
			cellIndexF.z - 1);
	cell += make_int3(
		relPos.x,
		relPos.y,
		relPos.z);

	// Check within grid.
	if (gridInfo.InsideGrid(cell))
	{
		uint3 cellU =
			make_uint3(
				cell.x,
				cell.y,
				cell.z);

		GridCell& voxel =
			gridCell[gridInfo.GetIndex(cellU)];

		float3 delta =
			cellIndexF -
			make_float3(
				cell.x,
				cell.y,
				cell.z);

		float weight;
		float3 weightGrad;
		computeWeightAndGrad(
			delta,
			weight,
			weightGrad);

		atomicAdd(
			&voxel.mass,
			particle.mass * weight);
		atomicAdd(
			&voxel.velocity,
			particle.velocity * weight);
		atomicAdd(
			&voxel.force,
			externalData[id].sigma * weightGrad);
	}
}

__global__
void ComputeCellVel(
	bool updateDeltaV,
	SnowParticle* particles,
	SnowParticleExternalData* externalData,
	GridCell* gridCell,
	GridInfo gridInfo,
	float deltaT,
	uint numParticles)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	// Bounds check.
	if (gridInfo.width * gridInfo.height * gridInfo.depth <= id)
	{
		return;
	}

	GridCell& voxel =
		gridCell[id];

	if (0.f < voxel.mass)
	{
		float inv =
			1.f / voxel.mass;

		voxel.velocity *= inv;
		voxel.deltaV =
			voxel.velocity;
		voxel.force +=
			make_float3(
				voxel.mass * GRAVITY.x,
				voxel.mass * GRAVITY.y,
				voxel.mass * GRAVITY.z);
		voxel.velocity +=
			deltaT *
			inv *
			voxel.force;

		// Collisions.
		{
			// TODO:
		}

		if (updateDeltaV)
		{
			voxel.deltaV =
				voxel.velocity - voxel.deltaV;
		}
	}
}

__global__
void UpdateParticles(
	SnowParticle* particles,
	SnowParticleExternalData* externalData,
	float deltaT,
	uint numParticles)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	// Bounds check.
	if (numParticles <= id)
	{
		return;
	}

	const SnowParticle& particle =
		particles[id];

	float3x3 grad(0.f);


	// TODO: collisions

	//particle.position +=
		//deltaT * particle.velocity;
}

void Simulation::StepSim(
	float deltaT)
{
	// Allocate cache.
	SnowParticleExternalData* externalData;
	{
		cudaError(
			cudaMalloc(
				&externalData,
				numParticles * sizeof(SnowParticleExternalData)));
	}

	/*
	{
		const dim3 threads(
			NUM_THREADS,
			1,
			1);
		const dim3 blocks(
			(numObstacles + NUM_THREADS - 1) / NUM_THREADS,
			1,
			1);

		UpdateObstacles<<<blocks, threads>>>>(
			obstacles,
			numObstacles,
			deltaT);
	}*/

	{
		const dim3 threads(
			NUM_THREADS,
			1,
			1);
		const dim3 blocks(
			(numParticles + NUM_THREADS - 1) / NUM_THREADS,
			1,
			1);

		SolveSystem<<<blocks, threads>>>(
			particles,
			externalData,
			numParticles);
	}

	{
		const dim3 threads(
			NUM_THREADS / 64,
			64,
			1);
		const dim3 blocks(
			(numParticles + NUM_THREADS - 1) / NUM_THREADS,
			64,
			1);
		
		ComputeSim<<<blocks, threads>>> (
			particles,
			externalData,
			grid->Data(),
			*grid->GetGridInfo(),
			numParticles);
	}

	{
		const dim3 threads(
			NUM_THREADS,
			1,
			1);
		const dim3 blocks(
			(grid->GetWidth() * grid->GetHeight() * grid->GetDepth()
				+ NUM_THREADS - 1) / NUM_THREADS,
			1,
			1);

		ComputeCellVel<<<blocks, threads>>>(
			true,
			particles,
			externalData,
			grid->Data(),
			*grid->GetGridInfo(),
			deltaT,
			numParticles);
	}

	// TODO: implicit.
	{

	}

	// TODO: Add.
	{
		const dim3 threads(
			NUM_THREADS,
			1,
			1);
		const dim3 blocks(
			(numParticles + NUM_THREADS - 1) / NUM_THREADS,
			1,
			1);
		/*
		UpdateParticles<<<blocks, threads>>>(
			particles,
			externalData,
			numParticles);*/
	}
}