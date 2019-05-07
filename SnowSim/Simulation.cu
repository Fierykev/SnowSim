#include <cuda_runtime_api.h>
#include <iostream>

#include "Obstacle.h"
#include "Simulation.cuh"
#include "Cube.h"

#define NUM_THREADS 128

#define GRAVITY make_float3(0.f, -9.8f, 0.f)

#define ALPHA .05f

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
		((0 <= d && d < 1) * (.5*d*d*d - d * d + 2.f / 3.f) +
		(1 <= d && d < 2) * (-1.f / 6.f*d*d*d + d * d - 2 * d + 4.f / 3.f));

	return fabs(_d);
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
	const float3& val2,
	float& weight,
	float3& weightGrad)
{
	// TODO: rename shit
	const float3 N =
		make_float3(
			bSplineFalloff(val2.x),
			bSplineFalloff(val2.y),
			bSplineFalloff(val2.z));

	weight =
		N.x *
		N.y *
		N.z;

	const float3 Nx =
		val *
		make_float3(
			bSplineGradFalloff(val2.x),
			bSplineGradFalloff(val2.y),
			bSplineGradFalloff(val2.z));
	weightGrad.x = Nx.x * N.y * N.z;
	weightGrad.y = N.x * Nx.y * N.z;
	weightGrad.z = N.x * N.y * Nx.z;
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
		float3 delta =
			cellIndexF -
			make_float3(
				cell.x,
				cell.y,
				cell.z);

		uint3 cellU =
			make_uint3(
				cell.x,
				cell.y,
				cell.z);
		
		GridCell& voxel =
			gridCell[GridInfo::GetIndex(
				cellU, make_uint3(
					gridInfo.width + 1,
					gridInfo.height + 1,
					gridInfo.depth + 1))];
		
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
			particle.velocity * particle.mass * weight);
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
	if ((gridInfo.width + 1) *
		(gridInfo.height + 1) *
		(gridInfo.depth + 1) <= id)
	{
		return;
	}

	GridCell& voxel =
		gridCell[id];

	if (0.f < voxel.mass)
	{
		float inv =
			1.f / voxel.mass;

		voxel.velocity *=
			inv;
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

__device__
float3x3 ComputeVelGrad(
	SnowParticle& particle,
	GridCell* gridCell,
	GridInfo gridInfo)
{
	float3 cellF =
		gridInfo.GetCellPosF(particle.position);

	// Bounds for looping.
	uint3 minIndx, maxIndx;
	{
		minIndx = clamp(
			make_uint3(
				ceilf(cellF.x),
				ceilf(cellF.y),
				ceilf(cellF.z))
			- make_uint3(2, 2, 2),
			make_uint3(0, 0, 0),
			make_uint3(
				gridInfo.width,
				gridInfo.height,
				gridInfo.depth));
		maxIndx = clamp(
			make_uint3(
				floorf(cellF.x),
				floorf(cellF.y),
				floorf(cellF.z))
			+ make_uint3(2, 2, 2),
			make_uint3(0, 0, 0),
			make_uint3(
				gridInfo.width,
				gridInfo.height,
				gridInfo.depth));
	}

	// PIC / FLIP sim.
	float3 pic, flip;
	float3x3 velGrad(0.f);
	{
		pic = flip = make_float3(0.f, 0.f, 0.f);

		for (uint x = minIndx.x; x <= maxIndx.x; x++)
		{
			float3 data, s;

			for (uint y = minIndx.y; y <= maxIndx.y; y++)
			{
				for (uint z = minIndx.z; z <= maxIndx.z; z++)
				{
					{
						data = cellF - make_float3(x, y, z);
						s = sign(data);

						// Abs.
						data *= s;
					}
					
					float weight;
					float3 wGrad;
					{
						computeWeightAndGrad(
							s,
							data,
							weight,
							wGrad);
					}

					const GridCell& voxel =
						gridCell[
							GridInfo::GetIndex(
								make_uint3(x, y, z),
								make_uint3(
									gridInfo.width + 1,
									gridInfo.height + 1,
									gridInfo.depth + 1))];

					velGrad =
						velGrad +
						float3x3::outerProduct(
							voxel.velocity,
							wGrad);

					pic += voxel.velocity * weight;
					flip += voxel.deltaV * weight;
				}
			}
		}
	}
	
	particle.velocity =
		(1.f - ALPHA) * pic +
		ALPHA * (particle.velocity + flip);
}

__device__
void ComputeDeformGrad(
	SnowParticle& particle,
	float3x3 velGrad,
	float deltaT)
{
	particle.elasticity =
		(deltaT * velGrad + float3x3()) *
		particle.elasticity;
	
	const Mat& material =
		particle.material;

	float3x3 u, s, v;
	particle.elasticity.svdDecomp(
		u, s, v);

	float3x3 sClamp;
	{
		sClamp.d[0] = clamp(s.d[0],
			material.compressionRatio,
			material.stretchRatio);
		sClamp.d[4] = clamp(s.d[4],
			material.compressionRatio,
			material.stretchRatio);
		sClamp.d[8] = clamp(s.d[8],
			material.compressionRatio,
			material.stretchRatio);
	}

	float3x3 sClampInv;
	{
		sClampInv.d[0] = 1.f / sClamp.d[0];
		sClampInv.d[4] = 1.f / sClamp.d[4];
		sClampInv.d[8] = 1.f / sClamp.d[8];
	}

	particle.plasticity =
		v.multABCt(sClampInv, u);
	particle.elasticity =
		u.multABCt(sClamp, v);
}

__global__
void UpdateParticles(
	SnowParticle* particles,
	SnowParticleExternalData* externalData,
	GridCell* gridCell,
	GridInfo gridInfo,
	float deltaT,
	uint numParticles)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	// Bounds check.
	if (numParticles <= id)
	{
		return;
	}

	SnowParticle& particle =
		particles[id];

	float3x3 velGrad(0.f);

	velGrad =
		ComputeVelGrad(
			particle,
			gridCell,
			gridInfo);

	ComputeDeformGrad(
		particle,
		velGrad,
		deltaT);
	
	// TODO: collisions

	particle.position +=
		deltaT * particle.velocity;
}

void Simulation::StepSim(
	float deltaT)
{
	// Clear data.
	{
		cudaError(
			cudaMemset(
				grid->Data(),
				0,
				(grid->GetWidth() + 1) *
				(grid->GetHeight() + 1) *
				(grid->GetDepth() + 1) *
				sizeof(GridCell)));
	}

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
	/*
	{
		SnowParticleExternalData* externalDataCPU =
			new SnowParticleExternalData[numParticles];

		cudaError(cudaMemcpy(
			externalDataCPU,
			externalData,
			numParticles * sizeof(SnowParticleExternalData),
			cudaMemcpyDeviceToHost));

		for (uint i = 0; i < numParticles; i++)
		{
			std::cout << i << std::endl;
			externalDataCPU[i].sigma.print();
		}
	}
	*/
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
	/*
	{
		uint numNodes =
			(grid->GetWidth() + 1) *
			(grid->GetHeight() + 1) *
			(grid->GetDepth() + 1);

		GridCell* nodeCPU =
			new GridCell[numNodes];

		cudaError(cudaMemcpy(
			nodeCPU,
			grid->Data(),
			numNodes * sizeof(GridCell),
			cudaMemcpyDeviceToHost));

		for (uint i = 0; i < numNodes; i++)
		{
			if (nodeCPU[i].mass != 0)
			{
				std::cout << i << " " <<
					nodeCPU[i].velocity.x << " " <<
					nodeCPU[i].velocity.y << " " <<
					nodeCPU[i].velocity.z << " ";
				std::cout << std::endl;
			}
		}
	}
	*/
	{
		const dim3 threads(
			NUM_THREADS,
			1,
			1);
		const dim3 blocks(
			((grid->GetWidth() + 1) *
				(grid->GetHeight() + 1) *
				(grid->GetDepth() + 1)
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
	/*
	{
		uint numNodes =
			(grid->GetWidth() + 1) *
			(grid->GetHeight() + 1) *
			(grid->GetDepth() + 1);

		GridCell* nodeCPU =
			new GridCell[numNodes];

		cudaError(cudaMemcpy(
			nodeCPU,
			grid->Data(),
			numNodes * sizeof(GridCell),
			cudaMemcpyDeviceToHost));
		
		int m = 0;

		for (uint i = 0; i < numNodes; i++)
		{
			if (nodeCPU[i].mass != 0)
			{
				m++;
				std::cout << i << " " <<
					nodeCPU[i].mass << " " <<
					nodeCPU[i].velocity.x << " " <<
					nodeCPU[i].velocity.y << " " <<
					nodeCPU[i].velocity.z << " ";
				std::cout << std::endl;
			}
		}

		std::cout << m << std::endl;
	}
	*/
	// TODO: implicit.
	{

	}

	{
		const dim3 threads(
			NUM_THREADS,
			1,
			1);
		const dim3 blocks(
			(numParticles + NUM_THREADS - 1) / NUM_THREADS,
			1,
			1);
		
		UpdateParticles<<<blocks, threads>>>(
			particles,
			externalData,
			grid->Data(),
			*grid->GetGridInfo(),
			deltaT,
			numParticles);
	}
	/*
	{
		SnowParticle* particlesCPU =
			new SnowParticle[numParticles];

		cudaError(cudaMemcpy(
			particlesCPU,
			particles,
			numParticles * sizeof(SnowParticle),
			cudaMemcpyDeviceToHost));

		for (uint i = 0; i < numParticles; i++)
		{
			std::cout << i << " " <<
				particlesCPU[i].position.x << " " <<
				particlesCPU[i].position.y << " " <<
				particlesCPU[i].position.z << " " << std::endl;
		}
	}*/
}

void Simulation::Draw()
{
	{
		SnowParticle* particlesCPU =
			new SnowParticle[numParticles];

		cudaError(cudaMemcpy(
			particlesCPU,
			particles,
			numParticles * sizeof(GridCell),
			cudaMemcpyDeviceToHost));

		glColor3f(0.f, 1.f, 0.f);

		for (uint i = 0; i < numParticles; i++)
		{
			glPushMatrix();
			{
				glTranslatef(
					particlesCPU->position.x,
					particlesCPU->position.y,
					particlesCPU->position.z);

				Cube::Render(.1f);
			}
			glPopMatrix();
		}

		glColor3f(1.f, 1.f, 1.f);
	}
}