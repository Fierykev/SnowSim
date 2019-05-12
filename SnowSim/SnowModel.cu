#include <curand.h>
#include <curand_kernel.h>

#include "SnowModel.cuh"
#include "Cube.h"

__constant__ float epsilon =.001f;

__device__
bool RayTriangleIntersection(
	float3 origin,
	float3 direction,
	float3 a,
	float3 b,
	float3 c,
	float& t)
{
	// Edge deltas.
	const float3 edge1 = b - a;
	const float3 edge2 = c - a;

	float3 tmpVar =
		cross(direction, edge2);
	const double detX =
		dot(edge1, tmpVar);

	if (-epsilon < detX &&
		detX < epsilon)
	{
		return false;
	}
	
	const double inverseDeterminant =
		1.f / detX;

	float3 rayToTri =
		origin - a;
	float u =
		dot(rayToTri, tmpVar) *
		inverseDeterminant;

	if (u < 0.f ||
		1.f < u)
	{
		return false;
	}

	float3 q =
		cross(rayToTri, edge1);
	float v =
		dot(direction, q) *
		inverseDeterminant;

	if (v < 0.f ||
		1.f < u + v)
	{
		return false;
	}

	t =
		dot(edge2, q) *
		inverseDeterminant;

	if (epsilon < t)
	{
		return true;
	}

	return false;
}

__global__
void ModelToVoxels(
	const Vertex* verts,
	const unsigned int* indices,
	unsigned int numIndices,
	GridInfo gridInfo,
	bool* occupied)
{
	int2 location =
		make_int2(
			threadIdx.x + blockIdx.x * blockDim.x,
			threadIdx.y + blockIdx.y * blockDim.y);

	// Out of bounds check.
	if (gridInfo.height <= location.x ||
		gridInfo.depth <= location.y)
	{
		return;
	}

	unsigned int lookup =
		location.x * gridInfo.width +
		location.y * gridInfo.width * gridInfo.height;

	// Voxelization approx by shooting ray.
	unsigned int xEndLoc =
		lookup +
		gridInfo.width;
	unsigned int xLoc =
		xEndLoc;
	{
		float3 origin =
			gridInfo.position +
			gridInfo.scale *
			make_float3(
				0.f,
				float(location.x) + .5f,
				float(location.y) + .5f);
		float3 direction =
			make_float3(1.f, .0f, .0f);

		for (unsigned int index = 0;
			index < numIndices;
			index += 3)
		{
			const float3 v0 = verts[indices[index]].position;
			const float3 v1 = verts[indices[index + 1]].position;
			const float3 v2 = verts[indices[index + 2]].position;

			float t;
			
			if (RayTriangleIntersection(
				origin,
				direction,
				v0,
				v1,
				v2,
				t))
			{
				unsigned int cell = t / gridInfo.scale;
				unsigned int location = lookup + cell;
				occupied[location] = true;

				xLoc = min(xLoc, location);
			}
		}
	}

	// Fill voxels.
	{
		unsigned int xExtentLoc;

		for (; xLoc < xEndLoc; xLoc++)
		{
			// Skip filled in areas.
			while (
				occupied[xLoc] &&
				xLoc < xEndLoc)
			{
				xLoc++;
			}

			// Scan unfilled areas.
			xExtentLoc =
				xLoc;
			while (
				!occupied[xExtentLoc] &&
				xExtentLoc < xEndLoc)
			{
				xExtentLoc++;
			}

			// Stop.
			if (xExtentLoc == xEndLoc)
			{
				break;
			}

			while (
				xLoc < xExtentLoc)
			{
				occupied[xLoc] =
					true;
				xLoc++;
			}
		}
	}
}

__global__
void SnowSamples(
	GridInfo gridInfo,
	SnowParticle* particle,
	float mass,
	const unsigned int numSamples,
	const bool* occupied,
	curandState* state,
	const uint seed)
{
	int sampleNumber =
		threadIdx.x + blockIdx.x * blockDim.x;

	if (numSamples <= sampleNumber)
	{
		return;
	}

	curandState& cState = state[sampleNumber];
	curand_init(seed, sampleNumber, 0, &cState);

	uint voxelID;

	uint dim =
		gridInfo.width *
		gridInfo.height *
		gridInfo.depth;

	do
	{
		voxelID = curand(&cState) % dim;
	} while (!occupied[voxelID]);

	// Sample inside the voxel.
	float3 position =
		gridInfo.GetGrid(voxelID);

	{
		position +=
			make_float3(
				curand_uniform(&cState),
				curand_uniform(&cState),
				curand_uniform(&cState)) * gridInfo.scale;
	}

	SnowParticle tmpParticle;
	{
		tmpParticle.position =
			position;
		tmpParticle.mass =
			mass;
		tmpParticle.velocity =
			make_float3(0.f, 0.f, 10.f);
	}

	particle[sampleNumber] =
		tmpParticle;
}

SnowModel::SnowModel()
{

}

SnowModel::SnowModel(
	const char* filename)
{
	Load(filename);
}

void SnowModel::Load(const char* filename)
{
	//obj.reset(); TODO:
	obj.Load(filename);
}

void SnowModel::SampleParticles(
	Grid<GridCell>* grid,
	SnowParticle* particle,
	float density,
	uint numParticles,
	short display)
{
	if (display & MODEL)
	{
		glColor3f(1, 0, 0);
		obj.Draw();
		glColor3f(1, 1, 1);
	}
	
	// Copy verts to the GPU.
	Vertex* gpuVerts;
	{
		cudaError(
			cudaMalloc(
				&gpuVerts,
				obj.numberofVerts() * sizeof(Vertex)));
		cudaError(
			cudaMemcpy(
				gpuVerts,
				obj.getVertices(),
				obj.numberofVerts() * sizeof(Vertex),
				cudaMemcpyHostToDevice));
	}

	// Copy indices to the GPU.
	unsigned int* gpuIndices;
	{
		cudaError(
			cudaMalloc(
				&gpuIndices,
				obj.getNumIndices() * sizeof(unsigned int)));

		unsigned int offset = 0;

		for (unsigned int matIndex = 0; matIndex < obj.getNumMaterials(); matIndex++)
		{
			cudaError(
				cudaMemcpy(
					gpuIndices + offset,
					obj.getIndices(matIndex),
					obj.getNumIndices(matIndex) * sizeof(unsigned int),
					cudaMemcpyHostToDevice));

			offset += obj.getNumIndices(matIndex);
		}
	}

	// Create occupied slots.
	bool* occupied;
	{
		auto size =
			grid->GetWidth() *
			grid->GetHeight() *
			grid->GetDepth();
		
		cudaError(
			cudaMalloc(
				(void**)&occupied,
				size * sizeof(bool)));

		cudaError(
			cudaMemset(
				occupied,
				0,
				size * sizeof(bool)));
	}

	// Find occupied voxels.
	{
		const dim3 threads(
			numThreads,
			numThreads,
			1);
		const dim3 blocks(
			(grid->GetHeight() + threads.x - 1) / threads.x,
			(grid->GetDepth() + threads.y - 1) / threads.y,
			1);
		
		ModelToVoxels<<<blocks, threads>>>(
			gpuVerts,
			gpuIndices,
			obj.getNumIndices(),
			*grid->GetGridInfo(),
			occupied);

		if (display & VOXELS)
		{
			RenderVoxels(
				grid,
				occupied);
		}

	}

	// Get mass.
	float mass;
	{
		// TODO: move to GPU.
		bool* occupiedCPU;
		{
			unsigned int size =
				grid->GetWidth() * grid->GetHeight() * grid->GetDepth();
			occupiedCPU = new bool[size];

			cudaError(
				cudaMemcpy(
					occupiedCPU,
					occupied,
					size * sizeof(bool),
					cudaMemcpyDeviceToHost));
		}

		uint count = 0;
		{
			for (int i = 0;
				i < grid->GetWidth() *
				grid->GetHeight() *
				grid->GetDepth();
				i++)
			{
				if (occupiedCPU[i])
				{
					count++;
				}
			}
		}

		float volume = count *
			grid->GetScale() *
			grid->GetScale() *
			grid->GetScale();
		mass = density * volume / (float)numParticles;

		delete[] occupiedCPU;
	}

	// Create particles.
	{
		const dim3 threads(
			numThreads,
			1,
			1);
		const dim3 blocks(
			(numParticles + threads.x - 1) / threads.x,
			1,
			1);

		curandState* state;
		{
			cudaError(
				cudaMalloc(
					&state,
					numParticles * sizeof(curandState)));
		}

		SnowSamples<<<blocks, threads>>>(
			*grid->GetGridInfo(),
			particle,
			mass,
			numParticles,
			occupied,
			state,
			0);// time(NULL));

		{
			cudaError(
				cudaFree(state));
		}

		if (display & PARTICLES)
		{
			RenderParticles(
				particle,
				numParticles);
		}
	}

	// Free data.
	{
		cudaError(
			cudaFree(gpuVerts));
		cudaError(
			cudaFree(gpuIndices));
		cudaError(
			cudaFree(occupied));
	}
}

void SnowModel::RenderVoxels(
	Grid<GridCell>* grid,
	bool* occupied)
{
	bool* occupiedCPU;
	{
		unsigned int size =
			grid->GetWidth() * grid->GetHeight() * grid->GetDepth();
		occupiedCPU = new bool[size];

		cudaError(
			cudaMemcpy(
				occupiedCPU,
				occupied,
				size * sizeof(bool),
				cudaMemcpyDeviceToHost));
	}

	{
		glColor3f(1.f, 0.f, 0.f);

		glPushMatrix();
		{
			glTranslatef(
				grid->GetPosition().x,
				grid->GetPosition().y,
				grid->GetPosition().z);

			for (int z = 0; z < grid->GetDepth(); z++)
			{
				for (int y = 0; y < grid->GetHeight(); y++)
				{
					for (int x = 0; x < grid->GetWidth(); x++)
					{
						if (occupiedCPU[
								x +
								y * grid->GetWidth() +
								z * grid->GetWidth() * grid->GetHeight()])
						{
							glPushMatrix();
							{
								glTranslatef(
									float(x) * grid->GetScale(),
									float(y) * grid->GetScale(),
									float(z) * grid->GetScale());

								Cube::Render(grid->GetScale());
							}
							glPopMatrix();
						}
					}
				}
			}
		}
		glPopMatrix();

		glColor3f(1.f, 1.f, 1.f);
	}

	delete[] occupiedCPU;
}

void SnowModel::RenderParticles(
	SnowParticle* particle,
	uint numParticles)
{
	SnowParticle* particleCPU;
	{
		unsigned int size =
			numParticles;
		particleCPU = new SnowParticle[size];

		cudaError(
			cudaMemcpy(
				particleCPU,
				particle,
				size * sizeof(SnowParticle),
				cudaMemcpyDeviceToHost));
	}

	{
		glColor3f(0.f, 0.f, 1.f);

		for (uint i = 0; i < numParticles; i++)
		{
			glPushMatrix();
			{
				auto& pos =
					particleCPU[i].position;

				glTranslatef(
					pos.x,
					pos.y,
					pos.z);

				Cube::Render(.1f);
			}
			glPopMatrix();
		}

		glColor3f(1.f, 1.f, 1.f);
	}

	delete[] particleCPU;
}