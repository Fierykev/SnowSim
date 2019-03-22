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

	if (detX < -epsilon ||
		epsilon < detX)
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
	float v =
		dot(direction,
			cross(rayToTri, edge1)) *
		inverseDeterminant;

	if (u < 0.f ||
		1.f < u + v)
	{
		return false;
	}

	t =
		dot(edge2, tmpVar) *
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
	{
		float3 origin =
			gridInfo.position +
			gridInfo.scale *
			make_float3(
				0.f,
				location.x + .5f,
				location.y + .5f);
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
				int cell = t / gridInfo.scale;
				occupied[lookup + cell] = true;
			}
		}
	}

	// Fill voxels.
	{
		bool inside = false;

		for (unsigned int x = 0; x < gridInfo.width; x++)
		{
			if (occupied[lookup + x])
			{
				inside = !inside;
			}
			else if (inside)
			{
				occupied[lookup + x] = true;
			}
		}
	}
}

__global__
void SnowSamples(
	GridInfo gridInfo,
	const unsigned int numSamples,
	bool* occupied)
{

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

void SnowModel::Voxelize(
	Grid<SnowParticle>* grid,
	DisplayType display)
{
	if (display == MODEL)
	{
		obj.Draw();
	}

	// Copy verts to the GPU.
	Vertex* gpuVerts;
	{
		cudaError(
			cudaMalloc(
				(void**)&gpuVerts,
				obj.numberofVerts() * sizeof(Vertex)));
		cudaError(
			cudaMemcpy(
				gpuVerts,
				(const void*)obj.getVertices(),
				obj.numberofVerts() * sizeof(Vertex),
				cudaMemcpyHostToDevice));
	}

	// Copy indices to the GPU.
	unsigned int* gpuIndices;
	{
		cudaError(
			cudaMalloc(
				(void**)&gpuIndices,
				obj.getNumIndices() * sizeof(unsigned int)));

		unsigned int offset = 0;

		for (unsigned int matIndex = 0; matIndex < obj.getNumMaterials(); matIndex++)
		{
			cudaError(
				cudaMemcpy(
					gpuIndices + offset,
					(const void*)obj.getIndices(matIndex),
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
			numThreads >> 1,
			numThreads >> 1,
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

		if (display == VOXELS)
		{
			RenderVoxels(
				grid,
				occupied);
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
	Grid<SnowParticle>* grid,
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