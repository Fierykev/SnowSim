#pragma once

#include <cuda_runtime.h>
#include <GL/glew.h>

// loading files

#include <fstream>

// get line in file

#include <sstream>

// Need these headers to support the array types I want

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <string>
#include <stdint.h>
#include <algorithm>

#include <iostream>
#include <fstream>

#include "Helper.h"
#include "Image.h"

using namespace std;

namespace std
{
	template <>
	struct equal_to<float3> : public unary_function<float3, bool>
	{
		bool operator() (const float3 &a, const float3 &b) const
		{
			return a.x == b.x && a.y == b.y && a.z == b.z;
		}
	};

	template<>
	struct hash<float3> : public unary_function<float3, size_t>
	{
		std::size_t operator() (const float3& a) const
		{
			return std::hash<float>{}(a.x) ^ std::hash<float>{}(a.y) ^ std::hash<float>{}(a.z);
		}
	};
};

struct Material
{
	std::string name;

	float4 ambient;

	float4 diffuse;

	float4 specular;

	float4 radiance;

	float shininess;

	float opticalDensity;

	float alpha;

	bool specularb;

	std::string texture_path;

	GLuint texID = 0;

	Image texture;

	bool usesTex()
	{
		return !texture.empty();
	}
};

struct MaterialUpload
{
	float4 ambient;

	float4 diffuse;

	float4 specular;

	float shininess;

	float opticalDensity;

	float alpha;

	bool specularb;

	GLuint texID;
};

struct Vertex
{
	float3 position;
	float3 normal;
	float2 texcoord;
	float3 tangent;
};

struct VertexDataforMap
{
	float3 normal;
	float2 texcoord;
	unsigned int index;
};

class ObjLoader
{
public:

	ObjLoader();

	~ObjLoader(); // destruction method

	void Load(const char *filename); // Load the object with its materials
									 // get the number of materials used

	void reset();

	void Draw(); // regular draw

	const size_t getMatNum()
	{
		return material.size();
	}

	// get the material pointer

	Material* getMaterials()
	{
		return &material.at(0);
	}

	// get the number of vertices in the object

	const unsigned int getNumVertices()
	{
		return numVerts;
	}

	// get the number of indices in a material

	const unsigned int getNumIndices(size_t mat_num)
	{
		return vx_array_i[mat_num].size();
	}

	// get the number of indices in the object

	const size_t getNumIndices()
	{
		size_t indices = 0;

		for (size_t i = 0; i < vx_array_i.size(); i++)
			indices += vx_array_i[i].size();

		return indices;
	}

	// get the number of material indices in the object

	const size_t getNumMaterialIndices(size_t mat_num)
	{
		return vx_array_i[mat_num].size();
	}

	// get the number of materials in the object

	const size_t getNumMaterials()
	{
		return material.size();
	}

	// get a pointer to the verticies

	const Vertex* getVertices()
	{
		return vertex_final_array;
	}

	// get the vertex stride

	unsigned int getVertexStride()
	{
		return sizeof(Vertex);
	}

	// get a pointer to the indices

	const unsigned int* getIndices(size_t mat_num)
	{
		return vx_array_i[mat_num].data();
	}

	// get the number of meshes used to draw the object

	const unsigned int getNumMesh()
	{
		return mesh_num;
	}

	// get a pointer to a certain material

	Material* getMaterial(unsigned int mat_num)
	{
		return &material.at(mat_num);
	}

	// setup Geometry accessors
	float3 getBoundingBoxMin()
	{
		return minBB;
	}

	float3 getBoundingBoxMax()
	{
		return maxBB;
	}

	int2 findByIndex(size_t index)
	{
		index *= 3;

		// find index
		size_t mat;
		for (mat = 0; mat < vx_array_i.size(); mat++)
		{
			if (vx_array_i[mat].size() <= index)
				index -= vx_array_i[mat].size();
			else
				break;
		}

		return make_int2(int(index), int(mat));
	}

	float3 getMinBB(size_t index)
	{
		int2 indexData = findByIndex(index);
		index = indexData.x;
		int mat = indexData.y;

		size_t vIndex[3] =
		{
			vx_array_i[mat][index],
			vx_array_i[mat][index + 1],
			vx_array_i[mat][index + 2]
		};

		float3 bb = { FLT_MAX, FLT_MAX, FLT_MAX };
		for (size_t i = 0; i < 3; i++)
		{
			bb = {
				min(bb.x, vertex_final_array[vIndex[i]].position.x),
				min(bb.y, vertex_final_array[vIndex[i]].position.y),
				min(bb.z, vertex_final_array[vIndex[i]].position.z)
			};
		}

		return bb;
	}

	float3 getMaxBB(size_t index)
	{
		int2 indexData = findByIndex(index);
		index = indexData.x;
		int mat = indexData.y;

		size_t vIndex[3] =
		{
			vx_array_i[mat][index],
			vx_array_i[mat][index + 1],
			vx_array_i[mat][index + 2]
		};

		float3 bb = { FLT_MIN, FLT_MIN, FLT_MIN };
		for (size_t i = 0; i < 3; i++)
		{
			bb = {
				max(bb.x, vertex_final_array[vIndex[i]].position.x),
				max(bb.y, vertex_final_array[vIndex[i]].position.y),
				max(bb.z, vertex_final_array[vIndex[i]].position.z)
			};
		}

		return bb;
	}

	float3 getCentroid(size_t index)
	{
		return (getBoundingBoxMin() + getBoundingBoxMax()) / 2.f;
	}

	float3 getNormal(size_t vertID)
	{
		return vertex_final_array[vertID].normal;
	}

	float3 getPosition(size_t vertID)
	{
		return vertex_final_array[vertID].position;
	}

	float4 getColor(size_t vertID)
	{
		return vertexToColor[vertex_final_array[vertID].position];
	}

	size_t numberOfObjects()
	{
		return getNumIndices() / 3.f;
	}

	size_t numberofVerts()
	{
		return getNumVertices();
	}

private:

	// Create a vector to store the verticies

	void Load_Geometry(const char *filename); // load the verticies and indices

	void Material_File(string filename, string matfile, unsigned long* tex_num); // load the material file

	void Base_Mat(Material *mat); // the basic material

	unordered_map <unsigned int, vector<unsigned int>> vx_array_i; // store the indecies for the vertex

	unordered_map  <float3, float4> vertexToColor; // associate color with vertex for PRT

	vector <float> vx_array; // store the verticies in the mesh

	Vertex* vertex_final_array = nullptr; // the final verticies organized for Direct3D to draw

	vector <Material> material; // the materials used on the object

	unsigned int numVerts;

	unsigned int mesh_num; // the number of meshes

	float3 minBB, maxBB;
};