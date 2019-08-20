#include <GL/glew.h>
#include "ObjectFileLoader.h" // link to the header
#include "ShaderHelper.h"
#include <climits>

#define EPSILON .001

/***************************************************************************
OBJ Loading
***************************************************************************/

ObjLoader::ObjLoader()
{

}

ObjLoader::~ObjLoader()
{
	// delete all data

	reset();
}

void ObjLoader::reset()
{
	delete[] vertex_final_array;

	for (auto i : material)
	{
		if (i.texID != 0)
		{
			glDeleteTextures(1, &i.texID);
		}
	}
}

void ObjLoader::Base_Mat(Material *mat)
{
	mat->ambient = make_float4(0.2f, 0.2f, 0.2f, 1.f);
	mat->diffuse = make_float4(0.8f, 0.8f, 0.8f, 1.f);
	mat->specular = make_float4(1.0f, 1.0f, 1.0f, 1.f);
	mat->shininess = 0;
	mat->opticalDensity = 0;
	mat->alpha = 1.0f;
	mat->specularb = false;
}

void ObjLoader::Material_File(string filename, string matfile, unsigned long* tex_num)
{
	// find the directory to the material file

	string directory = filename.substr(0, filename.find_last_of('/') + 1);

	matfile = directory + matfile; // the location of the material file to the program

								   // open the file

	ifstream matFile_2(matfile);

	if (matFile_2.is_open()) // If obj file is open, continue
	{
		string line_material;// store each line of the file here

		while (!matFile_2.eof()) // Start reading file data as long as we have not reached the end
		{
			getline(matFile_2, line_material); // Get a line from file
											   // convert to a char to do pointer arithmetics

			char* ptr = (char*)line_material.c_str();

			// remove leading white space
			while (iswspace(ptr[0]))
				ptr++;

			// This program is for standard Wavefront Objects that are triangulated and have normals stored in the file.  This reader has been tested with 3ds Max and Blender.

			if (ptr[0] == 'n' && ptr[1] == 'e' && ptr[2] == 'w' && ptr[3] == 'm'
				&& ptr[4] == 't' && ptr[5] == 'l') // new material
			{
				ptr += 7;// move address up

				Material mat; // allocate memory to create a new material

				Base_Mat(&mat); // init the material

				mat.name = ptr; // set the name of the material

				material.push_back(mat); // add to the vector

				*tex_num = material.size() - 1;
			}
			else if (ptr[0] == 'K' && ptr[1] == 'a') // ambient
			{
				ptr += 2;// move address up

				sscanf(ptr, "%f %f %f ",							// Read floats from the line: v X Y Z
					&material.at(*tex_num).ambient.x,
					&material.at(*tex_num).ambient.y,
					&material.at(*tex_num).ambient.z);

				material.at(*tex_num).ambient.z = 1.f;
			}
			else if (ptr[0] == 'K' && ptr[1] == 'd') // diffuse
			{
				ptr += 2;// move address up

				sscanf(ptr, "%f %f %f ",							// Read floats from the line: v X Y Z
					&material.at(*tex_num).diffuse.x,
					&material.at(*tex_num).diffuse.y,
					&material.at(*tex_num).diffuse.z);

				material.at(*tex_num).diffuse.z = 1.f;
			}
			else if (ptr[0] == 'K' && ptr[1] == 's') // specular
			{
				ptr += 2;// move address up

				sscanf(ptr, "%f %f %f ",							// Read floats from the line: v X Y Z
					&material.at(*tex_num).specular.x,
					&material.at(*tex_num).specular.y,
					&material.at(*tex_num).specular.z);

				material.at(*tex_num).specular.z = 1.f;
			}
			else if (ptr[0] == 'N' && ptr[1] == 's') // shininess
			{
				ptr += 2;// move address up

				sscanf(ptr, "%f ",							// Read floats from the line: v X Y Z
					&material.at(*tex_num).shininess);
			}
			else if (ptr[0] == 'N' && ptr[1] == 'i') // refraction
			{
				ptr += 2;// move address up

				sscanf(ptr, "%f ",							// Read floats from the line: v X Y Z
					&material.at(*tex_num).opticalDensity);
			}
			else if (ptr[0] == 'd') // transparency
			{
				ptr++;// move address up

				sscanf(ptr, "%f ",							// Read floats from the line: v X Y Z
					&material.at(*tex_num).alpha);
			}
			else if (ptr[0] == 'T' && ptr[1] == 'r') // another way to store transparency
			{
				ptr += 2;// move address up

				sscanf(ptr, "%f ",							// Read floats from the line: v X Y Z
					&material.at(*tex_num).alpha);
			}
			else if (ptr[0] == 'm' && ptr[1] == 'a' && ptr[2] == 'p' && ptr[3] == '_'
				&& ptr[4] == 'K' && ptr[5] == 'd') // image texture
			{
				ptr += 7;// move address up

				material.at(*tex_num).texture_path = ptr; // the material file
														  // load the file
														  // convert to a LPWSTR

				string filename;
				filename = directory + material.at(*tex_num).texture_path;

				// load image file
				if (!material.at(*tex_num).texture.load(filename.c_str()))
					cout << "Error (OBJECT LOADER): Cannot load image " << filename << endl;

				if (!material.at(*tex_num).texture.convert4f())
					cout << "Error (OBJECT LOADER): Cannot convert to float4 image " << filename << endl;

				// find avg color for radiance
				material.at(*tex_num).radiance = make_float4(0.f, 0.f, 0.f, 0.f);

				size_t numPixels = material.at(*tex_num).texture.getWidth()
					* material.at(*tex_num).texture.getHeight();

				for (size_t i = 0; i < numPixels; i++)
				{
					material.at(*tex_num).radiance +=
						material.at(*tex_num).texture.getData4f()[i];
				}

				material.at(*tex_num).radiance /= numPixels;

				// setup texture
				material.at(*tex_num).texID = material.at(*tex_num).texture.genGlImage();
			}
		}

		matFile_2.close(); // close the file
	}
	else
	{
		cout << "Error (OBJECT LOADER): Cannot Find Material File- " << matfile << endl;
	}
}

void ObjLoader::Load_Geometry(const char *filename)
{
	// delete past memory

	if (vertex_final_array != nullptr)
		delete vertex_final_array;

	// allocate memory to the vectors on the heap

	vx_array_i.clear();

	vx_array.clear();

	material.clear();

	unordered_map <float3, vector<VertexDataforMap>> vertexmap; // map for removing doubles

	mesh_num = 0;

	minBB = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
	maxBB = make_float3(FLT_MIN, FLT_MIN, FLT_MIN);

	// create maps to store the lighting values for the material

	ifstream objFile(filename); // open the object file

	if (objFile.is_open()) // If the obj file is open, continue
	{
		// initialize the strings needed to read the file

		string line;

		string mat;

		// the material that is used

		unsigned long material_num = 0;

		unsigned long tex_num = 0;

		numVerts = 0;

		// Store the coordinates

		vector <float> vn_array;

		vector <float> vt_array;

		while (!objFile.eof()) // start reading file data
		{
			getline(objFile, line);	// get line from file

									// convert to a char to do pointers

			const char* ptr = line.c_str();

			if (ptr[0] == 'm' && ptr[1] == 't' && ptr[2] == 'l' && ptr[3] == 'l'  && ptr[4] == 'i' && ptr[5] == 'b' && ptr[6] == ' ') // load the material file
			{
				ptr += 7; // move the address up

				const string material_file = ptr;// the material file

				Material_File(filename, material_file, &tex_num); // read the material file and update the number of materials
			}
			if (ptr[0] == 'v' && ptr[1] == ' ') // the first character is a v: on this line is a vertex stored.
			{
				ptr += 2; // move address up

						  // store the three tmp's into the verticies

				float tmp[3];

				sscanf(ptr, "%f %f %f ", // read floats from the line: X Y Z
					&tmp[0],
					&tmp[1],
					&tmp[2]);

				vx_array.push_back(tmp[0]);
				vx_array.push_back(tmp[1]);
				vx_array.push_back(tmp[2]);
			}

			else if (ptr[0] == 'v' && ptr[1] == 'n') // the vertex normal
			{
				ptr += 2;

				// store the three tmp's into the verticies

				float tmp[3];

				sscanf(ptr, "%f %f %f ", // read floats from the line: X Y Z
					&tmp[0],
					&tmp[1],
					&tmp[2]);

				vn_array.push_back(tmp[0]);
				vn_array.push_back(tmp[1]);
				vn_array.push_back(tmp[2]);
			}

			else if (ptr[0] == 'v' && ptr[1] == 't') // texture coordinate for a vertex
			{
				ptr += 2;

				// store the two tmp's into the verticies

				float tmp[2];

				sscanf(ptr, "%f %f ",	// read floats from the line: X Y Z
					&tmp[0],
					&tmp[1]);

				vt_array.push_back(tmp[0]);
				vt_array.push_back(tmp[1]);
			}
			else if (ptr[0] == 'u' && ptr[1] == 's' && ptr[2] == 'e' && ptr[3] == 'm' && ptr[4] == 't' && ptr[5] == 'l') // which material is being used
			{
				mat = line.substr(6 + 1, line.length());// save so the comparison will work

														// add new to the material name so that it matches the names of the materials in the mtl file

				for (unsigned long num = 0; num < tex_num + 1; num++)// find the material
				{
					if (mat == material.at(num).name)// matches material in mtl file
					{
						material_num = num;
					}
				}
			}
			else if (ptr[0] == 'f') // store the faces in the object
			{
				ptr++;

				int vertexNumber[3] = { 0, 0, 0 };
				int normalNumber[3] = { 0, 0, 0 };
				int textureNumber[3] = { 0, 0, 0 };

				// no texture
				if (string(ptr).find("//") != -1)
				{
					sscanf(ptr, "%i//%i %i//%i %i//%i ",
						&vertexNumber[0],
						&normalNumber[0],
						&vertexNumber[1],
						&normalNumber[1],
						&vertexNumber[2],
						&normalNumber[2]
					);

					textureNumber[0] = INT_MAX;
					textureNumber[1] = INT_MAX;
					textureNumber[2] = INT_MAX;
				}
				else
				{
					sscanf(ptr, "%i/%i/%i %i/%i/%i %i/%i/%i ",
						&vertexNumber[0],
						&textureNumber[0],
						&normalNumber[0],
						&vertexNumber[1],
						&textureNumber[1],
						&normalNumber[1],
						&vertexNumber[2],
						&textureNumber[2],
						&normalNumber[2]
					); // each point represents an X,Y,Z.
				}

				// create a vertex for this area

				for (int i = 0; i < 3; i++) // loop for each triangle
				{
					Vertex vert;

					vert.position = make_float3(vx_array.at((vertexNumber[i] - 1) * 3), vx_array.at((vertexNumber[i] - 1) * 3 + 1), vx_array.at((vertexNumber[i] - 1) * 3 + 2));

					vert.normal = make_float3(vn_array[(normalNumber[i] - 1) * 3], vn_array[(normalNumber[i] - 1) * 3 + 1], vn_array[(normalNumber[i] - 1) * 3 + 2]);

					if (textureNumber[i] != INT_MAX)
						vert.texcoord = make_float2(vt_array[(textureNumber[i] - 1) * 2], vt_array[(textureNumber[i] - 1) * 2 + 1]);
					else
						vert.texcoord = make_float2(0.f, 0.f);

					unsigned int index = 0;

					bool indexupdate = false;

					if (vertexmap.find(vert.position) != vertexmap.end())
						for (VertexDataforMap vdm : vertexmap[vert.position])
						{
							if (vert.normal == vdm.normal && vert.texcoord == vdm.texcoord) // found the index
							{
								index = vdm.index;

								indexupdate = true;
								break;
							}
						}

					// nothing found

					if (!indexupdate)
					{
						VertexDataforMap tmp;

						index = numVerts;
						tmp.normal = vert.normal;

						tmp.texcoord = vert.texcoord;

						tmp.index = index;

						minBB = {
							min(minBB.x, vert.position.x),
							min(minBB.y, vert.position.y),
							min(minBB.z, vert.position.z)
						};

						maxBB = {
							max(maxBB.x, vert.position.x),
							max(maxBB.y, vert.position.y),
							max(maxBB.z, vert.position.z)
						};

						vertexmap[vert.position].push_back(tmp);

						numVerts++;
					}

					vx_array_i[material_num].push_back(index);
				}
			}
		}

		// create the final verts

		vertex_final_array = new Vertex[numVerts];

		for (unordered_map<float3, vector<VertexDataforMap>>::iterator i = vertexmap.begin(); i != vertexmap.end(); i++)
		{
			for (VertexDataforMap vdm : i->second)
			{
				vertex_final_array[vdm.index].position = i->first;

				vertex_final_array[vdm.index].normal = vdm.normal;

				vertex_final_array[vdm.index].texcoord = vdm.texcoord;

				vertex_final_array[vdm.index].tangent = { 0.f, 0.f, 0.f };
			}
		}

		// compute tangent
		float3* sTmp = new float3[numVerts];
		float3* tTmp = new float3[numVerts];

		for (size_t mat = 0; mat < vx_array_i.size(); mat++)
		{
			for (size_t index = 0; index < vx_array_i[mat].size(); index += 3)
			{
				Vertex* verts[3] = {
					&vertex_final_array[vx_array_i[mat][index]],
					&vertex_final_array[vx_array_i[mat][index + 1]],
					&vertex_final_array[vx_array_i[mat][index + 2]]
				};

				float3 delta0 = verts[1]->position - verts[0]->position;
				float3 delta1 = verts[2]->position - verts[0]->position;

				float2 deltaUV0 = verts[1]->texcoord - verts[0]->texcoord;
				float2 deltaUV1 = verts[2]->texcoord - verts[0]->texcoord;

				float scale = 1.f /
					(deltaUV0.x * deltaUV1.y - deltaUV0.y * deltaUV1.x);

				float3 dirS = {
					deltaUV1.y * delta0.x + deltaUV0.y * delta1.x,
					deltaUV1.y * delta0.y + deltaUV0.y * delta1.y,
					deltaUV1.y * delta0.z + deltaUV0.y * delta1.z
				};
				dirS *= scale;

				sTmp[vx_array_i[mat][index]] += dirS;
				sTmp[vx_array_i[mat][index + 1]] += dirS;
				sTmp[vx_array_i[mat][index + 2]] += dirS;

				float3 dirT = {
					deltaUV0.x * delta1.x - deltaUV1.x * delta0.x,
					deltaUV0.x * delta1.y - deltaUV1.x * delta0.y,
					deltaUV0.x * delta1.z - deltaUV1.x * delta0.z
				};
				dirT *= scale;

				tTmp[vx_array_i[mat][index]] += dirT;
				tTmp[vx_array_i[mat][index + 1]] += dirT;
				tTmp[vx_array_i[mat][index + 2]] += dirT;
			}
		}

		// run tangent calc
		for (size_t i = 0; i < numVerts; i++)
		{
			vertex_final_array[i].tangent =
				sTmp[i] - vertex_final_array[i].normal * (
					vertex_final_array[i].normal * sTmp[i]);
			
			vertex_final_array[i].tangent =
				normalize(vertex_final_array[i].tangent);

			// hardness not needed
		}

		delete[] sTmp, tTmp;

		// associate color with each vertex (for PRT only)
		for (int i = 0; i < getNumMaterials(); i++)
		{
			for (int j = 0; j < vx_array_i[i].size(); j++)
			{
				if (material[i].texture.empty())
					vertexToColor[vertex_final_array[vx_array_i[i][j]].position] =
						material[i].diffuse;
				else
					vertexToColor[vertex_final_array[vx_array_i[i][j]].position] =
						material[i].radiance;
			}
		}

	}
	else
	{
		printf("Error (OBJECT LOADER):  Cannot Find Object File- %s\n", filename);
	}
}

void ObjLoader::Load(const char *filename)
{
	Load_Geometry(filename);
}

void ObjLoader::Draw()
{
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);
	glEnable(GL_TEXTURE_2D);
	glEnable(GL_AMBIENT);
	glEnable(GL_DIFFUSE);
	glEnable(GL_SPECULAR);
	glEnable(GL_SHININESS);

	// setup normals, verts, and texcroods
	glVertexPointer(3, GL_FLOAT, sizeof(Vertex), &vertex_final_array[0].position);
	glNormalPointer(GL_FLOAT, sizeof(Vertex), &vertex_final_array[0].normal);
	glTexCoordPointer(2, GL_FLOAT, sizeof(Vertex), &vertex_final_array[0].texcoord);

	for (int i = 0; i < getNumMaterials(); i++)
	{
		// set texture
		glBindTexture(GL_TEXTURE_2D, material[i].texID);

		// setup mat
		glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, &material[i].ambient.x);

		if (material[i].texID == 0)
			glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, &material[i].diffuse.x);
		else
		{
			float4 white = { 1, 1, 1, 1 };
			glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, (GLfloat*)&white);
		}
		glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (GLfloat*)&material[i].specular);
		glMaterialf(GL_FRONT, GL_SHININESS, material[i].shininess);

		glDrawElements(GL_TRIANGLES, vx_array_i[i].size(), GL_UNSIGNED_INT, vx_array_i[i].data());
	}

	glDisable(GL_SHININESS);
	glDisable(GL_SPECULAR);
	glDisable(GL_DIFFUSE);
	glDisable(GL_AMBIENT);
	glDisable(GL_TEXTURE_2D);
	glDisableClientState(GL_TEXTURE_COORD_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
}