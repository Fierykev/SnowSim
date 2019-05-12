#pragma once

#include <iostream>
#include <fstream>
#include <string>

#include "Helper.h"

class Serializable
{
public:
	static void Store(
		float3* arr,
		uint length,
		const char* fileName,
		uint frame = 0)
	{
		std::string fileNameC =
			std::to_string(frame);
		fileNameC += fileName;

		std::ofstream file(fileNameC);

		for (uint i = 0; i < length; i++)
		{
			file << arr[i].x <<
				", " << arr[i].y <<
				", " << arr[i].z << std::endl;
		}

		file.close();
	}

	static void Compare(
		float3* arr,
		const char* fileName,
		uint frame = 0)
	{
		std::string fileNameC =
			std::to_string(frame);
		fileNameC += fileName;

		std::ifstream file(fileNameC);
		std::string line;

		uint index = 0;

		while (std::getline(file, line))
		{
			if (line.empty())
				continue;

			float3 val;
			sscanf(
				line.c_str(),
				"%f, %f, %f",
				&val.x,
				&val.y,
				&val.z);

			if (fabs(val.x - arr[index].x) > .001 &&
				fabs(val.y - arr[index].y) > .001 &&
				fabs(val.z - arr[index].z) > .001)
			{
				std::cout << "Index: " <<
					index << std::endl;
				std::cout <<
					val.x << ", " <<
					val.y << ", " <<
					val.z << std::endl;
				std::cout <<
					arr[index].x << ", " <<
					arr[index].y << ", " <<
					arr[index].z << std::endl;

				system("PAUSE");
			}

			index++;
		}

		file.close();
	}
};