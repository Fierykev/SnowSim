#pragma once

#include <stdlib.h>

#define GLEW_STATIC
#include <GL/glew.h>

#include "Helper.h"

class Cube
{
public:
	static void Render(float size = 1.f)
	{
		if (!init)
		{
			for (int i = 0; i < _countof(cubeBuffer); i += 9)
			{
				float3 v[3];
				for (int j = 0; j < 3; j++)
				{
					const float* cubeOff =
						&cubeBuffer[i + 3 * j];
					v[j] =
						make_float3(
							cubeOff[0],
							cubeOff[1],
							cubeOff[2]);
				}

				float3 total =
					v[0] - v[1] - v[2];

				float3 normal =
					make_float3(
						total.x != 0.f * sign(v[0].x),
						total.y != 0.f * sign(v[0].x),
						total.z != 0.f * sign(v[0].x));

				int normalWrite = i / 3;
				cubeNormalBuffer[normalWrite] =
					normal.x;
				cubeNormalBuffer[normalWrite + 1] =
					normal.y;
				cubeNormalBuffer[normalWrite + 2] =
					normal.z;
			}
		}

		glPushMatrix();
		{
			glScalef(size / 2.f, size / 2.f, size / 2.f);

			glEnableClientState(GL_VERTEX_ARRAY);
			glVertexPointer(3, GL_FLOAT, 0, cubeBuffer);

			glEnableClientState(GL_NORMAL_ARRAY);
			glNormalPointer(GL_FLOAT, 0, cubeNormalBuffer);

			glDrawArrays(
				GL_TRIANGLES,
				0,
				_countof(cubeBuffer) / 3);

			glDisableClientState(GL_VERTEX_ARRAY);
			glDisableClientState(GL_NORMAL_ARRAY);
		}
		glPopMatrix();
	}

private:
	static const float cubeBuffer[108];
	static bool init;

	static float cubeNormalBuffer[_countof(cubeBuffer) / 3];
};