#pragma once

#include <stdlib.h>

#define GLEW_STATIC
#include <GL/glew.h>

class Cube
{
public:
	static void Render(float size = 1.f)
	{
		glPushMatrix();
		{
			glScalef(size / 2.f, size / 2.f, size / 2.f);

			glEnableClientState(GL_VERTEX_ARRAY);
			glVertexPointer(3, GL_FLOAT, 0, cubeBuffer);

			glDrawArrays(
				GL_TRIANGLES,
				0,
				_countof(cubeBuffer) / 3);

			glDisableClientState(GL_VERTEX_ARRAY);
		}
		glPopMatrix();
	}

private:
	static const float cubeBuffer[108];
};