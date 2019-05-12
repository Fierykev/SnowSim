#pragma once

#include <stdlib.h>

#define GLEW_STATIC
#include <GL/glew.h>

class Sphere
{
public:
	static void Render(float radius = 1.f)
	{
		glPushMatrix();
		{
			GLUquadric* quad =
				gluNewQuadric();
			gluSphere(
				quad,
				radius,
				20,
				20);

			gluQuadricDrawStyle(
				quad,
				GLU_FILL);

			gluDeleteQuadric(quad);
		}
		glPopMatrix();
	}

private:
	static const float cubeBuffer[108];
};