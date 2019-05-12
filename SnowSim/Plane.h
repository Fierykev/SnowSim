#pragma once

#include <stdlib.h>

#define GLEW_STATIC
#include <GL/glew.h>

#include "Helper.h"

class Plane
{
public:
	static void Render(
		float3 pos,
		float3 normal,
		float size = 1.f,
		float3 up = make_float3(0.f, 1.f, 0.f))
	{
		size /= 2.f;

		glPushMatrix();
		{
			float3 forward =
				normal;
			float3 right =
				cross(forward, up);
			up =
				cross(right, forward);

			float rot[16] =
				{ right.x, right.y, right.z, 0.f,
					up.x, up.y, up.z, 0.f,
					-forward.x, -forward.y, -forward.z, 0.f,
					.0f, .0f, .0f, 1.f };

			glTranslatef(
				pos.x,
				pos.y,
				pos.z);

			glMultMatrixf(rot);

			glBegin(GL_QUADS);
			{
				glNormal3f(
					normal.x,
					normal.y,
					normal.z);
				glVertex3f(
					-size,
					size,
					0);

				glNormal3f(
					normal.x,
					normal.y,
					normal.z);
				glVertex3f(
					-size,
					-size,
					0);

				glNormal3f(
					normal.x,
					normal.y,
					normal.z);
				glVertex3f(
					size,
					-size,
					0);

				glNormal3f(
					normal.x,
					normal.y,
					normal.z);
				glVertex3f(
					size,
					size,
					0);
			}
			glEnd();
		}
		glPopMatrix();
	}

private:
};