#include <iostream>
#include <string>

#define GLEW_STATIC
#include <GL/glew.h>
#include <GL/glut.h>

// CUDA headers
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

// DevIL
#include <IL/il.h>

#include "SnowParticle.cuh"
#include "SnowModel.cuh"
#include "Simulation.cuh"
#include "Grid.h"
#include "GridCell.cuh"
#include "Cube.h"

#define CENTER make_float3(0.f, 0.f, 0.f)
#define GRID_SIZE 100
#define SCALE .1f

#define NUM_PARTICLES 1000

int windowHandle = 0;
int windowSizeX = 512, windowSizeY = 512;

SnowModel snowModel;
Grid<GridCell> grid;
SnowParticle* particles;
Simulation simulation;

static float angle = 0.f;

void Render()
{
	glClear(GL_COLOR_BUFFER_BIT
		| GL_DEPTH_BUFFER_BIT);

	glLoadIdentity();

	glTranslatef(0.f, 0.f, -10.f);
	glRotatef(angle, 0.f, 1.f, 0.f);

	simulation.StepSim(.1f);
	//StepSimulation();

	angle += 1.f;

	glFlush();
	glutSwapBuffers();

	glutPostRedisplay();
}

void Keyboard(unsigned char key, int x, int y)
{

}

void Reshape(int w, int h)
{
	// TODO:
}

void Idle()
{
	// TODO:
}

int main(int argc, char* argv[])
{
	// Setup.
	{
		{
			glutInit(&argc, argv);

			// init DevIL
			ilInit();

			glutInitDisplayMode(
				GLUT_RGBA |
				GLUT_ALPHA |
				GLUT_DOUBLE |
				GLUT_DEPTH);

			glutInitWindowSize(
				windowSizeX,
				windowSizeY);
			windowHandle =
				glutCreateWindow("Snow Sim");
			
			// Callbacks.
			{
				glutDisplayFunc(Render);
				glutKeyboardFunc(Keyboard);
				glutReshapeFunc(Reshape);
				glutIdleFunc(Idle);
			}

			// Glew.
			GLenum err = glewInit();

			if (GLEW_OK != err)
			{
				printf("GLEW ERROR: %s\n", glewGetErrorString(err));
			}

			{
				// setup for OpenGL
				ilEnable(IL_ORIGIN_SET);
				ilOriginFunc(IL_ORIGIN_LOWER_LEFT);
				
				// TMP
				glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
				glEnable(GL_DEPTH_TEST);
				glDepthFunc(GL_LESS);

				glMatrixMode(GL_PROJECTION);
				glLoadIdentity();

				gluPerspective(
					45.f,
					float(windowSizeX) / float(windowSizeY),
					.1f,
					100.f);

				glMatrixMode(GL_MODELVIEW);

				glEnable(GL_COLOR);
				glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
			}
		}
	}

	// Create grid.
	
	/*Grid<SnowParticle> grid(
		CENTER,
		GRID_SIZE, GRID_SIZE, GRID_SIZE,
		SCALE);*/
	grid.Resize(
		CENTER,
		GRID_SIZE, GRID_SIZE, GRID_SIZE,
		SCALE);

	// Setup scene.
	{
		snowModel.Load("Models/Monkey.obj");

		{
			cudaError(
				cudaMalloc(
					&particles,
					NUM_PARTICLES * sizeof(SnowParticle)));
		}

		snowModel.SampleParticles(
			&grid,
			particles,
			NUM_PARTICLES,
			SnowModel::DisplayType::NONE);
	}

	// Setup simulation.
	{
		simulation.SetupSim(
			&grid,
			particles,
			10);// NUM_PARTICLES);
	}

	// Run.
	{
		glutMainLoop();
	}

	// Cleanup.
	{
		cudaError(
			cudaFree(particles));
	}

	return 0;
}