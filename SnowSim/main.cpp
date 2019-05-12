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

#include "Simulation.cuh"
#include "SnowParticle.cuh"
#include "SnowModel.cuh"
#include "Grid.h"
#include "GridCell.cuh"
#include "Cube.h"
#include "Serializable.h"
#include "Global.h"

#define CENTER make_float3(0.f, 0.f, 0.f)
#define GRID_SIZE 100
#define SCALE .1f

int NUM_OBSTACLES = (6 + 1);

int windowHandle = 0;
int windowSizeX = 512, windowSizeY = 512;

SnowModel snowModel;
Grid<GridCell> grid;
SnowParticle* particles;
Simulation simulation;
Obstacle* obstacles;

static float angle = 0.f;
int frame = 0;// , time, timebase = 0;

void Render()
{
	frame++;
	/*
	time = glutGet(GLUT_ELAPSED_TIME);
	
	if ((time - timebase) % 1000 == 0) {
		printf("FPS:%4.2f, Frame:%i\n",
			frame*1000.0 / (time - timebase),
			frame);
		//timebase = time;
		//frame = 0;
	}*/

	glClear(GL_COLOR_BUFFER_BIT
		| GL_DEPTH_BUFFER_BIT);

	glLoadIdentity();

	glTranslatef(0.f, 0.f, -20.f);
	//glRotatef(angle, 0.f, 1.f, 0.f);

	if (Global::SCENE == 0)
	{
		glRotatef(40.f, 0.f, 1.f, 0.f);//-60
	}
	else
	{
		glRotatef(-60.f, 0.f, 1.f, 0.f);
	}

	simulation.StepSim(
		Global::TIME_STEP,
		frame);
	simulation.Draw();

	angle += 1.f;

	glFlush();
	glutSwapBuffers();
	glutPostRedisplay();

	if ((frame - 1) % 10 == 0) {
		printf("Frame: %i\n", frame);
		system("PAUSE");
	}
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
	{
		for (int i = 0; i < argc; i++)
		{
			char* s = argv[i];
			if (s[0] == '-')
			{
				if (argc <= i + 1)
					continue;

				std::string str;
				std::istringstream stream;
				uint num;
				float numF;

				switch (s[1])
				{
				case 'n':
					str = argv[i + 1];
					stream.str(str);
					stream >> num;
					
					if (!stream.bad())
					{
						Global::NUM_PARTICLES =
							num;
					}

					break;
				case 's':
					str = argv[i + 1];
					stream.str(str);
					stream >> num;

					if (!stream.bad())
					{
						Global::SCENE =
							num;
					}

					break;
				case 't':
					str = argv[i + 1];
					stream.str(str);
					stream >> numF;

					if (!stream.bad())
					{
						Global::TIME_STEP =
							numF;
					}

					break;
				case 'w':
					str = argv[i + 1];
					stream.str(str);
					stream >> numF;

					if (!stream.bad())
					{
						Global::ALPHA =
							numF;
					}

					break;
				case 'h':
					str = argv[i + 1];
					stream.str(str);
					stream >> numF;

					if (!stream.bad())
					{
						Global::HARDENING_COEFF =
							numF;
					}

					break;
				case 'r':
					str = argv[i + 1];
					stream.str(str);
					stream >> numF;

					if (!stream.bad())
					{
						Global::POISSON_RATIO =
							numF;
					}

					break;
				case 'C':
					str = argv[i + 1];
					stream.str(str);
					stream >> numF;

					if (!stream.bad())
					{
						Global::CRIT_COMP =
							numF;
					}

					break;
				case 'S':
					str = argv[i + 1];
					stream.str(str);
					stream >> numF;

					if (!stream.bad())
					{
						Global::CRIT_STRETCH =
							numF;
					}

					break;
				}
			}
		}
	}

	{
		if (1 < Global::SCENE)
		{
			Global::SCENE = 0;
		}
	}

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
				
				glEnable(GL_BLEND);
				glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
				
				glEnable(GL_CULL_FACE);

				glEnable(GL_DEPTH_TEST);
				glDepthFunc(GL_LESS);

				glShadeModel(GL_SMOOTH);

				{
					GLfloat lSpecular[] = { 1.0, 1.0, 1.0, 1.0 };
					GLfloat lShininess[] = { 50.0 };
					GLfloat lPosition[] = { -1.0, 10.0, -1.0, 0.0 };
					glMaterialfv(GL_FRONT, GL_SPECULAR, lSpecular);
					glMaterialfv(GL_FRONT, GL_SHININESS, lShininess);
					glLightfv(GL_LIGHT0, GL_POSITION, lPosition);

					{
						GLfloat lAmbient[] = {
									1.0,
									1.0,
									1.0,
									1.0 };
						glMaterialfv(GL_FRONT, GL_AMBIENT, lAmbient);
					}

					glEnable(GL_LIGHTING);
					glEnable(GL_LIGHT0);
				}

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
		if (Global::SCENE == 0)
		{
			snowModel.Load("Models/Monkey.obj");
		}
		else
		{
			snowModel.Load("Models/Sphere.obj");
		}

		{
			cudaError(
				cudaMalloc(
					&particles,
					Global::NUM_PARTICLES * sizeof(SnowParticle)));
		}

		snowModel.SampleParticles(
			&grid,
			particles,
			Global::TARGET_DENSITY,
			Global::NUM_PARTICLES,
			SnowModel::DisplayType::NONE);

#ifdef CHECK
		{
			unsigned int size =
				NUM_PARTICLES;
			auto particleCPU =
				new SnowParticle[size];

			cudaError(
				cudaMemcpy(
					particleCPU,
					particles,
					size * sizeof(SnowParticle),
					cudaMemcpyDeviceToHost));

			float3* vals =
				new float3[size];

			for (uint i = 0; i < size; i++)
			{
				vals[i] = particleCPU[i].position;
			}
			
#ifndef _DEBUG
			Serializable::Store(
				vals,
				NUM_PARTICLES,
				"sampleTest.txt");
#else
			Serializable::Compare(
				vals,
				"sampleTest.txt");
#endif

			delete[] particleCPU;
			delete[] vals;
		}
#endif
	}

	// Setup simulation.
	{
		// Create floor.
		{
			if (Global::SCENE == 1)
			{
				NUM_OBSTACLES++;
			}

			Obstacle* obstaclesCpu =
				new Obstacle[NUM_OBSTACLES];

			// Grid bounds.
			uint boundNum = 0;

			if (Global::SCENE == 1)
			{
				obstaclesCpu[boundNum].pos =
					make_float3(0.f, -2.f, 0.f);
				obstaclesCpu[boundNum].vel =
					make_float3(0.f, 0.f, 0.f);
				obstaclesCpu[boundNum].misc =
					normalize(make_float3(0.2f, 1.f, -.3f));
				obstaclesCpu[boundNum].friction = .001f;
				obstaclesCpu[boundNum].type = 0;

				boundNum++;

				obstaclesCpu[boundNum].pos =
					make_float3(-20.f, -5.f, 1.5f);
				obstaclesCpu[boundNum].vel =
					make_float3(10.f, 1.f, 0.f);
				obstaclesCpu[boundNum].misc.x = 2.f;
				obstaclesCpu[boundNum].friction = .1f;
				obstaclesCpu[boundNum].type = 1;

				boundNum++;
			}
			
			if (Global::SCENE == 0)
			{
				obstaclesCpu[boundNum].pos =
					make_float3(0.f, -2.f, -5.f);
				obstaclesCpu[boundNum].vel =
					make_float3(0.f, 0.f, 10.f);
				obstaclesCpu[boundNum].misc.x = 1.f;
				obstaclesCpu[boundNum].friction = .1f;
				obstaclesCpu[boundNum].type = 1;

				boundNum++;
			}

			{
				float3 bounds = make_float3(
					grid.GetWidth() / 2.f,
					grid.GetHeight() / 2.f,
					grid.GetDepth() / 2.f);

				float3 normal[] = {
					make_float3(1, 0, 0),
					make_float3(-1, 0, 0),
					make_float3(0, 1, 0),
					make_float3(0, -1, 0),
					make_float3(0, 0, 1),
					make_float3(0, 0, -1)
				};

				float3 center =
					grid.GetPosition()
					+ bounds * grid.GetScale();

				for (uint i = 0; i < _countof(normal); i++)
				{
					{
						obstaclesCpu[boundNum].pos =
							center +
							bounds *
							grid.GetScale() *
							-normal[i] * .9f;
						obstaclesCpu[boundNum].vel =
							make_float3(0.f, 0.f, 0.f);
						obstaclesCpu[boundNum].misc =
							-normal[i];
						obstaclesCpu[boundNum].friction = 1.001f;
						obstaclesCpu[boundNum].type = 2;
					}

					boundNum++;
				}
			}

			{
				cudaError(
					cudaMalloc(
						&obstacles,
						NUM_OBSTACLES * sizeof(Obstacle)));

				cudaError(cudaMemcpy(
					obstacles,
					obstaclesCpu,
					NUM_OBSTACLES * sizeof(Obstacle),
					cudaMemcpyHostToDevice));
			}

			delete[] obstaclesCpu;
		}

		simulation.SetupSim(
			&grid,
			particles,
			Global::NUM_PARTICLES,
			obstacles,
			NUM_OBSTACLES);
	}
	/*
	simulation.StepSim(
		.1f,
		0);
	exit(0);*/

	// Run.
	{
		glutMainLoop();
	}

	// Cleanup.
	{
		cudaError(
			cudaFree(particles));

		cudaError(
			cudaFree(obstacles));
	}

	return 0;
}
