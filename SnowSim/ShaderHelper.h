#pragma once

#include <GL/glew.h>

#define BUFFER_OFFSET(_i) (reinterpret_cast<char *>(NULL) + (_i))

void createProgram(GLuint* program, GLuint* vs, GLuint* fs, GLuint* gs,
	char* fileVS, char* fileGS, char* fileFS);