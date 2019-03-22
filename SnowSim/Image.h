#pragma once

#include <GL/glew.h>
#include <IL/il.h>

#include "Helper.h"

class Image
{
public:

	Image();
	~Image();

	bool empty();

	bool load(const char* file);
	bool convert(ILenum format, ILenum type);
	bool convert4f();
	float4* getData4f();
	GLuint genGlImage();

	ILubyte* getData();
	int getWidth();
	int getHeight();

private:
	ILuint imageID = 0;
	GLuint glImageID = 0;

	ILubyte* data;
	int width, height;
};