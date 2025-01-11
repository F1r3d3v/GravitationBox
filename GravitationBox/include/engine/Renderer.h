#pragma once
#include "glm.hpp"
#include "Window.h"

#include <cuda_runtime.h>

struct ParticleData;

class Renderer
{
public:
	Renderer(Window *window);
	Renderer() = default;
	~Renderer();

	static unsigned int LoadShader(const char *vertexShaderSource, const char *fragmentShaderSource);
	static unsigned int LoadShaderFromFile(const char *vertexShaderFilepath, const char *fragmentShaderFilepath);
	static void UnloadShader(unsigned int shaderProgram);
	void Clear(glm::vec4 color);

	static glm::vec2 GetViewportSize();

private:
	static Window *m_Window;
};

