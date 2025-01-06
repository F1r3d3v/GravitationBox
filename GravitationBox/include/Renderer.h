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

	void InitializeParticleInstancing(size_t instanceCount);
	void UninitializeParticleInstancing();
	void UpdateParticleInstancesCPU(ParticleData *pData);
	void UpdateParticleInstancesCUDA(ParticleData *pData);
	void RenderParticles(unsigned int shaderProgram, size_t instanceCount);

	static glm::vec2 GetViewportSize() {
		if (m_Window) return glm::vec2(m_Window->GetWidth(), m_Window->GetHeight());
		else return glm::vec2(0.0f);
	}

private:
	static Window *m_Window;

	unsigned int m_ParticleVAO;
	unsigned int m_ParticleVBO;
	unsigned int m_InstanceVBO;

	cudaGraphicsResource_t m_CudaVBOResource;
};

