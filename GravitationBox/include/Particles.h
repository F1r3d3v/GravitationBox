#pragma once

#include "Renderer.h"
#include <glm.hpp>
#include <random>
#include <algorithm>

#include "Config.h"

#include <cuda_runtime.h>


struct ParticleData {
	float *PosX;
	float *PosY;
	float2 Scale;
	float4 *Color;
	size_t Count;
};

struct Particles
{
public:
	size_t Count;
	float Radius;
	float *PosX, *PosY;
	float *VelX, *VelY;
	float *ForceX, *ForceY;
	float *Mass;
	glm::vec4 *Color;

	Particles(size_t count, float radius, bool isCUDA);
	~Particles();

	static Particles *RandomCPU(size_t count, float radius, glm::ivec2 dim);
	static Particles *RandomCircleCPU(size_t count, float radius, glm::ivec2 dim);
	static Particles *RandomBoxCPU(size_t count, float radius, glm::ivec2 dim);
	static Particles *RandomCUDA(size_t count, float radius, glm::ivec2 dim);
	static Particles *RandomCircleCUDA(size_t count, float radius, glm::ivec2 dim);
	static Particles *RandomBoxCUDA(size_t count, float radius, glm::ivec2 dim);

	void DrawCPU(Renderer *renderer);
	void DrawCUDA(Renderer *renderer);

private:
	bool m_IsCuda;
	unsigned int m_ShaderProgram = Renderer::LoadShaderFromFile("shaders/particle.vert", "shaders/particle.frag");
	ParticleData m_ParticleData{};

	void InitDrawingData();
};
