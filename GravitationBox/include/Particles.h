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
	uint32_t Count;
};

struct Particles
{
public:
	uint32_t Count;
	float Radius;
	float *PosX, *PosY, *SortedPosX, *SortedPosY;
	float *VelX, *VelY, *SortedVelX, *SortedVelY;
	float *ForceX, *ForceY, *SortedForceX, *SortedForceY;
	float *Mass;
	glm::vec4 *Color;

	Particles(uint32_t count, float radius, bool isCUDA);
	~Particles();

	static Particles *RandomCPU(uint32_t count, float radius, glm::ivec2 dim);
	static Particles *RandomCircleCPU(uint32_t count, float radius, glm::ivec2 dim);
	static Particles *RandomBoxCPU(uint32_t count, float radius, glm::ivec2 dim);
	static Particles *RandomCUDA(uint32_t count, float radius, glm::ivec2 dim);
	static Particles *RandomCircleCUDA(uint32_t count, float radius, glm::ivec2 dim);
	static Particles *RandomBoxCUDA(uint32_t count, float radius, glm::ivec2 dim);

	void DrawCPU(Renderer *renderer);
	void DrawCUDA(Renderer *renderer);

private:
	bool m_IsCuda;
	unsigned int m_ShaderProgram = Renderer::LoadShaderFromFile("shaders/particle.vert", "shaders/particle.frag");
	ParticleData m_ParticleData{};

	void InitDrawingData();
};
