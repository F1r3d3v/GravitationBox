#pragma once

#include "Renderer.h"
#include <glm.hpp>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>
#include "InstancedParticles.h"
#include "Config.h"

struct ParticleData {
	uint32_t Count;
	float *PosX;
	float *PosY;
	float2 Scale;
	float4 *Color;

	bool RandomColor;
	float4 StillColor;
};

struct Particles
{
public:
	uint32_t TotalCount;
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
	static Particles *WaterfallCPU(uint32_t count, float radius, glm::ivec2 dim, float velocity, int rows);
	static Particles *WaterfallCUDA(uint32_t count, float radius, glm::ivec2 dim, float velocity, int rows);

	void DrawCPU(Renderer *renderer, InstancedParticles *instancedParticles);
	void DrawCUDA(Renderer *renderer, InstancedParticles *instancedParticles);

	void SetCount(uint32_t count);
	void SetStillColor(glm::vec4 color) { m_ParticleData.StillColor = *(float4*)&color; }
	void SetRandomColor(bool random) { m_ParticleData.RandomColor = random; }

private:
	bool m_IsCuda;
	ParticleData m_ParticleData{};

	void InitDrawingData();
};
