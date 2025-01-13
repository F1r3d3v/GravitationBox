#pragma once
#include "ParticleSystem.h"
#include "ParticleSolver.h"

class CudaParticleSystem;

class CpuParticleSystem : public ParticleSystem
{
public:
	CpuParticleSystem(uint32_t count, float radius, ParticleSolver *solver);
	~CpuParticleSystem();

	static CpuParticleSystem *CreateRandom(uint32_t count, float radius, glm::ivec2 dim, ParticleSolver *solver);
	static CpuParticleSystem *CreateCircle(uint32_t count, float radius, glm::ivec2 dim, ParticleSolver *solver);
	static CpuParticleSystem *CreateBox(uint32_t count, float radius, glm::ivec2 dim, ParticleSolver *solver);
	static CpuParticleSystem *CreateWaterfall(uint32_t count, float radius, glm::ivec2 dim, float velocity, int rows, ParticleSolver *solver);

	static CpuParticleSystem *CreateFromCuda(CudaParticleSystem *p, ParticleSolver *solver);
};

