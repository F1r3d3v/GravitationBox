#pragma once
#include "ParticleSystem.h"
#include "ParticleSolver.h"

class CudaParticleSystem : public ParticleSystem
{
public:
	CudaParticleSystem(uint32_t count, float radius, ParticleSolver *solver);
	~CudaParticleSystem();

	static CudaParticleSystem *CreateRandom(uint32_t count, float radius, glm::ivec2 dim, ParticleSolver *solver);
	static CudaParticleSystem *CreateCircle(uint32_t count, float radius, glm::ivec2 dim, ParticleSolver *solver);
	static CudaParticleSystem *CreateBox(uint32_t count, float radius, glm::ivec2 dim, ParticleSolver *solver);
	static CudaParticleSystem *CreateWaterfall(uint32_t count, float radius, glm::ivec2 dim, float velocity, int rows, ParticleSolver *solver);
};
