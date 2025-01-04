#pragma once
#include "Grid.h"
#include "Particles.h"
#include <glm.hpp>

class VerletSolver
{
public:
	VerletSolver(Particles *p, Grid *g);
	~VerletSolver();

	void VerletCPU(float dt);
	cudaError_t VerletCuda(float dt);

	void SetParticlesInstance(Particles *p) { m_Particles = p; }

private:
	Particles *m_Particles;
	Grid *m_Grid;

	void SetParticlePosition(uint32_t id, glm::vec2 pos);
	glm::vec2 GetParticlePosition(uint32_t id);
	void SetParticleVelocity(uint32_t id, glm::vec2 vel);
	glm::vec2 GetParticleVelocity(uint32_t id);

	void CheckCollisionsWithWalls();
	void CheckCollisionsWithParticles();
	void UpdateParticles(float dt);
};

