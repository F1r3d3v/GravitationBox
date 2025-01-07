#pragma once
#include "Grid.h"
#include "Particles.h"
#include <glm.hpp>

struct SimulationParams
{
	float Timestep = 1.0f / 60.0f;
	float Radius = 10.0f;
	float Gravity = 981.0f;
	float WallDampening = 0.5f;
	float ParicleDampening = 0.5f;
	float ParticleStiffness = 0.5f;
	float ParticleShear = 0.5f;
	int DimX = 1600;
	int DimY = 900;
};

class VerletSolver
{
public:
	VerletSolver(Particles *p, Grid *g);
	~VerletSolver();

	void VerletCPU(float dt);
	cudaError_t VerletCuda(float dt);

	void SetParticlesInstance(Particles *p) { m_Particles = p; }

	SimulationParams Params;
private:
	Particles *m_Particles;
	Grid *m_Grid;

	void SetParticlePosition(uint32_t id, glm::vec2 pos);
	glm::vec2 GetParticlePosition(uint32_t id);
	void SetParticleVelocity(uint32_t id, glm::vec2 vel);
	glm::vec2 GetParticleVelocity(uint32_t id);

	void UpdateParticles(float dt);
	void CheckCollisionsWithWalls(uint32_t id);
	void CheckCollisions();
};
