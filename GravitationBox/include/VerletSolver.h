#pragma once
#include "Grid.h"
#include "Particles.h"
#include <glm.hpp>

class VerletSolver
{
public:
	VerletSolver(Particles *p, Grid *g);
	~VerletSolver();

	struct SimulationParams
	{
		float Timestep = 0.2f;
		float Radius = 5.0;
		float Gravity = 1.0f;
		float WallDampening = 0.75f;
		float ParticleDampening = 5.0f;
		float ParticleStiffness = 50.0f;
		float ParticleShear = 0.1f;
		int DimX = 1600;
		int DimY = 900;
	};

	void VerletCPU();
	cudaError_t VerletCuda();

	void SetParticlesInstance(Particles *p) { m_Particles = p; }
	void SetParams(const SimulationParams &params);

private:
	Particles *m_Particles;
	Grid *m_Grid;
	SimulationParams m_Params;

	void SetParticlePosition(uint32_t id, glm::vec2 pos);
	glm::vec2 GetParticlePosition(uint32_t id);
	void SetParticleVelocity(uint32_t id, glm::vec2 vel);
	glm::vec2 GetParticleVelocity(uint32_t id);

	glm::vec2 SolveCollision(glm::vec2 positionA, glm::vec2 velocityA, glm::vec2 positionB, glm::vec2 velocityB);
	glm::vec2 CheckCollisionInCell(uint32_t tid, uint32_t cellId, glm::vec2 position, glm::vec2 velocity);

	template <bool stage1>
	void UpdateParticles();

	void CheckCollisionsWithWalls(uint32_t id);
	void CheckCollisions();
};
