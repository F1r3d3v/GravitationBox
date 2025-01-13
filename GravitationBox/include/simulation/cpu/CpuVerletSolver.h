#pragma once
#include "Grid.h"
#include "ParticleSolver.h"
#include <glm.hpp>

class CpuVerletSolver : public ParticleSolver
{
public:
	CpuVerletSolver(Grid *g);
	~CpuVerletSolver() = default;

	void Solve(ParticleSystem *p) override;

private:
	Grid *m_Grid;

	template <bool floor = false>
	glm::vec2 CollideWall(glm::vec2 position, glm::vec2 velocity, float mass, glm::vec2 floorPosition);
	glm::vec2 SolveCollision(glm::vec2 positionA, glm::vec2 velocityA, glm::vec2 positionB, glm::vec2 velocityB);
	glm::vec2 CheckCollisionsInCell(uint32_t tid, uint32_t cellId, glm::vec2 position, glm::vec2 velocity, ParticleSystem *p);
	glm::vec2 CheckCollisionsWithWalls(uint32_t id, ParticleSystem *p);

	template <bool stage1>
	void UpdateParticles(ParticleSystem *p);
	void CheckCollisions(ParticleSystem *p);
};
