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
		float Timestep = Config::TIMESTEP;
		float Radius = Config::PARTICLE_RADIUS;
		float Gravity = Config::GRAVITY;
		float WallDampening = Config::WALL_DAMPENING;
		float WallFriction = Config::WALL_FRICTION;
		float ParticlesDampening = Config::PARTICLES_DAMPENING;
		float ParticlesStiffness = Config::PARTICLES_STIFFNESS;
		float ParticlesFriction = Config::PARTICLES_FRICTION;
		int DimX = Config::WINDOW_WIDTH;
		int DimY = Config::WINDOW_HEIGHT;
	};

	void VerletCPU();
	cudaError_t VerletCuda();

	void SetParticlesInstance(Particles *p) { m_Particles = p; }
	void SetParams(const SimulationParams &params);

private:
	Particles *m_Particles;
	Grid *m_Grid;
	SimulationParams m_Params;

	template <bool floor = false>
	glm::vec2 CollideWall(glm::vec2 position, glm::vec2 velocity, float mass, glm::vec2 floorPosition);
	glm::vec2 SolveCollision(glm::vec2 positionA, glm::vec2 velocityA, glm::vec2 positionB, glm::vec2 velocityB);
	glm::vec2 CheckCollisionsInCell(uint32_t tid, uint32_t cellId, glm::vec2 position, glm::vec2 velocity);
	glm::vec2 CheckCollisionsWithWalls(uint32_t id);

	template <bool stage1>
	void UpdateParticles();
	void CheckCollisions();
};
