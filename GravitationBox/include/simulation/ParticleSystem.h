#pragma once
#include <glm.hpp>
#include <cstdint>
#include "Config.h"
#include "ParticleSolver.h"

class ParticleSystem
{
public:
	ParticleSystem(uint32_t count, float radius, ParticleSolver *solver);
	virtual ~ParticleSystem() = default;

	uint32_t TotalCount;
	uint32_t Count;
	float Radius;
	float *PosX = nullptr, *PosY = nullptr, *SortedPosX = nullptr, *SortedPosY = nullptr;
	float *VelX = nullptr, *VelY = nullptr, *SortedVelX = nullptr, *SortedVelY = nullptr;
	float *ForceX = nullptr, *ForceY = nullptr, *SortedForceX = nullptr, *SortedForceY = nullptr;
	float *Mass = nullptr;
	glm::vec4 *Color = nullptr;

	struct Parameters
	{
		uint8_t Substeps = Config::SUBSTEPS;
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
	} m_Params;

	void SetParameters(const Parameters &params) { m_Params = params; }
	Parameters GetParameters() const { return m_Params; }
	void Update();

	static ParticleSystem *CreateRandom(uint32_t count, float radius, glm::ivec2 dim, ParticleSolver *solver, bool useCuda);
	static ParticleSystem *CreateCircle(uint32_t count, float radius, glm::ivec2 dim, ParticleSolver *solver, bool useCuda);
	static ParticleSystem *CreateBox(uint32_t count, float radius, glm::ivec2 dim, ParticleSolver *solver, bool useCuda);
	static ParticleSystem *CreateWaterfall(uint32_t count, float radius, glm::ivec2 dim, float velocity, int rows, ParticleSolver *solver, bool useCuda);

protected:
	ParticleSolver *m_Solver;
};