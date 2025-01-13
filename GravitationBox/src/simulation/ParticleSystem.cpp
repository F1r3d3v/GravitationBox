#include "ParticleSystem.h"
#include "cpu/CpuParticleSystem.h"
#include "cuda/CudaParticleSystem.h"

ParticleSystem::ParticleSystem(uint32_t count, float radius, ParticleSolver *solver)
	: TotalCount(count), Count(count), Radius(radius), m_Solver(solver)
{
}

void ParticleSystem::Update()
{
	for (uint8_t i = 0; i < m_Params.Substeps; ++i)
	{
		m_Solver->Solve(this);
	}
}

ParticleSystem *ParticleSystem::CreateRandom(uint32_t count, float radius, glm::ivec2 dim, ParticleSolver *solver, bool useCuda)
{
	return (useCuda) ? static_cast<ParticleSystem *>(CudaParticleSystem::CreateRandom(count, radius, dim, solver))
		: static_cast<ParticleSystem *>(CpuParticleSystem::CreateRandom(count, radius, dim, solver));
}

ParticleSystem *ParticleSystem::CreateCircle(uint32_t count, float radius, glm::ivec2 dim, ParticleSolver *solver, bool useCuda)
{
	return (useCuda) ? static_cast<ParticleSystem *>(CudaParticleSystem::CreateCircle(count, radius, dim, solver))
		: static_cast<ParticleSystem *>(CpuParticleSystem::CreateCircle(count, radius, dim, solver));
}

ParticleSystem *ParticleSystem::CreateBox(uint32_t count, float radius, glm::ivec2 dim, ParticleSolver *solver, bool useCuda)
{
	return (useCuda) ? static_cast<ParticleSystem *>(CudaParticleSystem::CreateBox(count, radius, dim, solver))
		: static_cast<ParticleSystem *>(CpuParticleSystem::CreateBox(count, radius, dim, solver));
}

ParticleSystem *ParticleSystem::CreateWaterfall(uint32_t count, float radius, glm::ivec2 dim, float velocity, int rows, ParticleSolver *solver, bool useCuda)
{
	return (useCuda) ? static_cast<ParticleSystem *>(CudaParticleSystem::CreateWaterfall(count, radius, dim, velocity, rows, solver))
		: static_cast<ParticleSystem *>(CpuParticleSystem::CreateWaterfall(count, radius, dim, velocity, rows, solver));
}
