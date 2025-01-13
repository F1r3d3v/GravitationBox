#pragma once
#include "InstancedParticles.h"

class CpuInstancedParticles : public InstancedParticles
{
public:
	CpuInstancedParticles(ParticleSystem *p, uint32_t ShaderProgram);
	~CpuInstancedParticles() = default;

	void UpdateParticleInstances() override;
};