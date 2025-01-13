#pragma once
#include "InstancedParticles.h"

class CudaInstancedParticles : public InstancedParticles
{
public:
	CudaInstancedParticles(ParticleSystem *p, uint32_t ShaderProgram);
	~CudaInstancedParticles();

	void UpdateParticleInstances() override;

private:
	cudaGraphicsResource_t m_CudaVBOResource = nullptr;
};
