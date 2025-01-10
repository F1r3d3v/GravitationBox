#pragma once
#include "InstancedObject.h"

#include <cuda_runtime.h>

struct Particles;
struct ParticleData;

class InstancedParticles : public InstancedObject
{
public:
	InstancedParticles(Particles *p, uint32_t ShaderProgram);
	~InstancedParticles();
	void Draw() override;

	void UpdateParticleInstancesCPU(ParticleData *pData);
	cudaError_t UpdateParticleInstancesCUDA(ParticleData *pData);

protected:
	void BindBuffers() override;

private:
	cudaGraphicsResource_t m_CudaVBOResource = nullptr;
};

