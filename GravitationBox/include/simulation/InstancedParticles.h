#pragma once
#include "engine/InstancedObject.h"
#include "ParticleSystem.h"
#include <cuda_runtime.h>

class InstancedParticles : public InstancedObject
{
public:
	InstancedParticles(ParticleSystem *p, uint32_t ShaderProgram);
	~InstancedParticles();

	struct GraphicsData
	{
		uint32_t Count;
		float *PosX;
		float *PosY;
		float2 Scale;
		float4 *Color;

		bool RandomColor;
		float4 StillColor;
	};

	void Draw() override;
	void UpdateParticleInstancesCPU();
	cudaError_t UpdateParticleInstancesCUDA();

	void SetStillColor(glm::vec4 color) { m_ParticleData.StillColor = *(float4 *)&color; }
	void SetRandomColor(bool random) { m_ParticleData.RandomColor = random; }

private:
	void UpdateGraphicsData();

	cudaGraphicsResource_t m_CudaVBOResource = nullptr;
	GraphicsData m_ParticleData;
	ParticleSystem *m_Particles;
};

