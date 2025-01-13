#pragma once
#include "engine/InstancedObject.h"
#include "ParticleSystem.h"
#include <cuda_runtime.h>

class InstancedParticles : public InstancedObject
{
public:
	InstancedParticles(ParticleSystem *p, uint32_t ShaderProgram);
	virtual ~InstancedParticles() = default;

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
	virtual void UpdateParticleInstances() = 0;

	void SetStillColor(glm::vec4 color) { m_ParticleData.StillColor = *(float4 *)&color; }
	void SetRandomColor(bool random) { m_ParticleData.RandomColor = random; }

protected:
	void UpdateGraphicsData();

	GraphicsData m_ParticleData{};
	ParticleSystem *m_Particles;
};

