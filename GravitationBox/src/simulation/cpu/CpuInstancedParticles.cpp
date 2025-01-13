#include "cpu/CpuInstancedParticles.h"
#include "cuda/CudaInstancedParticles.h"
#include <glad/gl.h>

CpuInstancedParticles::CpuInstancedParticles(ParticleSystem *p, uint32_t ShaderProgram)
	: InstancedParticles(p, ShaderProgram)
{
}

void CpuInstancedParticles::UpdateParticleInstances()
{
	UpdateGraphicsData();
	glBindBuffer(GL_ARRAY_BUFFER, m_InstanceVBO);
	for (size_t i = 0; i < m_ParticleData.Count; ++i)
	{
		float instanceData[8] =
		{
			m_ParticleData.PosX[i], m_ParticleData.PosY[i],
			m_ParticleData.Scale.x, m_ParticleData.Scale.y,
			m_ParticleData.RandomColor ? m_ParticleData.Color[i].x : m_ParticleData.StillColor.x,
			m_ParticleData.RandomColor ? m_ParticleData.Color[i].y : m_ParticleData.StillColor.y,
			m_ParticleData.RandomColor ? m_ParticleData.Color[i].z : m_ParticleData.StillColor.z,
			m_ParticleData.RandomColor ? m_ParticleData.Color[i].w : m_ParticleData.StillColor.w
		};
		glBufferSubData(GL_ARRAY_BUFFER, i * 8 * sizeof(float), 8 * sizeof(float), instanceData);
	}
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}
