#pragma once
#include "GraphicsObject.h"
#include <cstdint>

class InstancedObject : public GraphicsObject
{
public:
	InstancedObject(size_t InstanceCount, uint32_t ShaderProgram);
	~InstancedObject();

protected:
	virtual void BindBuffers() = 0;

	size_t m_InstanceCount;
	unsigned int m_ShaderProgram;
	unsigned int m_ParticleVAO;
	unsigned int m_ParticleVBO;
	unsigned int m_ParticleEBO;
	unsigned int m_InstanceVBO;
};

