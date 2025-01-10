#include "InstancedObject.h"
#include "glad/gl.h"

InstancedObject::InstancedObject(size_t InstanceCount, uint32_t ShaderProgram)
	: m_InstanceCount(InstanceCount), m_ShaderProgram(ShaderProgram)
{
	glGenVertexArrays(1, &m_ParticleVAO);
	glGenBuffers(1, &m_ParticleVBO);
	glGenBuffers(1, &m_ParticleEBO);
	glGenBuffers(1, &m_InstanceVBO);

	BindBuffers();
}

InstancedObject::~InstancedObject()
{
	glDeleteBuffers(1, &m_InstanceVBO);
	glDeleteBuffers(1, &m_ParticleEBO);
	glDeleteBuffers(1, &m_ParticleVBO);
	glDeleteVertexArrays(1, &m_ParticleVAO);
}
