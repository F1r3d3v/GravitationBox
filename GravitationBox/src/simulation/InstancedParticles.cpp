#include "InstancedParticles.h"
#include "glad/gl.h"
#include "cuda/cuda_helper.h"
#include "cuda_gl_interop.h"
#include "ParticleSystem.h"
#include "engine/Renderer.h"
#include "glm.hpp"

InstancedParticles::InstancedParticles(ParticleSystem *p, uint32_t ShaderProgram)
	: InstancedObject(p->TotalCount, ShaderProgram), m_Particles(p)
{
	// Vertex data for a quad (centered at origin)
	float quadVertices[] = {
		// positions     // texture coords
		-1.0f, -1.0f,   0.0f, 0.0f,
		 1.0f, -1.0f,   1.0f, 0.0f,
		 1.0f,  1.0f,   1.0f, 1.0f,
		-1.0f,  1.0f,   0.0f, 1.0f
	};

	unsigned int indices[] = {
		0, 1, 2,
		2, 3, 0
	};

	// Create buffers
	glGenVertexArrays(1, &m_ParticleVAO);
	glGenBuffers(1, &m_ParticleVBO);
	unsigned int EBO;
	glGenBuffers(1, &EBO);
	glGenBuffers(1, &m_InstanceVBO);

	// Bind vertex array
	glBindVertexArray(m_ParticleVAO);

	// Set up vertex buffer
	glBindBuffer(GL_ARRAY_BUFFER, m_ParticleVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);

	// Set up element buffer
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

	// Position attribute
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0);
	glEnableVertexAttribArray(0);

	// Texture coord attribute
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)(2 * sizeof(float)));
	glEnableVertexAttribArray(1);

	// Create and register instance buffer for CUDA
	glBindBuffer(GL_ARRAY_BUFFER, m_InstanceVBO);
	glBufferData(GL_ARRAY_BUFFER, m_InstanceCount * 8 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);

	// Register buffer with CUDA
	CUDA_CHECK_NR(cudaGraphicsGLRegisterBuffer(&m_CudaVBOResource, m_InstanceVBO, cudaGraphicsMapFlagsWriteDiscard));

	// Position (per instance)
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *)0);
	glEnableVertexAttribArray(2);
	glVertexAttribDivisor(2, 1);

	// Scale (per instance)
	glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *)(2 * sizeof(float)));
	glEnableVertexAttribArray(3);
	glVertexAttribDivisor(3, 1);

	// Color RGBA (per instance)
	glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *)(4 * sizeof(float)));
	glEnableVertexAttribArray(4);
	glVertexAttribDivisor(4, 1);

	// Unbind buffers
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}

InstancedParticles::~InstancedParticles()
{
	cudaGraphicsUnregisterResource(m_CudaVBOResource);
	m_CudaVBOResource = nullptr;
}

void InstancedParticles::Draw()
{
	// Bind shader
	glm::vec2 viewport = Renderer::GetViewportSize();
	glUseProgram(m_ShaderProgram);
	glUniform2f(glGetUniformLocation(m_ShaderProgram, "Viewport"), viewport.x, viewport.y);

	// Enable blending
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	// Draw particles
	glBindVertexArray(m_ParticleVAO);
	glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0, static_cast<GLsizei>(m_InstanceCount));
	glBindVertexArray(0);

	// Disable blending
	glDisable(GL_BLEND);

	// Unbind shader
	glUseProgram(0);
}

void InstancedParticles::UpdateParticleInstancesCPU()
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

void InstancedParticles::UpdateGraphicsData()
{
	glm::vec2 viewport = Renderer::GetViewportSize();
	m_ParticleData.PosX = m_Particles->PosX;
	m_ParticleData.PosY = m_Particles->PosY;
	m_ParticleData.Scale = make_float2(2 * m_Particles->Radius / viewport.x, 2 * m_Particles->Radius / viewport.y);
	m_ParticleData.Color = (float4 *)m_Particles->Color;
	m_ParticleData.Count = m_Particles->Count;
}
