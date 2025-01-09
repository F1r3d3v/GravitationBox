#include "Renderer.h"
#include "Log.h"
#include "glad/gl.h"
#include "Config.h"
#include "Particles.h"

#include <cuda_gl_interop.h>

Window *Renderer::m_Window = nullptr;

Renderer::Renderer(Window *window)
{
	m_Window = window;
}

Renderer::~Renderer()
{
}

void Renderer::UninitializeParticleInstancing()
{
	cudaGraphicsUnregisterResource(m_CudaVBOResource);
	glDeleteBuffers(1, &m_ParticleVBO);
	glDeleteBuffers(1, &m_InstanceVBO);
	glDeleteVertexArrays(1, &m_ParticleVAO);
}

void Renderer::InitializeParticleInstancing(size_t instanceCount)
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
	glBufferData(GL_ARRAY_BUFFER, instanceCount * 8 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);

	// Register buffer with CUDA
	cudaError_t err = cudaGraphicsGLRegisterBuffer(&m_CudaVBOResource, m_InstanceVBO, cudaGraphicsMapFlagsWriteDiscard);
	if (err != cudaSuccess) {
		Log::Error("Failed to register OpenGL buffer with CUDA");
		return;
	}

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

	// Enable blending
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void Renderer::UpdateParticleInstancesCPU(ParticleData *pData)
{
	glBindBuffer(GL_ARRAY_BUFFER, m_InstanceVBO);
	for (size_t i = 0; i < pData->Count; ++i)
	{
		float instanceData[8] = {
			pData->PosX[i], pData->PosY[i],
			pData->Scale.x, pData->Scale.y,
			pData->RandomColor ? pData->Color[i].x : pData->StillColor.x,
			pData->RandomColor ? pData->Color[i].y : pData->StillColor.y,
			pData->RandomColor ? pData->Color[i].z : pData->StillColor.z,
			pData->RandomColor ? pData->Color[i].w : pData->StillColor.w
		};
		glBufferSubData(GL_ARRAY_BUFFER, i * 8 * sizeof(float), 8 * sizeof(float), instanceData);
	}
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Renderer::RenderParticles(unsigned int shaderProgram, size_t instanceCount)
{
	glm::vec2 viewport = GetViewportSize();
	glUseProgram(shaderProgram);
	glUniform2f(glGetUniformLocation(shaderProgram, "Viewport"), viewport.x, viewport.y);
	glBindVertexArray(m_ParticleVAO);
	glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0, static_cast<GLsizei>(instanceCount));
	glBindVertexArray(0);
}

unsigned int Renderer::LoadShader(const char *vertexShaderSource, const char *fragmentShaderSource)
{
	unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
	glCompileShader(vertexShader);

	int success;
	char infoLog[512];
	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
	if (!success) {
		glGetShaderInfoLog(vertexShader, 512, nullptr, infoLog);
		Log::Error("Vertex shader compilation failed: " + std::string(infoLog));
	}

	unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
	glCompileShader(fragmentShader);

	glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
	if (!success) {
		glGetShaderInfoLog(fragmentShader, 512, nullptr, infoLog);
		Log::Error("Fragment shader compilation failed: " + std::string(infoLog));
	}

	unsigned int shaderProgram = glCreateProgram();
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);
	glLinkProgram(shaderProgram);

	glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
	if (!success) {
		glGetProgramInfoLog(shaderProgram, 512, nullptr, infoLog);
		Log::Error("Shader program linking failed: " + std::string(infoLog));
	}

	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);

	return shaderProgram;
}

unsigned int Renderer::LoadShaderFromFile(const char *vertexShaderFilepath, const char *fragmentShaderFilepath)
{
	std::ifstream vertexShaderFile(vertexShaderFilepath);
	std::ifstream fragmentShaderFile(fragmentShaderFilepath);
	if (!vertexShaderFile.is_open())
		Log::Error("Failed to open vertex shader file: " + std::string(vertexShaderFilepath));
	if (!fragmentShaderFile.is_open())
		Log::Error("Failed to open fragment shader file: " + std::string(fragmentShaderFilepath));
	std::string vertexShaderSource((std::istreambuf_iterator<char>(vertexShaderFile)), std::istreambuf_iterator<char>());
	std::string fragmentShaderSource((std::istreambuf_iterator<char>(fragmentShaderFile)), std::istreambuf_iterator<char>());
	return LoadShader(vertexShaderSource.c_str(), fragmentShaderSource.c_str());
}

void Renderer::UnloadShader(unsigned int shaderProgram)
{
	glDeleteProgram(shaderProgram);
}

void Renderer::Clear(glm::vec4 color)
{
	glClearColor(color.x * color.w, color.y * color.w, color.z * color.w, color.w);
	glClear(GL_COLOR_BUFFER_BIT);
}
