#pragma once
#include "engine/Simulation.h"
#include "ParticleSystem.h"
#include "ParticleSolver.h"
#include "InstancedParticles.h"
#include "Grid.h"

#include <imgui.h>
#include <memory>
#include <glad/gl.h>

class MySimulation : public Simulation
{
public:
	MySimulation(std::string title, int width, int height);
	~MySimulation();
	void OnStart() override;
	void OnUpdate(float deltaTime) override;
	void OnRender(Renderer* renderer) override;
	void OnImGuiRender() override;
	void OnCleanup() override;
	void OnResize(int width, int height) override;

	void ChangeCuda(bool isCuda);

private:
	GLuint m_ParticleShader = Renderer::LoadShaderFromFile("shaders/particle.vert", "shaders/particle.frag");
	cudaError_t InitCUDA();

	glm::vec4 m_ClearColor = Config::CLEAR_COLOR;
	glm::vec4 m_ParticleColor = Config::PARTICLE_COLOR;
	uint32_t m_ParticleCount = Config::PARTICLE_COUNT;
	float m_ParticleRadius = Config::PARTICLE_RADIUS;
	int m_Substeps = Config::SUBSTEPS;

	ParticleSystem::Parameters m_Params;
	std::unique_ptr<ParticleSystem> m_Particles;
	std::unique_ptr<InstancedParticles> m_InstancedParticles;
	std::unique_ptr<ParticleSolver> m_Solver;
	std::unique_ptr<Grid> m_Grid;

	bool m_IsPaused;
	bool m_IsCuda = true;
	bool m_VSync;
	bool m_IsWaterfall = false;
	bool m_RandomColor = false;

	int m_Selecteditem = Config::STARTING_PRESET;
	uint32_t m_WaterfallRows;
	float m_WaterfallDelay;
	float m_WaterfalVelocity = Config::WATERFALL_VELOCITY;
	float m_WaterfallAccumulator = 0.0f;
};

