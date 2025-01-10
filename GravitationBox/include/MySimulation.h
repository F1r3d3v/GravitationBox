#pragma once
#include "Simulation.h"
#include "Particles.h"
#include "VerletSolver.h"
#include "InstancedParticles.h"
#include <imgui.h>

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

	cudaError_t ChangeCuda(bool isCuda);

private:
	unsigned int m_ParticleShader = Renderer::LoadShaderFromFile("shaders/particle.vert", "shaders/particle.frag");
	cudaError_t InitCUDA();

	glm::vec4 m_ClearColor = glm::vec4(100.0f / 255.0f, 149.0f / 255.0f, 237.0f / 255.0f, 1.0f);
	glm::vec4 m_ParticleColor = glm::vec4(40.0f / 255.0f, 12.0f / 255.0f, 221.0f / 255.0f, 1.0f);
	uint32_t m_ParticleCount = 100000;
	float m_ParticleRadius = 1.5f;
	int m_Substeps = 1;

	Particles *m_ParticlesCPU, *m_ParticlesCUDA;
	InstancedParticles *m_InstancedParticles;
	Grid *m_Grid;
	VerletSolver *m_Solver;
	VerletSolver::SimulationParams m_Params;

	bool m_IsPaused;
	bool m_IsCuda = true;
	bool m_VSync;
	bool m_IsWaterfall = false;
	bool m_RandomColor = false;

	int m_Selecteditem = 3;
	int m_WaterfallRows;
	float m_WaterfallDelay;
	float m_WaterfalVelocity = 20.0f;
	float m_WaterfallAccumulator = 0.0f;
};

