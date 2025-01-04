#pragma once
#include "Simulation.h"
#include "Particles.h"
#include "VerletSolver.h"
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
	cudaError_t InitCUDA();

	ImVec4 clear_color = ImVec4(100.0f / 255.0f, 149.0f / 255.0f, 237.0f / 255.0f, 1.0f);
	Particles *m_ParticlesCPU, *m_ParticlesCUDA;
	Grid *m_Grid;
	VerletSolver *m_Solver;
	bool m_IsPaused;
	bool m_IsCuda = true;
};

