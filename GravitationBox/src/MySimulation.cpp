#include "MySimulation.h"
#include "Log.h"
#include "Renderer.h"
#include "Input.h"
#include <glm.hpp>
#include "Config.h"
#include "cuda_helper.h"

MySimulation::MySimulation(std::string title, int width, int height)
	: Simulation(title, width, height)
{
	InitCUDA();
}

MySimulation::~MySimulation()
{

}

void MySimulation::OnStart()
{
	m_IsPaused = true;
	glm::ivec2 size = GetViewport();
	GetRenderer()->InitializeParticleInstancing(Config::PARTICLE_COUNT);
	m_Grid = new Grid(make_int2(size.x, size.y), 2 * Config::PARTICLE_RADIUS, m_IsCuda);
	if (m_IsCuda)
	{
		switch (m_Selecteditem)
		{
		case 0:
			m_ParticlesCUDA = Particles::RandomCUDA(Config::PARTICLE_COUNT, Config::PARTICLE_RADIUS, size);
			break;
		case 1:
			//m_ParticlesCUDA = Particles::RandomCircleCUDA(Config::PARTICLE_COUNT, Config::PARTICLE_RADIUS, size);
			break;
		case 2:
			//m_ParticlesCUDA = Particles::RandomBoxCUDA(Config::PARTICLE_COUNT, Config::PARTICLE_RADIUS, size);
			break;
		}
		m_ParticlesCPU = new Particles(m_ParticlesCUDA->Count, m_ParticlesCUDA->Radius, false);
		m_Solver = new VerletSolver(m_ParticlesCUDA, m_Grid);
	}
	else
	{
		switch (m_Selecteditem)
		{
		case 0:
			m_ParticlesCPU = Particles::RandomCPU(Config::PARTICLE_COUNT, Config::PARTICLE_RADIUS, size);
			break;
		case 1:
			m_ParticlesCPU = Particles::RandomCircleCPU(Config::PARTICLE_COUNT, Config::PARTICLE_RADIUS, size);
			break;
		case 2:
			m_ParticlesCPU = Particles::RandomBoxCPU(Config::PARTICLE_COUNT, Config::PARTICLE_RADIUS, size);
			break;
		}
		m_ParticlesCUDA = new Particles(m_ParticlesCPU->Count, m_ParticlesCPU->Radius, true);
		m_Solver = new VerletSolver(m_ParticlesCPU, m_Grid);
	}
	Log::Info("Simulation initialized");
}

void MySimulation::OnUpdate(float deltaTime)
{
	if (Input::IsKeyPressed(GLFW_KEY_ESCAPE))
		Close();

	if (Input::IsKeyPressed(GLFW_KEY_P))
	{
		if (m_IsPaused = !m_IsPaused)
			Log::Info("Simulation paused");
		else
			Log::Info("Simulation resumed");
	}

	if (Input::IsKeyPressed(GLFW_KEY_C))
		ChangeCuda(m_IsCuda = !m_IsCuda);

	if (Input::IsKeyPressed(GLFW_KEY_R))
		Reset();

	if (!m_IsPaused)
	{
		float substep_dt = deltaTime / Config::SUBSTEPS;
		for (int i = 0; i < Config::SUBSTEPS; i++)
		{
			if (m_IsCuda)
				m_Solver->VerletCuda(substep_dt);
			else
				m_Solver->VerletCPU(substep_dt);
		}
	}
}

void MySimulation::OnRender(Renderer *renderer)
{
	renderer->Clear(Config::CLEAR_COLOR);
	if (m_IsCuda)
		m_ParticlesCUDA->DrawCUDA(renderer);
	else
		m_ParticlesCPU->DrawCPU(renderer);
}

void MySimulation::OnCleanup()
{
	Log::Info("Simulation Ended");
	delete m_ParticlesCPU;
	delete m_ParticlesCUDA;
	delete m_Grid;
	delete m_Solver;
	GetRenderer()->UninitializeParticleInstancing();
}

void MySimulation::OnResize(int width, int height)
{
	m_Grid->Resize(make_int2(width, height), 2*Config::PARTICLE_RADIUS);
}

cudaError_t MySimulation::ChangeCuda(bool isCuda)
{
	m_IsCuda = isCuda;
	if (m_IsCuda)
	{
		// Copy data from CPU to GPU
		CUDA_CHECK(cudaMemcpy(m_ParticlesCUDA->PosX, m_ParticlesCPU->PosX, m_ParticlesCPU->Count * sizeof(float), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(m_ParticlesCUDA->PosY, m_ParticlesCPU->PosY, m_ParticlesCPU->Count * sizeof(float), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(m_ParticlesCUDA->VelX, m_ParticlesCPU->VelX, m_ParticlesCPU->Count * sizeof(float), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(m_ParticlesCUDA->VelY, m_ParticlesCPU->VelY, m_ParticlesCPU->Count * sizeof(float), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(m_ParticlesCUDA->Mass, m_ParticlesCPU->Mass, m_ParticlesCPU->Count * sizeof(float), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(m_ParticlesCUDA->Color, m_ParticlesCPU->Color, m_ParticlesCPU->Count * sizeof(glm::vec4), cudaMemcpyHostToDevice));
		m_Solver->SetParticlesInstance(m_ParticlesCUDA);
	}
	else
	{
		// Copy data from GPU to CPU
		CUDA_CHECK(cudaMemcpy(m_ParticlesCPU->PosX, m_ParticlesCUDA->PosX, m_ParticlesCUDA->Count * sizeof(float), cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaMemcpy(m_ParticlesCPU->PosY, m_ParticlesCUDA->PosY, m_ParticlesCUDA->Count * sizeof(float), cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaMemcpy(m_ParticlesCPU->VelX, m_ParticlesCUDA->VelX, m_ParticlesCUDA->Count * sizeof(float), cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaMemcpy(m_ParticlesCPU->VelY, m_ParticlesCUDA->VelY, m_ParticlesCUDA->Count * sizeof(float), cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaMemcpy(m_ParticlesCPU->Mass, m_ParticlesCUDA->Mass, m_ParticlesCUDA->Count * sizeof(float), cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaMemcpy(m_ParticlesCPU->Color, m_ParticlesCUDA->Color, m_ParticlesCUDA->Count * sizeof(glm::vec4), cudaMemcpyDeviceToHost));
		m_Solver->SetParticlesInstance(m_ParticlesCPU);
	}
	m_Grid->setDevice(m_IsCuda);

	return cudaSuccess;
}

void MySimulation::OnImGuiRender()
{
	ImGuiIO &io = ImGui::GetIO(); (void)io;

	ImGui::SetNextWindowSize(ImVec2(364.0, 0.0), ImGuiCond_Once);
	ImGui::SetNextWindowCollapsed(false, ImGuiCond_Once);
	ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x, 0), ImGuiCond_Always, ImVec2(1.0f, 0.0f));
	ImGui::Begin("Control Panel", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize);


	ImGui::Checkbox("Pause Simulation", &m_IsPaused);
	ImGui::SameLine();
	if (ImGui::Checkbox("Use CUDA", &m_IsCuda))
		ChangeCuda(m_IsCuda);
	ImGui::Spacing();
	ImGui::PushItemWidth(100.0f);
	ImGui::InputScalar("Count", ImGuiDataType_U32, &Config::PARTICLE_COUNT);
	if (Config::PARTICLE_COUNT < 1)
		Config::PARTICLE_COUNT = 1;
	ImGui::SameLine();
	ImGui::InputFloat("Radius", &Config::PARTICLE_RADIUS);
	ImGui::PopItemWidth();
	if (ImGui::Button("Reset simulation"))
		Reset();
	ImGui::Spacing();
	if (ImGui::CollapsingHeader("Scene"))
	{
		ImGui::ColorEdit3("Background Color", &Config::CLEAR_COLOR.x);
		const char *items[]{ "Random","Circle","Box" };
		ImGui::Combo("Preset", &m_Selecteditem, items, IM_ARRAYSIZE(items));
	}
	ImGui::Spacing();
	if (ImGui::CollapsingHeader("Info"))
	{
		ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
		ImGui::Text("Window size: %.0f x %.0f", io.DisplaySize.x, io.DisplaySize.y);
		ImGui::Text("Particle count: %zd", m_ParticlesCPU->Count);
		ImGui::Text("Particle radius: %.1f", m_ParticlesCPU->Radius);
		ImGui::Text("Particle mass range: %.1f - %.1f", Config::PARTICLE_MASS_MIN, Config::PARTICLE_MASS_MAX);
	}
	ImGui::Spacing();
	if (ImGui::CollapsingHeader("Controls"))
	{
		ImGui::BulletText("Press 'R' to reset simulation");
		ImGui::BulletText("Press 'C' to toggle between CPU and CUDA");
		ImGui::BulletText("Press 'P' to pause/resume simulation");
		ImGui::BulletText("Press 'ESC' to close the application");
	}

	ImGui::SetWindowSize(ImVec2(ImGui::GetWindowWidth(), 0.0));
	ImGui::End();
}

cudaError_t MySimulation::InitCUDA()
{
	Log::Info("Initializing CUDA");

	// Initialize CUDA device
	CUDA_CHECK(cudaSetDevice(0));

	Log::Info("CUDA initialized successfully");
	return cudaSuccess;
}
