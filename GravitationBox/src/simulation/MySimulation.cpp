#include "MySimulation.h"
#include "engine/Log.h"
#include "engine/Renderer.h"
#include "engine/Input.h"
#include "Config.h"
#include "utils/cuda_helper.h"
#include "cuda/CudaGrid.h"
#include "cuda/CudaVerletSolver.h"
#include "cuda/CudaParticleSystem.h"
#include "cuda/CudaInstancedParticles.h"
#include "cpu/CpuGrid.h"
#include "cpu/CpuVerletSolver.h"
#include "cpu/CpuParticleSystem.h"
#include "cpu/CpuInstancedParticles.h"
#include <glm.hpp>

MySimulation::MySimulation(std::string title, int width, int height)
	: Simulation(title, width, height)
{
	InitCUDA();
	SetVSync(false);
	m_VSync = false;
}

MySimulation::~MySimulation()
{
	CUDA_CHECK_NR(cudaDeviceReset());
}

void MySimulation::OnStart()
{
	glm::ivec2 size = GetViewport();
	m_IsPaused = true;
	m_IsWaterfall = false;
	m_WaterfallRows = static_cast<uint32_t>(1.0 / 4.0 * (size.y / (2.0 * m_ParticleRadius)));
	m_WaterfallDelay = 3 * m_ParticleRadius / m_WaterfalVelocity;
	m_Params.Radius = m_ParticleRadius;

	if (m_IsCuda)
	{
		m_Grid = std::make_unique<CudaGrid>(size, 2 * m_ParticleRadius);
		m_Solver = std::make_unique<CudaVerletSolver>(m_Grid.get());
	}
	else
	{
		m_Grid = std::make_unique<CpuGrid>(size, 2 * m_ParticleRadius);
		m_Solver = std::make_unique<CpuVerletSolver>(m_Grid.get());
	}

	switch (m_Selecteditem)
	{
	case 0:
		m_Particles = std::unique_ptr<ParticleSystem>(ParticleSystem::CreateRandom(m_ParticleCount, m_ParticleRadius, size, m_Solver.get(), m_IsCuda));
		break;
	case 1:
		m_Particles = std::unique_ptr<ParticleSystem>(ParticleSystem::CreateCircle(m_ParticleCount, m_ParticleRadius, size, m_Solver.get(), m_IsCuda));
		break;
	case 2:
		m_Particles = std::unique_ptr<ParticleSystem>(ParticleSystem::CreateBox(m_ParticleCount, m_ParticleRadius, size, m_Solver.get(), m_IsCuda));
		break;
	case 3:
		m_IsWaterfall = true;
		m_Particles = std::unique_ptr<ParticleSystem>(ParticleSystem::CreateWaterfall(m_ParticleCount, m_ParticleRadius, size, m_WaterfalVelocity, m_WaterfallRows, m_Solver.get(), m_IsCuda));
		break;
	}

	if (m_IsCuda)
		m_InstancedParticles = std::make_unique<CudaInstancedParticles>(m_Particles.get(), m_ParticleShader);
	else
		m_InstancedParticles = std::make_unique<CpuInstancedParticles>(m_Particles.get(), m_ParticleShader);

	m_InstancedParticles->SetRandomColor(m_RandomColor);
	m_InstancedParticles->SetStillColor(m_ParticleColor);

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

	if (Input::IsKeyPressed(GLFW_KEY_V))
		SetVSync(m_VSync = !m_VSync);

	if (!m_IsPaused)
	{
		if (m_IsWaterfall)
		{
			m_WaterfallAccumulator += m_Params.Timestep;
			if (m_WaterfallAccumulator >= m_WaterfallDelay)
			{
				m_WaterfallAccumulator -= m_WaterfallDelay;

				m_Particles->Count = std::min(m_Particles->TotalCount, m_Particles->Count + m_WaterfallRows);
			}
		}

		ParticleSystem::Parameters params = m_Params;
		params.Timestep /= m_Substeps;
		m_Particles->SetParameters(params);
		m_Particles->Update();
	}
}

void MySimulation::OnRender(Renderer *renderer)
{
	renderer->Clear(m_ClearColor);
	m_InstancedParticles->UpdateParticleInstances();
	m_InstancedParticles->Draw();
}

void MySimulation::OnCleanup()
{
	Log::Info("Simulation Ended");
	m_InstancedParticles.reset();
	m_Particles.reset();
	m_Solver.reset();
	m_Grid.reset();
}

void MySimulation::OnResize(int width, int height)
{
	if (width == 0 && height == 0) return;
	m_Grid->Resize(glm::ivec2(width, height), 2 * m_ParticleRadius);
	m_Params.DimX = width;
	m_Params.DimY = height;
}

void MySimulation::ChangeCuda(bool isCuda)
{
	m_IsCuda = isCuda;

	if (m_IsCuda)
	{
		m_Grid.reset(new CudaGrid(GetViewport(), 2 * m_ParticleRadius));
		m_Solver.reset(new CudaVerletSolver(m_Grid.get()));
		m_Particles.reset(CudaParticleSystem::CreateFromCPU(dynamic_cast<CpuParticleSystem *>(m_Particles.get()), m_Solver.get()));
		m_InstancedParticles.reset(new CudaInstancedParticles(m_Particles.get(), m_ParticleShader));
	}
	else
	{
		m_Grid.reset(new CpuGrid(GetViewport(), 2 * m_ParticleRadius));
		m_Solver.reset(new CpuVerletSolver(m_Grid.get()));
		m_Particles.reset(CpuParticleSystem::CreateFromCuda(dynamic_cast<CudaParticleSystem *>(m_Particles.get()), m_Solver.get()));
		m_InstancedParticles.reset(new CpuInstancedParticles(m_Particles.get(), m_ParticleShader));
	}

	m_InstancedParticles->SetRandomColor(m_RandomColor);
	m_InstancedParticles->SetStillColor(m_ParticleColor);
}

void MySimulation::OnImGuiRender()
{
	ImGuiIO &io = ImGui::GetIO(); (void)io;

	ImGui::SetNextWindowSize(ImVec2(448.0, 0.0), ImGuiCond_Once);
	ImGui::SetNextWindowCollapsed(false, ImGuiCond_Once);
	ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x, 0), ImGuiCond_Always, ImVec2(1.0f, 0.0f));
	ImGui::Begin("Control Panel", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize);

	ImGui::Checkbox("Pause Simulation", &m_IsPaused);
	ImGui::SameLine();
	if (ImGui::Checkbox("Use CUDA", &m_IsCuda))
		ChangeCuda(m_IsCuda);
	ImGui::SameLine();
	if (ImGui::Checkbox("VSync", &m_VSync))
		SetVSync(m_VSync);

	ImGui::Spacing();

	ImGui::PushItemWidth(100.0f);
	ImGui::InputScalar("Count", ImGuiDataType_U32, &m_ParticleCount);
	if (m_ParticleCount < 1)
		m_ParticleCount = 1;
	ImGui::SameLine();
	ImGui::InputFloat("Radius", &m_ParticleRadius);
	ImGui::PopItemWidth();
	if (ImGui::Button("Reset simulation"))
		Reset();

	ImGui::Spacing();

	if (ImGui::CollapsingHeader("Scene"))
	{
		if (ImGui::Checkbox("Random Color", &m_RandomColor))
		{
			m_InstancedParticles->SetRandomColor(m_RandomColor);
			if (!m_RandomColor)
			{
				m_InstancedParticles->SetStillColor(m_ParticleColor);
			}
		}

		if (!m_RandomColor)
		{
			if (ImGui::ColorEdit3("Particle Color", &m_ParticleColor.x))
			{
				m_InstancedParticles->SetStillColor(m_ParticleColor);
			}
		}

		ImGui::ColorEdit3("Background Color", &m_ClearColor.x);
		const char *items[]{ "Random","Circle","Box","Waterfall" };
		ImGui::Combo("Preset", &m_Selecteditem, items, IM_ARRAYSIZE(items));
		if (m_Selecteditem == 3)
			ImGui::InputFloat("Velocity", &m_WaterfalVelocity);
	}

	ImGui::Spacing();

	if (ImGui::CollapsingHeader("Parameters"))
	{
		ImGui::InputInt("Substeps", &m_Substeps);
		if (m_Substeps < 1)
			m_Substeps = 1;
		ImGui::InputFloat("Timestep", &m_Params.Timestep);
		ImGui::InputFloat("Gravity", &m_Params.Gravity);
		ImGui::InputFloat("Walls Dampening", &m_Params.WallDampening);
		ImGui::InputFloat("Walls Friction", &m_Params.WallFriction);
		ImGui::InputFloat("Particles Dampening", &m_Params.ParticlesDampening);
		ImGui::InputFloat("Particles Stiffness", &m_Params.ParticlesStiffness);
		ImGui::InputFloat("Particles Friction", &m_Params.ParticlesFriction);
	}

	ImGui::Spacing();

	if (ImGui::CollapsingHeader("Info"))
	{
		ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
		ImGui::Text("Window size: %.0f x %.0f", io.DisplaySize.x, io.DisplaySize.y);
		ImGui::Text("Particle count: %zd", m_Particles->Count);
		ImGui::Text("Particle mass range: %.1f - %.1f", Config::PARTICLE_MASS_MIN, Config::PARTICLE_MASS_MAX);
	}

	ImGui::Spacing();

	if (ImGui::CollapsingHeader("Controls"))
	{
		ImGui::BulletText("Press 'P' to pause/resume simulation");
		ImGui::BulletText("Press 'R' to reset simulation");
		ImGui::BulletText("Press 'C' to toggle between CPU and CUDA");
		ImGui::BulletText("Press 'V' to toggle VSync");
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
