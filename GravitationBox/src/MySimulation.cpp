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
	m_VSync = GetVSync();
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
	m_WaterfallRows = 1.0 / 4.0 * (size.y / (2 * m_ParticleRadius));
	m_WaterfallDelay = 3 * m_ParticleRadius / m_WaterfalVelocity;
	m_Params.Radius = m_ParticleRadius;
	m_Grid = new Grid(make_int2(size.x, size.y), 2 * m_ParticleRadius, m_IsCuda);
	if (m_IsCuda)
	{
		switch (m_Selecteditem)
		{
		case 0:
			m_ParticlesCUDA = Particles::RandomCUDA(m_ParticleCount, m_ParticleRadius, size);
			break;
		case 1:
			m_ParticlesCUDA = Particles::RandomCircleCUDA(m_ParticleCount, m_ParticleRadius, size);
			break;
		case 2:
			m_ParticlesCUDA = Particles::RandomBoxCUDA(m_ParticleCount, m_ParticleRadius, size);
			break;
		case 3:
			m_IsWaterfall = true;
			m_ParticlesCUDA = Particles::WaterfallCUDA(m_ParticleCount, m_ParticleRadius, size, m_WaterfalVelocity, m_WaterfallRows);
			break;
		}
		m_ParticlesCPU = new Particles(m_ParticleCount, m_ParticleRadius, false);
		m_Solver = new VerletSolver(m_ParticlesCUDA, m_Grid);
	}
	else
	{
		switch (m_Selecteditem)
		{
		case 0:
			m_ParticlesCPU = Particles::RandomCPU(m_ParticleCount, m_ParticleRadius, size);
			break;
		case 1:
			m_ParticlesCPU = Particles::RandomCircleCPU(m_ParticleCount, m_ParticleRadius, size);
			break;
		case 2:
			m_ParticlesCPU = Particles::RandomBoxCPU(m_ParticleCount, m_ParticleRadius, size);
			break;
		case 3:
			m_IsWaterfall = true;
			m_ParticlesCPU = Particles::WaterfallCPU(m_ParticleCount, m_ParticleRadius, size, m_WaterfalVelocity, m_WaterfallRows);
			break;
		}
		m_ParticlesCUDA = new Particles(m_ParticleCount, m_ParticleRadius, true);
		m_Solver = new VerletSolver(m_ParticlesCPU, m_Grid);
	}
	m_ParticlesCPU->SetRandomColor(m_RandomColor);
	m_ParticlesCUDA->SetRandomColor(m_RandomColor);
	m_ParticlesCPU->SetStillColor(m_ParticleColor);
	m_ParticlesCUDA->SetStillColor(m_ParticleColor);

	m_InstancedParticles = new InstancedParticles(m_ParticlesCPU, m_ParticleShader);
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

				m_ParticlesCUDA->SetCount(m_ParticlesCUDA->Count + m_WaterfallRows);
				m_ParticlesCPU->SetCount(m_ParticlesCPU->Count + m_WaterfallRows);
			}
		}
		VerletSolver::SimulationParams params = m_Params;
		params.Timestep /= m_Substeps;
		m_Solver->SetParams(params);
		for (int i = 0; i < m_Substeps; i++)
		{
			if (m_IsCuda)
			{
				m_Solver->VerletCuda();
			}
			else
			{
				m_Solver->VerletCPU();
			}
		}
	}
}

void MySimulation::OnRender(Renderer *renderer)
{
	renderer->Clear(m_ClearColor);
	if (m_IsCuda)
		m_ParticlesCUDA->DrawCUDA(renderer, m_InstancedParticles);
	else
		m_ParticlesCPU->DrawCPU(renderer, m_InstancedParticles);
}

void MySimulation::OnCleanup()
{
	Log::Info("Simulation Ended");
	delete m_ParticlesCPU;
	delete m_ParticlesCUDA;
	delete m_Grid;
	delete m_Solver;
	delete m_InstancedParticles;
}

void MySimulation::OnResize(int width, int height)
{
	if (width == 0 && height == 0) return;
	m_Grid->Resize(make_int2(width, height), 2 * m_ParticleRadius);
	m_Params.DimX = width;
	m_Params.DimY = height;
}

cudaError_t MySimulation::ChangeCuda(bool isCuda)
{
	m_IsCuda = isCuda;
	if (m_IsCuda)
	{
		// Copy data from CPU to GPU
		CUDA_CHECK(cudaMemcpy(m_ParticlesCUDA->PosX, m_ParticlesCPU->PosX, m_ParticlesCPU->TotalCount * sizeof(float), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(m_ParticlesCUDA->PosY, m_ParticlesCPU->PosY, m_ParticlesCPU->TotalCount * sizeof(float), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(m_ParticlesCUDA->VelX, m_ParticlesCPU->VelX, m_ParticlesCPU->TotalCount * sizeof(float), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(m_ParticlesCUDA->VelY, m_ParticlesCPU->VelY, m_ParticlesCPU->TotalCount * sizeof(float), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(m_ParticlesCUDA->Mass, m_ParticlesCPU->Mass, m_ParticlesCPU->TotalCount * sizeof(float), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(m_ParticlesCUDA->Color, m_ParticlesCPU->Color, m_ParticlesCPU->TotalCount * sizeof(glm::vec4), cudaMemcpyHostToDevice));
		m_ParticlesCUDA->SetCount(m_ParticlesCPU->Count);
		m_Solver->SetParticlesInstance(m_ParticlesCUDA);
	}
	else
	{
		// Copy data from GPU to CPU
		CUDA_CHECK(cudaMemcpy(m_ParticlesCPU->PosX, m_ParticlesCUDA->PosX, m_ParticlesCUDA->TotalCount * sizeof(float), cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaMemcpy(m_ParticlesCPU->PosY, m_ParticlesCUDA->PosY, m_ParticlesCUDA->TotalCount * sizeof(float), cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaMemcpy(m_ParticlesCPU->VelX, m_ParticlesCUDA->VelX, m_ParticlesCUDA->TotalCount * sizeof(float), cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaMemcpy(m_ParticlesCPU->VelY, m_ParticlesCUDA->VelY, m_ParticlesCUDA->TotalCount * sizeof(float), cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaMemcpy(m_ParticlesCPU->Mass, m_ParticlesCUDA->Mass, m_ParticlesCUDA->TotalCount * sizeof(float), cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaMemcpy(m_ParticlesCPU->Color, m_ParticlesCUDA->Color, m_ParticlesCUDA->TotalCount * sizeof(glm::vec4), cudaMemcpyDeviceToHost));
		m_ParticlesCPU->SetCount(m_ParticlesCUDA->Count);
		m_Solver->SetParticlesInstance(m_ParticlesCPU);
	}
	m_Grid->SetDevice(m_IsCuda);

	return cudaSuccess;
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
			m_ParticlesCPU->SetRandomColor(m_RandomColor);
			m_ParticlesCUDA->SetRandomColor(m_RandomColor);
			if (!m_RandomColor)
			{
				m_ParticlesCPU->SetStillColor(m_ParticleColor);
				m_ParticlesCUDA->SetStillColor(m_ParticleColor);
			}
		}

		if (!m_RandomColor)
		{
			if (ImGui::ColorEdit3("Particle Color", &m_ParticleColor.x))
			{
				m_ParticlesCUDA->SetStillColor(m_ParticleColor);
				m_ParticlesCPU->SetStillColor(m_ParticleColor);
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
		ImGui::InputFloat("Wall Dampening", &m_Params.WallDampening);
		ImGui::InputFloat("Particle Dampening", &m_Params.ParticleDampening);
		ImGui::InputFloat("Particle Stiffness", &m_Params.ParticleStiffness);
		ImGui::InputFloat("Particle Shear", &m_Params.ParticleShear);
	}

	ImGui::Spacing();

	if (ImGui::CollapsingHeader("Info"))
	{
		ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
		ImGui::Text("Window size: %.0f x %.0f", io.DisplaySize.x, io.DisplaySize.y);
		ImGui::Text("Particle count: %zd", m_IsCuda ? m_ParticlesCUDA->Count : m_ParticlesCPU->Count);
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
