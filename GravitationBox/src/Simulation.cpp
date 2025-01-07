#include "Simulation.h"
#include "Log.h"

#include <glad/gl.h>
#include <GLFW/glfw3.h>

#include "ImGUI_Impl.h"
#include "Window.h"
#include "Renderer.h"
#include "Input.h"

Simulation::Simulation(std::string title, int width, int height)
{
	Initialize();

	Log::Info("Creating window");
	m_Window = new Window(width, height, title);

	// Register resize callback
	glfwSetWindowUserPointer(m_Window->GetHandle(), this);
	glfwSetFramebufferSizeCallback(m_Window->GetHandle(), [](GLFWwindow *window, int width, int height)
		{
			Simulation *sim = static_cast<Simulation *>(glfwGetWindowUserPointer(window));
			sim->OnResize(width, height);
		});

	glfwMakeContextCurrent(m_Window->GetHandle());
	glfwSwapInterval(m_IsVsync);

	if (!gladLoadGL(glfwGetProcAddress))
		throw std::runtime_error("Failed to initialize GLAD");

	Log::Info("Initializing renderer");
	m_Renderer = new Renderer(m_Window);
	Log::Info("Renderer initialized");

	// Setup Dear ImGui context
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();

	Log::Info("Initializing input");
	Input::Init(m_Window->GetHandle());

	Log::Info("Simulation created");
}

Simulation::~Simulation()
{
	delete m_Window;
	delete m_Renderer;
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
	Log::Shutdown();
}

void Simulation::Run()
{
	ImGuiIO &io = ImGui::GetIO(); (void)io;

	OnStart();

	ImGuiStyle &style = ImGui::GetStyle();
	if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
	{
		style.WindowRounding = 0.0f;
		style.Colors[ImGuiCol_WindowBg].w = 1.0f;
	}

	// Setup Platform/Renderer backends
	ImGui_ImplGlfw_InitForOpenGL(m_Window->GetHandle(), true);
	ImGui_ImplOpenGL3_Init("#version 330 core");

	float currentTime = static_cast<float>(glfwGetTime());

	while (!glfwWindowShouldClose(m_Window->GetHandle()))
	{
		float newTime = static_cast<float>(glfwGetTime());
		float deltaTime = newTime - currentTime;
		currentTime = newTime;

		Input::Update();
		deltaTime = std::min(deltaTime, 1.0f/30.0f);
		OnUpdate(deltaTime);

		glfwPollEvents();
		if (glfwGetWindowAttrib(m_Window->GetHandle(), GLFW_ICONIFIED) != 0)
		{
			ImGui_ImplGlfw_Sleep(10);
			continue;
		}

		// Start the Dear ImGui frame
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();
		OnImGuiRender();
		ImGui::Render();

		int display_w, display_h;
		glfwGetFramebufferSize(m_Window->GetHandle(), &display_w, &display_h);
		glViewport(0, 0, display_w, display_h);

		OnRender(m_Renderer);

		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
		if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
		{
			GLFWwindow *backup_current_context = glfwGetCurrentContext();
			ImGui::UpdatePlatformWindows();
			ImGui::RenderPlatformWindowsDefault();
			glfwMakeContextCurrent(backup_current_context);
		}

		glfwSwapBuffers(m_Window->GetHandle());
	}

	OnCleanup();
}

void Simulation::Reset()
{
	OnCleanup();
	OnStart();
}

void Simulation::Close()
{
	glfwSetWindowShouldClose(m_Window->GetHandle(), true);
}

void Simulation::SetVSync(bool enabled)
{
	m_IsVsync = enabled;
	glfwSwapInterval(enabled);
}

void Simulation::Initialize()
{
	Log::Init();

	if (!glfwInit())
		throw std::runtime_error("Failed to initialize GLFW");

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
}
