#include "Window.h"
#include <glad/gl.h>
#include <GLFW/glfw3.h>

Window::Window(int Width, int Height, std::string Title, bool VSync)
	: m_Title(Title)
{
	if (!glfwInit())
		throw std::runtime_error("Failed to initialize GLFW");

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	m_Handle = glfwCreateWindow(Width, Height, Title.c_str(), nullptr, nullptr);
	if (!m_Handle)
		throw std::runtime_error("Failed to create window");

	glfwMakeContextCurrent(m_Handle);
	glfwSwapInterval(VSync);
}

Window::~Window()
{
	glfwDestroyWindow(m_Handle);
	m_Handle = nullptr;
	glfwTerminate();
}
