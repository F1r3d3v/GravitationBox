#pragma once

#include <glad/gl.h>
#include <GLFW/glfw3.h>

#include <stdexcept>
#include <string>

class Window
{
public:
	Window(int Width, int Height, std::string Title, bool VSync = true);
	~Window();

	GLFWwindow* GetHandle() const { return m_Handle; }
private:
	GLFWwindow* m_Handle;
	int m_Width;
	int m_Height;
	std::string m_Title;
};

