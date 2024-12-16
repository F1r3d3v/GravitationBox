#pragma once

#include <stdexcept>
#include <string>

struct GLFWwindow;

class Window
{

public:
	Window(int Width, int Height, std::string Title, bool VSync = true);
	~Window();

	GLFWwindow* GetHandle() const { return m_Handle; }
	std::string GetTitle() const { return m_Title; }
private:
	GLFWwindow* m_Handle;
	std::string m_Title;
};

