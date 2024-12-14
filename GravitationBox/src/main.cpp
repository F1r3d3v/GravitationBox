#include <iostream>
#include "Window.h"

#define WIDTH 1600
#define HEIGHT 900

int main(int argc, char* argv[])
{
	try
	{
		Window window(WIDTH, HEIGHT, "Gravitation Box");
		glViewport(0, 0, WIDTH, HEIGHT);
		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);

		while (!glfwWindowShouldClose(window.GetHandle()))
		{
			glfwPollEvents();
			glClear(GL_COLOR_BUFFER_BIT);
			glfwSwapBuffers(window.GetHandle());
		}
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << std::endl;
	}

	return EXIT_SUCCESS;
}