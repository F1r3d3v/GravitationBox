#include <iostream>
#include "MySimulation.h"

#ifdef _WIN32
#include <windows.h>
int WINAPI WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPSTR lpCmdLine, _In_ int nCmdShow)
#else
int main(int argc, char **argv)
#endif
{
	try
	{
		MySimulation simulation = MySimulation("Gravitation Box", 1600, 900);
		simulation.Run();
	}
	catch (const std::exception &e)
	{
		std::cerr << e.what() << std::endl;
		getchar();
	}

	return EXIT_SUCCESS;
}