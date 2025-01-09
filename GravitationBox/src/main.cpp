#include <iostream>
#include "MySimulation.h"

int WinMain()
{
	//try
	//{
	//	MySimulation simulation = MySimulation("Gravitation Box", 1600, 900);
	//	simulation.Run();
	//}
	//catch (const std::exception &e)
	//{
	//	std::cerr << e.what() << std::endl;
	//	getchar();
	//}

	MySimulation simulation = MySimulation("Gravitation Box", 1600, 900);
	simulation.Run();

	return EXIT_SUCCESS;
}