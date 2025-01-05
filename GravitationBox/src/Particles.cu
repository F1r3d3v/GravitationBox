#include "Particles.h"  
#include "Log.h"
#include "cuda_helper.h"  
#include <device_launch_parameters.h>  
#include <curand_kernel.h>  

__global__ void randomParticlesKernel(float *posX, float *posY, float *velX, float *velY, float *mass, glm::vec4 *color, size_t count, float radius, glm::ivec2 dim, unsigned long seed)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	curandState state;
	curand_init(seed, idx, 0, &state);

	float sqrt = sqrtf(count);
	int gridWidth = sqrt;
	int gridHeight = ceilf(sqrt);
	float cellWidth = (float)dim.x / gridWidth;
	float cellHeight = (float)dim.y / gridHeight;

	for (; idx < count; idx += stride)
	{
		int row = idx / gridWidth;
		int col = idx % gridWidth;
		int baseX = col * cellWidth;
		int baseY = row * cellHeight;

		posX[idx] = baseX + curand_uniform(&state) * (cellWidth - 2 * radius) + radius;
		posY[idx] = baseY + curand_uniform(&state) * (cellHeight - 2 * radius) + radius;
		velX[idx] = curand_uniform(&state) * 2 * Config::RAND_PARTICLE_VELOCITY_MAX - Config::RAND_PARTICLE_VELOCITY_MAX;
		velY[idx] = curand_uniform(&state) * 2 * Config::RAND_PARTICLE_VELOCITY_MAX - Config::RAND_PARTICLE_VELOCITY_MAX;
		float t = curand_uniform(&state);
		mass[idx] = t * Config::PARTICLE_MASS_MAX + (1.0 - t) * Config::PARTICLE_MASS_MIN;
		color[idx] = glm::vec4(curand_uniform(&state), curand_uniform(&state), curand_uniform(&state), 1.0f);
	}
}

Particles *Particles::RandomCUDA(size_t count, float radius, glm::ivec2 dim)
{
	cudaError_t cudaStatus;
	Particles *p = new Particles(count, radius, true);
	unsigned long seed = time(NULL);

	// Launch kernel
	randomParticlesKernel << <BLOCKS_PER_GRID(count), THREADS_PER_BLOCK >> > (p->PosX, p->PosY, p->VelX, p->VelY, p->Mass, p->Color, count, radius, dim, seed);
	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		Log::Error("CUDA Error " + std::to_string(cudaStatus) + ": " + cudaGetErrorString(cudaStatus) + ". In file '" + __FILE__ + "' on line " + std::to_string(__LINE__));
		return nullptr;
	}

	p->InitDrawingData();

	return p;
}
