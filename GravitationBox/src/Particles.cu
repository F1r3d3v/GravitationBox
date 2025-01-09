#include "Particles.h"  
#include "Log.h"
#include "cuda_helper.h"  
#include <device_launch_parameters.h>  
#include <curand_kernel.h>  

__global__ void randomParticlesKernel(float *posX, float *posY, float *velX, float *velY, float *mass, glm::vec4 *color, size_t count, float radius, glm::ivec2 dim, uint64_t seed)
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

Particles *Particles::RandomCUDA(uint32_t count, float radius, glm::ivec2 dim)
{
	cudaError_t cudaStatus;
	Particles *p = new Particles(count, radius, true);
	time_t seed = time(NULL);

	// Launch kernel
	randomParticlesKernel << <BLOCKS_PER_GRID(count), THREADS_PER_BLOCK >> > (p->PosX, p->PosY, p->VelX, p->VelY, p->Mass, p->Color, count, radius, dim, seed);
	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		Log::Error("CUDA Error " + std::to_string(cudaStatus) + ": " + cudaGetErrorString(cudaStatus) + ". In file '" + __FILE__ + "' on line " + std::to_string(__LINE__));
		return nullptr;
	}

	return p;
}

Particles *Particles::RandomCircleCUDA(uint32_t count, float radius, glm::ivec2 dim)
{
	Particles *p = RandomCircleCPU(count, radius, dim);
	Particles *pGPU = new Particles(p->Count, radius, true);
	cudaMemcpy(pGPU->PosX, p->PosX, pGPU->Count * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(pGPU->PosY, p->PosY, pGPU->Count * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(pGPU->VelX, p->VelX, pGPU->Count * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(pGPU->VelY, p->VelY, pGPU->Count * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(pGPU->Mass, p->Mass, pGPU->Count * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(pGPU->Color, p->Color, pGPU->Count * sizeof(glm::vec4), cudaMemcpyHostToDevice);
	delete p;
	return pGPU;
}

Particles *Particles::RandomBoxCUDA(uint32_t count, float radius, glm::ivec2 dim)
{
	Particles *p = RandomBoxCPU(count, radius, dim);
	Particles *pGPU = new Particles(p->Count, radius, true);
	cudaMemcpy(pGPU->PosX, p->PosX, pGPU->Count * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(pGPU->PosY, p->PosY, pGPU->Count * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(pGPU->VelX, p->VelX, pGPU->Count * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(pGPU->VelY, p->VelY, pGPU->Count * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(pGPU->Mass, p->Mass, pGPU->Count * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(pGPU->Color, p->Color, pGPU->Count * sizeof(glm::vec4), cudaMemcpyHostToDevice);
	delete p;
	return pGPU;
}

Particles *Particles::WaterfallCUDA(uint32_t count, float radius, glm::ivec2 dim, float velocity, int rows)
{
	Particles *p = WaterfallCPU(count, radius, dim, velocity, rows);
	Particles *pGPU = new Particles(count, radius, true);
	cudaMemcpy(pGPU->PosX, p->PosX, pGPU->Count * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(pGPU->PosY, p->PosY, pGPU->Count * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(pGPU->VelX, p->VelX, pGPU->Count * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(pGPU->VelY, p->VelY, pGPU->Count * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(pGPU->Mass, p->Mass, pGPU->Count * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(pGPU->Color, p->Color, pGPU->Count * sizeof(glm::vec4), cudaMemcpyHostToDevice);
	pGPU->SetCount(0);
	delete p;
	return pGPU;
}
