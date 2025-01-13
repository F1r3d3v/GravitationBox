#include "cuda/CudaParticleSystem.h"
#include "cpu/CpuParticleSystem.h"
#include "utils/cuda_helper.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>

CudaParticleSystem::CudaParticleSystem(uint32_t count, float radius, ParticleSolver *solver)
	: ParticleSystem(count, radius, solver)
{
	CUDA_CHECK_NR(cudaMalloc(&PosX, Count * sizeof(float)));
	CUDA_CHECK_NR(cudaMalloc(&SortedPosX, Count * sizeof(float)));
	CUDA_CHECK_NR(cudaMalloc(&PosY, Count * sizeof(float)));
	CUDA_CHECK_NR(cudaMalloc(&SortedPosY, Count * sizeof(float)));
	CUDA_CHECK_NR(cudaMalloc(&VelX, Count * sizeof(float)));
	CUDA_CHECK_NR(cudaMalloc(&SortedVelX, Count * sizeof(float)));
	CUDA_CHECK_NR(cudaMalloc(&VelY, Count * sizeof(float)));
	CUDA_CHECK_NR(cudaMalloc(&SortedVelY, Count * sizeof(float)));
	CUDA_CHECK_NR(cudaMalloc(&ForceX, Count * sizeof(float)));
	CUDA_CHECK_NR(cudaMemset(ForceX, 0, Count * sizeof(float)));
	CUDA_CHECK_NR(cudaMalloc(&SortedForceX, Count * sizeof(float)));
	CUDA_CHECK_NR(cudaMalloc(&ForceY, Count * sizeof(float)));
	CUDA_CHECK_NR(cudaMemset(ForceY, 0, Count * sizeof(float)));
	CUDA_CHECK_NR(cudaMalloc(&SortedForceY, Count * sizeof(float)));
	CUDA_CHECK_NR(cudaMalloc(&Mass, Count * sizeof(float)));
	CUDA_CHECK_NR(cudaMalloc(&Color, Count * sizeof(glm::vec4)));
}

CudaParticleSystem::~CudaParticleSystem()
{
	CUDA_CHECK_NR(cudaFree(PosX));
	CUDA_CHECK_NR(cudaFree(SortedPosX));
	CUDA_CHECK_NR(cudaFree(PosY));
	CUDA_CHECK_NR(cudaFree(SortedPosY));
	CUDA_CHECK_NR(cudaFree(VelX));
	CUDA_CHECK_NR(cudaFree(SortedVelX));
	CUDA_CHECK_NR(cudaFree(VelY));
	CUDA_CHECK_NR(cudaFree(SortedVelY));
	CUDA_CHECK_NR(cudaFree(ForceX));
	CUDA_CHECK_NR(cudaFree(SortedForceX));
	CUDA_CHECK_NR(cudaFree(ForceY));
	CUDA_CHECK_NR(cudaFree(SortedForceY));
	CUDA_CHECK_NR(cudaFree(Mass));
	CUDA_CHECK_NR(cudaFree(Color));
}

__global__ void RandomParticlesKernel(float *posX, float *posY, float *velX, float *velY, float *mass, glm::vec4 *color, size_t count, float radius, glm::ivec2 dim, uint64_t seed)
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

CudaParticleSystem *CudaParticleSystem::CreateRandom(uint32_t count, float radius, glm::ivec2 dim, ParticleSolver *solver)
{
	cudaError_t cudaStatus;
	CudaParticleSystem *p = new CudaParticleSystem(count, radius, solver);
	time_t seed = time(NULL);

	RandomParticlesKernel << <BLOCKS_PER_GRID(count), THREADS_PER_BLOCK >> > (p->PosX, p->PosY, p->VelX, p->VelY, p->Mass, p->Color, count, radius, dim, seed);
	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		Log::Error("CUDA Error " + std::to_string(cudaStatus) + ": " + cudaGetErrorString(cudaStatus) + ". In file '" + __FILE__ + "' on line " + std::to_string(__LINE__));
		return nullptr;
	}

	return p;
}

CudaParticleSystem *CudaParticleSystem::CreateCircle(uint32_t count, float radius, glm::ivec2 dim, ParticleSolver *solver)
{
	CpuParticleSystem *p = CpuParticleSystem::CreateCircle(count, radius, dim, nullptr);
	CudaParticleSystem *pGPU = new CudaParticleSystem(count, radius, solver);
	cudaMemcpy(pGPU->PosX, p->PosX, count * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(pGPU->PosY, p->PosY, count * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(pGPU->VelX, p->VelX, count * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(pGPU->VelY, p->VelY, count * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(pGPU->Mass, p->Mass, count * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(pGPU->Color, p->Color, count * sizeof(glm::vec4), cudaMemcpyHostToDevice);
	pGPU->Count = p->Count;
	delete p;
	return pGPU;
}

CudaParticleSystem *CudaParticleSystem::CreateBox(uint32_t count, float radius, glm::ivec2 dim, ParticleSolver *solver)
{
	CpuParticleSystem *p = CpuParticleSystem::CreateBox(count, radius, dim, nullptr);
	CudaParticleSystem *pGPU = new CudaParticleSystem(count, radius, solver);
	cudaMemcpy(pGPU->PosX, p->PosX, count * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(pGPU->PosY, p->PosY, count * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(pGPU->VelX, p->VelX, count * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(pGPU->VelY, p->VelY, count * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(pGPU->Mass, p->Mass, count * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(pGPU->Color, p->Color, count * sizeof(glm::vec4), cudaMemcpyHostToDevice);
	pGPU->Count = p->Count;
	delete p;
	return pGPU;
}

CudaParticleSystem *CudaParticleSystem::CreateWaterfall(uint32_t count, float radius, glm::ivec2 dim, float velocity, int rows, ParticleSolver *solver)
{
	CpuParticleSystem *p = CpuParticleSystem::CreateWaterfall(count, radius, dim, velocity, rows, nullptr);
	CudaParticleSystem *pGPU = new CudaParticleSystem(count, radius, solver);
	cudaMemcpy(pGPU->PosX, p->PosX, count * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(pGPU->PosY, p->PosY, count * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(pGPU->VelX, p->VelX, count * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(pGPU->VelY, p->VelY, count * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(pGPU->Mass, p->Mass, count * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(pGPU->Color, p->Color, count * sizeof(glm::vec4), cudaMemcpyHostToDevice);
	pGPU->Count = p->Count;
	delete p;
	return pGPU;
}

CudaParticleSystem *CudaParticleSystem::CreateFromCPU(CpuParticleSystem *p, ParticleSolver *solver)
{
	CudaParticleSystem *pGPU = new CudaParticleSystem(p->TotalCount, p->Radius, solver);
	cudaMemcpy(pGPU->PosX, p->PosX, p->TotalCount * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(pGPU->PosY, p->PosY, p->TotalCount * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(pGPU->VelX, p->VelX, p->TotalCount * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(pGPU->VelY, p->VelY, p->TotalCount * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(pGPU->Mass, p->Mass, p->TotalCount * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(pGPU->Color, p->Color, p->TotalCount * sizeof(glm::vec4), cudaMemcpyHostToDevice);
	pGPU->Count = p->Count;
	return pGPU;
}
