#include "VerletSolver.h"
#include "cuda_helper.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "Config.h"

__global__ void UpdateParticlesKernel(
	float *PosX, float *PosY,
	float *VelX, float *VelY,
	float *Mass,
	float dt, size_t count)
{
	uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t stride = blockDim.x * gridDim.x;

	float accX = 0.0f;
	float accY = Config::GRAVITY;

	for (; tid < count; tid += stride)
	{
		// Update positions
		PosX[tid] += VelX[tid] * dt + 0.5f * accX * dt * dt;
		PosY[tid] += VelY[tid] * dt + 0.5f * accY * dt * dt;

		// Update velocities
		VelX[tid] += accX * dt;
		VelY[tid] += accY * dt;
	}
}

__global__ void CheckCollisionsWithParticlesKernel(
	float *PosX, float *PosY,
	float *VelX, float *VelY,
	float *Mass,
	unsigned int *cellStart, unsigned int *cellEnd,
	unsigned int *indices,
	float radius, int2 gridDim, float cellSize,
	size_t count)
{
	uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t stride = blockDim.x * gridDim.x;

	for (; tid < count; tid += stride)
	{
		// Get particle's grid cell
		float x = PosX[tid];
		float y = PosY[tid];
		int2 cell = make_int2(x / cellSize, y / cellSize);

		// Loop over neighboring cells
#pragma unroll
		for (int i = -1; i <= 1; ++i) {
#pragma unroll
			for (int j = -1; j <= 1; ++j) {
				int2 neighborCell = make_int2(cell.x + i, cell.y + j);

				// Skip invalid cells
				if (neighborCell.x < 0 || neighborCell.x >= gridDim.x ||
					neighborCell.y < 0 || neighborCell.y >= gridDim.y) continue;

				int cellHash = neighborCell.y * gridDim.x + neighborCell.x;

				unsigned int startIdx = cellStart[cellHash];
				unsigned int endIdx = cellEnd[cellHash];

				// Loop over particles in the neighboring cell
				for (unsigned int idx = startIdx; idx < endIdx; ++idx) {
					uint32_t jId = indices[idx];
					if (tid >= jId || jId >= count) continue;

					// Check for collision
					float dx = x - PosX[jId];
					float dy = y - PosY[jId];
					float dist2 = dx * dx + dy * dy;
					float minDist = 2.0f * radius;

					if (dist2 < minDist * minDist) {
						float dist = sqrtf(dist2);
						if (dist == 0.0f) dist = minDist;

						// Collision response
						float nx = dx / dist;
						float ny = dy / dist;

						float relVelX = VelX[tid] - VelX[jId];
						float relVelY = VelY[tid] - VelY[jId];
						float relVel = nx * relVelX + ny * relVelY;

						float restitution = Config::DAMPENING;
						float impulse = -(1.0f + restitution) * relVel;
						impulse /= (1.0f / Mass[tid] + 1.0f / Mass[jId]);

						float impulseX = impulse * nx;
						float impulseY = impulse * ny;

						// Update velocities using atomic operations
						atomicAdd(&VelX[tid], impulseX / Mass[tid]);
						atomicAdd(&VelY[tid], impulseY / Mass[tid]);
						atomicAdd(&VelX[jId], -impulseX / Mass[jId]);
						atomicAdd(&VelY[jId], -impulseY / Mass[jId]);

						// Position correction to resolve overlap
						float overlap = 0.5f * (minDist - dist);
						atomicAdd(&PosX[tid], nx * overlap);
						atomicAdd(&PosY[tid], ny * overlap);
						atomicAdd(&PosX[jId], -nx * overlap);
						atomicAdd(&PosY[jId], -ny * overlap);
					}
				}
			}
		}
	}
}

__global__ void CheckCollisionsWithWallsKernel(
	float *PosX, float *PosY,
	float *VelX, float *VelY,
	float radius, int2 worldDim,
	size_t count)
{
	uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t stride = blockDim.x * gridDim.x;

	for (; tid < count; tid += stride)
	{
		// Left and right walls
		if (PosX[tid] < radius)
		{
			PosX[tid] = radius;
			VelX[tid] = -VelX[tid] * Config::DAMPENING;
		}
		else if (PosX[tid] > worldDim.x - radius)
		{
			PosX[tid] = worldDim.x - radius;
			VelX[tid] = -VelX[tid] * Config::DAMPENING;
		}

		// Top and bottom walls
		if (PosY[tid] < radius)
		{
			PosY[tid] = radius;
			VelY[tid] = -VelY[tid] * Config::DAMPENING;
		}
		else if (PosY[tid] > worldDim.y - radius)
		{
			PosY[tid] = worldDim.y - radius;
			VelY[tid] = -VelY[tid] * Config::DAMPENING;
		}
	}
}

cudaError_t VerletSolver::VerletCuda(float dt)
{
	UpdateParticlesKernel << <BLOCKS_PER_GRID(m_Particles->Count), THREADS_PER_BLOCK >> > (
		m_Particles->PosX, m_Particles->PosY,
		m_Particles->VelX, m_Particles->VelY,
		m_Particles->Mass, dt, m_Particles->Count
		);
	cudaDeviceSynchronize();
	CUDA_CHECK(cudaGetLastError());

	m_Grid->updateGrid(*m_Particles);

	CheckCollisionsWithParticlesKernel << <BLOCKS_PER_GRID(m_Particles->Count), THREADS_PER_BLOCK >> > (
		m_Particles->PosX, m_Particles->PosY,
		m_Particles->VelX, m_Particles->VelY,
		m_Particles->Mass,
		m_Grid->d_cellStart, m_Grid->d_cellEnd,
		m_Grid->d_indices,
		m_Particles->Radius, m_Grid->m_Dim, m_Grid->m_cellSize,
		m_Particles->Count
		);
	cudaDeviceSynchronize();
	CUDA_CHECK(cudaGetLastError());

	CheckCollisionsWithWallsKernel << <BLOCKS_PER_GRID(m_Particles->Count), THREADS_PER_BLOCK >> > (
		m_Particles->PosX, m_Particles->PosY,
		m_Particles->VelX, m_Particles->VelY,
		m_Particles->Radius, m_Grid->m_WorldDim,
		m_Particles->Count
		);
	cudaDeviceSynchronize();
	CUDA_CHECK(cudaGetLastError());

	return cudaSuccess;
}