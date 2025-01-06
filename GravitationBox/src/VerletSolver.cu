#include "VerletSolver.h"
#include "cuda_helper.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "Config.h"

__device__ void SetParticleVelocity(float* prevPosX, float* prevPosY, float* PosX, float* PosY, uint32_t id, float2 vel)
{
	prevPosX[id] = PosX[id] - vel.x;
	prevPosY[id] = PosY[id] - vel.y;
}

__device__ float2 GetParticleVelocity(float *prevPosX, float *prevPosY, float *PosX, float *PosY, uint32_t id)
{
	return make_float2(PosX[id] - prevPosX[id], PosY[id] - prevPosY[id]);
}

template <bool inertia>
__global__ void UpdateParticlesKernel(
	float *PosX, float *PosY,
	float *prevPosX, float *prevPosY,
	float *Mass,
	float dt, size_t count)
{
	uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t stride = blockDim.x * gridDim.x;

	float accX = 0.0f;
	float accY = Config::GRAVITY;

	for (; tid < count; tid += stride)
	{
		if (inertia)
		{
			float posX = PosX[tid];
			float posY = PosY[tid];

			// Update positions
			PosX[tid] += (PosX[tid] - prevPosX[tid]);
			PosY[tid] += (PosY[tid] - prevPosY[tid]);

			// Update velocities
			prevPosX[tid] = posX;
			prevPosY[tid] = posY;
		}
		else
		{
			// Update positions
			PosX[tid] += accX * dt * dt;
			PosY[tid] += accY * dt * dt;
		}
	}
}

//__global__ void CheckCollisionsWithParticlesKernel(
//	float *PosX, float *PosY,
//	float *prevPosX, float *prevPosY,
//	float *Mass,
//	unsigned int *cellStart, unsigned int *cellEnd,
//	unsigned int *indices,
//	float radius, int2 gridDim, float cellSize,
//	size_t count)
//{
//	uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
//	uint32_t stride = blockDim.x * gridDim.x;
//
//	for (; tid < count; tid += stride)
//	{
//		// Get particle's grid cell
//		float x = PosX[tid];
//		float y = PosY[tid];
//		int2 cell = make_int2(x / cellSize, y / cellSize);
//
//		// Loop over neighboring cells
//#pragma unroll
//		for (int i = -1; i <= 1; ++i) {
//#pragma unroll
//			for (int j = -1; j <= 1; ++j) {
//				int2 neighborCell = make_int2(cell.x + i, cell.y + j);
//
//				// Skip invalid cells
//				if (neighborCell.x < 0 || neighborCell.x >= gridDim.x ||
//					neighborCell.y < 0 || neighborCell.y >= gridDim.y) continue;
//
//				int cellHash = neighborCell.y * gridDim.x + neighborCell.x;
//
//				unsigned int startIdx = cellStart[cellHash];
//				if (startIdx == 0xffffffff) continue;
//				unsigned int endIdx = cellEnd[cellHash];
//
//				// Loop over particles in the neighboring cell
//				for (unsigned int idx = startIdx; idx < endIdx; ++idx) {
//					uint32_t jId = indices[idx];
//					if (tid >= jId || jId >= count) continue;
//
//					// Check for collision
//					float dx = PosX[jId] - x;
//					float dy = PosY[jId] - y;
//					float dist2 = dx * dx + dy * dy;
//					float minDist = 2.0f * radius;
//
//					if (dist2 < minDist * minDist) {
//						float dist = sqrtf(dist2);
//						if (dist == 0.0f) dist = minDist;
//
//						// Collision response
//						float nx = dx / dist;
//						float ny = dy / dist;
//
//						float overlap = minDist - dist;
//
//						float SLOP = 0.04f;
//						float PERCENT = 0.2900f;
//						float correctionX = (fmaxf(overlap - SLOP, 0.0f) / (inv_mass1 + inv_mass2)) * PERCENT * nx;
//						float correctionY = (fmaxf(overlap - SLOP, 0.0f) / (inv_mass1 + inv_mass2)) * PERCENT * ny;
//
//						atomicAdd(&PosX[tid], (inv_mass1) * correctionX);
//						atomicAdd(&PosY[tid], (inv_mass1) * correctionY);
//
//						atomicAdd(&PosX[jId], -(inv_mass2) * correctionX);
//						atomicAdd(&PosY[jId], -(inv_mass2) * correctionY);
//					}
//				}
//			}
//		}
//	}
//}

__global__ void CheckCollisionsWithParticlesKernel(
	float *PosX, float *PosY,
	float *prevPosX, float *prevPosY,
	float *Mass,
	unsigned int *cellStart, unsigned int *cellEnd,
	unsigned int *indices,
	float radius, int2 gridDim, float cellSize,
	size_t count,
	bool preserve_impulse
)
{
	uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t stride = blockDim.x * gridDim.x;

	for (; tid < count; tid += stride)
	{
		// Get particle's position
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
				if (startIdx == 0xffffffff) continue;
				unsigned int endIdx = cellEnd[cellHash];

				// Loop over particles in the neighboring cell
				for (unsigned int idx = startIdx; idx < endIdx; ++idx) {
					uint32_t jId = indices[idx];
					if (tid >= jId || jId >= count) continue;

					// Compute displacement and distance
					float dx = x - PosX[jId];
					float dy = y - PosY[jId];
					float slength = dx * dx + dy * dy;
					float length = sqrtf(slength);
					float target = 2.0f * radius;

					if (length < target) {
						// Calculate velocities
						float v1x = x - prevPosX[tid];
						float v1y = y - prevPosY[tid];
						float v2x = PosX[jId] - prevPosX[jId];
						float v2y = PosY[jId] - prevPosY[jId];

						// Position correction
						float factor = (length - target) / length;
						float correctionX = dx * factor * 0.5f;
						float correctionY = dy * factor * 0.5f;
						PosX[tid] -= correctionX;
						PosY[tid] -= correctionY;
						PosX[jId] += correctionX;
						PosY[jId] += correctionY;

						if (preserve_impulse) {
							// Impulse preservation
							float f1 = (Config::DAMPENING * (dx * v1x + dy * v1y)) / slength;
							float f2 = (Config::DAMPENING * (dx * v2x + dy * v2y)) / slength;

							v1x += f2 * dx - f1 * dx;
							v1y += f2 * dy - f1 * dy;
							v2x += f1 * dx - f2 * dx;
							v2y += f1 * dy - f2 * dy;

							prevPosX[tid] = PosX[tid] - v1x;
							prevPosY[tid] = PosY[tid] - v1y;
							prevPosX[jId] = PosX[jId] - v2x;
							prevPosY[jId] = PosY[jId] - v2y;
						}
					}
				}
			}
		}
	}
}

template <bool preserveImpulse>
__global__ void CheckCollisionsWithWallsKernel(
	float *PosX, float *PosY,
	float *prevPosX, float *prevPosY,
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
			if (preserveImpulse)
			{
				float vx = (prevPosX[tid] - PosX[tid]) * Config::DAMPENING;
				PosX[tid] = radius;
				prevPosX[tid] = PosX[tid] - vx;
			}
			else
			{
				PosX[tid] = radius;
			}
		}
		else if (PosX[tid] > worldDim.x - radius)
		{
			if (preserveImpulse)
			{
				float vx = (prevPosX[tid] - PosX[tid]) * Config::DAMPENING;
				PosX[tid] = worldDim.x - radius;
				prevPosX[tid] = PosX[tid] - vx;
			}
			else
			{
				PosX[tid] = worldDim.x - radius;
			}
		}

		// Top and bottom walls
		if (PosY[tid] < radius)
		{
			if (preserveImpulse)
			{
				float vy = (prevPosY[tid] - PosY[tid]) * Config::DAMPENING;
				PosY[tid] = radius;
				prevPosY[tid] = PosY[tid] - vy;
			}
			else
			{
				PosY[tid] = radius;
			}
		}
		else if (PosY[tid] > worldDim.y - radius)
		{
			if (preserveImpulse)
			{
				float vy = (prevPosY[tid] - PosY[tid]) * Config::DAMPENING;
				PosY[tid] = worldDim.y - radius;
				prevPosY[tid] = PosY[tid] - vy;
			}
			else
			{
				PosY[tid] = worldDim.y - radius;
			}
		}
	}
}

cudaError_t VerletSolver::VerletCuda(float dt)
{
	UpdateParticlesKernel<false><< <BLOCKS_PER_GRID(m_Particles->Count), THREADS_PER_BLOCK >> > (
		m_Particles->PosX, m_Particles->PosY,
		m_Particles->prevPosX, m_Particles->prevPosY,
		m_Particles->Mass, dt, m_Particles->Count
		);
	cudaDeviceSynchronize();
	CUDA_CHECK(cudaGetLastError());

	m_Grid->updateGrid(*m_Particles);

	CheckCollisionsWithParticlesKernel<< <BLOCKS_PER_GRID(m_Particles->Count), THREADS_PER_BLOCK >> > (
		m_Particles->PosX, m_Particles->PosY,
		m_Particles->prevPosX, m_Particles->prevPosY,
		m_Particles->Mass,
		m_Grid->d_cellStart, m_Grid->d_cellEnd,
		m_Grid->d_indices,
		m_Particles->Radius, m_Grid->m_Dim, m_Grid->m_cellSize,
		m_Particles->Count, false
		);
	cudaDeviceSynchronize();
	CUDA_CHECK(cudaGetLastError());

	CheckCollisionsWithWallsKernel<false><< <BLOCKS_PER_GRID(m_Particles->Count), THREADS_PER_BLOCK >> > (
		m_Particles->PosX, m_Particles->PosY,
		m_Particles->prevPosX, m_Particles->prevPosY,
		m_Particles->Radius, m_Grid->m_WorldDim,
		m_Particles->Count
		);
	cudaDeviceSynchronize();
	CUDA_CHECK(cudaGetLastError());

	// ----------------------------

	UpdateParticlesKernel<true> << <BLOCKS_PER_GRID(m_Particles->Count), THREADS_PER_BLOCK >> > (
		m_Particles->PosX, m_Particles->PosY,
		m_Particles->prevPosX, m_Particles->prevPosY,
		m_Particles->Mass, dt, m_Particles->Count
		);
	cudaDeviceSynchronize();
	CUDA_CHECK(cudaGetLastError());

	m_Grid->updateGrid(*m_Particles);

	CheckCollisionsWithParticlesKernel << <BLOCKS_PER_GRID(m_Particles->Count), THREADS_PER_BLOCK >> > (
		m_Particles->PosX, m_Particles->PosY,
		m_Particles->prevPosX, m_Particles->prevPosY,
		m_Particles->Mass,
		m_Grid->d_cellStart, m_Grid->d_cellEnd,
		m_Grid->d_indices,
		m_Particles->Radius, m_Grid->m_Dim, m_Grid->m_cellSize,
		m_Particles->Count, true
		);
	cudaDeviceSynchronize();
	CUDA_CHECK(cudaGetLastError());

	CheckCollisionsWithWallsKernel<true> << <BLOCKS_PER_GRID(m_Particles->Count), THREADS_PER_BLOCK >> > (
		m_Particles->PosX, m_Particles->PosY,
		m_Particles->prevPosX, m_Particles->prevPosY,
		m_Particles->Radius, m_Grid->m_WorldDim,
		m_Particles->Count
		);
	cudaDeviceSynchronize();
	CUDA_CHECK(cudaGetLastError());

	return cudaSuccess;
}