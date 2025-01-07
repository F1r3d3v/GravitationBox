#include "VerletSolver.h"
#include "cuda_helper.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cuda_helper_math.h"
#include "Config.h"

__constant__ SimulationParams d_Params;

__global__ void UpdateParticlesKernel(float *PosX, float *PosY, float *VelX, float *VelY, float *ForceX, float *ForceY, float *Mass, size_t particleCount)
{
	uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t stride = blockDim.x * gridDim.x;

	for (; tid < particleCount; tid += stride)
	{
		float2 Position = make_float2(PosX[tid], PosY[tid]);
		float2 Velocity = make_float2(VelX[tid], VelY[tid]);
		float2 Force = make_float2(ForceX[tid], ForceY[tid]);

		// Integration
		Position += Velocity * d_Params.Timestep + Force * (0.5f * d_Params.Timestep * d_Params.Timestep);
		Velocity += Force * d_Params.Timestep;
		Force = make_float2(0.0f, d_Params.Gravity);
	}
}

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
//		// Get particle's position
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
//					// Compute displacement and distance
//					float dx = x - PosX[jId];
//					float dy = y - PosY[jId];
//					float slength = dx * dx + dy * dy;
//					float length = sqrtf(slength);
//					float target = 2.0f * radius;
//
//					if (length < target) {
//						// Calculate velocities
//						float v1x = x - prevPosX[tid];
//						float v1y = y - prevPosY[tid];
//						float v2x = PosX[jId] - prevPosX[jId];
//						float v2y = PosY[jId] - prevPosY[jId];
//
//						// Position correction
//						float factor = (length - target) / length;
//						float correctionX = dx * factor * 0.5f;
//						float correctionY = dy * factor * 0.5f;
//						PosX[tid] -= correctionX;
//						PosY[tid] -= correctionY;
//						PosX[jId] += correctionX;
//						PosY[jId] += correctionY;
//
//						if (preserve_impulse) {
//							// Impulse preservation
//							float f1 = (Config::DAMPENING * (dx * v1x + dy * v1y)) / slength;
//							float f2 = (Config::DAMPENING * (dx * v2x + dy * v2y)) / slength;
//
//							v1x += f2 * dx - f1 * dx;
//							v1y += f2 * dy - f1 * dy;
//							v2x += f1 * dx - f2 * dx;
//							v2y += f1 * dy - f2 * dy;
//
//							prevPosX[tid] = PosX[tid] - v1x;
//							prevPosY[tid] = PosY[tid] - v1y;
//							prevPosX[jId] = PosX[jId] - v2x;
//							prevPosY[jId] = PosY[jId] - v2y;
//						}
//					}
//				}
//			}
//		}
	}
}

__device__ void CheckCollisionsWithWallsKernel(uint32_t tid, float *PosX, float *PosY, float *VelX, float *VelY, float radius, int2 worldDim)
{
	// Left and right walls
	if (PosX[tid] < radius)
	{
		PosX[tid] = radius;
		VelX[tid] *= -d_Params.WallDampening;
	}
	else if (PosX[tid] > worldDim.x - radius)
	{
		PosX[tid] = worldDim.x - radius;
		VelX[tid] *= -d_Params.WallDampening;
	}

	// Top and bottom walls
	if (PosY[tid] < radius)
	{
		PosY[tid] = radius;
		VelY[tid] *= -d_Params.WallDampening;
	}
	else if (PosY[tid] > worldDim.y - radius)
	{
		PosY[tid] = worldDim.y - radius;
		VelY[tid] *= -d_Params.WallDampening;
	}
}

cudaError_t VerletSolver::VerletCuda(float dt)
{
	cudaMemcpyToSymbol(d_Params, &Params, sizeof(SimulationParams));

	UpdateParticlesKernel<< <BLOCKS_PER_GRID(m_Particles->Count), THREADS_PER_BLOCK >> >
	(
		m_Particles->PosX, m_Particles->PosY,
		m_Particles->VelX, m_Particles->VelY,
		m_Particles->ForceX, m_Particles->ForceY,
		m_Particles->Mass, m_Particles->Count
	);
	cudaDeviceSynchronize();
	CUDA_CHECK(cudaGetLastError());

	m_Grid->updateGrid(*m_Particles);

	CheckCollisionsWithParticlesKernel<< <BLOCKS_PER_GRID(m_Particles->Count), THREADS_PER_BLOCK >> > (
		m_Particles->PosX, m_Particles->PosY,
		m_Particles->VelX, m_Particles->VelY,
		m_Particles->Mass,
		m_Grid->d_cellStart, m_Grid->d_cellEnd,
		m_Grid->d_indices,
		m_Particles->Radius, m_Grid->m_Dim, m_Grid->m_cellSize,
		m_Particles->Count, false
		);
	cudaDeviceSynchronize();
	CUDA_CHECK(cudaGetLastError());

	return cudaSuccess;
}