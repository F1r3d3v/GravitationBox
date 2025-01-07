#include "VerletSolver.h"
#include "cuda_helper.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cuda_helper_math.h"
#include "Config.h"

__constant__ VerletSolver::SimulationParams d_Params;

__device__ void CheckCollisionsWithWalls(uint32_t tid, float *PosX, float *PosY, float *VelX, float *VelY, float radius, int2 worldDim)
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

__device__ float2 SolveCollision(float2 positionA, float2 velocityA, float2 positionB, float2 velocityB)
{
	float2 positionDelta = positionB - positionA;
	float distance = length(positionDelta);

	float2 Force = make_float2(0.0f, 0.0f);

	float collideDistance = d_Params.Radius * 2.0f;
	if (distance < collideDistance)
	{
		// Spring force
		float2 normal = positionDelta / distance;
		Force = -d_Params.ParticleStiffness * (collideDistance - distance) * normal;

		// Damping force
		float2 relativeVelocity = velocityB - velocityA;
		float velocityAlongNormal = dot(relativeVelocity, normal);
		float2 normalVelocity = velocityAlongNormal * normal;
		if (velocityAlongNormal < 0) Force += d_Params.ParticleDampening * normalVelocity;

		// Friction force
		float2 tangentVelocity = relativeVelocity - normalVelocity;
		Force += d_Params.ParticleShear * tangentVelocity;
	}

	return Force;
}

__device__ float2 CheckCollisionInCell(uint32_t tid, uint32_t cellId,
	float2 position, float2 velocity,
	const float *__restrict PosX, const float *__restrict PosY,
	const float *__restrict VelX, const float *__restrict VelY,
	uint32_t *cellStart, uint32_t *cellEnd)
{
	float2 Force = make_float2(0.0f, 0.0f);

	uint32_t startIdx = cellStart[cellId];

	if (startIdx == 0xffffffff) return Force;

	uint32_t endIdx = cellEnd[cellId];

	for (size_t i = startIdx; i < endIdx; ++i)
	{
		if (tid == i) continue;

		float2 otherPosition = make_float2(PosX[i], PosY[i]);
		float2 otherVelocity = make_float2(VelX[i], VelY[i]);

		Force += SolveCollision(position, velocity, otherPosition, otherVelocity);
	}

	return Force;
}

template <bool stage1>
__global__ void VelocityVerletIntegrationKernel(float *PosX, float *PosY, float *VelX, float *VelY, float *ForceX, float *ForceY, float *Mass, size_t particleCount)
{
	uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t stride = blockDim.x * gridDim.x;

	for (; tid < particleCount; tid += stride)
	{
		if constexpr (stage1)
		{
			float2 Position = make_float2(PosX[tid], PosY[tid]);
			float2 Velocity = make_float2(VelX[tid], VelY[tid]);
			float2 Force = make_float2(ForceX[tid], ForceY[tid]);

			Position += Velocity * d_Params.Timestep + Force * (0.5f * d_Params.Timestep * d_Params.Timestep) / Mass[tid];
			Velocity += 0.5f * Force * d_Params.Timestep / Mass[tid];
			Force = make_float2(0.0f, d_Params.Gravity);

			PosX[tid] = Position.x;
			PosY[tid] = Position.y;
			VelX[tid] = Velocity.x;
			VelY[tid] = Velocity.y;
			ForceX[tid] = Force.x;
			ForceY[tid] = Force.y;

			CheckCollisionsWithWalls(tid, PosX, PosY, VelX, VelY, d_Params.Radius, make_int2(d_Params.DimX, d_Params.DimY));
		}
		else
		{
			float2 Velocity = make_float2(VelX[tid], VelY[tid]);
			float2 Force = make_float2(ForceX[tid], ForceY[tid]);

			Velocity += 0.5f * Force * d_Params.Timestep / Mass[tid];

			VelX[tid] = Velocity.x;
			VelY[tid] = Velocity.y;
		}

	}
}

__global__ void CheckCollisionsKernel(
	float *newForceX, float *newForceY,
	const float * __restrict PosX, const float *__restrict PosY,
	const float *__restrict VelX, const float *__restrict VelY,
	const float *__restrict ForceX, const float *__restrict ForceY,
	unsigned int *cellStart, unsigned int *cellEnd,
	unsigned int *indices,
	int2 gridDimension, float cellSize,
	uint32_t particleCount)
{
	uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t stride = blockDim.x * gridDim.x;

	for (; tid < particleCount; tid += stride)
	{
		float2 Force = make_float2(0.0f, 0.0f);
		float x = PosX[tid];
		float y = PosY[tid];
		int2 cell = make_int2(x / cellSize, y / cellSize);
		clamp(cell.x, 0, gridDimension.x - 1);
		clamp(cell.y, 0, gridDimension.y - 1);

#pragma unroll
		for (int i = -1; i <= 1; ++i)
		{
#pragma unroll
			for (int j = -1; j <= 1; ++j)
			{
				int2 neighbour = cell + make_int2(i, j);
				if (neighbour.x < 0 || neighbour.x >= gridDimension.x || neighbour.y < 0 || neighbour.y >= gridDimension.y) continue;

				uint32_t cellId = neighbour.y * gridDimension.x + neighbour.x;
				Force += CheckCollisionInCell(tid, cellId, make_float2(x, y), make_float2(VelX[tid], VelY[tid]), PosX, PosY, VelX, VelY, cellStart, cellEnd);
			}
		}

		uint32_t index = indices[tid];
		newForceX[index] = ForceX[tid] + Force.x;
		newForceY[index] = ForceY[tid] + Force.y;
	}
}

void VerletSolver::SetParams(const SimulationParams &params)
{
	m_Params = params;
	cudaMemcpyToSymbol(d_Params, &params, sizeof(SimulationParams));
}

cudaError_t VerletSolver::VerletCuda()
{
	VelocityVerletIntegrationKernel<true> << <BLOCKS_PER_GRID(m_Particles->Count), THREADS_PER_BLOCK >> > (
		m_Particles->PosX, m_Particles->PosY,
		m_Particles->VelX, m_Particles->VelY,
		m_Particles->ForceX, m_Particles->ForceY,
		m_Particles->Mass, m_Particles->Count
		);
	cudaDeviceSynchronize();
	CUDA_CHECK(cudaGetLastError());

	m_Grid->updateGrid(*m_Particles);

	CheckCollisionsKernel << <BLOCKS_PER_GRID(m_Particles->Count), THREADS_PER_BLOCK >> > (
		m_Particles->ForceX, m_Particles->ForceY,
		m_Particles->SortedPosX, m_Particles->SortedPosY,
		m_Particles->SortedVelX, m_Particles->SortedVelY,
		m_Particles->SortedForceX, m_Particles->SortedForceY,
		m_Grid->d_cellStart, m_Grid->d_cellEnd,
		m_Grid->d_particleIndex, m_Grid->m_Dim,
		m_Grid->m_cellSize, m_Particles->Count
		);
	cudaDeviceSynchronize();
	CUDA_CHECK(cudaGetLastError());

	VelocityVerletIntegrationKernel<false> << <BLOCKS_PER_GRID(m_Particles->Count), THREADS_PER_BLOCK >> > (
		m_Particles->PosX, m_Particles->PosY,
		m_Particles->VelX, m_Particles->VelY,
		m_Particles->ForceX, m_Particles->ForceY,
		m_Particles->Mass, m_Particles->Count
		);
	cudaDeviceSynchronize();
	CUDA_CHECK(cudaGetLastError());

	return cudaSuccess;
}