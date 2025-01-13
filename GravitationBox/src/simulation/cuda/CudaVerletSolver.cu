#include "cuda/CudaVerletSolver.h"
#include "cuda/cuda_helper.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cuda/cuda_helper_math.h"
#include "Config.h"

__constant__ ParticleSystem::Parameters d_Params;

template <bool floor = false>
__device__ float2 CollideWall(float2 position, float2 velocity, float mass, float2 wallPosition)
{
	float2 positionDelta = wallPosition - position;

	float distance = length(positionDelta);

	float2 Force = make_float2(0.0f);

	if (distance < d_Params.Radius)
	{
		float2 normal = positionDelta / distance;

		float2 relativeVelocity = -velocity;
		float  velocityAlongNormal = dot(relativeVelocity, normal);
		float2 normalVelocity = velocityAlongNormal * normal;
		float2 tangentVelocity = relativeVelocity - normalVelocity;

		if constexpr (floor)
		{
			// Normal force
			Force = -mass * d_Params.Gravity * normal;
		}

		// Damping force
		if (velocityAlongNormal < 0.0f) Force += d_Params.WallDampening * -normalVelocity;

		// Friction force
		Force += d_Params.WallFriction * tangentVelocity;
	}

	return Force;
}

__device__ float2 CheckCollisionsWithWalls(uint32_t tid, float *PosX, float *PosY, float *VelX, float *VelY, float *Mass)
{
	float2 Force = make_float2(0.0f);

	// Left and right walls
	if (PosX[tid] < d_Params.Radius)
	{
		Force += CollideWall(
			make_float2(PosX[tid], PosY[tid]),
			make_float2(VelX[tid], VelY[tid]),
			Mass[tid],
			make_float2(0.0f, PosY[tid])
		);

		PosX[tid] = d_Params.Radius;
		VelX[tid] *= -1;
	}
	else if (PosX[tid] > d_Params.DimX - d_Params.Radius)
	{
		Force += CollideWall(
			make_float2(PosX[tid], PosY[tid]),
			make_float2(VelX[tid], VelY[tid]),
			Mass[tid],
			make_float2(d_Params.DimX, PosY[tid])
		);

		PosX[tid] = d_Params.DimX - d_Params.Radius;
		VelX[tid] *= -1;
	}

	// Top and bottom walls
	if (PosY[tid] < d_Params.Radius)
	{
		Force += CollideWall<true>(
			make_float2(PosX[tid], PosY[tid]),
			make_float2(VelX[tid], VelY[tid]),
			Mass[tid],
			make_float2(PosX[tid], 0.0f)
		);

		PosY[tid] = d_Params.Radius;
		VelY[tid] *= -1;
	}
	else if (PosY[tid] > d_Params.DimY - d_Params.Radius)
	{
		Force += CollideWall(
			make_float2(PosX[tid], PosY[tid]),
			make_float2(VelX[tid], VelY[tid]),
			Mass[tid],
			make_float2(PosX[tid], d_Params.DimY)
		);

		PosY[tid] = d_Params.DimY - d_Params.Radius;
		VelY[tid] *= -1;
	}

	return Force;
}

__device__ float2 SolveCollision(float2 positionA, float2 velocityA, float2 positionB, float2 velocityB)
{
	float2 positionDelta = positionB - positionA;
	float distance = length(positionDelta);

	float2 Force = make_float2(0.0f, 0.0f);

	float collideDistance = d_Params.Radius * 2.0f;
	if (distance < collideDistance)
	{
		float2 normal = positionDelta / distance;
		float2 relativeVelocity = velocityB - velocityA;
		float velocityAlongNormal = dot(relativeVelocity, normal);
		float2 normalVelocity = velocityAlongNormal * normal;
		float2 tangentVelocity = relativeVelocity - normalVelocity;

		// Spring force
		Force = -d_Params.ParticlesStiffness * (collideDistance - distance) * normal;

		// Damping force
		if (velocityAlongNormal < 0) Force += d_Params.ParticlesDampening * normalVelocity;

		// Friction force
		Force += d_Params.ParticlesFriction * tangentVelocity;
	}

	return Force;
}

__device__ float2 CheckCollisionsInCell(uint32_t tid, uint32_t cellId,
	float2 position, float2 velocity,
	float *PosX, float *PosY,
	float *VelX, float *VelY,
	uint32_t *cellStart, uint32_t *cellEnd)
{
	float2 Force = make_float2(0.0f, 0.0f);

	uint32_t startIdx = __ldg(&cellStart[cellId]);

	if (startIdx == 0xFFFFFFFF) return Force;

	uint32_t endIdx = __ldg(&cellEnd[cellId]);

	for (size_t i = startIdx; i < endIdx; ++i)
	{
		if (tid == i) continue;

		float2 otherPosition = make_float2(__ldg(&PosX[i]), __ldg(&PosY[i]));
		float2 otherVelocity = make_float2(__ldg(&VelX[i]), __ldg(&VelY[i]));

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

			Position += Velocity * d_Params.Timestep + Force * (0.5f * d_Params.Timestep * d_Params.Timestep) / __ldg(&Mass[tid]);
			Velocity += 0.5f * Force * d_Params.Timestep / __ldg(&Mass[tid]);
			Force = make_float2(0.0f, d_Params.Gravity * __ldg(&Mass[tid]));

			PosX[tid] = Position.x;
			PosY[tid] = Position.y;
			VelX[tid] = Velocity.x;
			VelY[tid] = Velocity.y;

			Force += CheckCollisionsWithWalls(tid, PosX, PosY, VelX, VelY, Mass);

			ForceX[tid] = Force.x;
			ForceY[tid] = Force.y;
		}
		else
		{
			float2 Velocity = make_float2(VelX[tid], VelY[tid]);
			float2 Force = make_float2(ForceX[tid], ForceY[tid]);

			Velocity += 0.5f * Force * d_Params.Timestep / __ldg(&Mass[tid]);

			VelX[tid] = Velocity.x;
			VelY[tid] = Velocity.y;
		}

	}
}

__global__ void CheckCollisionsKernel(
	float *newForceX, float *newForceY,
	float *PosX, float *PosY,
	float *VelX, float *VelY,
	float *ForceX, float *ForceY,
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
		float x = __ldg(&PosX[tid]);
		float y = __ldg(&PosY[tid]);
		float vx = __ldg(&VelX[tid]);
		float vy = __ldg(&VelY[tid]);
		int2 cell = make_int2(x / cellSize, y / cellSize);
		clamp(cell.x, 0, gridDimension.x - 1);
		clamp(cell.y, 0, gridDimension.y - 1);

#pragma unroll
		for (int i = -1; i <= 1; ++i)
		{
#pragma unroll
			for (int j = -1; j <= 1; ++j)
			{
				int2 neighbour = make_int2(cell.x + i, cell.y + j);
				if (neighbour.x < 0 || neighbour.x >= gridDimension.x || neighbour.y < 0 || neighbour.y >= gridDimension.y) continue;

				uint32_t cellId = neighbour.y * gridDimension.x + neighbour.x;
				Force += CheckCollisionsInCell(tid, cellId, make_float2(x, y), make_float2(vx, vy), PosX, PosY, VelX, VelY, cellStart, cellEnd);
			}
		}

		uint32_t index = __ldg(&indices[tid]);
		newForceX[index] = __ldg(&ForceX[tid]) + Force.x;
		newForceY[index] = __ldg(&ForceY[tid]) + Force.y;
	}
}

CudaVerletSolver::CudaVerletSolver(Grid *g) : m_Grid(g)
{
}

void CudaVerletSolver::Solve(ParticleSystem *p)
{
	if (!p->Count) return;

	cudaMemcpyToSymbol(d_Params, &p->m_Params, sizeof(ParticleSystem::Parameters));

	VelocityVerletIntegrationKernel<true> << <BLOCKS_PER_GRID(p->Count), THREADS_PER_BLOCK >> > (
		p->PosX, p->PosY,
		p->VelX, p->VelY,
		p->ForceX, p->ForceY,
		p->Mass, p->Count
		);
	cudaDeviceSynchronize();
	CUDA_CHECK_NR(cudaGetLastError());

	m_Grid->UpdateGrid(p);

	CheckCollisionsKernel << <BLOCKS_PER_GRID(p->Count), THREADS_PER_BLOCK >> > (
		p->ForceX, p->ForceY,
		p->SortedPosX, p->SortedPosY,
		p->SortedVelX, p->SortedVelY,
		p->SortedForceX, p->SortedForceY,
		m_Grid->d_cellStart, m_Grid->d_cellEnd,
		m_Grid->d_particleIndex, make_int2(m_Grid->m_Dim.x, m_Grid->m_Dim.y),
		m_Grid->m_CellSize, p->Count
		);
	cudaDeviceSynchronize();
	CUDA_CHECK_NR(cudaGetLastError());

	VelocityVerletIntegrationKernel<false> << <BLOCKS_PER_GRID(p->Count), THREADS_PER_BLOCK >> > (
		p->PosX, p->PosY,
		p->VelX, p->VelY,
		p->ForceX, p->ForceY,
		p->Mass, p->Count
		);
	cudaDeviceSynchronize();
	CUDA_CHECK_NR(cudaGetLastError());
}
