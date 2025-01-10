#include "Grid.h"
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>
#include <device_launch_parameters.h>
#include "cuda_helper.h"
#include "cuda_helper_math.h"

__global__ void calculateCellIds(float *posX, float *posY, uint32_t *cellIds, uint32_t particleCount, float cellSize, int2 gridDims)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDims.x;

	for (; tid < particleCount; tid += stride)
	{
		int x = clamp((int)(posX[tid] / cellSize), 0, gridDims.x - 1);
		int y = clamp((int)(posY[tid] / cellSize), 0, gridDims.y - 1);
		cellIds[tid] = y * gridDims.x + x;
	}
}

__global__ void findCellStartEnd(uint32_t *sortedCellIds, uint32_t *cellStart, uint32_t *cellEnd, uint32_t particleCount)
{
	__shared__ extern uint32_t sharedCellIds[];
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	uint32_t cellId;

	// Load prev cell ids to shared memory
	if (tid < particleCount)
	{
		cellId = __ldg(&sortedCellIds[tid]);
		sharedCellIds[threadIdx.x + 1] = cellId;

		if (threadIdx.x == 0 && tid > 0)
		{
			sharedCellIds[0] = __ldg(&sortedCellIds[tid - 1]);
		}
	}

	__syncthreads();

	if (tid < particleCount)
	{
		if (tid == 0)
		{
			cellStart[cellId] = tid;
		}
		else if (sharedCellIds[threadIdx.x] != cellId)
		{
			cellStart[cellId] = tid;
			cellEnd[sharedCellIds[threadIdx.x]] = tid;
		}

		if (tid == particleCount - 1)
		{
			cellEnd[cellId] = tid + 1;
		}
	}
}

__global__ void Reorder(uint32_t *particleIndex,
	float *posX, float *sortedPosX,
	float *posY, float *sortedPosY,
	float *velX, float *sortedVelX,
	float *velY, float *sortedVelY,
	float *forceX, float *sortedForceX,
	float *forceY, float *sortedForceY,
	uint32_t particleCount)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (; tid < particleCount; tid += stride)
	{
		uint32_t index = particleIndex[tid];
		sortedPosX[tid] = __ldg(&posX[index]);
		sortedPosY[tid] = __ldg(&posY[index]);
		sortedVelX[tid] = __ldg(&velX[index]);
		sortedVelY[tid] = __ldg(&velY[index]);
		sortedForceX[tid] = __ldg(&forceX[index]);
		sortedForceY[tid] = __ldg(&forceY[index]);
	}
}

cudaError_t Grid::allocateGPUMemory(size_t particleCount)
{
	CUDA_CHECK(cudaMalloc(&d_cellIds, particleCount * sizeof(uint32_t)));
	CUDA_CHECK(cudaMalloc(&d_particleIndex, particleCount * sizeof(uint32_t)));
	CUDA_CHECK(cudaMalloc(&d_cellStart, h_cellStart.size() * sizeof(uint32_t)));
	CUDA_CHECK(cudaMalloc(&d_cellEnd, h_cellEnd.size() * sizeof(uint32_t)));

	return cudaSuccess;
}

cudaError_t Grid::freeGPUMemory()
{
	CUDA_CHECK(cudaFree(d_cellIds));
	CUDA_CHECK(cudaFree(d_particleIndex));
	CUDA_CHECK(cudaFree(d_cellStart));
	CUDA_CHECK(cudaFree(d_cellEnd));
	d_cellIds = nullptr;
	d_particleIndex = nullptr;
	d_cellStart = nullptr;
	d_cellEnd = nullptr;

	return cudaSuccess;
}

cudaError_t Grid::transferToGPU()
{
	CUDA_CHECK(cudaMemcpy(d_cellIds, h_cellIds.data(), h_cellIds.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_particleIndex, h_indices.data(), h_indices.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_cellStart, h_cellStart.data(), h_cellStart.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_cellEnd, h_cellEnd.data(), h_cellEnd.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

	return cudaSuccess;
}

cudaError_t Grid::transferToCPU()
{
	CUDA_CHECK(cudaMemcpy(h_cellIds.data(), d_cellIds, h_cellIds.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(h_indices.data(), d_particleIndex, h_indices.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(h_cellStart.data(), d_cellStart, h_cellStart.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(h_cellEnd.data(), d_cellEnd, h_cellEnd.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));

	return cudaSuccess;
}

cudaError_t Grid::updateGridCUDA(const Particles &particles)
{
	if (d_cellIds == nullptr)
	{
		allocateGPUMemory(particles.TotalCount);
	}

	if (particles.Count == 0) return cudaSuccess;

	calculateCellIds << <BLOCKS_PER_GRID(particles.Count), THREADS_PER_BLOCK >> > (
		particles.PosX,
		particles.PosY,
		d_cellIds,
		particles.Count, m_cellSize, m_Dim);
	cudaDeviceSynchronize();
	CUDA_CHECK(cudaGetLastError());

	thrust::device_ptr<uint32_t> d_cellIds_ptr(d_cellIds);
	thrust::device_ptr<uint32_t> d_indices_ptr(d_particleIndex);
	thrust::device_ptr<uint32_t> d_cellStart_ptr(d_cellStart);
	thrust::sequence(thrust::device, d_indices_ptr, d_indices_ptr + particles.Count);
	thrust::sort_by_key(thrust::device, d_cellIds_ptr, d_cellIds_ptr + particles.Count, d_indices_ptr);
	thrust::fill(thrust::device, d_cellStart_ptr, d_cellStart_ptr + h_cellStart.size(), 0xFFFFFFFF);

	uint32_t memsize = (THREADS_PER_BLOCK + 1) * sizeof(uint32_t);
	findCellStartEnd << <BLOCKS_PER_GRID(particles.Count), THREADS_PER_BLOCK, memsize >> > (
		d_cellIds,
		d_cellStart,
		d_cellEnd,
		particles.Count);
	cudaDeviceSynchronize();
	CUDA_CHECK(cudaGetLastError());

	Reorder << <BLOCKS_PER_GRID(particles.Count), THREADS_PER_BLOCK >> > (
		d_particleIndex,
		particles.PosX, particles.SortedPosX,
		particles.PosY, particles.SortedPosY,
		particles.VelX, particles.SortedVelX,
		particles.VelY, particles.SortedVelY,
		particles.ForceX, particles.SortedForceX,
		particles.ForceY, particles.SortedForceY,
		particles.Count);
	cudaDeviceSynchronize();
	CUDA_CHECK(cudaGetLastError());

	return cudaSuccess;
}