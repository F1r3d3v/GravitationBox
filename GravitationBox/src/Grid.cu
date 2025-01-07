#include "Grid.h"
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>
#include <device_launch_parameters.h>
#include "cuda_helper.h"
#include "cuda_helper_math.h"

__global__ void calculateCellIds(float *posX, float *posY, uint32_t*cellIds, uint32_t particleCount, float cellSize, int2 gridDims) 
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

__global__ void findCellStartEnd(uint32_t *sortedCellIds, uint32_t*cellStart, uint32_t*cellEnd, uint32_t particleCount)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (; tid < particleCount; tid += stride)
	{
		if (tid == 0)
		{
			cellStart[sortedCellIds[0]] = 0;
		}
		else
		{
			int cell = sortedCellIds[tid];
			int prevCell = sortedCellIds[tid - 1];

			if (cell != prevCell)
			{
				cellEnd[prevCell] = tid;
				cellStart[cell] = tid;
			}
		}
		if (tid == particleCount - 1)
		{
			cellEnd[sortedCellIds[tid]] = particleCount;
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
		sortedPosX[tid] = posX[index];
		sortedPosY[tid] = posY[index];
		sortedVelX[tid] = velX[index];
		sortedVelY[tid] = velY[index];
		sortedForceX[tid] = forceX[index];
		sortedForceY[tid] = forceY[index];
	}
}

void Grid::allocateGPUMemory(size_t particleCount) {
	cudaMalloc(&d_cellIds, particleCount * sizeof(uint32_t));
	cudaMalloc(&d_particleIndex, particleCount * sizeof(uint32_t));
	cudaMalloc(&d_cellStart, h_cellStart.size() * sizeof(uint32_t));
	cudaMalloc(&d_cellEnd, h_cellEnd.size() * sizeof(uint32_t));
}

void Grid::freeGPUMemory() {
	cudaFree(d_cellIds);
	cudaFree(d_particleIndex);
	cudaFree(d_cellStart);
	cudaFree(d_cellEnd);
	d_cellIds = nullptr;
	d_particleIndex = nullptr;
	d_cellStart = nullptr;
	d_cellEnd = nullptr;
}

void Grid::transferToGPU() {
	cudaMemcpy(d_cellIds, h_cellIds.data(), h_cellIds.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_particleIndex, h_indices.data(), h_indices.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_cellStart, h_cellStart.data(), h_cellStart.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_cellEnd, h_cellEnd.data(), h_cellEnd.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
}

void Grid::transferToCPU() {
	cudaMemcpy(h_cellIds.data(), d_cellIds, h_cellIds.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_indices.data(), d_particleIndex, h_indices.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_cellStart.data(), d_cellStart, h_cellStart.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_cellEnd.data(), d_cellEnd, h_cellEnd.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
}

cudaError_t Grid::updateGridCUDA(const Particles &particles) {
	if (particles.Count == 0) return cudaSuccess;

	if (d_cellIds == nullptr) {
		allocateGPUMemory(particles.Count);
	}

	calculateCellIds<<<BLOCKS_PER_GRID(particles.Count), THREADS_PER_BLOCK>> > (
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

	findCellStartEnd<<<BLOCKS_PER_GRID(particles.Count), THREADS_PER_BLOCK >> > (
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