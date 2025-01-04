#include "Grid.h"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <device_launch_parameters.h>
#include "cuda_helper.h"

__device__ float clamp(float x, float a, float b)
{
	return max(a, min(b, x));
}

__global__ void calculateCellIds(float *posX, float *posY, unsigned int *cellIds,
	int numParticles, float cellSize, int2 gridDims) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDims.x;

	for (; tid < numParticles; tid += stride)
	{
		int x = clamp(posX[tid] / cellSize, 0, gridDims.x - 1);
		int y = clamp(posY[tid] / cellSize, 0, gridDims.y - 1);
		cellIds[tid] = y * gridDims.x + x;
	}
}

__global__ void findCellStartEnd(unsigned int *sortedCellIds, unsigned int *cellStart,
	unsigned int *cellEnd, int numParticles) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (; tid < numParticles; tid += stride)
	{
		if (tid == 0) {
			cellStart[sortedCellIds[0]] = 0;
		}
		else {
			int cell = sortedCellIds[tid];
			int prevCell = sortedCellIds[tid - 1];

			if (cell != prevCell) {
				cellEnd[prevCell] = tid;
				cellStart[cell] = tid;
			}
		}
		if (tid == numParticles - 1) {
			cellEnd[sortedCellIds[tid]] = numParticles;
		}
	}
}

void Grid::allocateGPUMemory(size_t particleCount) {
	cudaMalloc(&d_cellIds, particleCount * sizeof(unsigned int));
	cudaMalloc(&d_indices, particleCount * sizeof(unsigned int));
	cudaMalloc(&d_cellStart, h_cellStart.size() * sizeof(unsigned int));
	cudaMalloc(&d_cellEnd, h_cellEnd.size() * sizeof(unsigned int));
}

void Grid::freeGPUMemory() {
	cudaFree(d_cellIds);
	cudaFree(d_indices);
	cudaFree(d_cellStart);
	cudaFree(d_cellEnd);
	d_cellIds = nullptr;
	d_indices = nullptr;
	d_cellStart = nullptr;
	d_cellEnd = nullptr;
}

void Grid::transferToGPU() {
	cudaMemcpy(d_cellIds, h_cellIds.data(), h_cellIds.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_indices, h_indices.data(), h_indices.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_cellStart, h_cellStart.data(), h_cellStart.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_cellEnd, h_cellEnd.data(), h_cellEnd.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);
}

void Grid::transferToCPU() {
	cudaMemcpy(h_cellIds.data(), d_cellIds, h_cellIds.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_indices.data(), d_indices, h_indices.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_cellStart.data(), d_cellStart, h_cellStart.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_cellEnd.data(), d_cellEnd, h_cellEnd.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost);
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

	thrust::device_ptr<unsigned int> d_cellIds_ptr(d_cellIds);
	thrust::device_ptr<unsigned int> d_indices_ptr(d_indices);
	thrust::sequence(thrust::device, d_indices_ptr, d_indices_ptr + particles.Count);
	thrust::sort_by_key(thrust::device, d_cellIds_ptr, d_cellIds_ptr + particles.Count, d_indices_ptr);

	findCellStartEnd<<<BLOCKS_PER_GRID(particles.Count), THREADS_PER_BLOCK >> > (
		d_cellIds,
		d_cellStart,
		d_cellEnd,
		particles.Count);
	cudaDeviceSynchronize();
	CUDA_CHECK(cudaGetLastError());

	return cudaSuccess;
}