#include "cuda/CudaGrid.h"
#include "utils/cuda_helper.h"
#include "utils/cuda_helper_math.h"

#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>
#include <device_launch_parameters.h>

__global__ void CalculateCellIds(float *posX, float *posY, uint32_t *cellIds, uint32_t particleCount, float cellSize, int2 gridDims)
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

__global__ void SetCellStartEnd(uint32_t *sortedCellIds, uint32_t *cellStart, uint32_t *cellEnd, uint32_t particleCount)
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

CudaGrid::CudaGrid(glm::ivec2 dimensions, float size)
	: Grid(dimensions, size)
{
	CUDA_CHECK_NR(cudaMalloc(&d_cellStart, m_TotalCells * sizeof(uint32_t)));
	CUDA_CHECK_NR(cudaMalloc(&d_cellEnd, m_TotalCells * sizeof(uint32_t)));
}

CudaGrid::~CudaGrid()
{
	CUDA_CHECK_NR(cudaFree(d_cellIds));
	CUDA_CHECK_NR(cudaFree(d_particleIndex));
	CUDA_CHECK_NR(cudaFree(d_cellStart));
	CUDA_CHECK_NR(cudaFree(d_cellEnd));
}

void CudaGrid::UpdateGrid(ParticleSystem *p)
{
	if (d_cellIds == nullptr)
	{
		CUDA_CHECK_NR(cudaMalloc(&d_cellIds, p->TotalCount * sizeof(uint32_t)));
		CUDA_CHECK_NR(cudaMalloc(&d_particleIndex, p->TotalCount * sizeof(uint32_t)));
	}

	if (p->Count == 0) return;

	CalculateCellIds << <BLOCKS_PER_GRID(p->Count), THREADS_PER_BLOCK >> > (
		p->PosX,
		p->PosY,
		d_cellIds,
		p->Count,
		m_CellSize,
		make_int2(m_Dim.x, m_Dim.y));
	cudaDeviceSynchronize();
	CUDA_CHECK_NR(cudaGetLastError());

	thrust::device_ptr<uint32_t> d_cellIds_ptr(d_cellIds);
	thrust::device_ptr<uint32_t> d_indices_ptr(d_particleIndex);
	thrust::device_ptr<uint32_t> d_cellStart_ptr(d_cellStart);
	thrust::sequence(thrust::device, d_indices_ptr, d_indices_ptr + p->Count);
	thrust::sort_by_key(thrust::device, d_cellIds_ptr, d_cellIds_ptr + p->Count, d_indices_ptr);
	thrust::fill(thrust::device, d_cellStart_ptr, d_cellStart_ptr + m_TotalCells, 0xFFFFFFFF);

	uint32_t memsize = (THREADS_PER_BLOCK + 1) * sizeof(uint32_t);
	SetCellStartEnd << <BLOCKS_PER_GRID(p->Count), THREADS_PER_BLOCK, memsize >> > (
		d_cellIds,
		d_cellStart,
		d_cellEnd,
		p->Count);
	cudaDeviceSynchronize();
	CUDA_CHECK_NR(cudaGetLastError());

	Reorder << <BLOCKS_PER_GRID(p->Count), THREADS_PER_BLOCK >> > (
		d_particleIndex,
		p->PosX, p->SortedPosX,
		p->PosY, p->SortedPosY,
		p->VelX, p->SortedVelX,
		p->VelY, p->SortedVelY,
		p->ForceX, p->SortedForceX,
		p->ForceY, p->SortedForceY,
		p->Count);
	cudaDeviceSynchronize();
	CUDA_CHECK_NR(cudaGetLastError());
}

void CudaGrid::Resize(glm::ivec2 dimensions, float size)
{
	Grid::Resize(dimensions, size);
	CUDA_CHECK_NR(cudaFree(d_cellStart));
	CUDA_CHECK_NR(cudaFree(d_cellEnd));
	CUDA_CHECK_NR(cudaMalloc(&d_cellStart, m_TotalCells * sizeof(uint32_t)));
	CUDA_CHECK_NR(cudaMalloc(&d_cellEnd, m_TotalCells * sizeof(uint32_t)));
}