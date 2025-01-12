#pragma once
#include <vector>
#include <cuda_runtime.h>
#include "Particles.h"

class Grid {
public:
	Grid(int2 dimensions, float size, bool gpu = false);
	~Grid();
	void SetDevice(bool gpu);
	void UpdateGrid(const Particles &particles);
	void Resize(int2 dimensions, float size);

	// GPU data
	uint32_t *d_cellIds = nullptr;
	uint32_t *d_particleIndex = nullptr;
	uint32_t *d_cellStart = nullptr;
	uint32_t *d_cellEnd = nullptr;

	// CPU data
	std::vector<size_t> h_cellIds;
	std::vector<size_t> h_particleIndex;
	std::vector<size_t> h_cellStart;
	std::vector<size_t> h_cellEnd;

	int2 m_WorldDim;
	int2 m_Dim;
	float m_cellSize;
	bool m_IsCuda;

private:
	void updateGridCPU(const Particles &particles);
	cudaError_t updateGridCUDA(const Particles &particles);
	cudaError_t allocateGPUMemory(size_t particleCount);
	cudaError_t freeGPUMemory();
	cudaError_t transferToGPU();
	cudaError_t transferToCPU();
};
