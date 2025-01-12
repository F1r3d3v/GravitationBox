#include "Grid.h"
#include <algorithm>
#include <cmath>
#include <numeric>

Grid::Grid(int2 dimensions, float size, bool gpu)
	: m_WorldDim(dimensions), m_cellSize(size), m_IsCuda(gpu), h_cellIds(0), h_particleIndex(0), h_cellStart(0), h_cellEnd(0)
{
	m_Dim = make_int2(ceilf(dimensions.x / m_cellSize), ceilf(dimensions.y / m_cellSize));
	int totalCells = m_Dim.x * m_Dim.y;
	h_cellStart.resize(totalCells);
	h_cellEnd.resize(totalCells);
}

void Grid::SetDevice(bool gpu) {
	if (m_IsCuda == gpu) return;

	if (gpu) {
		allocateGPUMemory(h_cellIds.size());
		transferToGPU();
	}
	else {
		transferToCPU();
		freeGPUMemory();
	}
	m_IsCuda = gpu;
}

Grid::~Grid() {
	if (m_IsCuda) {
		freeGPUMemory();
	}
}

void Grid::updateGridCPU(const Particles &particles) {
	if (particles.Count == 0) return;

	h_cellIds.resize(particles.TotalCount);
	h_particleIndex.resize(particles.TotalCount);

	for (size_t i = 0; i < particles.Count; i++)
	{
		int x = std::clamp((int)(particles.PosX[i] / m_cellSize), 0, m_Dim.x - 1);
		int y = std::clamp((int)(particles.PosY[i] / m_cellSize), 0, m_Dim.y - 1);
		h_cellIds[i] = y * m_Dim.x + x;
	}

	std::iota(h_particleIndex.begin(), h_particleIndex.begin() + particles.Count, 0);
	std::sort(h_particleIndex.begin(), h_particleIndex.begin() + particles.Count,
		[this](unsigned int a, unsigned int b) {
			return h_cellIds[a] < h_cellIds[b];
		});
	std::sort(h_cellIds.begin(), h_cellIds.begin() + particles.Count);
	std::fill(h_cellStart.begin(), h_cellStart.end(), 0xFFFFFFFF);

	int currentCell = h_cellIds[0];
	h_cellStart[currentCell] = 0;
	for (size_t i = 1; i < particles.Count; i++)
	{
		if (h_cellIds[i] != currentCell)
		{
			h_cellEnd[currentCell] = i;
			currentCell = h_cellIds[i];
			h_cellStart[currentCell] = i;
		}
	}
	h_cellEnd[currentCell] = particles.Count;

	for (size_t i = 0; i < particles.Count; i++)
	{
		uint32_t index = h_particleIndex[i];
		particles.SortedPosX[i] = particles.PosX[index];
		particles.SortedPosY[i] = particles.PosY[index];
		particles.SortedVelX[i] = particles.VelX[index];
		particles.SortedVelY[i] = particles.VelY[index];
		particles.SortedForceX[i] = particles.ForceX[index];
		particles.SortedForceY[i] = particles.ForceY[index];
	}
}

void Grid::UpdateGrid(const Particles &particles) {
	if (m_IsCuda) {
		updateGridCUDA(particles);
	}
	else {
		updateGridCPU(particles);
	}
}

void Grid::Resize(int2 dimensions, float size)
{
	m_WorldDim = dimensions;
	m_cellSize = size;
	m_Dim = make_int2(ceilf(dimensions.x / m_cellSize), ceilf(dimensions.y / m_cellSize));
	int totalCells = m_Dim.x * m_Dim.y;
	h_cellStart.resize(totalCells);
	h_cellEnd.resize(totalCells);
	if (m_IsCuda)
	{
		freeGPUMemory();
	}
}
