#include "Grid.h"
#include <algorithm>
#include <cmath>

Grid::Grid(int2 dimensions, float size, bool gpu)
	: m_WorldDim(dimensions), m_cellSize(size), m_IsCuda(gpu), h_cellIds(0), h_indices(0), h_cellStart(0), h_cellEnd(0)
{
	m_Dim = make_int2(ceilf(dimensions.x / m_cellSize), ceilf(dimensions.y / m_cellSize));
	int totalCells = m_Dim.x * m_Dim.y;
	h_cellStart.resize(totalCells);
	h_cellEnd.resize(totalCells);
}

void Grid::setDevice(bool gpu) {
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

	h_cellIds.resize(particles.Count);
	h_indices.resize(particles.Count);

	for (size_t i = 0; i < particles.Count; i++) {
		int x = std::clamp((int)(particles.PosX[i] / m_cellSize), 0, m_Dim.x - 1);
		int y = std::clamp((int)(particles.PosY[i] / m_cellSize), 0, m_Dim.y - 1);
		h_cellIds[i] = y * m_Dim.x + x;
		h_indices[i] = i;
	}

	std::sort(h_indices.begin(), h_indices.end(),
		[this](unsigned int a, unsigned int b) {
			return h_cellIds[a] < h_cellIds[b];
		});

	std::sort(h_cellIds.begin(), h_cellIds.end());

	std::fill(h_cellStart.begin(), h_cellStart.end(), 0);
	std::fill(h_cellEnd.begin(), h_cellEnd.end(), 0);

	int currentCell = h_cellIds[0];
	h_cellStart[currentCell] = 0;

	for (size_t i = 1; i < particles.Count; i++) {
		if (h_cellIds[i] != currentCell) {
			h_cellEnd[currentCell] = i;
			currentCell = h_cellIds[i];
			h_cellStart[currentCell] = i;
		}
	}
	h_cellEnd[currentCell] = particles.Count;
}

void Grid::updateGrid(const Particles &particles) {
	if (m_IsCuda) {
		updateGridCUDA(particles);
	}
	else {
		updateGridCPU(particles);
	}
}

int Grid::getGridIndex(float x, float y) const {
	int cellX = x / m_cellSize;
	int cellY = y / m_cellSize;
	return cellY * m_Dim.x + cellX;
}

std::vector<int> Grid::getNeighborsCPU(float x, float y, float radius, const Particles &particles) {
	std::vector<int> neighborIndices;
	int cellId = getGridIndex(x, y);
	float radiusSq = radius * radius;

	for (int dy = -1; dy <= 1; dy++) {
		for (int dx = -1; dx <= 1; dx++) {
			int neighborCellId = cellId + (dy * m_Dim.x) + dx;

			if (neighborCellId < 0 || neighborCellId >= h_cellStart.size()) continue;

			unsigned int start = h_cellStart[neighborCellId];
			unsigned int end = h_cellEnd[neighborCellId];

			for (unsigned int i = start; i < end; i++) {
				unsigned int idx = h_indices[i];
				float dx = x - particles.PosX[idx];
				float dy = y - particles.PosY[idx];
				float distSq = dx * dx + dy * dy;

				if (distSq <= radiusSq) {
					neighborIndices.push_back(idx);
				}
			}
		}
	}
	return neighborIndices;
}

void Grid::Resize(int2 dimensions, float size)
{
	m_WorldDim = dimensions;
	m_cellSize = size;
	m_Dim = make_int2(ceilf(dimensions.x / m_cellSize), ceilf(dimensions.y / m_cellSize));
	int totalCells = m_Dim.x * m_Dim.y;
	h_cellStart.resize(totalCells);
	h_cellEnd.resize(totalCells);
}
