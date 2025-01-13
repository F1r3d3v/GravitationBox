#include "cpu/CpuGrid.h"
#include <algorithm>
#include <cmath>
#include <numeric>

CpuGrid::CpuGrid(glm::ivec2 dimensions, float size)
	: Grid(dimensions, size)
{
	h_cellStart.resize(m_TotalCells);
	h_cellEnd.resize(m_TotalCells);
}

void CpuGrid::UpdateGrid(ParticleSystem *p)
{
	if (p->Count == 0) return;

	h_cellIds.resize(p->TotalCount);
	h_particleIndex.resize(p->TotalCount);

	for (uint32_t i = 0; i < p->Count; i++)
	{
		int x = std::clamp((int)(p->PosX[i] / m_CellSize), 0, m_Dim.x - 1);
		int y = std::clamp((int)(p->PosY[i] / m_CellSize), 0, m_Dim.y - 1);
		h_cellIds[i] = y * m_Dim.x + x;
	}

	std::iota(h_particleIndex.begin(), h_particleIndex.begin() + p->Count, 0);
	std::sort(h_particleIndex.begin(), h_particleIndex.begin() + p->Count,
		[this](unsigned int a, unsigned int b) {
			return h_cellIds[a] < h_cellIds[b];
		});
	std::sort(h_cellIds.begin(), h_cellIds.begin() + p->Count);
	std::fill(h_cellStart.begin(), h_cellStart.end(), 0xFFFFFFFF);

	int currentCell = h_cellIds[0];
	h_cellStart[currentCell] = 0;
	for (uint32_t i = 1; i < p->Count; i++)
	{
		if (h_cellIds[i] != currentCell)
		{
			h_cellEnd[currentCell] = i;
			currentCell = h_cellIds[i];
			h_cellStart[currentCell] = i;
		}
	}
	h_cellEnd[currentCell] = p->Count;

	for (uint32_t i = 0; i < p->Count; i++)
	{
		uint32_t index = h_particleIndex[i];
		p->SortedPosX[i] = p->PosX[index];
		p->SortedPosY[i] = p->PosY[index];
		p->SortedVelX[i] = p->VelX[index];
		p->SortedVelY[i] = p->VelY[index];
		p->SortedForceX[i] = p->ForceX[index];
		p->SortedForceY[i] = p->ForceY[index];
	}
}

void CpuGrid::Resize(glm::ivec2 dimensions, float size)
{
	Grid::Resize(dimensions, size);
	h_cellStart.resize(m_TotalCells);
	h_cellEnd.resize(m_TotalCells);
}
