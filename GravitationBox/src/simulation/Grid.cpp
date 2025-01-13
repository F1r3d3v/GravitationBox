#include "Grid.h"

Grid::Grid(glm::ivec2 dimensions, float size)
	: m_WorldDim(dimensions), m_CellSize(size)
{
	m_Dim = glm::ivec2(ceilf(m_WorldDim.x / m_CellSize), ceilf(m_WorldDim.y / m_CellSize));
	m_TotalCells = m_Dim.x * m_Dim.y;
}

void Grid::Resize(glm::ivec2 dimensions, float size)
{
	m_WorldDim = dimensions;
	m_CellSize = size;
	m_Dim = glm::ivec2(ceilf(m_WorldDim.x / m_CellSize), ceilf(m_WorldDim.y / m_CellSize));
	m_TotalCells = m_Dim.x * m_Dim.y;
}
