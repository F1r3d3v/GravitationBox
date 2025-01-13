#pragma once
#include "ParticleSystem.h"
#include <glm.hpp>

class Grid
{
public:
	Grid(glm::ivec2 dimensions, float size);
	virtual ~Grid() = default;
	virtual void UpdateGrid(ParticleSystem *p) = 0;
	virtual void Resize(glm::ivec2 dimensions, float size);
	virtual bool GetStartEnd(uint32_t cellId, uint32_t &start, uint32_t &end) = 0;
	virtual uint32_t GetParticleIndex(uint32_t index) = 0;


	glm::ivec2 m_WorldDim;
	glm::ivec2 m_Dim;
	float m_CellSize;
	uint32_t m_TotalCells;
};
