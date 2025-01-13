#pragma once
#include <vector>
#include "Grid.h"
#include "ParticleSystem.h"
#include <glm.hpp>

class CpuGrid : public Grid
{
public:
	CpuGrid(glm::ivec2 dimensions, float size);
	~CpuGrid() = default;
	void UpdateGrid(ParticleSystem *p) override;
	void Resize(glm::ivec2 dimensions, float size) override;

	// CPU data
	std::vector<size_t> h_cellIds;
	std::vector<size_t> h_particleIndex;
	std::vector<size_t> h_cellStart;
	std::vector<size_t> h_cellEnd;
};
