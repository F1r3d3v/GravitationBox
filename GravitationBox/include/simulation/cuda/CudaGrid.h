#pragma once
#include "Grid.h"
#include "ParticleSystem.h"
#include <glm.hpp>

class CudaGrid : public Grid
{
public:
	CudaGrid(glm::ivec2 dimensions, float size);
	~CudaGrid();
	void UpdateGrid(ParticleSystem *p) override;
	void Resize(glm::ivec2 dimensions, float size) override;

	// GPU data
	uint32_t *d_cellIds;
	uint32_t *d_particleIndex;
	uint32_t *d_cellStart;
	uint32_t *d_cellEnd;
};
