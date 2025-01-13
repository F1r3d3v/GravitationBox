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

	uint32_t *GetCellIds() override {
		return d_cellIds;
	};

	uint32_t *GetParticleIndex() override {
		return d_particleIndex;
	};

	uint32_t *GetCellStart() override {
		return d_cellStart;
	};

	uint32_t *GetCellEnd() override {
		return d_cellEnd;
	};

	// GPU data
	uint32_t *d_cellIds = nullptr;
	uint32_t *d_particleIndex = nullptr;
	uint32_t *d_cellStart = nullptr;
	uint32_t *d_cellEnd = nullptr;
};
