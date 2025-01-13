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

	uint32_t *GetCellIds() override {
		return h_cellIds.data();
	};

	uint32_t *GetParticleIndex() override {
		return h_particleIndex.data();
	};

	uint32_t *GetCellStart() override {
		return h_cellStart.data();
	};

	uint32_t *GetCellEnd() override {
		return h_cellEnd.data();
	};

	// CPU data
	std::vector<uint32_t> h_cellIds;
	std::vector<uint32_t> h_particleIndex;
	std::vector<uint32_t> h_cellStart;
	std::vector<uint32_t> h_cellEnd;
};
