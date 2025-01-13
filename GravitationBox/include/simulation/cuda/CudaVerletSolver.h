#pragma once
#include "Grid.h"
#include "ParticleSolver.h"

class CudaVerletSolver : public ParticleSolver
{
public:
	CudaVerletSolver(Grid *g);
	~CudaVerletSolver() = default;

	void Solve(ParticleSystem *p) override;

private:
	Grid *m_Grid;

};