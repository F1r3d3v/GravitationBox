#pragma once

class ParticleSystem;

class ParticleSolver
{
public:
	ParticleSolver() = default;
	virtual ~ParticleSolver() = default;

	virtual void Solve(ParticleSystem *system) = 0;
};