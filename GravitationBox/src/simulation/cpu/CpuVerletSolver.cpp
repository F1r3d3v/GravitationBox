#include "cpu/CpuVerletSolver.h"
#include <algorithm>

CpuVerletSolver::CpuVerletSolver(Grid *g) : m_Grid(g)
{
}

void CpuVerletSolver::Solve(ParticleSystem *p)
{
	UpdateParticles<true>(p);
	m_Grid->UpdateGrid(p);
	CheckCollisions(p);
	UpdateParticles<false>(p);
}

template <bool floor>
glm::vec2 CpuVerletSolver::CollideWall(glm::vec2 position, glm::vec2 velocity, float mass, glm::vec2 floorPosition)
{
	glm::vec2 positionDelta = floorPosition - position;

	float distance = glm::length(positionDelta);

	glm::vec2 Force = glm::vec2(0.0f);

	if (distance < m_Params.Radius)
	{
		glm::vec2 normal = positionDelta / distance;

		glm::vec2 relativeVelocity = -velocity;
		float velocityAlongNormal = glm::dot(relativeVelocity, normal);
		glm::vec2 normalVelocity = velocityAlongNormal * normal;
		glm::vec2 tangentVelocity = relativeVelocity - normalVelocity;

		if constexpr (floor)
		{
			// Normal force
			Force = -mass * m_Params.Gravity * normal;
		}

		// Damping force
		if (velocityAlongNormal < 0.0f) Force += m_Params.WallDampening * -normalVelocity;

		// Friction force
		Force += m_Params.WallFriction * tangentVelocity;
	}

	return Force;
}

glm::vec2 CpuVerletSolver::SolveCollision(glm::vec2 positionA, glm::vec2 velocityA, glm::vec2 positionB, glm::vec2 velocityB, ParticleSystem *p)
{
	glm::vec2 positionDelta = positionB - positionA;
	float distance = glm::length(positionDelta);

	glm::vec2 Force(0.0f, 0.0f);

	float collideDistance = p->m_Params.Radius * 2.0f;
	if (distance < collideDistance)
	{
		glm::vec2 normal = positionDelta / distance;
		glm::vec2 relativeVelocity = velocityB - velocityA;
		float velocityAlongNormal = glm::dot(relativeVelocity, normal);
		glm::vec2 normalVelocity = velocityAlongNormal * normal;
		glm::vec2 tangentVelocity = relativeVelocity - normalVelocity;

		// Spring force
		Force = -p->m_Params.ParticlesStiffness * (collideDistance - distance) * normal;

		// Damping force
		if (velocityAlongNormal < 0.0f) Force += p->m_Params.ParticlesDampening * normalVelocity;

		// Friction force
		Force += p->m_Params.ParticlesFriction * tangentVelocity;
	}

	return Force;
}

glm::vec2 CpuVerletSolver::CheckCollisionsInCell(uint32_t id, uint32_t cellId, glm::vec2 position, glm::vec2 velocity, ParticleSystem *p)
{
	glm::vec2 Force(0.0f, 0.0f);

	uint32_t startIdx = m_Grid->h_cellStart[cellId];

	if (startIdx == 0xFFFFFFFF) return Force;

	uint32_t endIdx = m_Grid->h_cellEnd[cellId];

	for (size_t i = startIdx; i < endIdx; ++i)
	{
		if (id == i) continue;

		glm::vec2 otherPosition(p->SortedPosX[i], p->SortedPosY[i]);
		glm::vec2 otherVelocity(p->SortedVelX[i], p->SortedVelY[i]);

		Force += SolveCollision(position, velocity, otherPosition, otherVelocity);
	}

	return Force;
}

void CpuVerletSolver::CheckCollisions(ParticleSystem *p)
{
	for (uint32_t id = 0; id < p->Count; ++id)
	{
		glm::vec2 Force(0.0f, 0.0f);
		float x = p->SortedPosX[id];
		float y = p->SortedPosY[id];
		float vx = p->SortedVelX[id];
		float vy = p->SortedVelY[id];
		glm::ivec2 cell = glm::ivec2(x / m_Grid->m_CellSize, y / m_Grid->m_CellSize);
		cell.x = std::clamp(cell.x, 0, m_Grid->m_Dim.x - 1);
		cell.y = std::clamp(cell.y, 0, m_Grid->m_Dim.y - 1);

		for (int i = -1; i <= 1; ++i)
		{
			for (int j = -1; j <= 1; ++j)
			{
				glm::ivec2 neighbour = glm::ivec2(cell.x + i, cell.y + j);
				if (neighbour.x < 0 || neighbour.x >= m_Grid->m_Dim.x || neighbour.y < 0 || neighbour.y >= m_Grid->m_Dim.y) continue;

				uint32_t cellId = neighbour.y * m_Grid->m_Dim.x + neighbour.x;
				Force += CheckCollisionsInCell(id, cellId, glm::vec2(x, y), glm::vec2(vx, vy));
			}
		}

		uint32_t index = m_Grid->h_particleIndex[id];
		p->ForceX[index] = p->SortedForceX[id] + Force.x;
		p->ForceY[index] = p->SortedForceY[id] + Force.y;
	}
}

glm::vec2 CpuVerletSolver::CheckCollisionsWithWalls(uint32_t id, ParticleSystem *p)
{
	glm::vec2 Force = glm::vec2(0.0f);

	// Left and right walls
	if (p->PosX[id] < p->m_Params.Radius)
	{
		Force += CollideWall(
			glm::vec2(p->PosX[id], p->PosY[id]),
			glm::vec2(p->VelX[id], p->VelY[id]),
			p->Mass[id],
			glm::vec2(0.0f, p->PosY[id])
		);

		p->PosX[id] = p->m_Params.Radius;
		p->VelX[id] *= -1;
	}
	else if (p->PosX[id] > p->m_Params.DimX - p->m_Params.Radius)
	{
		Force += CollideWall(
			glm::vec2(p->PosX[id], p->PosY[id]),
			glm::vec2(p->VelX[id], p->VelY[id]),
			p->Mass[id],
			glm::vec2(p->m_Params.DimX, p->PosY[id])
		);

		p->PosX[id] = p->m_Params.DimX - p->m_Params.Radius;
		p->VelX[id] *= -1;
	}

	// Top and bottom walls
	if (p->PosY[id] < p->m_Params.Radius)
	{
		Force += CollideWall<true>(
			glm::vec2(p->PosX[id], p->PosY[id]),
			glm::vec2(p->VelX[id], p->VelY[id]),
			p->Mass[id],
			glm::vec2(p->PosX[id], 0.0f)
		);

		p->PosY[id] = p->m_Params.Radius;
		p->VelY[id] *= -1;
	}
	else if (p->PosY[id] > p->m_Params.DimY - p->m_Params.Radius)
	{
		Force += CollideWall(
			glm::vec2(p->PosX[id], p->PosY[id]),
			glm::vec2(p->VelX[id], p->VelY[id]),
			p->Mass[id],
			glm::vec2(p->PosX[id], p->m_Params.DimY)
		);

		p->PosY[id] = p->m_Params.DimY - p->m_Params.Radius;
		p->VelY[id] *= -1;
	}

	return Force;
}

template <bool stage1>
void CpuVerletSolver::UpdateParticles(ParticleSystem *p)
{
	for (uint32_t id = 0; id < p->Count; ++id)
	{
		if constexpr (stage1)
		{
			glm::vec2 Position(p->PosX[id], p->PosY[id]);
			glm::vec2 Velocity(p->VelX[id], p->VelY[id]);
			glm::vec2 Force(p->ForceX[id], p->ForceY[id]);

			Position += Velocity * p->m_Params.Timestep + Force * (0.5f * p->m_Params.Timestep * p->m_Params.Timestep) / p->Mass[id];
			Velocity += 0.5f * Force * p->m_Params.Timestep / p->Mass[id];
			Force = glm::vec2(0.0f, p->m_Params.Gravity * p->Mass[id]);

			p->PosX[id] = Position.x;
			p->PosY[id] = Position.y;
			p->VelX[id] = Velocity.x;
			p->VelY[id] = Velocity.y;

			Force += CheckCollisionsWithWalls(id);

			p->ForceX[id] = Force.x;
			p->ForceY[id] = Force.y;
		}
		else
		{
			glm::vec2 Velocity(p->VelX[id], p->VelY[id]);
			glm::vec2 Force(p->ForceX[id], p->ForceY[id]);

			Velocity += 0.5f * Force * p->m_Params.Timestep / p->Mass[id];

			p->VelX[id] = Velocity.x;
			p->VelY[id] = Velocity.y;
		}
	}
}
