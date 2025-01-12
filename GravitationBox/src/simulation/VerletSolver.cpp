#include "VerletSolver.h"

VerletSolver::VerletSolver(Particles *p, Grid *g) : m_Particles(p), m_Grid(g)
{
	g->UpdateGrid(*m_Particles);
}

VerletSolver::~VerletSolver()
{
}

void VerletSolver::VerletCPU()
{
	UpdateParticles<true>();
	m_Grid->UpdateGrid(*m_Particles);
	CheckCollisions();
	UpdateParticles<false>();
}

template <bool floor>
glm::vec2 VerletSolver::CollideWall(glm::vec2 position, glm::vec2 velocity, float mass, glm::vec2 floorPosition)
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

glm::vec2 VerletSolver::SolveCollision(glm::vec2 positionA, glm::vec2 velocityA, glm::vec2 positionB, glm::vec2 velocityB)
{
	glm::vec2 positionDelta = positionB - positionA;
	float distance = glm::length(positionDelta);

	glm::vec2 Force(0.0f, 0.0f);

	float collideDistance = m_Params.Radius * 2.0f;
	if (distance < collideDistance)
	{
		glm::vec2 normal = positionDelta / distance;
		glm::vec2 relativeVelocity = velocityB - velocityA;
		float velocityAlongNormal = glm::dot(relativeVelocity, normal);
		glm::vec2 normalVelocity = velocityAlongNormal * normal;
		glm::vec2 tangentVelocity = relativeVelocity - normalVelocity;

		// Spring force
		Force = -m_Params.ParticlesStiffness * (collideDistance - distance) * normal;

		// Damping force
		if (velocityAlongNormal < 0.0f) Force += m_Params.ParticlesDampening * normalVelocity;

		// Friction force
		Force += m_Params.ParticlesFriction * tangentVelocity;
	}

	return Force;
}

glm::vec2 VerletSolver::CheckCollisionsInCell(uint32_t id, uint32_t cellId, glm::vec2 position, glm::vec2 velocity)
{
	glm::vec2 Force(0.0f, 0.0f);

	uint32_t startIdx = m_Grid->h_cellStart[cellId];

	if (startIdx == 0xFFFFFFFF) return Force;

	uint32_t endIdx = m_Grid->h_cellEnd[cellId];

	for (size_t i = startIdx; i < endIdx; ++i)
	{
		if (id == i) continue;

		glm::vec2 otherPosition(m_Particles->SortedPosX[i], m_Particles->SortedPosY[i]);
		glm::vec2 otherVelocity(m_Particles->SortedVelX[i], m_Particles->SortedVelY[i]);

		Force += SolveCollision(position, velocity, otherPosition, otherVelocity);
	}

	return Force;
}

void VerletSolver::CheckCollisions()
{
	for (uint32_t id = 0; id < m_Particles->Count; ++id)
	{
		glm::vec2 Force(0.0f, 0.0f);
		float x = m_Particles->SortedPosX[id];
		float y = m_Particles->SortedPosY[id];
		float vx = m_Particles->SortedVelX[id];
		float vy = m_Particles->SortedVelY[id];
		glm::ivec2 cell = glm::ivec2(x / m_Grid->m_cellSize, y / m_Grid->m_cellSize);
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
		m_Particles->ForceX[index] = m_Particles->SortedForceX[id] + Force.x;
		m_Particles->ForceY[index] = m_Particles->SortedForceY[id] + Force.y;
	}
}

glm::vec2 VerletSolver::CheckCollisionsWithWalls(uint32_t id)
{
	glm::vec2 Force = glm::vec2(0.0f);

	// Left and right walls
	if (m_Particles->PosX[id] < m_Params.Radius)
	{
		Force += CollideWall(
			glm::vec2(m_Particles->PosX[id], m_Particles->PosY[id]),
			glm::vec2(m_Particles->VelX[id], m_Particles->VelY[id]),
			m_Particles->Mass[id],
			glm::vec2(0.0f, m_Particles->PosY[id])
		);

		m_Particles->PosX[id] = m_Params.Radius;
		m_Particles->VelX[id] *= -1;
	}
	else if (m_Particles->PosX[id] > m_Params.DimX - m_Params.Radius)
	{
		Force += CollideWall(
			glm::vec2(m_Particles->PosX[id], m_Particles->PosY[id]),
			glm::vec2(m_Particles->VelX[id], m_Particles->VelY[id]),
			m_Particles->Mass[id],
			glm::vec2(m_Params.DimX, m_Particles->PosY[id])
		);

		m_Particles->PosX[id] = m_Params.DimX - m_Params.Radius;
		m_Particles->VelX[id] *= -1;
	}

	// Top and bottom walls
	if (m_Particles->PosY[id] < m_Params.Radius)
	{
		Force += CollideWall<true>(
			glm::vec2(m_Particles->PosX[id], m_Particles->PosY[id]),
			glm::vec2(m_Particles->VelX[id], m_Particles->VelY[id]),
			m_Particles->Mass[id],
			glm::vec2(m_Particles->PosX[id], 0.0f)
		);

		m_Particles->PosY[id] = m_Params.Radius;
		m_Particles->VelY[id] *= -1;
	}
	else if (m_Particles->PosY[id] > m_Params.DimY - m_Params.Radius)
	{
		Force += CollideWall(
			glm::vec2(m_Particles->PosX[id], m_Particles->PosY[id]),
			glm::vec2(m_Particles->VelX[id], m_Particles->VelY[id]),
			m_Particles->Mass[id],
			glm::vec2(m_Particles->PosX[id], m_Params.DimY)
		);

		m_Particles->PosY[id] = m_Params.DimY - m_Params.Radius;
		m_Particles->VelY[id] *= -1;
	}

	return Force;
}

template <bool stage1>
void VerletSolver::UpdateParticles()
{
	for (uint32_t id = 0; id < m_Particles->Count; ++id)
	{
		if constexpr (stage1)
		{
			glm::vec2 Position(m_Particles->PosX[id], m_Particles->PosY[id]);
			glm::vec2 Velocity(m_Particles->VelX[id], m_Particles->VelY[id]);
			glm::vec2 Force(m_Particles->ForceX[id], m_Particles->ForceY[id]);

			Position += Velocity * m_Params.Timestep + Force * (0.5f * m_Params.Timestep * m_Params.Timestep) / m_Particles->Mass[id];
			Velocity += 0.5f * Force * m_Params.Timestep / m_Particles->Mass[id];
			Force = glm::vec2(0.0f, m_Params.Gravity * m_Particles->Mass[id]);

			m_Particles->PosX[id] = Position.x;
			m_Particles->PosY[id] = Position.y;
			m_Particles->VelX[id] = Velocity.x;
			m_Particles->VelY[id] = Velocity.y;

			Force += CheckCollisionsWithWalls(id);

			m_Particles->ForceX[id] = Force.x;
			m_Particles->ForceY[id] = Force.y;
		}
		else
		{
			glm::vec2 Velocity(m_Particles->VelX[id], m_Particles->VelY[id]);
			glm::vec2 Force(m_Particles->ForceX[id], m_Particles->ForceY[id]);

			Velocity += 0.5f * Force * m_Params.Timestep / m_Particles->Mass[id];

			m_Particles->VelX[id] = Velocity.x;
			m_Particles->VelY[id] = Velocity.y;
		}
	}
}
