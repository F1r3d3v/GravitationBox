#include "VerletSolver.h"

VerletSolver::VerletSolver(Particles *p, Grid *g) : m_Particles(p), m_Grid(g)
{
	g->updateGrid(*m_Particles);
}

VerletSolver::~VerletSolver()
{
}

void VerletSolver::VerletCPU()
{
	UpdateParticles<true>();
	m_Grid->updateGrid(*m_Particles);
	CheckCollisions();
	UpdateParticles<false>();
}

void VerletSolver::SetParticlePosition(uint32_t id, glm::vec2 pos)
{
	m_Particles->PosX[id] = pos.x;
	m_Particles->PosY[id] = pos.y;
}

glm::vec2 VerletSolver::GetParticlePosition(uint32_t id)
{
	return glm::vec2(m_Particles->PosX[id], m_Particles->PosY[id]);
}

void VerletSolver::SetParticleVelocity(uint32_t id, glm::vec2 vel)
{
	m_Particles->VelX[id] = vel.x;
	m_Particles->VelY[id] = vel.y;
}

glm::vec2 VerletSolver::GetParticleVelocity(uint32_t id)
{
	return glm::vec2(m_Particles->VelX[id], m_Particles->VelY[id]);
}

glm::vec2 VerletSolver::SolveCollision(glm::vec2 positionA, glm::vec2 velocityA, glm::vec2 positionB, glm::vec2 velocityB)
{
	glm::vec2 positionDelta = positionB - positionA;
	float distance = glm::length(positionDelta);

	glm::vec2 Force(0.0f, 0.0f);

	float collideDistance = m_Params.Radius * 2.0f;
	if (distance < collideDistance)
	{
		// Spring force
		glm::vec2 normal = positionDelta / distance;
		Force = -m_Params.ParticleStiffness * (collideDistance - distance) * normal;

		// Damping force
		glm::vec2 relativeVelocity = velocityB - velocityA;
		float velocityAlongNormal = glm::dot(relativeVelocity, normal);
		glm::vec2 normalVelocity = velocityAlongNormal * normal;
		if (velocityAlongNormal < 0)
		{
			Force += m_Params.ParticleDampening * normalVelocity;
		}

		// Friction force
		glm::vec2 tangentVelocity = relativeVelocity - normalVelocity;
		Force += m_Params.ParticleShear * tangentVelocity;
	}

	return Force;
}

glm::vec2 VerletSolver::CheckCollisionInCell(uint32_t id, uint32_t cellId, glm::vec2 position, glm::vec2 velocity)
{
	glm::vec2 Force(0.0f, 0.0f);

	uint32_t startIdx = m_Grid->h_cellStart[cellId];

	if (startIdx == 0xffffffff) return Force;

	uint32_t endIdx = m_Grid->h_cellEnd[cellId];

	for (size_t i = startIdx; i < endIdx; ++i)
	{
		uint32_t j = m_Grid->h_indices[i];
		if (id == j) continue;

		glm::vec2 otherPosition(m_Particles->PosX[j], m_Particles->PosY[j]);
		glm::vec2 otherVelocity(m_Particles->VelX[j], m_Particles->VelY[j]);

		Force += SolveCollision(position, velocity, otherPosition, otherVelocity);
	}

	return Force;
}

void VerletSolver::CheckCollisions()
{
	for (uint32_t id = 0; id < m_Particles->Count; ++id)
	{
		glm::vec2 Force(0.0f, 0.0f);
		float x = m_Particles->PosX[id];
		float y = m_Particles->PosY[id];
		glm::ivec2 cell = { static_cast<int>(x / m_Grid->m_cellSize), static_cast<int>(y / m_Grid->m_cellSize) };
		cell.x = std::clamp(cell.x, 0, m_Grid->m_Dim.x - 1);
		cell.y = std::clamp(cell.y, 0, m_Grid->m_Dim.y - 1);

		for (int i = -1; i <= 1; ++i)
		{
			for (int j = -1; j <= 1; ++j)
			{
				int2 neighbour = { cell.x + i, cell.y + j };
				if (neighbour.x < 0 || neighbour.x >= m_Grid->m_Dim.x || neighbour.y < 0 || neighbour.y >= m_Grid->m_Dim.y) continue;

				uint32_t cellId = neighbour.y * m_Grid->m_Dim.x + neighbour.x;
				Force += CheckCollisionInCell(id, cellId, { x, y }, { m_Particles->VelX[id], m_Particles->VelY[id] });
			}
		}

		m_Particles->ForceX[id] += Force.x;
		m_Particles->ForceY[id] += Force.y;
	}
}

void VerletSolver::CheckCollisionsWithWalls(uint32_t id)
{
	// Left and right walls
	if (m_Particles->PosX[id] < m_Params.Radius)
	{
		m_Particles->PosX[id] = m_Params.Radius;
		m_Particles->VelX[id] *= -m_Params.WallDampening;
	}
	else if (m_Particles->PosX[id] > m_Params.DimX - m_Params.Radius)
	{
		m_Particles->PosX[id] = m_Params.DimX - m_Params.Radius;
		m_Particles->VelX[id] *= -m_Params.WallDampening;
	}

	// Top and bottom walls
	if (m_Particles->PosY[id] < m_Params.Radius)
	{
		m_Particles->PosY[id] = m_Params.Radius;
		m_Particles->VelY[id] *= -m_Params.WallDampening;
	}
	else if (m_Particles->PosY[id] > m_Params.DimY - m_Params.Radius)
	{
		m_Particles->PosY[id] = m_Params.DimY - m_Params.Radius;
		m_Particles->VelY[id] *= -m_Params.WallDampening;
	}
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
			Force = glm::vec2(0.0f, m_Params.Gravity);

			m_Particles->PosX[id] = Position.x;
			m_Particles->PosY[id] = Position.y;
			m_Particles->VelX[id] = Velocity.x;
			m_Particles->VelY[id] = Velocity.y;
			m_Particles->ForceX[id] = Force.x;
			m_Particles->ForceY[id] = Force.y;

			CheckCollisionsWithWalls(id);
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
