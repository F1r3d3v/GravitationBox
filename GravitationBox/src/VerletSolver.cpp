#include "VerletSolver.h"

VerletSolver::VerletSolver(Particles *p, Grid *g) : m_Particles(p), m_Grid(g)
{
	g->updateGrid(*m_Particles);
}

VerletSolver::~VerletSolver()
{
}

void VerletSolver::VerletCPU(float dt)
{
	UpdateParticles(dt);
	m_Grid->updateGrid(*m_Particles);
	CheckCollisionsWithParticles();
	CheckCollisionsWithWalls();
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

void VerletSolver::CheckCollisionsWithWalls()
{
	for (uint32_t id = 0; id < m_Particles->Count; ++id)
	{
		if (m_Particles->PosX[id] < m_Particles->Radius || m_Particles->PosX[id] > m_Grid->m_WorldDim.x - m_Particles->Radius) { // Bounce off left/right walls
			if (m_Particles->PosX[id] < m_Particles->Radius) m_Particles->PosX[id] = m_Particles->Radius;
			if (m_Particles->PosX[id] > m_Grid->m_WorldDim.x - m_Particles->Radius) m_Particles->PosX[id] = m_Grid->m_WorldDim.x - m_Particles->Radius;
			SetParticleVelocity(id, glm::vec2(-m_Particles->VelX[id] * Config::DAMPENING, m_Particles->VelY[id]));
		}

		if (m_Particles->PosY[id] <  m_Particles->Radius || m_Particles->PosY[id] > m_Grid->m_WorldDim.y - m_Particles->Radius) { // Bounce off top/bottom walls
			if (m_Particles->PosY[id] < m_Particles->Radius) m_Particles->PosY[id] = m_Particles->Radius;
			if (m_Particles->PosY[id] > m_Grid->m_WorldDim.y - m_Particles->Radius) m_Particles->PosY[id] = m_Grid->m_WorldDim.y - m_Particles->Radius;
			SetParticleVelocity(id, glm::vec2(m_Particles->VelX[id], -m_Particles->VelY[id] * Config::DAMPENING));
		}
	}
}

void VerletSolver::CheckCollisionsWithParticles()
{
	for (uint32_t id = 0; id < m_Particles->Count; ++id)
	{
		glm::vec2 pos_i = GetParticlePosition(id);
		glm::vec2 vel_i = GetParticleVelocity(id);
		float radius = m_Particles->Radius;

		// Use the grid to get neighboring particles for efficiency
		std::vector<int> neighbors = m_Grid->getNeighborsCPU(pos_i.x, pos_i.y, radius * 2.0f, *m_Particles);

		for (int j : neighbors)
		{
			if (j == id) continue;

			glm::vec2 pos_j = GetParticlePosition(j);
			glm::vec2 vel_j = GetParticleVelocity(j);
			glm::vec2 delta = pos_i - pos_j;
			float dist2 = delta.x * delta.x + delta.y * delta.y;
			float minDist = 2.0f * radius;

			if (dist2 < minDist * minDist)
			{
				float dist = sqrtf(dist2);
				if (dist == 0.0f) dist = minDist;

				// Normalized collision vector
				glm::vec2 normal = delta / dist;

				// Relative velocity along the normal direction
				float relVel = glm::dot(vel_i - vel_j, normal);

				// Apply impulse
				float restitution = Config::DAMPENING;
				float impulse = -(1.0f + restitution) * relVel;
				impulse /= (1.0f / m_Particles->Mass[id] + 1.0f / m_Particles->Mass[j]);

				glm::vec2 impulseVec = normal * impulse;

				// Update velocities
				vel_i += impulseVec / m_Particles->Mass[id];
				vel_j -= impulseVec / m_Particles->Mass[j];

				SetParticleVelocity(id, vel_i);
				SetParticleVelocity(j, vel_j);

				// Position correction
				float overlap = 0.5f * (minDist - dist);
				pos_i += normal * overlap;
				pos_j -= normal * overlap;

				SetParticlePosition(id, pos_i);
				SetParticlePosition(j, pos_j);
			}
		}
	}
}

void VerletSolver::UpdateParticles(float dt)
{
	for (uint32_t id = 0; id < m_Particles->Count; ++id)
	{
		float2 acc = make_float2(0.0f, Config::GRAVITY);
		float new_posx = m_Particles->PosX[id] + m_Particles->VelX[id] * dt + acc.x * (dt * dt * 0.5);
		float new_posy = m_Particles->PosY[id] + m_Particles->VelY[id] * dt + acc.y * (dt * dt * 0.5);
		float new_vely = m_Particles->VelY[id] + acc.y * dt;
		m_Particles->PosX[id] = new_posx;
		m_Particles->PosY[id] = new_posy;
		m_Particles->VelY[id] = new_vely;
	}
}
