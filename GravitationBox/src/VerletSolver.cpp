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
	UpdateParticles<false>(dt);
	m_Grid->updateGrid(*m_Particles);
	CheckCollisionsWithParticles<false>();
	CheckCollisionsWithWalls<false>();

	UpdateParticles<true>(dt);
	m_Grid->updateGrid(*m_Particles);
	CheckCollisionsWithParticles<true>();
	CheckCollisionsWithWalls<true>();
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
	m_Particles->prevPosX[id] = m_Particles->PosX[id] - vel.x;
	m_Particles->prevPosY[id] = m_Particles->PosY[id] - vel.y;
}

glm::vec2 VerletSolver::GetParticleVelocity(uint32_t id)
{
	//return glm::vec2(m_Particles->prevPosX[id], m_Particles->prevPosY[id]);
	return glm::vec2(m_Particles->PosX[id] - m_Particles->prevPosX[id], m_Particles->PosY[id] - m_Particles->prevPosY[id]);
}

template <bool preserveImpulse>
void VerletSolver::CheckCollisionsWithParticles()
{
	for (uint32_t id = 0; id < m_Particles->Count; ++id)
	{
		glm::vec2 pos_i = GetParticlePosition(id);

		// Retrieve neighboring particles
		std::vector<int> neighbors = m_Grid->getNeighborsCPU(pos_i.x, pos_i.y, m_Particles->Radius * 2.0f, *m_Particles);

		for (int j : neighbors)
		{
			if (j <= id) continue; // Avoid double-checking pairs

			glm::vec2 pos_j = GetParticlePosition(j);
			glm::vec2 delta = pos_i - pos_j;
			float dist2 = glm::dot(delta, delta);
			float minDist = m_Particles->Radius * 2.0f;

			if (dist2 < minDist * minDist)
			{
				float dist = sqrtf(dist2);
				if (dist == 0.0f) dist = minDist;

				glm::vec2 normal = delta / dist;
				float inv_mass_i = 1.0f / m_Particles->Mass[id];
				float inv_mass_j = 1.0f / m_Particles->Mass[j];
				constexpr float slop = 0.05f;
				constexpr float percent = 0.5f;
				glm::vec2 correction = (fmaxf(minDist - dist - slop, 0.0f) / (inv_mass_i + inv_mass_j)) * percent * normal;

				// Adjust positions to resolve overlap
				pos_i += inv_mass_i * correction;
				pos_j -= inv_mass_j * correction;

				SetParticlePosition(id, pos_i);
				SetParticlePosition(j, pos_j);

				if constexpr (preserveImpulse)
				{
					// Preserve impulse
					glm::vec2 vel_i = pos_i - glm::vec2(m_Particles->prevPosX[id], m_Particles->prevPosY[id]);
					glm::vec2 vel_j = pos_j - glm::vec2(m_Particles->prevPosX[j], m_Particles->prevPosY[j]);

					glm::vec2 relVel = vel_i - vel_j;
					float velAlongNormal = glm::dot(relVel, normal);
					if (velAlongNormal > 0) continue;

					float restitution = Config::DAMPENING;
					float impulseMag = -(1.0f + restitution) * velAlongNormal;
					impulseMag /= (1.0f / m_Particles->Mass[id] + 1.0f / m_Particles->Mass[j]);

					glm::vec2 impulse = impulseMag * normal;

					vel_i += impulse / m_Particles->Mass[id];
					vel_j -= impulse / m_Particles->Mass[j];

					// Update previous positions based on new velocities
					m_Particles->prevPosX[id] = m_Particles->PosX[id] - vel_i.x;
					m_Particles->prevPosY[id] = m_Particles->PosY[id] - vel_i.y;
					m_Particles->prevPosX[j] = m_Particles->PosX[j] - vel_j.x;
					m_Particles->prevPosY[j] = m_Particles->PosY[j] - vel_j.y;
				}
			}
		}
	}
}

template <bool preserveImpulse>
void VerletSolver::CheckCollisionsWithWalls()
{
	for (uint32_t id = 0; id < m_Particles->Count; ++id)
	{
		float radius = m_Particles->Radius;
		float posX = m_Particles->PosX[id];
		float posY = m_Particles->PosY[id];
		float prevPosX = m_Particles->prevPosX[id];
		float prevPosY = m_Particles->prevPosY[id];

		// Handle left and right walls
		if (posX - radius < 0.0f)
		{
			if constexpr (preserveImpulse)
			{
				float velX = (prevPosX - posX) * Config::DAMPENING;
				posX = radius;
				prevPosX = posX - velX;
			}
			else
			{
				posX = radius;
			}
		}
		else if (posX + radius > m_Grid->m_WorldDim.x)
		{
			if constexpr (preserveImpulse)
			{
				float velX = (prevPosX - posX) * Config::DAMPENING;
				posX = m_Grid->m_WorldDim.x - radius;
				prevPosX = posX - velX;
			}
			else
			{
				posX = m_Grid->m_WorldDim.x - radius;
			}
		}

		// Handle top and bottom walls
		if (posY - radius < 0.0f)
		{
			if constexpr (preserveImpulse)
			{
				float velY = (prevPosY - posY) * Config::DAMPENING;
				posY = radius;
				prevPosY = posY - velY;
			}
			else
			{
				posY = radius;
			}
		}
		else if (posY + radius > m_Grid->m_WorldDim.y)
		{
			if constexpr (preserveImpulse)
			{
				float velY = (prevPosY - posY) * Config::DAMPENING;
				posY = m_Grid->m_WorldDim.y - radius;
				prevPosY = posY - velY;
			}
			else
			{
				posY = m_Grid->m_WorldDim.y - radius;
			}
		}

		// Update positions and previous positions
		m_Particles->PosX[id] = posX;
		m_Particles->PosY[id] = posY;
		m_Particles->prevPosX[id] = prevPosX;
		m_Particles->prevPosY[id] = prevPosY;
	}
}

template <bool preserveImpulse>
void VerletSolver::UpdateParticles(float dt)
{
	for (uint32_t id = 0; id < m_Particles->Count; ++id)
	{
		float accX = 0.0f;
		float accY = Config::GRAVITY;

		if constexpr (preserveImpulse)
		{
			float newPosX = 2.0f * m_Particles->PosX[id] - m_Particles->prevPosX[id];
			float newPosY = 2.0f * m_Particles->PosY[id] - m_Particles->prevPosY[id];

			// Update previous positions
			m_Particles->prevPosX[id] = m_Particles->PosX[id];
			m_Particles->prevPosY[id] = m_Particles->PosY[id];

			// Update current positions
			m_Particles->PosX[id] = newPosX;
			m_Particles->PosY[id] = newPosY;
		}
		else
		{
			m_Particles->PosX[id] += accX * dt * dt;
			m_Particles->PosY[id] += accY * dt * dt;
		}
	}
}
