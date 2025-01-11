#include "Particles.h"
#include "engine/Renderer.h"
#include "cuda/cuda_helper.h"

#define _USE_MATH_DEFINES	
#include <math.h>

Particles::Particles(uint32_t count, float radius, bool isCUDA) : TotalCount(count), Count(count), Radius(radius), m_IsCuda(isCUDA)
{
	if (m_IsCuda)
	{
		CUDA_CHECK_NR(cudaMalloc(&PosX, Count * sizeof(float)));
		CUDA_CHECK_NR(cudaMalloc(&SortedPosX, Count * sizeof(float)));
		CUDA_CHECK_NR(cudaMalloc(&PosY, Count * sizeof(float)));
		CUDA_CHECK_NR(cudaMalloc(&SortedPosY, Count * sizeof(float)));
		CUDA_CHECK_NR(cudaMalloc(&VelX, Count * sizeof(float)));
		CUDA_CHECK_NR(cudaMalloc(&SortedVelX, Count * sizeof(float)));
		CUDA_CHECK_NR(cudaMalloc(&VelY, Count * sizeof(float)));
		CUDA_CHECK_NR(cudaMalloc(&SortedVelY, Count * sizeof(float)));
		CUDA_CHECK_NR(cudaMalloc(&ForceX, Count * sizeof(float)));
		CUDA_CHECK_NR(cudaMemset(ForceX, 0, Count * sizeof(float)));
		CUDA_CHECK_NR(cudaMalloc(&SortedForceX, Count * sizeof(float)));
		CUDA_CHECK_NR(cudaMalloc(&ForceY, Count * sizeof(float)));
		CUDA_CHECK_NR(cudaMemset(ForceY, 0, Count * sizeof(float)));
		CUDA_CHECK_NR(cudaMalloc(&SortedForceY, Count * sizeof(float)));
		CUDA_CHECK_NR(cudaMalloc(&Mass, Count * sizeof(float)));
		CUDA_CHECK_NR(cudaMalloc(&Color, Count * sizeof(glm::vec4)));
	}
	else
	{
		PosX = new float[Count];
		PosY = new float[Count];
		VelX = new float[Count];
		VelY = new float[Count];
		ForceX = new float[Count];
		memset(ForceX, 0, Count * sizeof(float));
		ForceY = new float[Count];
		memset(ForceY, 0, Count * sizeof(float));
		Mass = new float[Count];
		Color = new glm::vec4[Count];
	}

	InitDrawingData();
}

Particles::~Particles()
{
	if (m_IsCuda)
	{
		CUDA_CHECK_NR(cudaFree(PosX));
		CUDA_CHECK_NR(cudaFree(SortedPosX));
		CUDA_CHECK_NR(cudaFree(PosY));
		CUDA_CHECK_NR(cudaFree(SortedPosY));
		CUDA_CHECK_NR(cudaFree(VelX));
		CUDA_CHECK_NR(cudaFree(SortedVelX));
		CUDA_CHECK_NR(cudaFree(VelY));
		CUDA_CHECK_NR(cudaFree(SortedVelY));
		CUDA_CHECK_NR(cudaFree(ForceX));
		CUDA_CHECK_NR(cudaFree(SortedForceX));
		CUDA_CHECK_NR(cudaFree(ForceY));
		CUDA_CHECK_NR(cudaFree(SortedForceY));
		CUDA_CHECK_NR(cudaFree(Mass));
		CUDA_CHECK_NR(cudaFree(Color));
	}
	else
	{
		delete[] PosX;
		delete[] PosY;
		delete[] VelX;
		delete[] VelY;
		delete[] ForceX;
		delete[] ForceY;
		delete[] Mass;
		delete[] Color;
	}
}

Particles *Particles::RandomCPU(uint32_t count, float radius, glm::ivec2 dim)
{
	Particles *p = new Particles(count, radius, false);

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> x_pos_dist(0.0f + radius, dim.x - radius);
	std::uniform_real_distribution<float> y_pos_dist(0.0f + radius, dim.y - radius);
	std::uniform_real_distribution<float> vel_dist(-Config::RAND_PARTICLE_VELOCITY_MAX, Config::RAND_PARTICLE_VELOCITY_MAX);
	std::uniform_real_distribution<float> mass_dist(Config::PARTICLE_MASS_MIN, Config::PARTICLE_MASS_MAX);
	std::uniform_real_distribution<float> color_dist(0.0f, 1.0f);

	for (size_t i = 0; i < count; ++i)
	{
		glm::vec2 pos;
		bool too_close;
		size_t max_iter = 1000;
		size_t iter = 0;
		do
		{
			too_close = false;
			pos = glm::vec2(x_pos_dist(gen), y_pos_dist(gen));
			for (size_t j = 0; j < i; ++j)
			{
				if (glm::distance(pos, glm::vec2(p->PosX[j], p->PosY[j])) < radius * 2)
				{
					too_close = true;
					break;
				}
			}
			iter++;
			if (iter >= max_iter) break;
		} while (too_close);

		p->PosX[i] = pos.x;
		p->PosY[i] = pos.y;
		p->VelX[i] = vel_dist(gen);
		p->VelY[i] = vel_dist(gen);
		p->Mass[i] = mass_dist(gen);
		p->Color[i] = glm::vec4(color_dist(gen), color_dist(gen), color_dist(gen), 1.0f);
	}

	p->InitDrawingData();
	return p;
}

Particles *Particles::RandomCircleCPU(uint32_t count, float radius, glm::ivec2 dim)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	int cr = ceilf(sqrtf(count));
	float circle_radius = cr * radius;
	std::uniform_real_distribution<float> center_dist_x(circle_radius, dim.x - circle_radius);
	std::uniform_real_distribution<float> center_dist_y(circle_radius, dim.y - circle_radius);
	glm::vec2 center(center_dist_x(gen), center_dist_y(gen));

	Particles *p = new Particles(count, radius, false);
	std::uniform_real_distribution<float> vel_dist(-Config::RAND_PARTICLE_VELOCITY_MAX, Config::RAND_PARTICLE_VELOCITY_MAX);
	std::uniform_real_distribution<float> mass_dist(Config::PARTICLE_MASS_MIN, Config::PARTICLE_MASS_MAX);
	std::uniform_real_distribution<float> color_dist(0.0f, 1.0f);
	std::uniform_real_distribution<float> jitter(-radius * 0.01f, radius * 0.01f);

	int index = 0;
	float spacing = 2 * radius;
	for (int y = -cr; y <= cr; y++)
	{
		for (int x = -cr; x <= cr; x++)
		{
			float dx = x * spacing;
			float dy = y * spacing;
			float l = sqrtf(dx * dx + dy * dy);
			if ((l <= circle_radius - radius) && (index < count))
			{
				p->PosX[index] = center.x + dx + jitter(gen);
				p->PosY[index] = center.y + dy + jitter(gen);
				p->VelX[index] = 0;
				p->VelY[index] = 0;
				p->Mass[index] = mass_dist(gen);
				p->Color[index] = glm::vec4(color_dist(gen), color_dist(gen), color_dist(gen), 1.0f);
				index++;
			}
		}
	}
	p->Count = index;
	p->InitDrawingData();
	return p;
}

Particles *Particles::RandomBoxCPU(uint32_t count, float radius, glm::ivec2 dim)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	int side_count = sqrtf(count);
	float square_side = side_count * radius * 2.0f;
	std::uniform_real_distribution<float> center_dist_x(radius, dim.x - (square_side - radius));
	std::uniform_real_distribution<float> center_dist_y(radius, dim.y - (square_side - radius));
	glm::vec2 center(center_dist_x(gen), center_dist_y(gen));

	Particles *p = new Particles(side_count * side_count, radius, false);
	std::uniform_real_distribution<float> vel_dist(-Config::RAND_PARTICLE_VELOCITY_MAX, Config::RAND_PARTICLE_VELOCITY_MAX);
	std::uniform_real_distribution<float> mass_dist(Config::PARTICLE_MASS_MIN, Config::PARTICLE_MASS_MAX);
	std::uniform_real_distribution<float> color_dist(0.0f, 1.0f);
	std::uniform_real_distribution<float> jitter(-radius * 0.01f, radius * 0.01f);

	int index = 0;
	float spacing = 2 * radius;
	for (int y = 0; y < side_count; y++)
	{
		for (int x = 0; x < side_count; x++)
		{
			float dx = x * spacing;
			float dy = y * spacing;
			if (index < count)
			{
				p->PosX[index] = center.x + dx + jitter(gen);
				p->PosY[index] = center.y + dy + jitter(gen);
				p->VelX[index] = 0;
				p->VelY[index] = 0;
				p->Mass[index] = mass_dist(gen);
				p->Color[index] = glm::vec4(color_dist(gen), color_dist(gen), color_dist(gen), 1.0f);
				index++;
			}
		}
	}
	p->InitDrawingData();
	return p;
}

Particles *Particles::WaterfallCPU(uint32_t count, float radius, glm::ivec2 dim, float velocity, int rows)
{
	Particles *p = new Particles(count, radius, false);
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> mass_dist(Config::PARTICLE_MASS_MIN, Config::PARTICLE_MASS_MAX);
	std::uniform_real_distribution<float> color_dist(0.0f, 1.0f);
	std::uniform_real_distribution<float> jitter(-radius * 0.01f, radius * 0.01f);

	for (size_t i = 0; i < count; ++i)
	{
		p->PosX[i] = radius + jitter(gen);
		p->PosY[i] = 1.5f * radius + (i % rows) * (3.0f * radius) + jitter(gen);
		p->VelX[i] = velocity;
		p->VelY[i] = 0.0f;
		p->Mass[i] = mass_dist(gen);
		p->Color[i] = glm::vec4(color_dist(gen), color_dist(gen), color_dist(gen), 1.0f);
	}

	p->Count = 0;
	p->InitDrawingData();
	return p;
}

void Particles::InitDrawingData()
{
	m_ParticleData.PosX = PosX;
	m_ParticleData.PosY = PosY;
	m_ParticleData.Color = (float4 *)Color;
	m_ParticleData.Count = Count;
}

void Particles::DrawCPU(Renderer *renderer, InstancedParticles *instancedParticles)
{
	glm::vec2 viewport = renderer->GetViewportSize();
	m_ParticleData.Scale = make_float2(2 * Radius / viewport.x, 2 * Radius / viewport.y);
	instancedParticles->UpdateParticleInstancesCPU(&m_ParticleData);
	instancedParticles->Draw();
}

void Particles::DrawCUDA(Renderer *renderer, InstancedParticles *instancedParticles)
{
	glm::vec2 viewport = renderer->GetViewportSize();
	m_ParticleData.Scale = make_float2(2 * Radius / viewport.x, 2 * Radius / viewport.y);
	instancedParticles->UpdateParticleInstancesCUDA(&m_ParticleData);
	instancedParticles->Draw();
}

void Particles::SetCount(uint32_t count)
{
	if (count <= TotalCount)
	{
		Count = count;
		m_ParticleData.Count = count;
	}
	else
	{
		Count = TotalCount;
		m_ParticleData.Count = TotalCount;
	}
}
