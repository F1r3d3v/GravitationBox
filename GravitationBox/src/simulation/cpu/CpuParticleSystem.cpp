#include "cpu/CpuParticleSystem.h"
#include "cuda/CudaParticleSystem.h"
#include <cuda_runtime.h>
#include <random>

CpuParticleSystem::CpuParticleSystem(uint32_t count, float radius, ParticleSolver *solver)
	: ParticleSystem(count, radius, solver)
{
	PosX = new float[Count];
	SortedPosX = new float[Count];
	PosY = new float[Count];
	SortedPosY = new float[Count];
	VelX = new float[Count];
	SortedVelX = new float[Count];
	VelY = new float[Count];
	SortedVelY = new float[Count];
	ForceX = new float[Count];
	memset(ForceX, 0, Count * sizeof(float));
	SortedForceX = new float[Count];
	ForceY = new float[Count];
	memset(ForceY, 0, Count * sizeof(float));
	SortedForceY = new float[Count];
	Mass = new float[Count];
	Color = new glm::vec4[Count];
}

CpuParticleSystem::~CpuParticleSystem()
{
	delete[] PosX;
	delete[] SortedPosX;
	delete[] PosY;
	delete[] SortedPosY;
	delete[] VelX;
	delete[] SortedVelX;
	delete[] VelY;
	delete[] SortedVelY;
	delete[] ForceX;
	delete[] SortedForceX;
	delete[] ForceY;
	delete[] SortedForceY;
	delete[] Mass;
	delete[] Color;
}

CpuParticleSystem *CpuParticleSystem::CreateRandom(uint32_t count, float radius, glm::ivec2 dim, ParticleSolver *solver)
{

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> x_pos_dist(0.0f + radius, dim.x - radius);
	std::uniform_real_distribution<float> y_pos_dist(0.0f + radius, dim.y - radius);
	std::uniform_real_distribution<float> vel_dist(-Config::RAND_PARTICLE_VELOCITY_MAX, Config::RAND_PARTICLE_VELOCITY_MAX);
	std::uniform_real_distribution<float> mass_dist(Config::PARTICLE_MASS_MIN, Config::PARTICLE_MASS_MAX);
	std::uniform_real_distribution<float> color_dist(0.0f, 1.0f);

	CpuParticleSystem *p = new CpuParticleSystem(count, radius, solver);
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

	return p;
}

CpuParticleSystem *CpuParticleSystem::CreateCircle(uint32_t count, float radius, glm::ivec2 dim, ParticleSolver *solver)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	int cr = static_cast<int>(ceil(sqrt(count)));
	float circle_radius = cr * radius;

	glm::vec2 minPos(circle_radius), maxPos(dim.x - circle_radius, dim.y - circle_radius);
	if (2 * circle_radius > dim.x)
	{
		minPos.x = dim.x / 2.0f;
		maxPos.x = dim.x / 2.0f;
	}
	if (2 * circle_radius > dim.y)
	{
		minPos.y = dim.y / 2.0f;
		maxPos.y = dim.y / 2.0f;
	}

	std::uniform_real_distribution<float> center_dist_x(minPos.x, maxPos.x);
	std::uniform_real_distribution<float> center_dist_y(minPos.y, maxPos.y);
	glm::vec2 center(center_dist_x(gen), center_dist_y(gen));

	CpuParticleSystem *p = new CpuParticleSystem(count, radius, solver);
	std::uniform_real_distribution<float> vel_dist(-Config::RAND_PARTICLE_VELOCITY_MAX, Config::RAND_PARTICLE_VELOCITY_MAX);
	std::uniform_real_distribution<float> mass_dist(Config::PARTICLE_MASS_MIN, Config::PARTICLE_MASS_MAX);
	std::uniform_real_distribution<float> color_dist(0.0f, 1.0f);
	std::uniform_real_distribution<float> jitter(-radius * 0.01f, radius * 0.01f);

	uint32_t index = 0;
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
	return p;
}

CpuParticleSystem *CpuParticleSystem::CreateBox(uint32_t count, float radius, glm::ivec2 dim, ParticleSolver *solver)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	uint32_t side_count = static_cast<uint32_t>(sqrt(count));
	float square_side = side_count * radius * 2.0f;

	glm::vec2 minPos(radius), maxPos(dim.x - (square_side - radius), dim.y - (square_side - radius));
	if (square_side > dim.x)
	{
		maxPos.x = minPos.x;
	}
	if (square_side > dim.y)
	{
		maxPos.y = minPos.y;
	}

	std::uniform_real_distribution<float> center_dist_x(minPos.x, maxPos.x);
	std::uniform_real_distribution<float> center_dist_y(minPos.y, maxPos.y);
	glm::vec2 center(center_dist_x(gen), center_dist_y(gen));

	CpuParticleSystem *p = new CpuParticleSystem(side_count * side_count, radius, solver);
	std::uniform_real_distribution<float> vel_dist(-Config::RAND_PARTICLE_VELOCITY_MAX, Config::RAND_PARTICLE_VELOCITY_MAX);
	std::uniform_real_distribution<float> mass_dist(Config::PARTICLE_MASS_MIN, Config::PARTICLE_MASS_MAX);
	std::uniform_real_distribution<float> color_dist(0.0f, 1.0f);
	std::uniform_real_distribution<float> jitter(-radius * 0.01f, radius * 0.01f);

	uint32_t index = 0;
	float spacing = 2 * radius;
	for (uint32_t y = 0; y < side_count; y++)
	{
		for (uint32_t x = 0; x < side_count; x++)
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

	return p;
}

CpuParticleSystem *CpuParticleSystem::CreateWaterfall(uint32_t count, float radius, glm::ivec2 dim, float velocity, int rows, ParticleSolver *solver)
{
	CpuParticleSystem *p = new CpuParticleSystem(count, radius, solver);
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
	return p;
}

CpuParticleSystem *CpuParticleSystem::CreateFromCuda(CudaParticleSystem *pGPU, ParticleSolver *solver)
{
	CpuParticleSystem *p = new CpuParticleSystem(pGPU->TotalCount, pGPU->Radius, solver);
	cudaMemcpy(p->PosX, pGPU->PosX, pGPU->TotalCount * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(p->PosY, pGPU->PosY, pGPU->TotalCount * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(p->VelX, pGPU->VelX, pGPU->TotalCount * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(p->VelY, pGPU->VelY, pGPU->TotalCount * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(p->Mass, pGPU->Mass, pGPU->TotalCount * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(p->Color, pGPU->Color, pGPU->TotalCount * sizeof(glm::vec4), cudaMemcpyDeviceToHost);
	p->Count = pGPU->Count;
	return p;
}
