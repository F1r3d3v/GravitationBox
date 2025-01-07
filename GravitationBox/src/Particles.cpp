#include "Particles.h"
#include "Renderer.h"
#include "cuda_helper.h"

#define _USE_MATH_DEFINES	
#include <math.h>

Particles::Particles(size_t count, float radius, bool isCUDA) : Count(count), Radius(radius), m_IsCuda(isCUDA)
{
	if (m_IsCuda)
	{
		cudaMalloc(&PosX, Count * sizeof(float));
		cudaMalloc(&PosY, Count * sizeof(float));
		cudaMalloc(&VelX, Count * sizeof(float));
		cudaMalloc(&VelY, Count * sizeof(float));
		cudaMalloc(&ForceX, Count * sizeof(float));
		cudaMalloc(&ForceY, Count * sizeof(float));
		cudaMalloc(&Mass, Count * sizeof(float));
		cudaMalloc(&Color, Count * sizeof(glm::vec4));
	}
	else
	{
		PosX = new float[Count];
		PosY = new float[Count];
		VelX = new float[Count];
		VelY = new float[Count];
		ForceX = new float[Count];
		ForceY = new float[Count];
		Mass = new float[Count];
		Color = new glm::vec4[Count];
	}

	InitDrawingData();
}

Particles::~Particles()
{
	if (m_IsCuda)
	{
		cudaFree(PosX);
		cudaFree(PosY);
		cudaFree(VelX);
		cudaFree(VelY);
		cudaFree(ForceX);
		cudaFree(ForceY);
		cudaFree(Mass);
		cudaFree(Color);
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

	Renderer::UnloadShader(m_ShaderProgram);
}

Particles *Particles::RandomCPU(size_t count, float radius, glm::ivec2 dim)
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

Particles *Particles::RandomCircleCPU(size_t count, float radius, glm::ivec2 dim)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	int cr = ceilf(sqrtf(count));
	float circle_radius = cr*radius;
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
	for (int y = -cr; y <= cr; y++) {
		for (int x = -cr; x <= cr; x++) {
			float dx = x * spacing;
			float dy = y * spacing;
			float l = sqrtf(dx * dx + dy * dy);
			if ((l <= circle_radius - radius) && (index < count)) {
				p->PosX[index] = center.x + dx + (1.0 + jitter(gen));
				p->PosY[index] = center.y + dy + (1.0 + jitter(gen));
				p->VelX[index] = 0;
				p->VelY [index] = 0;
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

Particles *Particles::RandomBoxCPU(size_t count, float radius, glm::ivec2 dim)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	int side_count = sqrtf(count);
	float square_side = side_count * radius * 2;
	std::uniform_real_distribution<float> center_dist_x(square_side / 2, dim.x - square_side / 2);
	std::uniform_real_distribution<float> center_dist_y(square_side / 2, dim.y - square_side / 2);
	glm::vec2 center(center_dist_x(gen), center_dist_y(gen));

	Particles *p = new Particles(count, radius, false);
	std::uniform_real_distribution<float> vel_dist(-Config::RAND_PARTICLE_VELOCITY_MAX, Config::RAND_PARTICLE_VELOCITY_MAX);
	std::uniform_real_distribution<float> mass_dist(Config::PARTICLE_MASS_MIN, Config::PARTICLE_MASS_MAX);
	std::uniform_real_distribution<float> color_dist(0.0f, 1.0f);
	std::uniform_real_distribution<float> jitter(-radius * 0.01f, radius * 0.01f);

	int index = 0;
	float spacing = 2 * radius;
	int sr1 = ceilf(side_count / 2.0f);
	int sr2 = floorf(side_count / 2.0f);
	for (int y = -sr1; y < sr2; y++) {
		for (int x = -sr1; x < sr2; x++) {
			float dx = x * spacing;
			float dy = y * spacing;
			if (index < count) {
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

//Particles *Particles::LoadFromFile(const char *filename, bool isCUDA)
//{
//	FILE *file;
//	fopen_s(&file, filename, "r");
//	if (!file)
//	{
//		return nullptr;
//	}
//	glm::vec4 color;
//	size_t count;
//	float radius;
//	fscanf_s(file, "%zu %f", &count, &radius);
//	fscanf_s(file, "%f %f %f", &color.r, &color.g, &color.b);
//	color.a = 1.0f;
//	Particles *p = new Particles(count, radius, isCUDA);
//	p->PosX = new float[p->Count];
//	p->PosY = new float[p->Count];
//	p->VelX = new float[p->Count];
//	p->VelY = new float[p->Count];
//	p->Mass = new float[p->Count];
//	for (size_t i = 0; i < p->Count; ++i)
//	{
//		fscanf_s(file, "%f %f %f %f %f", &p->PosX[i], &p->PosY[i], &p->VelX[i], &p->VelY[i], &p->Mass[i]);
//		p->Color[i] = color;
//	}
//	fclose(file);
//	return p;
//}

void Particles::InitDrawingData()
{
	m_ParticleData.PosX = PosX;
	m_ParticleData.PosY = PosY;
	m_ParticleData.Color = (float4 *)Color;
	m_ParticleData.Count = Count;
}

void Particles::DrawCPU(Renderer *renderer)
{
	glm::vec2 viewport = renderer->GetViewportSize();
	m_ParticleData.Scale = make_float2(2 * Radius / viewport.x, 2 * Radius / viewport.y);
	renderer->UpdateParticleInstancesCPU(&m_ParticleData);
	renderer->RenderParticles(m_ShaderProgram, Count);
}

void Particles::DrawCUDA(Renderer *renderer)
{
	glm::vec2 viewport = renderer->GetViewportSize();
	m_ParticleData.Scale = make_float2(2 * Radius / viewport.x, 2 * Radius / viewport.y);
	renderer->UpdateParticleInstancesCUDA(&m_ParticleData);
	renderer->RenderParticles(m_ShaderProgram, Count);
}