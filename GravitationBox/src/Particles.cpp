#include "Particles.h"
#include "Renderer.h"
#include "cuda_helper.h"

Particles::Particles(size_t count, float radius, bool isCUDA) : Count(count), Radius(radius), m_IsCuda(isCUDA)
{
	if (m_IsCuda)
	{
		cudaMalloc(&PosX, Count * sizeof(float));
		cudaMalloc(&PosY, Count * sizeof(float));
		cudaMalloc(&VelX, Count * sizeof(float));
		cudaMalloc(&VelY, Count * sizeof(float));
		cudaMalloc(&Mass, Count * sizeof(float));
		cudaMalloc(&Color, Count * sizeof(glm::vec4));
	}
	else
	{
		PosX = new float[Count];
		PosY = new float[Count];
		VelX = new float[Count];
		VelY = new float[Count];
		Mass = new float[Count];
		Color = new glm::vec4[Count];
	}
}

Particles::~Particles()
{
	if (m_IsCuda)
	{
		cudaFree(PosX);
		cudaFree(PosY);
		cudaFree(VelX);
		cudaFree(VelY);
		cudaFree(Mass);
		cudaFree(Color);
	}
	else
	{
		delete[] PosX;
		delete[] PosY;
		delete[] VelX;
		delete[] VelY;
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

Particles *Particles::LoadFromFile(const char *filename, bool isCUDA)
{
	FILE *file;
	fopen_s(&file, filename, "r");
	if (!file)
	{
		return nullptr;
	}
	glm::vec4 color;
	size_t count;
	float radius;
	fscanf_s(file, "%zu %f", &count, &radius);
	fscanf_s(file, "%f %f %f", &color.r, &color.g, &color.b);
	color.a = 1.0f;
	Particles *p = new Particles(count, radius, isCUDA);
	p->PosX = new float[p->Count];
	p->PosY = new float[p->Count];	
	p->VelX = new float[p->Count];
	p->VelY = new float[p->Count];
	p->Mass = new float[p->Count];
	for (size_t i = 0; i < p->Count; ++i)
	{
		fscanf_s(file, "%f %f %f %f %f", &p->PosX[i], &p->PosY[i], &p->VelX[i], &p->VelY[i], &p->Mass[i]);
		p->Color[i] = color;
	}
	fclose(file);
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