#pragma once

#include <cstdint>
#include <glm.hpp>

static inline consteval glm::vec4 make_color(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255)
{
	glm::vec4 v; v.x = r / 255.0f; v.y = g / 255.0f; v.z = b / 255.0f; v.w = a / 255.0f; return v;
}

namespace Config 
{
	static constexpr int WINDOW_WIDTH = 1600;
	static constexpr int WINDOW_HEIGHT = 900;
	static constexpr float RAND_PARTICLE_VELOCITY_MAX = 50.0f;
	static constexpr float PARTICLE_MASS_MIN = 5.0f;
	static constexpr float PARTICLE_MASS_MAX = 10.0f;
	static constexpr glm::vec4 CLEAR_COLOR = make_color(100, 149, 237);
	static constexpr glm::vec4 PARTICLE_COLOR = make_color(40, 12, 221);

	// Simulation default parameters
	static constexpr uint32_t PARTICLE_COUNT = 150000;
	static constexpr float PARTICLE_RADIUS = 1.5f;
	static constexpr uint8_t SUBSTEPS = 2;
	static constexpr float TIMESTEP = 0.01f;
	static constexpr float GRAVITY = 10.0f;
	static constexpr float WALL_DAMPENING = 50.0f;
	static constexpr float WALL_FRICTION = 100.0f;
	static constexpr float PARTICLES_DAMPENING = 20.0f;
	static constexpr float PARTICLES_STIFFNESS = 150000.0f;
	static constexpr float PARTICLES_FRICTION = 100.0f;


	// Presets
	// 0 - Random
	// 1 - Circle
	// 2 - Box
	// 3 - Waterfall
	static constexpr uint8_t STARTING_PRESET = 3;
	static constexpr float WATERFALL_VELOCITY = 80.0f;
}