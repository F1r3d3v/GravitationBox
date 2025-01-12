#pragma once

#include <glm.hpp>

namespace Config {
	static constexpr int WINDOW_WIDTH = 1600;
	static constexpr int WINDOW_HEIGHT = 900;
	static constexpr float RAND_PARTICLE_VELOCITY_MAX = 50.0f;
	static constexpr float PARTICLE_MASS_MIN = 5.0f;
	static constexpr float PARTICLE_MASS_MAX = 10.0f;

	// Simulation default parameters
	static constexpr float PARTICLE_COUNT = 150000;
	static constexpr float PARTICLE_RADIUS = 1.5f;
	static constexpr float SUBSTEPS = 2;
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
	static constexpr int STARTING_PRESET = 3;
	static constexpr float WATERFALL_VELOCITY = 80.0f;
}