#pragma once

#include <glm.hpp>

namespace Config {
	static constexpr int WINDOW_WIDTH = 1600;
	static constexpr int WINDOW_HEIGHT = 900;
	static constexpr float GRAVITY = 981.0f;
	static constexpr float DAMPENING = 0.5f;
	static constexpr float RAND_PARTICLE_VELOCITY_MAX = 1.0f;
	static constexpr float PARTICLE_MASS_MIN = 5.0f;
	static constexpr float PARTICLE_MASS_MAX = 10.0f;
	static constexpr int SUBSTEPS = 8;

	static glm::vec4 CLEAR_COLOR = glm::vec4(100.0f / 255.0f, 149.0f / 255.0f, 237.0f / 255.0f, 1.0f);
	static uint32_t PARTICLE_COUNT = 5;
	static float PARTICLE_RADIUS = 50.0f;
}