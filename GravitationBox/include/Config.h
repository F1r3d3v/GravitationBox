#pragma once

#include <glm.hpp>

namespace Config {
	static constexpr int WINDOW_WIDTH = 1600;
	static constexpr int WINDOW_HEIGHT = 900;
	static constexpr float RAND_PARTICLE_VELOCITY_MAX = 10.0f;
	static constexpr float PARTICLE_MASS_MIN = 5.0f;
	static constexpr float PARTICLE_MASS_MAX = 10.0f;

	static glm::vec4 CLEAR_COLOR = glm::vec4(100.0f / 255.0f, 149.0f / 255.0f, 237.0f / 255.0f, 1.0f);
	static uint32_t PARTICLE_COUNT = 100000;
	static float PARTICLE_RADIUS = 1.0f;
	static int SUBSTEPS = 2;
}