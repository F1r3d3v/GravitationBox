#pragma once
#include "Renderer.h"

class Simulation
{
public:
	Simulation();
	~Simulation();

	void Run();
	void Reset();

	virtual void Initialize() = 0;
	virtual void Update(float deltaTime) = 0;
	virtual void Render(Renderer renderer) = 0;
	virtual void Cleanup() = 0;
	virtual void OnImGuiRender() = 0;

	void SetFixedTimeStep(bool isFixedTimeStep) { m_IsFixedTimeStep = isFixedTimeStep; }
	bool IsFixedTimeStep() const { return m_IsFixedTimeStep; }
	void TooglePause() { m_IsPaused = !m_IsPaused; }

private:
	bool m_IsFixedTimeStep;
	bool m_IsPaused;
};

