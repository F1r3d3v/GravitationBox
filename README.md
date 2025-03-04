# Gravitation Box

Gravitation Box is a 2D particle simulation project that demonstrates gravitational interactions between particles.  It offers both CPU and GPU (CUDA) implementations for performance comparison and features a user-friendly interface built with ImGui for parameter control and visualization.  The simulation uses a Verlet integration scheme and a spatial grid for efficient collision detection.

## Features

*   **2D Particle Simulation:** Simulates the movement of particles under the influence of gravity and collisions.
*   **CPU and CUDA Implementations:** Provides both CPU and GPU-accelerated versions of the simulation, allowing users to compare performance.  The CUDA implementation leverages `thrust` for efficient parallel operations.
*   **Verlet Integration:** Employs the Verlet integration method for accurate and stable time integration.
*   **Spatial Grid:** Utilizes a spatial grid data structure to optimize collision detection between particles, improving performance, especially for large numbers of particles.
*   **Interactive Controls (ImGui):** Offers a user-friendly interface for:
    *   Pausing and resuming the simulation.
    *   Switching between CPU and CUDA implementations.
    *   Adjusting simulation parameters (gravity, particle radius, damping, stiffness, friction, etc.).
    *   Resetting the simulation.
    *   Toggling VSync.
    *   Selecting different initial particle arrangements (presets).
    *   Changing particle and background colors.
*   **Multiple Presets:** Includes pre-configured particle setups:
    *   Random: Particles are randomly distributed within the simulation space.
    *   Circle: Particles are arranged in a circular formation.
    *   Box: Particles are arranged in a rectangular box formation.
    *   Waterfall: Simulates a continuous stream of particles falling from the top of the screen.
*   **Collision Handling:** Implements collision detection and response between particles and with the boundaries of the simulation window.
*   **Configurable Parameters:** Allows users to fine-tune simulation parameters through the ImGui interface and the `Config.h` file.
*   **OpenGL Rendering:** Uses OpenGL for rendering the particles as instanced quads, with blending for smooth visual appearance.
*   **Logging:** Includes a logging system (using the custom `Log` class) to record simulation events, errors, and debugging information to the console and a log file.
*   **Cross-Platform Compatibility:**  Designed to work on Windows (using `WinMain`) and other platforms (using `main`).  Requires a C++ compiler, CUDA toolkit (for GPU implementation), GLFW, and GLAD.

## Table of Contents

*   [Features](#features)
*   [Table of Contents](#table-of-contents)
*   [Project Structure](#project-structure)
*   [Dependencies](#dependencies)
*   [Building and Running](#building-and-running)
    *   [Visual Studio (Windows)](#visual-studio-windows)
*   [Usage](#usage)
*   [Implementation Details](#implementation-details)
    *   [Verlet Integration](#verlet-integration)
    *   [Spatial Grid](#spatial-grid)
    *   [Collision Handling](#collision-handling)
    *   [CUDA Implementation](#cuda-implementation)
    *   [CPU Implementation](#cpu-implementation)
    *   [Rendering](#rendering)
## Project Structure

The project is organized into the following directories and files:

```
GravitationBox/
├── include/
│   ├── engine/
│   │   ├── GraphicsObject.h             - Base class for drawable objects.
│   │   ├── Input.h                      - Input handling (keyboard).
│   │   ├── InstancedObject.h            - Base class for instanced rendering.
│   │   ├── Log.h                        - Logging functionality.
│   │   ├── Renderer.h                   - OpenGL rendering utilities.
│   │   ├── Simulation.h                 - Base class for simulations.
│   │   ├── Window.h                     - Window management (GLFW).
│   ├── simulation/
│   │   ├── cpu/
│   │   │   ├── CpuGrid.h                - CPU implementation of the spatial grid.
│   │   │   ├── CpuInstancedParticles.h  - CPU implementation of instanced particle rendering.
│   │   │   ├── CpuParticleSystem.h      - CPU implementation of the particle system.
│   │   │   ├── CpuVerletSolver.h        - CPU implementation of the Verlet solver.
│   │   ├── cuda/
│   │   │   ├── CudaGrid.h               - CUDA implementation of the spatial grid.
│   │   │   ├── CudaInstancedParticles.h - CUDA implementation of instanced particle rendering.
│   │   │   ├── CudaParticleSystem.h     - CUDA implementation of the particle system.
│   │   │   ├── CudaVerletSolver.h       - CUDA implementation of the Verlet solver.
│   │   ├── utils/
│   │   │   ├── cuda_helper_math.h       - CUDA math utility functions.
│   │   │   ├── cuda_helper.h            - CUDA error checking macros.
│   │   ├── Config.h                     - Simulation configuration parameters.
│   │   ├── Grid.h                       - Base class for the spatial grid.
│   │   ├── InstancedParticles.h         - Base class for instanced particle rendering.
│   │   ├── MySimulation.h               - Main simulation class (inherits from `Simulation`).
│   │   ├── ParticleSolver.h             - Base class for particle solvers.
│   │   ├── ParticleSystem.h             - Base class for particle systems.
├── shaders/
│   ├── particle.frag                    - Fragment shader for particle rendering.
│   ├── particle.vert                    - Vertex shader for particle rendering.
├── src/
│   ├── engine/
│   │   ├── Input.cpp
│   │   ├── InstancedObject.cpp
│   │   ├── Log.cpp
│   │   ├── Renderer.cpp
│   │   ├── Simulation.cpp
│   │   ├── Window.cpp
│   ├── simulation/
│   │   ├── cpu/
│   │   │   ├── CpuGrid.cpp
│   │   │   ├── CpuInstancedParticles.cpp
│   │   │   ├── CpuParticleSystem.cpp
│   │   │   ├── CpuVerletSolver.cpp
│   │   ├── cuda/
│   │   │   ├── CudaGrid.cu
│   │   │   ├── CudaInstancedParticles.cu
│   │   │   ├── CudaParticleSystem.cu
│   │   │   ├── CudaVerletSolver.cu
│   │   ├── Grid.cpp
│   │   ├── InstancedParticles.cpp
│   │   ├── MySimulation.cpp
│   │   ├── ParticleSystem.cpp
│   ├── main.cpp                         - Entry point of the application.
```

## Dependencies

*   **C++ Compiler:** A C++ compiler with support for C++17 or later (e.g., GCC, Clang, MSVC).
*   **CUDA Toolkit:** NVIDIA CUDA Toolkit (version 11.0 or later recommended).
*   **GLFW:** A library for creating and managing OpenGL contexts and windows.  [GLFW Download](https://www.glfw.org/download.html)
*   **GLAD:** An OpenGL loading library.  [GLAD Generator](https://glad.dav1d.de/)
*   **GLM:** OpenGL Mathematics library.  [GLM Download](https://glm.g-truc.net/0.9.9/index.html)
*   **ImGui:** A bloat-free graphical user interface library for C++.  [ImGui GitHub](https://github.com/ocornut/imgui)

## Building and Running

### Visual Studio (Windows) ###

1.  **Install Dependencies:** Ensure you have installed the CUDA Toolkit.

2.  **Open the Solution:** Open the provided Visual Studio solution file (`.sln`).

3.  **Build the Project:**
    *   Select "Build" -> "Build Solution" (or press Ctrl+Shift+B).
    *   Choose either the "Debug" or "Release" configuration.

4.  **Run the Executable:**
    *   After a successful build, the executable (`GravitationBox.exe`) will be located in the `bin\x64\Debug` or `bin\x64\Release` directory (relative to the solution directory).    
## Usage

1.  **Run the Executable:** Launch the compiled executable.
2.  **Interact with the Simulation:**
    *   Use the ImGui interface (on the right side of the window) to control the simulation.
    *   Press 'P' to pause/resume.
    *   Press 'R' to reset.
    *   Press 'C' to switch between CPU and CUDA implementations.
    *   Press 'V' to toggle VSync.
    *   Press 'ESC' to close the application.

## Implementation Details

### Verlet Integration

The simulation uses the Velocity Verlet integration scheme, a numerical method for integrating Newton's equations of motion.  It is commonly used in molecular dynamics and game physics due to its stability and accuracy.  The algorithm is implemented in `CpuVerletSolver.cpp` (CPU) and `CudaVerletSolver.cu` (CUDA).  Each step involves:

1.  **Calculate New Positions:** Update particle positions based on current velocities and forces.
2.  **Calculate Forces:** Compute forces acting on particles (gravity, collisions).
3.  **Calculate New Velocities:** Update particle velocities based on the forces calculated in the previous step.

### Spatial Grid

A spatial grid is used to accelerate collision detection.  The simulation space is divided into a grid of cells. Each particle is assigned to a cell based on its position.  Collision detection is then performed only between particles within the same cell or neighboring cells, significantly reducing the number of comparisons required.  The grid implementation is in `CpuGrid.cpp` and `CudaGrid.cu`. The grid is updated in each time step. `thrust::sort_by_key` is used in the CUDA implementation to efficiently sort particles based on their cell IDs.

### Collision Handling

Collision detection is performed between particles and with the walls of the simulation window.  The collision response is modeled using a spring-damper system. When a collision is detected:

1.  **Normal Calculation:** The collision normal is calculated (the direction of the force).
2.  **Relative Velocity:** The relative velocity between the colliding entities is calculated.
3.  **Normal and Tangential Velocities:** The relative velocity is decomposed into normal and tangential components.
4.  **Force Calculation:**  A force is calculated based on:
    *   **Spring Force:** Proportional to the interpenetration distance.
    *   **Damping Force:** Proportional to the normal component of the relative velocity.
    *   **Friction Force:** Proportional to the tangential component of the relative velocity.

The collision handling logic is found within the `CpuVerletSolver` and `CudaVerletSolver` classes.

### CUDA Implementation

The CUDA implementation utilizes the GPU for parallel computation. Key aspects:

*   **Kernels:**  CUDA kernels (functions executed on the GPU) are defined for calculating cell IDs, updating particle positions and velocities, and handling collisions.  These are found in `.cu` files.
*   **Memory Management:** Particle data (position, velocity, force, mass, color) is allocated on the GPU using `cudaMalloc`. Data transfer between the host (CPU) and device (GPU) is managed using `cudaMemcpy`.
*   **Thrust:** The Thrust library is used for efficient parallel operations, particularly for sorting particles by cell ID.
*   **CUDA-OpenGL Interoperability:**  The `CudaInstancedParticles` class demonstrates CUDA-OpenGL interoperability.  The vertex buffer object (VBO) used for rendering is registered with CUDA, allowing the GPU to directly update the particle positions and colors without transferring data back to the CPU.  This is handled by `cudaGraphicsGLRegisterBuffer` and `cudaGraphicsResourceGetMappedPointer`.
*   **Error Handling:**  The `CUDA_CHECK` and `CUDA_CHECK_NR` macros are used to check for CUDA errors and report them through the logging system.

### CPU Implementation
The CPU implementation uses standard C++ and STL containers. Key files are in the `simulation/cpu/` folder. The collision, grid, and update algorithms are implemented for the CPU.

### Rendering

*   **Instanced Rendering:** Particles are rendered using instanced drawing.  A single quad is defined, and multiple instances of this quad are drawn, each with its own position, scale, and color.
*   **Shaders:**  Simple vertex and fragment shaders (`particle.vert`, `particle.frag`) are used to render the particles. The vertex shader transforms the particle positions, and the fragment shader applies the color and creates a circular shape with alpha blending for a smooth appearance.
*   **OpenGL Functions:**  OpenGL functions (using GLAD) are used to create and manage vertex array objects (VAOs), vertex buffer objects (VBOs), and element buffer objects (EBOs).
