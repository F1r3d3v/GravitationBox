#include "cuda/CudaInstancedParticles.h"
#include "cpu/CpuInstancedParticles.h"
#include "utils/cuda_helper.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>

CudaInstancedParticles::CudaInstancedParticles(ParticleSystem *p, uint32_t ShaderProgram)
	: InstancedParticles(p, ShaderProgram)
{
	// Register buffer with CUDA
	CUDA_CHECK_NR(cudaGraphicsGLRegisterBuffer(&m_CudaVBOResource, m_InstanceVBO, cudaGraphicsMapFlagsWriteDiscard));
}

CudaInstancedParticles::~CudaInstancedParticles()
{
	cudaGraphicsUnregisterResource(m_CudaVBOResource);
}

__global__ void UpdateInstanceDataKernel(float *vboPtr, float *PosX, float *PosY, float2 Scale, float4 *Color, bool RandomColor, float4 StillColor, size_t count)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (; tid < count; tid += stride)
	{
		int vboIdx = 8 * tid;
		// Update position
		*(float2 *)(&vboPtr[vboIdx]) = make_float2(__ldg(&PosX[tid]), __ldg(&PosY[tid]));
		// Update scale
		*(float2 *)(&vboPtr[vboIdx + 2]) = Scale;
		// Update color
		*(float4 *)(&vboPtr[vboIdx + 4]) = RandomColor ? __ldg(&Color[tid]) : StillColor;
	}
}

void CudaInstancedParticles::UpdateParticleInstances()
{
	UpdateGraphicsData();
	if (!m_ParticleData.Count) return;

	// Map OpenGL buffer for writing from CUDA
	float *dPtr;
	size_t numBytes;
	CUDA_CHECK_NR(cudaGraphicsMapResources(1, &m_CudaVBOResource));
	CUDA_CHECK_NR(cudaGraphicsResourceGetMappedPointer((void **)&dPtr, &numBytes, m_CudaVBOResource));

	// Launch kernel to update instance data
	UpdateInstanceDataKernel << <BLOCKS_PER_GRID(m_ParticleData.Count), THREADS_PER_BLOCK >> > (
		dPtr,
		m_ParticleData.PosX,
		m_ParticleData.PosY,
		m_ParticleData.Scale,
		m_ParticleData.Color,
		m_ParticleData.RandomColor,
		m_ParticleData.StillColor,
		m_ParticleData.Count);
	cudaDeviceSynchronize();
	CUDA_CHECK_NR(cudaGetLastError());

	// Unmap buffer
	CUDA_CHECK_NR(cudaGraphicsUnmapResources(1, &m_CudaVBOResource));
}
