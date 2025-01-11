#include "InstancedParticles.h"
#include "glad/gl.h"
#include "cuda/cuda_helper.h"
#include "Particles.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>

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

cudaError_t InstancedParticles::UpdateParticleInstancesCUDA(ParticleData *pData)
{
	if (!pData->Count) return cudaSuccess;

	// Map OpenGL buffer for writing from CUDA
	float *dPtr;
	size_t numBytes;
	CUDA_CHECK(cudaGraphicsMapResources(1, &m_CudaVBOResource));
	CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void **)&dPtr, &numBytes, m_CudaVBOResource));

	// Launch kernel to update instance data
	UpdateInstanceDataKernel << <BLOCKS_PER_GRID(pData->Count), THREADS_PER_BLOCK >> > (
		dPtr,
		pData->PosX,
		pData->PosY,
		pData->Scale,
		pData->Color,
		pData->RandomColor,
		pData->StillColor,
		pData->Count);
	cudaDeviceSynchronize();
	CUDA_CHECK(cudaGetLastError());

	// Unmap buffer
	CUDA_CHECK(cudaGraphicsUnmapResources(1, &m_CudaVBOResource));

	return cudaSuccess;
}
