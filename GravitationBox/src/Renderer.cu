#include "Renderer.h"
#include "glad/gl.h"
#include "Particles.h"
#include "cuda_helper.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>

// CUDA kernel to update instance data
__global__ void updateInstanceDataKernel(float *vboPtr, ParticleData pData)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (; idx < pData.Count; idx += stride)
	{
		int vboIdx = 8 * idx;

		// Update position
		*(float2 *)(&vboPtr[vboIdx]) = make_float2(pData.PosX[idx], pData.PosY[idx]);
		// Update scale
		*(float2 *)(&vboPtr[vboIdx + 2]) = pData.Scale;
		// Update color
		*(float4 *)(&vboPtr[vboIdx + 4]) = pData.Color[idx];
	}
}

void Renderer::UpdateParticleInstancesCUDA(ParticleData *pData)
{
	// Map OpenGL buffer for writing from CUDA
	float *dPtr;
	size_t numBytes;
	cudaGraphicsMapResources(1, &m_CudaVBOResource);
	cudaGraphicsResourceGetMappedPointer((void **)&dPtr, &numBytes, m_CudaVBOResource);

	// Launch kernel to update instance data
	updateInstanceDataKernel<<<BLOCKS_PER_GRID(pData->Count), THREADS_PER_BLOCK>>> (dPtr, *pData);
	cudaDeviceSynchronize();

	// Unmap buffer
	cudaGraphicsUnmapResources(1, &m_CudaVBOResource);
}