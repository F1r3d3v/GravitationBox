#include "Renderer.h"
#include "glad/gl.h"
#include "Particles.h"
#include "cuda_helper.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>

__global__ void updateInstanceDataKernel(float *vboPtr, float *PosX, float *PosY, float2 Scale, float4 *Color, size_t count)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (; tid < count; tid += stride)
	{
		int vboIdx = 8 * tid;
		// Update position
		*(float2 *)(&vboPtr[vboIdx]) = make_float2(PosX[tid], PosY[tid]);
		// Update scale
		*(float2 *)(&vboPtr[vboIdx + 2]) = Scale;
		// Update color
		*(float4 *)(&vboPtr[vboIdx + 4]) = Color[tid];
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
	updateInstanceDataKernel << <BLOCKS_PER_GRID(pData->Count), THREADS_PER_BLOCK >> >(dPtr, pData->PosX, pData->PosY, pData->Scale, pData->Color, pData->Count);
	cudaDeviceSynchronize();

	// Unmap buffer
	cudaGraphicsUnmapResources(1, &m_CudaVBOResource);
}