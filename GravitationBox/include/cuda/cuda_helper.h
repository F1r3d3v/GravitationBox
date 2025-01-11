#pragma once
#include "engine/Log.h"

constexpr int THREADS_PER_BLOCK = 256;
static int BLOCKS_PER_GRID(int x) { return (x + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK; }

#define CUDA_CHECK(err)                                                                                                                                                      \
do                                                                                                                                                                           \
{                																																		                     \
    cudaError_t cudaStatus = (err);                                                                                                                                          \
    if (cudaStatus != cudaSuccess)                                                                                                                                           \
    {                                                                                                                                                                        \
        Log::Error("CUDA Error " + std::to_string(cudaStatus) + ": " + cudaGetErrorString(cudaStatus) + ". In file '" + __FILE__ + "' on line " + std::to_string(__LINE__)); \
        return cudaStatus;                                                                                                                                                   \
    }                                                                                                                                                                        \
} while (false)

#define CUDA_CHECK_NR(err)                                                                                                                                                   \
do                                                                                                                                                                           \
{                																																		                     \
    cudaError_t cudaStatus = (err);                                                                                                                                          \
    if (cudaStatus != cudaSuccess)                                                                                                                                           \
    {                                                                                                                                                                        \
        Log::Error("CUDA Error " + std::to_string(cudaStatus) + ": " + cudaGetErrorString(cudaStatus) + ". In file '" + __FILE__ + "' on line " + std::to_string(__LINE__)); \
    }                                                                                                                                                                        \
} while (false)