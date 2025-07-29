#ifndef OPENPGL_BUILD
#include <openpgl/defines.h>
#endif

#if defined(OPENPGL_GPU_SYCL)
#include <sycl/sycl.hpp>
#endif

#define USE_TREELETS
#ifndef ONE_OVER_FOUR_PI
#define ONE_OVER_FOUR_PI 0.07957747154594767F
#endif

#ifndef FLT_EPSILON
#define FLT_EPSILON 1.19209290E-07F
#endif

#if defined(OPENPGL_GPU_CPU)
#ifndef FLT_EPSILON
#define FLT_EPSILON 1.19209290E-07F
#endif
#include <cstring>
#endif

#if defined(OPENPGL_GPU_SYCL)
#ifndef FLT_EPSILON
#define FLT_EPSILON 1.19209290E-07F
#endif
#ifdef OPENPGL_GPU_CALLABLE
#undef OPENPGL_GPU_CALLABLE
#endif
#define OPENPGL_GPU_CALLABLE
#elif defined(OPENPGL_GPU_CUDA)// && defined(__CUDACC__)
#ifndef FLT_EPSILON
#define FLT_EPSILON 1.19209290E-07F
#endif
#ifdef OPENPGL_GPU_CALLABLE
#undef OPENPGL_GPU_CALLABLE
#endif
#define OPENPGL_GPU_CALLABLE __host__ __device__
#else
#ifdef OPENPGL_GPU_CALLABLE
#undef OPENPGL_GPU_CALLABLE
#endif
#define OPENPGL_GPU_CALLABLE
#endif

#if defined(OPENPGL_GPU_SYCL)
#ifdef OPENPGL_CPU_GPU_LAMBDA
#undef OPENPGL_CPU_GPU_LAMBDA
#endif
#define OPENPGL_CPU_GPU_LAMBDA(...) [=, *this ] OPENPGL_GPU_CALLABLE(__VA_ARGS__)
#else
#ifdef OPENPGL_CPU_GPU_LAMBDA
#undef OPENPGL_CPU_GPU_LAMBDA
#endif
#define OPENPGL_CPU_GPU_LAMBDA(...) [=, *this ] OPENPGL_GPU_CALLABLE(__VA_ARGS__) mutable
#endif

#if defined(OPENPGL_GPU_CUDA)
template <typename F>
inline int CUDAGetBlockSize(const char *description, F kernel) {
    (void)description;
    // Note: this isn't reentrant, but that's fine for our purposes...
    static std::map<std::type_index, int> kernelBlockSizes;

    std::type_index index = std::type_index(typeid(F));

    auto iter = kernelBlockSizes.find(index);
    if (iter != kernelBlockSizes.end())
        return iter->second;

    int minGridSize, blockSize;
    OPENPGL_CUDA_CHECK(
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, 0, 0));
    kernelBlockSizes[index] = blockSize;
    //LOG_VERBOSE("[%s]: block size %d", description, blockSize);

    return blockSize;
}

// GPU Launch Function Declarations
template <typename F>
void ParallelFor(openpgl::gpu::Device* device, const char *description, int nItems, F &&func);

#ifdef __NVCC__
template <typename F>
__global__ void Kernel(F func, int nItems) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nItems)
        return;
    func(tid);
}

template <typename F>
void ParallelFor(openpgl::gpu::Device* device, const char *description, int nItems, F &&func) {
    (void)device;
    CUDAParallelFor(description, nItems, func);
}
template <typename F>
void CUDAParallelFor(const char *description, int nItems, F func) {
    (void)description;
    auto kernel = &Kernel<F>;
    int blockSize = CUDAGetBlockSize(description, kernel);
    int gridSize = (nItems + blockSize - 1) / blockSize;
    kernel<<<gridSize, blockSize>>>(func, nItems);
}
#else
template <typename F>
void ParallelFor(openpgl::gpu::Device* device, const char *description, int nItems, F &&func) {
    (void)device;
    (void)description;
    (void)nItems;
    (void)func;
    //CUDAParallelFor(description, nItems, func);
}

#endif
#elif defined(OPENPGL_GPU_SYCL)
template <typename F>
void SYCLParallelFor(openpgl::gpu::Device* device, const char *description, int nItems, F func) {
    device->q->parallel_for(::sycl::range<1>(nItems), [=](::sycl::id<1> i) {
        int idx = i.get(0);
        func(idx);
    });
}

template <typename F>
void ParallelFor(openpgl::gpu::Device* device, const char *description, int nItems, F &&func) {
    SYCLParallelFor(device, description, nItems, func);
}
#endif
#if defined(OPENPGL_GPU_CPU)

//#ifndef __NVCC__
//#include <tbb/parallel_for.h>
//#endif

template <typename F>
void CPUParallelFor(openpgl::gpu::Device* device, const char *description, int nItems, F func) {
    (void)device;
    (void)description;
#ifndef __NVCC__
    tbb::parallel_for(tbb::blocked_range<int>(0, nItems), [&](tbb::blocked_range<int> r) {
        for (int idx = r.begin(); idx < r.end(); idx++){
            func(idx);
        }
    });
#endif
}

template <typename F>
void ParallelFor(openpgl::gpu::Device* device, const char *description, int nItems, F &&func) {
    CPUParallelFor(device, description, nItems, func);
}
#endif