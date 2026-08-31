#pragma once

#include <cstring>
#if defined(OPENPGL_GPU_SYCL_SUPPORT)
#include <sycl/sycl.hpp>
#endif

#if defined(OPENPGL_GPU_CUDA_SUPPORT)
#include <cuda_runtime.h>

#define cudaErrchk(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=false)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"cudaAssert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

inline void checkCudaError() {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cout << " CUDAError " << std::endl;
    }
}

#define OPENPGL_CUDA_CHECK(EXPR)                                        \
    if (EXPR != cudaSuccess) {                                  \
        cudaError_t error = cudaGetLastError();                 \
        std::cout << "CUDA error: "<< cudaGetErrorString(error) << std::endl; \
    } else /* eat semicolon */

#endif

namespace openpgl
{
namespace gpu
{
struct Device
{
    enum DeviceTypes
    {
        EDeviceType_CPU = 0,
#if defined(OPENPGL_GPU_SYCL_SUPPORT)
        EDeviceType_SYCL,
#endif
#if defined(OPENPGL_GPU_CUDA_SUPPORT)
        EDeviceType_CUDA,
#endif
    };

    DeviceTypes m_deviceType{EDeviceType_CPU};
#if defined(OPENPGL_GPU_SYCL_SUPPORT)
    ::sycl::queue* q;
    ::sycl::queue _q;
#endif
    Device(DeviceTypes dt, void* ptr = nullptr) : m_deviceType(dt) {
        (void)ptr;
#if defined(OPENPGL_GPU_SYCL_SUPPORT)
    if (ptr) {
        q = reinterpret_cast<::sycl::queue*>(ptr);
    } else {
        q = &_q;
    }
#endif        
    }

    ~Device() {}

    void wait()
    {
        switch (m_deviceType)
        {
            case EDeviceType_CPU:
            {
                break;
            }
#if defined(OPENPGL_GPU_SYCL_SUPPORT)
            case EDeviceType_SYCL:
            {
                try {
                    q->wait();
                } catch (::sycl::exception const &e) {
                    std::cout << "Caught sync SYCL exception: " << e.what() << "\n";
                }
                break;
            }
#endif
#if defined(OPENPGL_GPU_CUDA_SUPPORT)
            case EDeviceType_CUDA:
            {
                OPENPGL_CUDA_CHECK(cudaDeviceSynchronize());
                break;
            }
#endif
            default:
            {
                break;
            }
        }
    }

    template <class T>
    T *mallocArray(size_t numElements, bool shared = false)
    {
        T* ptr = nullptr;
        (void)ptr;
        switch (m_deviceType)
        {
            case EDeviceType_CPU:
            {
                return new T[numElements];
                break;
            }
#if defined(OPENPGL_GPU_SYCL_SUPPORT)
            case EDeviceType_SYCL:
            {
                try{
                    if(shared) {
                        ptr =  ::sycl::malloc_shared<T>(numElements, *q);
                    } else {
                        ptr =  ::sycl::malloc_device<T>(numElements, *q);
                    }
                } catch (::sycl::exception const &e) {
                    std::cout << "Caught sync SYCL exception: " << e.what() << "\n";
                    ptr =nullptr;
                }
                return ptr;
                break;
            }
#endif
#if defined(OPENPGL_GPU_CUDA_SUPPORT)
            case EDeviceType_CUDA:
            {
                void *devPtr;
                if(!shared) {
                    cudaErrchk(cudaMalloc(&devPtr, numElements * sizeof(T)));
                } else {
                    cudaErrchk(cudaMallocManaged(&devPtr, numElements * sizeof(T)));
                }
                checkCudaError();
                return (T *)devPtr;
                break;
            }
#endif
            default:
            {
                return new T[numElements];
                break;
            }
        }
    }

    template <class T>
    void freeArray(T *ptr)
    {
        switch (m_deviceType)
        {
            case EDeviceType_CPU:
            {
                delete[] ptr;
                break;
            }
#if defined(OPENPGL_GPU_SYCL_SUPPORT)
            case EDeviceType_SYCL:
            {
                try {
                    ::sycl::free(ptr, *q);
                } catch (::sycl::exception const &e) {
                    std::cout << "Caught sync SYCL exception: " << e.what() << "\n";
                }
                break;
            }
#endif
#if defined(OPENPGL_GPU_CUDA_SUPPORT)
            case EDeviceType_CUDA:
            {
                cudaErrchk(cudaFree(ptr));
                checkCudaError();
                break;
            }
#endif
            default:
            {
                delete[] ptr;
                break;
            }
        }
    }

    template <class T>
    void memcpyArrayToGPU(T *devicePtr, const T *hostPtr, size_t numElements)
    {
        switch (m_deviceType)
        {
            case EDeviceType_CPU:
            {
                std::memcpy(devicePtr, hostPtr, numElements * sizeof(T));
                break;
            }
#if defined(OPENPGL_GPU_SYCL_SUPPORT)
            case EDeviceType_SYCL:
            {
                try{
                    q->memcpy(devicePtr, hostPtr, numElements * sizeof(T));
                } catch (::sycl::exception const &e) {
                    std::cout << "Caught sync SYCL exception: " << e.what() << "\n";
                }
                break;
            }
#endif
#if defined(OPENPGL_GPU_CUDA_SUPPORT)
            case EDeviceType_CUDA:
            {
                cudaErrchk(cudaMemcpy(devicePtr, hostPtr, numElements * sizeof(T), cudaMemcpyHostToDevice));
                checkCudaError();
                break;
            }
#endif
            default:
            {
                std::memcpy(devicePtr, hostPtr, numElements * sizeof(T));
                break;
            }
        }
    }

    template <class T>
    void memcpyArrayFromGPU(T *devicePtr, T *hostPtr, size_t numElements)
    {
        switch (m_deviceType)
        {
            case EDeviceType_CPU:
            {
                std::memcpy(hostPtr, devicePtr, numElements * sizeof(T));
                break;
            }
#if defined(OPENPGL_GPU_SYCL_SUPPORT)
            case EDeviceType_SYCL:
            {
                try{
                    q->memcpy(hostPtr, devicePtr, numElements * sizeof(T));
                } catch (::sycl::exception const &e) {
                    std::cout << "Caught sync SYCL exception: " << e.what() << "\n";
                }
                break;
            }
#endif
#if defined(OPENPGL_GPU_CUDA_SUPPORT)
            case EDeviceType_CUDA:
            {
                cudaErrchk(cudaMemcpy(hostPtr, devicePtr, numElements * sizeof(T), cudaMemcpyDeviceToHost));
                checkCudaError();
                break;
            }
#endif
            default:
            {
                std::memcpy(hostPtr, devicePtr, numElements * sizeof(T));
                break;
            }
        }
    }
};
}  // namespace gpu
}  // namespace openpgl