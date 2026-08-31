#pragma once

#include "../common.h"
#include "../data.h"

#define KERNEL_FUNCTION __device__

namespace openpgl {
namespace cuda {

struct SampleStorageCUDAAlloc {
    uint32_t capacity;
    uint32_t sizeSurface;
    uint32_t sizeVolume;
};

struct SampleStorageCUDADesc{
    SampleStorageCUDAAlloc* alloc;
    PGLSampleData* samplesSurface;
    PGLSampleData* samplesVolume;

    KERNEL_FUNCTION inline void AddSample(const PGLSampleData &sd);
};

#ifdef __CUDACC__
KERNEL_FUNCTION void SampleStorageCUDADesc::AddSample(const PGLSampleData &sd) {
    bool isInsideVolume = sd.flags & PGLSampleData::EInsideVolume;
    uint32_t &atomic = isInsideVolume ? alloc->sizeVolume : alloc->sizeSurface;
    uint32_t offset = atomicAdd(&atomic, 1);
    PGLSampleData *arr = isInsideVolume ? samplesVolume : samplesSurface;
    if (offset < alloc->capacity)
        arr[offset] = sd;
}
#endif

}
}
