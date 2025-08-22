#ifndef KERNEL_H
#define KERNEL_H

#include "openpgl/cpp/SampleStorage.h"

namespace openpgl{
namespace gpu {
namespace cuda {

class GPUField;
class SamplesDevice;

GPUField* GPUFieldCreate();
void GPUFieldDestroy(GPUField* field);
void GPUFieldUpdate(openpgl::gpu::cuda::GPUField *field, SamplesDevice* samplesDevice);

SamplesDevice* SamplesDeviceCreate(const openpgl::cpp::SampleStorage &sampleStorage);
void SamplesDeviceDestroy(SamplesDevice* samplesDevice);
}
} 
}

#endif // KERNEL_H

