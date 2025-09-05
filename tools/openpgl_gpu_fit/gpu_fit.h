#ifndef KERNEL_H
#define KERNEL_H

#include "../../openpgl/include/openpgl/sdump.h"

namespace openpgl{
namespace gpu {
namespace cuda {

class GPUField;
class SamplesDevice;

GPUField* GPUFieldCreate();
void GPUFieldDestroy(GPUField* field);
void GPUFieldUpdate(openpgl::gpu::cuda::GPUField *field, SamplesDevice* samplesDevice);
void GPUFieldSDump(openpgl::gpu::cuda::GPUField *field, SDump* sDump);
void checkUsage();

SamplesDevice* SamplesDeviceCreate(const std::string& path);
void SamplesDeviceDestroy(SamplesDevice* samplesDevice);

}
} 
}

#endif // KERNEL_H

