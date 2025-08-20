#ifndef KERNEL_H
#define KERNEL_H

#include "openpgl/cpp/SampleStorage.h"

namespace openpgl{
namespace gpu {
namespace cuda {

class GPUField;

GPUField* GPUFieldCreate();
void GPUFieldDestroy(GPUField* field);
void GPUFieldUpdate(GPUField *field, const openpgl::cpp::SampleStorage &sampleStorage);

}
} 
}

#endif // KERNEL_H

