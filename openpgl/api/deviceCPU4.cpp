#define OPENPGL_VEC_SIZE 4
#include "kernel/cpu.h"
#include "device/Device.h"

namespace openpgl
{

IDevice *newDeviceCPU4(size_t numThreads)
{
    return (IDevice *)new OPENPGL_KERNEL_NS::Device<OPENPGL_KERNEL_NS::Kernel>(numThreads);
}

}  // namespace openpgl