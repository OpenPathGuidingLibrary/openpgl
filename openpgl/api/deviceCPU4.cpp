#define VEC_SIZE 4
#include "kernel/cpu.h"
#include "device/Device.h"

namespace openpgl
{

IDevice *newDeviceCPU4(size_t numThreads)
{
    return (IDevice *)new Device<Kernel>(numThreads);
}

}  // namespace openpgl