#define VEC_SIZE 8
#include "kernel/cpu.h"
#include "device/Device.h"

namespace openpgl
{

IDevice *newDeviceCPU8(size_t numThreads)
{
    return (IDevice *)new Device<Kernel>(numThreads);
}

}  // namespace openpgl