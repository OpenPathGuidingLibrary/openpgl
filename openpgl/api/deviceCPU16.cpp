#define OPENPGL_VEC_SIZE 16
#include "kernel/cpu.h"
#include "device/Device.h"

namespace openpgl
{

IDevice *newDeviceCPU16(size_t numThreads)
{
    return (IDevice *)new Device<Kernel>(numThreads);
}

}  // namespace openpgl