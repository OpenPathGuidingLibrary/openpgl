#pragma once

#include "field/ISurfaceVolumeField.h"

namespace openpgl
{

struct IDevice
{
    virtual ~IDevice(){};
    virtual ISurfaceVolumeField *newField(PGLFieldArguments args) const = 0;
    virtual ISurfaceVolumeField *newFieldFromFile(const std::string fieldFileName) const = 0;
};

#ifdef OPENPGL_DEVICE_TYPE_CPU_4
IDevice *newDeviceCPU4(size_t numThreads = 0);
#endif
#ifdef OPENPGL_DEVICE_TYPE_CPU_8
IDevice *newDeviceCPU8(size_t numThreads = 0);
#endif
#ifdef OPENPGL_DEVICE_TYPE_CPU_16
IDevice *newDeviceCPU16(size_t numThreads = 0);
#endif

}