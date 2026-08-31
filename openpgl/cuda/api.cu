#include "../include/openpgl/cuda.h"

#define OPENPGL_VEC_SIZE 1
#include "kernel/cuda.h"

#include "device/IDevice.h"
#include "../include/openpgl/config.h"
#include "SurfaceVolumeFieldCUDA.h"
#include "../include/openpgl/config.h"
#include "../openpgl_common.h"
#include "../include/openpgl/samplestorage.h"
#include "../field/ISurfaceVolumeField.h"

using namespace openpgl::cuda;

extern "C" OPENPGL_DLLEXPORT
PGLFieldCUDA pglNewFieldCUDA(PGLFieldArguments cfg)
OPENPGL_CATCH_BEGIN
{
    return (PGLFieldCUDA)new SurfaceVolumeFieldCUDA(cfg);
}
OPENPGL_CATCH_END(nullptr)

extern "C" OPENPGL_DLLEXPORT
void pglReleaseFieldCUDA(PGLFieldCUDA field)
OPENPGL_CATCH_BEGIN
{
    delete (SurfaceVolumeFieldCUDA*)field;
}
OPENPGL_CATCH_END_VOID

extern "C" OPENPGL_DLLEXPORT
void pglFieldCUDAUpdate(PGLFieldCUDA field, PGLSampleStorageCUDA sampleStorage)
OPENPGL_CATCH_BEGIN
{
    ((SurfaceVolumeFieldCUDA*)field)->update(*(SampleStorageCUDA*)sampleStorage);
}
OPENPGL_CATCH_END_VOID

extern "C" OPENPGL_DLLEXPORT
void pglFieldCUDAFillFieldData(PGLFieldCUDA field, void* fieldData)
OPENPGL_CATCH_BEGIN
{
    ((SurfaceVolumeFieldCUDA*)field)->fillFieldData((openpgl::gpu::FieldData*)fieldData);
}
OPENPGL_CATCH_END_VOID

extern "C" OPENPGL_DLLEXPORT
void pglFieldCUDAReleaseFieldData(PGLFieldCUDA field, void* fieldData)
OPENPGL_CATCH_BEGIN
{
    ((SurfaceVolumeFieldCUDA*)field)->releaseFieldData((openpgl::gpu::FieldData*)fieldData);
}
OPENPGL_CATCH_END_VOID

extern "C" OPENPGL_DLLEXPORT
void pglFieldCUDADump(PGLFieldCUDA field, const char* fileName)
OPENPGL_CATCH_BEGIN
{
    ((SurfaceVolumeFieldCUDA*)field)->dump(fileName);
}
OPENPGL_CATCH_END_VOID

extern "C" OPENPGL_DLLEXPORT
void pglFieldCUDATransferToCPU(PGLFieldCUDA field, PGLField fieldCPU)
OPENPGL_CATCH_BEGIN
{
    ((SurfaceVolumeFieldCUDA*)field)->transferToCPUField((openpgl::ISurfaceVolumeField*)fieldCPU);
}
OPENPGL_CATCH_END_VOID

extern "C" OPENPGL_DLLEXPORT
PGLSampleStorageCUDA pglNewSampleStorageCUDA(size_t size)
OPENPGL_CATCH_BEGIN
{
    return (PGLSampleStorageCUDA)new SampleStorageCUDA(size);
}
OPENPGL_CATCH_END(nullptr)

extern "C" OPENPGL_DLLEXPORT
PGLSampleStorageCUDA pglNewSampleStorageCUDAFromFile(const char *sampleStorageFileName)
OPENPGL_CATCH_BEGIN
{
    return (PGLSampleStorageCUDA)new SampleStorageCUDA(sampleStorageFileName);
}
OPENPGL_CATCH_END(nullptr)

extern "C" OPENPGL_DLLEXPORT
void pglReleaseSampleStorageCUDA(PGLSampleStorageCUDA sampleStorage)
OPENPGL_CATCH_BEGIN
{
    delete (SampleStorageCUDA*)sampleStorage;
}
OPENPGL_CATCH_END_VOID

extern "C" OPENPGL_DLLEXPORT
void pglSampleStorageCUDAStore(PGLSampleStorageCUDA sampleStorage, const char* fileName)
OPENPGL_CATCH_BEGIN
{
    ((SampleStorageCUDA*)sampleStorage)->store(std::string(fileName));
}
OPENPGL_CATCH_END_VOID

extern "C" OPENPGL_DLLEXPORT
void pglSampleStorageCUDAReset(PGLSampleStorageCUDA sampleStorage)
OPENPGL_CATCH_BEGIN
{
    ((SampleStorageCUDA*)sampleStorage)->reset();
}
OPENPGL_CATCH_END_VOID

extern "C" OPENPGL_DLLEXPORT 
void pglSampleStorageCUDAFillDesc(PGLSampleStorageCUDA sampleStorage, void *desc)
OPENPGL_CATCH_BEGIN
{
    ((SampleStorageCUDA*)sampleStorage)->fillGPUDesc((openpgl::cuda::SampleStorageCUDADesc*)desc);
}
OPENPGL_CATCH_END_VOID

extern "C" OPENPGL_DLLEXPORT 
void pglSampleStorageCUDATransferToCPU(PGLSampleStorageCUDA sampleStorage, PGLSampleStorage sampleStorageCPU)
OPENPGL_CATCH_BEGIN
{
    ((SampleStorageCUDA*)sampleStorage)->transferToCPU(*(openpgl::SampleDataStorage*)sampleStorageCPU);
}
OPENPGL_CATCH_END_VOID

extern "C" OPENPGL_DLLEXPORT 
void pglSampleStorageCUDATransferFromCPU(PGLSampleStorageCUDA sampleStorage, PGLSampleStorage sampleStorageCPU)
OPENPGL_CATCH_BEGIN
{
    ((SampleStorageCUDA*)sampleStorage)->transferFromCPU(*(openpgl::SampleDataStorage*)sampleStorageCPU);
}
OPENPGL_CATCH_END_VOID