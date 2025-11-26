#pragma once

#include "common.h"
#include "data.h"
#include "config.h"
#include "samplestorage.h"

#ifdef __cplusplus
    struct FieldCUDA;
#else
typedef ManagedObject FieldCUDA;
#endif

typedef struct PGLFieldCUDA_t *PGLFieldCUDA;

typedef struct PGLSampleStorageCUDA_t *PGLSampleStorageCUDA;

#ifdef __cplusplus
extern "C"
{
#endif

OPENPGL_CORE_INTERFACE PGLFieldCUDA pglNewFieldCUDA(PGLFieldArguments cfg);

OPENPGL_CORE_INTERFACE void pglReleaseFieldCUDA(PGLFieldCUDA field);

OPENPGL_CORE_INTERFACE void pglFieldCUDAUpdate(PGLFieldCUDA field, PGLSampleStorageCUDA sampleStorage);

OPENPGL_CORE_INTERFACE void pglFieldCUDAGetDesc(PGLFieldCUDA field, void *desc);

OPENPGL_CORE_INTERFACE void pglFieldCUDADump(PGLFieldCUDA field, const char* fileName);

OPENPGL_CORE_INTERFACE void pglFieldCUDAFillFieldData(PGLFieldCUDA field, void* fieldData);

OPENPGL_CORE_INTERFACE void pglFieldCUDAReleaseFieldData(PGLFieldCUDA field, void* fieldData);

OPENPGL_CORE_INTERFACE PGLSampleStorageCUDA pglNewSampleStorageCUDA(size_t size);

OPENPGL_CORE_INTERFACE PGLSampleStorageCUDA pglNewSampleStorageCUDAFromFile(const char *sampleStorageFileName);

OPENPGL_CORE_INTERFACE void pglReleaseSampleStorageCUDA(PGLSampleStorageCUDA sampleStorage);

OPENPGL_CORE_INTERFACE void pglSampleStorageCUDAStore(PGLSampleStorageCUDA sampleStorage, const char* fileName);

OPENPGL_CORE_INTERFACE void pglSampleStorageCUDAReset(PGLSampleStorageCUDA sampleStorage);

OPENPGL_CORE_INTERFACE void pglSampleStorageCUDAFillDesc(PGLSampleStorageCUDA sampleStorage, void *desc);

OPENPGL_CORE_INTERFACE void pglSampleStorageCUDATransferToCPU(PGLSampleStorageCUDA sampleStorage, PGLSampleStorage sampleStorageCPU);

OPENPGL_CORE_INTERFACE void pglSampleStorageCUDATransferFromCPU(PGLSampleStorageCUDA sampleStorage, PGLSampleStorage sampleStorageCPU);

#ifdef __cplusplus
}  // extern "C"
#endif
