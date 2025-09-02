// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../include/openpgl/openpgl.h"
// #include "../openpglTypes.h"

#include "kernel/cpu.h"

#include "data/PathSegmentData.h"
#include "data/PathSegmentDataStorage.h"
#include "data/SampleData.h"
#include "data/SampleDataStorage.h"
#include "device/IDevice.h"
#include "directional/ISurfaceSamplingDistribution.h"
#include "directional/IVolumeSamplingDistribution.h"
#include "field/FieldStatistics.h"
#include "field/ISurfaceVolumeField.h"
#include "openpgl_common.h"

#if defined(OPENPGL_IMAGE_SPACE_GUIDING_BUFFER)
#include "imagespace/ImageSpaceGuidingBuffer.h"
#endif

#include <cstring>

using namespace openpgl;

#define THROW_IF_NULL(obj, name) \
    if (obj == nullptr)          \
    throw std::runtime_error(std::string("null ") + name + std::string(" provided to ") + __FUNCTION__)

// convenience macros for repeated use of the above
#define THROW_IF_NULL_OBJECT(obj) THROW_IF_NULL(obj, "handle")
#define THROW_IF_NULL_STRING(str) THROW_IF_NULL(str, "string")

#define OPENPGL_CATCH_BEGIN \
    try                     \
    {
#define OPENPGL_CATCH_END(a)                                                \
    }                                                                       \
    catch (const std::bad_alloc &)                                          \
    {                                                                       \
        std::cout << "Open PGL was unable to allocate memory" << std::endl; \
        return a;                                                           \
    }                                                                       \
    catch (const std::exception &e)                                         \
    {                                                                       \
        std::cout << e.what() << std::endl;                                 \
        return a;                                                           \
    }                                                                       \
    catch (...)                                                             \
    {                                                                       \
        std::cout << "an unrecognized exception was caught" << std::endl;   \
        return a;                                                           \
    }

#define OPENPGL_CATCH_END_VOID                                              \
    }                                                                       \
    catch (const std::bad_alloc &)                                          \
    {                                                                       \
        std::cout << "Open PGL was unable to allocate memory" << std::endl; \
        return;                                                             \
    }                                                                       \
    catch (const std::exception &e)                                         \
    {                                                                       \
        std::cout << e.what() << std::endl;                                 \
        return;                                                             \
    }                                                                       \
    catch (...)                                                             \
    {                                                                       \
        std::cout << "an unrecognized exception was caught" << std::endl;   \
        return;                                                             \
    }

typedef ISurfaceVolumeField IGuidingField;

///////////////////////////////////////////////////////////////////////////////
// Field //////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

#define OPENPGL_FIELD_STRING "OPENPGL_" OPENPGL_VERSION_STRING "_FIELD"

extern "C" OPENPGL_DLLEXPORT PGLDevice pglNewDevice(PGL_DEVICE_TYPE deviceType, size_t numThreads) OPENPGL_CATCH_BEGIN
{
#ifdef OPENPGL_DEVICE_TYPE_CPU_4
    if (deviceType == PGL_DEVICE_TYPE_CPU_4)
        return (PGLDevice)newDeviceCPU4(numThreads);
#endif
#ifdef OPENPGL_DEVICE_TYPE_CPU_8
    if (deviceType == PGL_DEVICE_TYPE_CPU_8)
        return (PGLDevice)newDeviceCPU8(numThreads);
#endif
#ifdef OPENPGL_DEVICE_TYPE_CPU_16
    if (deviceType == PGL_DEVICE_TYPE_CPU_16)
        return (PGLDevice)newDeviceCPU16(numThreads);
#endif

    throw std::runtime_error("invalid vectorSize parameter!");
}
OPENPGL_CATCH_END(nullptr)

extern "C" OPENPGL_DLLEXPORT void pglReleaseDevice(PGLDevice device)
{
    auto *gDevice = (IDevice *)device;
    delete gDevice;
}

extern "C" OPENPGL_DLLEXPORT PGLField pglDeviceNewField(PGLDevice device, PGLFieldArguments args) OPENPGL_CATCH_BEGIN
{
    THROW_IF_NULL_OBJECT(device);
    auto *gDevice = (IDevice *)device;
    return (PGLField)gDevice->newField(args);
}
OPENPGL_CATCH_END(nullptr)

extern "C" OPENPGL_DLLEXPORT PGLField pglDeviceNewFieldFromFile(PGLDevice device, const char *fieldFileName) OPENPGL_CATCH_BEGIN
{
    THROW_IF_NULL_OBJECT(device);
    THROW_IF_NULL_STRING(fieldFileName);
    auto *gDevice = (IDevice *)device;
    return (PGLField)gDevice->newFieldFromFile(fieldFileName);
}
OPENPGL_CATCH_END(nullptr)

extern "C" OPENPGL_DLLEXPORT void pglReleaseField(PGLField field) OPENPGL_CATCH_BEGIN
{
    auto *gField = (IGuidingField *)field;
    delete gField;
}
OPENPGL_CATCH_END_VOID

extern "C" OPENPGL_DLLEXPORT bool pglFieldStoreToFile(PGLField field, const char *fieldFileName) OPENPGL_CATCH_BEGIN
{
    THROW_IF_NULL_OBJECT(field);
    THROW_IF_NULL_STRING(fieldFileName);
    ((IGuidingField *)field)->storeToFile(fieldFileName);
    return true;
}
OPENPGL_CATCH_END(false)

extern "C" OPENPGL_DLLEXPORT size_t pglFieldGetIteration(PGLField field) OPENPGL_CATCH_BEGIN
{
    auto *gField = (IGuidingField *)field;
    return gField->getIteration();
}
OPENPGL_CATCH_END(0)

extern "C" OPENPGL_DLLEXPORT void pglFieldSetSceneBounds(PGLField field, pgl_box3f bounds)
{
    auto *gField = (IGuidingField *)field;
    openpgl::BBox sceneBounds;
    sceneBounds.lower = openpgl::Vector3(bounds.lower.x, bounds.lower.y, bounds.lower.z);
    sceneBounds.upper = openpgl::Vector3(bounds.upper.x, bounds.upper.y, bounds.upper.z);
    gField->setSceneBounds(sceneBounds);
}

extern "C" OPENPGL_DLLEXPORT pgl_box3f pglFieldGetSceneBounds(PGLField field)
{
    auto *gField = (IGuidingField *)field;
    openpgl::BBox sceneBounds = gField->getSceneBounds();
    pgl_box3f bounds;
    bounds.lower.x = sceneBounds.lower.x;
    bounds.lower.y = sceneBounds.lower.y;
    bounds.lower.z = sceneBounds.lower.z;

    bounds.upper.x = sceneBounds.upper.x;
    bounds.upper.y = sceneBounds.upper.y;
    bounds.upper.z = sceneBounds.upper.z;

    return bounds;
}

extern "C" OPENPGL_DLLEXPORT void pglFieldUpdate(PGLField field, PGLSampleStorage sampleStorage) OPENPGL_CATCH_BEGIN
{
    auto *gField = (IGuidingField *)field;
    auto *gSampleStorage = (openpgl::SampleDataStorage *)sampleStorage;
    gField->updateField(gSampleStorage->m_surfaceContainer, gSampleStorage->m_volumeContainer);
}
OPENPGL_CATCH_END_VOID

extern "C" OPENPGL_DLLEXPORT void pglFieldUpdateSurface(PGLField field, PGLSampleStorage sampleStorage) OPENPGL_CATCH_BEGIN
{
    auto *gField = (IGuidingField *)field;
    auto *gSampleStorage = (openpgl::SampleDataStorage *)sampleStorage;
    gField->updateFieldSurface(gSampleStorage->m_surfaceContainer);
}
OPENPGL_CATCH_END_VOID

extern "C" OPENPGL_DLLEXPORT void pglFieldUpdateVolume(PGLField field, PGLSampleStorage sampleStorage) OPENPGL_CATCH_BEGIN
{
    auto *gField = (IGuidingField *)field;
    auto *gSampleStorage = (openpgl::SampleDataStorage *)sampleStorage;
    gField->updateFieldVolume(gSampleStorage->m_volumeContainer);
}
OPENPGL_CATCH_END_VOID

extern "C" OPENPGL_DLLEXPORT void pglFieldReset(PGLField field) OPENPGL_CATCH_BEGIN
{
    auto *gField = (IGuidingField *)field;
    gField->resetField();
}
OPENPGL_CATCH_END_VOID

extern "C" OPENPGL_DLLEXPORT PGLSurfaceSamplingDistribution pglFieldNewSurfaceSamplingDistribution(PGLField field) OPENPGL_CATCH_BEGIN
{
    auto *gField = (IGuidingField *)field;
    ISurfaceSamplingDistribution *surfaceSamplingDistribution = gField->newSurfaceSamplingDistribution();
    return (PGLSurfaceSamplingDistribution)surfaceSamplingDistribution;
}
OPENPGL_CATCH_END(nullptr)

extern "C" OPENPGL_DLLEXPORT bool pglFieldInitSurfaceSamplingDistribution(PGLField field, PGLSurfaceSamplingDistribution surfaceSamplingDistribution, pgl_point3f position,
                                                                          float *sample1D)
{
    const openpgl::Point3 pos(position.x, position.y, position.z);
    auto *gField = (IGuidingField *)field;
    ISurfaceSamplingDistribution *gSurfaceSamplingDistribution = (ISurfaceSamplingDistribution *)surfaceSamplingDistribution;
    return gField->initSurfaceSamplingDistribution(gSurfaceSamplingDistribution, pos, sample1D);
}
extern "C" OPENPGL_DLLEXPORT PGLVolumeSamplingDistribution pglFieldNewVolumeSamplingDistribution(PGLField field) OPENPGL_CATCH_BEGIN
{
    auto *gField = (IGuidingField *)field;
    IVolumeSamplingDistribution *volumeSamplingDistribution = gField->newVolumeSamplingDistribution();
    return (PGLVolumeSamplingDistribution)volumeSamplingDistribution;
}
OPENPGL_CATCH_END(nullptr)

extern "C" OPENPGL_DLLEXPORT bool pglFieldInitVolumeSamplingDistribution(PGLField field, PGLVolumeSamplingDistribution volumeSamplingDistribution, pgl_point3f position,
                                                                         float *sample1D)
{
    const openpgl::Point3 pos(position.x, position.y, position.z);
    auto *gField = (IGuidingField *)field;
    IVolumeSamplingDistribution *gVolumeSamplingDistribution = (IVolumeSamplingDistribution *)volumeSamplingDistribution;
    return gField->initVolumeSamplingDistribution(gVolumeSamplingDistribution, pos, sample1D);
}

extern "C" OPENPGL_DLLEXPORT bool pglFieldValidate(PGLField field)
{
    const auto *gField = (const IGuidingField *)field;
    return gField->validate(true, true);
}

extern "C" OPENPGL_DLLEXPORT bool pglFieldCompare(PGLField fieldA, PGLField fieldB)
{
    auto *gFieldA = (const IGuidingField *)fieldA;
    auto *gFieldB = (const IGuidingField *)fieldB;
    return gFieldA->operator==(gFieldB);
}

extern "C" OPENPGL_DLLEXPORT PGLFieldStatistics pglFieldGetSurfaceStatistics(PGLField field)
{
    const auto *gField = (const IGuidingField *)field;
    return (PGLFieldStatistics)gField->getSurfaceStatistics();
}

extern "C" OPENPGL_DLLEXPORT PGLFieldStatistics pglFieldGetVolumeStatistics(PGLField field)
{
    const auto *gField = (const IGuidingField *)field;
    return (PGLFieldStatistics)gField->getVolumeStatistics();
}

// Stefan SYCL
/*
extern "C" OPENPGL_DLLEXPORT  int pglFieldGetNumSurfaceNodes(PGLField field)
{
    const auto *gField = (const IGuidingField *)field;
    return gField->GetNumNodes(true);
}
extern "C" OPENPGL_DLLEXPORT  void* pglFieldGetSurfaceNodes(PGLField field)
{
    const auto *gField = (const IGuidingField *)field;
    return gField->GetNodes(true);
}

extern "C" OPENPGL_DLLEXPORT  int pglFieldGetNumSurfaceDistributions(PGLField field)
{
    const auto *gField = (const IGuidingField *)field;
    return gField->GetNumDistributions(true);
}
extern "C" OPENPGL_DLLEXPORT  void pglFieldCopySurfaceDistributions(PGLField field, void *p)
{
    const auto *gField = (const IGuidingField *)field;
    return gField->CopyDistributionsTo(p, true);
}

extern "C" OPENPGL_DLLEXPORT  int pglFieldGetNumVolumeNodes(PGLField field)
{
    const auto *gField = (const IGuidingField *)field;
    return gField->GetNumNodes(false);
}
extern "C" OPENPGL_DLLEXPORT  void* pglFieldGetVolumeNodes(PGLField field)
{
    const auto *gField = (const IGuidingField *)field;
    return gField->GetNodes(false);
}

extern "C" OPENPGL_DLLEXPORT  int pglFieldGetNumVolumeDistributions(PGLField field)
{
    const auto *gField = (const IGuidingField *)field;
    return gField->GetNumDistributions(false);
}
extern "C" OPENPGL_DLLEXPORT  void pglFieldCopyVolumeDistributions(PGLField field, void *p)
{
    const auto *gField = (const IGuidingField *)field;
    return gField->CopyDistributionsTo(p, false);
}

*/
extern "C" OPENPGL_DLLEXPORT void pglFieldFillFieldGPU(PGLField field, void *fieldData, void *deviceGPU)
{
    const auto *gField = (const IGuidingField *)field;
    return gField->FillFieldData(static_cast<openpgl::gpu::FieldData *>(fieldData), static_cast<openpgl::gpu::Device *>(deviceGPU));
}

extern "C" OPENPGL_DLLEXPORT void pglFieldReleaseFieldGPU(PGLField field, void *fieldData, void *deviceGPU)
{
    const auto *gField = (const IGuidingField *)field;
    return gField->ReleaseFieldData(static_cast<openpgl::gpu::FieldData *>(fieldData), static_cast<openpgl::gpu::Device *>(deviceGPU));
}

extern "C" OPENPGL_DLLEXPORT PGLRange pglFieldGetSurfaceSampleRange(PGLField field, size_t id)
{
    const auto *gField = (const IGuidingField *)field;
    return gField->getSurfaceSampleRange(id);
}

extern "C" OPENPGL_DLLEXPORT PGLRange pglFieldGetVolumeSampleRange(PGLField field, size_t id)
{
    const auto *gField = (const IGuidingField *)field;
    return gField->getVolumeSampleRange(id);
}

extern "C" OPENPGL_DLLEXPORT void pglFieldRunUpdateDump(PGLField field, const char *updateDumpFileName, bool surface)
{
    const auto *gField = (const IGuidingField *)field;
    gField->runUpdateDump(updateDumpFileName, surface);
}

extern "C" OPENPGL_DLLEXPORT void pglFieldSDump(PGLField field, SDump* sdump)
{
    const auto *gField = (const IGuidingField *)field;
    gField->sDump(sdump);
}

///////////////////////////////////////////////////////////////////////////////
// SampleStorage //////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

#define OPENPGL_SAMPLE_STORAGE_STRING "OPENPGL_" OPENPGL_VERSION_STRING "_SAMPLE_STORAGE"

extern "C" OPENPGL_DLLEXPORT PGLSampleStorage pglNewSampleStorage() OPENPGL_CATCH_BEGIN
{
    return (PGLSampleStorage)SampleDataStorage::newSampleDataStorage();
}
OPENPGL_CATCH_END(nullptr)

extern "C" OPENPGL_DLLEXPORT PGLSampleStorage pglNewSampleStorageFromFile(const char *sampleStorageFileName) OPENPGL_CATCH_BEGIN
{
    THROW_IF_NULL_STRING(sampleStorageFileName);
    return (PGLSampleStorage)openpgl::SampleDataStorage::newSampleDataStorageFromFile(sampleStorageFileName);
}
OPENPGL_CATCH_END(nullptr)

extern "C" OPENPGL_DLLEXPORT void pglReleaseSampleStorage(PGLSampleStorage sampleStorage)
{
    auto *gSampleStorage = (openpgl::SampleDataStorage *)sampleStorage;
    delete gSampleStorage;
}

extern "C" OPENPGL_DLLEXPORT bool pglSampleStorageStoreToFile(PGLSampleStorage sampleStorage, const char *sampleStorageFileName) OPENPGL_CATCH_BEGIN
{
    THROW_IF_NULL_OBJECT(sampleStorage);
    THROW_IF_NULL_STRING(sampleStorageFileName);
    openpgl::SampleDataStorage::storeSampleDataStorageToFile((openpgl::SampleDataStorage *)sampleStorage, sampleStorageFileName);
    return true;
}
OPENPGL_CATCH_END(false)

extern "C" OPENPGL_DLLEXPORT void pglSampleStorageAddSample(PGLSampleStorage sampleStorage, const PGLSampleData &sample)
{
    auto *gSampleStorage = (openpgl::SampleDataStorage *)sampleStorage;
    openpgl::SampleData opglSample = /**(openpgl::SampleData*)*/ sample;
    gSampleStorage->addSample(opglSample);
}

extern "C" OPENPGL_DLLEXPORT void pglSampleStorageAddSamples(PGLSampleStorage sampleStorage, const PGLSampleData *samples, size_t numSamples)
{
    auto *gSampleStorage = (openpgl::SampleDataStorage *)sampleStorage;

    openpgl::SampleData *opglSamples = (openpgl::SampleData *)samples;
    gSampleStorage->addSamples(opglSamples, numSamples);
}

extern "C" OPENPGL_DLLEXPORT void pglSampleStorageAddZeroValueSample(PGLSampleStorage sampleStorage, const PGLZeroValueSampleData &sample)
{
    auto *gSampleStorage = (openpgl::SampleDataStorage *)sampleStorage;
    openpgl::ZeroValueSampleData opglSample = /**(openpgl::SampleData*)*/ sample;
    gSampleStorage->addZeroValueSample(opglSample);
}

extern "C" OPENPGL_DLLEXPORT void pglSampleStorageAddZeroValueSamples(PGLSampleStorage sampleStorage, const PGLZeroValueSampleData *samples, size_t numSamples)
{
    auto *gSampleStorage = (openpgl::SampleDataStorage *)sampleStorage;

    openpgl::ZeroValueSampleData *opglSamples = (openpgl::ZeroValueSampleData *)samples;
    gSampleStorage->addZeroValueSamples(opglSamples, numSamples);
}

extern "C" OPENPGL_DLLEXPORT void pglSampleStorageReserve(PGLSampleStorage sampleStorage, const size_t sizeSurface, const size_t sizeVolume)
{
    auto *gSampleStorage = (openpgl::SampleDataStorage *)sampleStorage;
    gSampleStorage->reserveSurface(sizeSurface);
    gSampleStorage->reserveVolume(sizeVolume);

    gSampleStorage->reserveInvalidSurface(sizeSurface);
    gSampleStorage->reserveInvalidVolume(sizeVolume);
}

extern "C" OPENPGL_DLLEXPORT void pglSampleStorageClear(PGLSampleStorage sampleStorage)
{
    auto *gSampleStorage = (openpgl::SampleDataStorage *)sampleStorage;
    gSampleStorage->clearSurface();
    gSampleStorage->clearVolume();

    gSampleStorage->clearInvalidSurface();
    gSampleStorage->clearInvalidVolume();
}

extern "C" OPENPGL_DLLEXPORT void pglSampleStorageClearSurface(PGLSampleStorage sampleStorage)
{
    auto *gSampleStorage = (openpgl::SampleDataStorage *)sampleStorage;
    gSampleStorage->clearSurface();
}

extern "C" OPENPGL_DLLEXPORT void pglSampleStorageClearVolume(PGLSampleStorage sampleStorage)
{
    auto *gSampleStorage = (openpgl::SampleDataStorage *)sampleStorage;
    gSampleStorage->clearVolume();
}

extern "C" OPENPGL_DLLEXPORT size_t pglSampleStorageGetSizeSurface(PGLSampleStorage sampleStorage)
{
    auto *gSampleStorage = (openpgl::SampleDataStorage *)sampleStorage;
    return gSampleStorage->sizeSurface();
}

extern "C" OPENPGL_DLLEXPORT size_t pglSampleStorageGetSizeVolume(PGLSampleStorage sampleStorage)
{
    auto *gSampleStorage = (openpgl::SampleDataStorage *)sampleStorage;
    return gSampleStorage->sizeVolume();
}

extern "C" OPENPGL_DLLEXPORT size_t pglSampleStorageGetSizeZeroValueSurface(PGLSampleStorage sampleStorage)
{
    auto *gSampleStorage = (openpgl::SampleDataStorage *)sampleStorage;
    return gSampleStorage->sizeInvalidSurface();
}

extern "C" OPENPGL_DLLEXPORT size_t pglSampleStorageGetSizeZeroValueVolume(PGLSampleStorage sampleStorage)
{
    auto *gSampleStorage = (openpgl::SampleDataStorage *)sampleStorage;
    return gSampleStorage->sizeInvalidVolume();
}

extern "C" OPENPGL_DLLEXPORT PGLSampleData pglSampleStorageGetSampleSurface(PGLSampleStorage sampleStorage, const int idx)
{
    auto *gSampleStorage = (openpgl::SampleDataStorage *)sampleStorage;
    return gSampleStorage->getSampleSurface(idx);
}

extern "C" OPENPGL_DLLEXPORT PGLSampleData pglSampleStorageGetSampleVolume(PGLSampleStorage sampleStorage, const int idx)
{
    auto *gSampleStorage = (openpgl::SampleDataStorage *)sampleStorage;
    return gSampleStorage->getSampleVolume(idx);
}

extern "C" OPENPGL_DLLEXPORT PGLZeroValueSampleData pglSampleStorageGetZeroValueSampleSurface(PGLSampleStorage sampleStorage, const int idx)
{
    auto *gSampleStorage = (openpgl::SampleDataStorage *)sampleStorage;
    return gSampleStorage->getZeroValueSampleSurface(idx);
}

extern "C" OPENPGL_DLLEXPORT PGLZeroValueSampleData pglSampleStorageGetZeroValueSampleVolume(PGLSampleStorage sampleStorage, const int idx)
{
    auto *gSampleStorage = (openpgl::SampleDataStorage *)sampleStorage;
    return gSampleStorage->getZeroValueSampleVolume(idx);
}

extern "C" OPENPGL_DLLEXPORT bool pglSampleStorageValidate(PGLSampleStorage sampleStorage)
{
    auto *gSampleStorage = (openpgl::SampleDataStorage *)sampleStorage;
    return gSampleStorage->validate();
}

extern "C" OPENPGL_DLLEXPORT void pglSampleStorageMerge(PGLSampleStorage sampleStorageA, PGLSampleStorage sampleStorageB)
{
    auto *gSampleStorageA = (openpgl::SampleDataStorage *)sampleStorageA;
    auto *gSampleStorageB = (openpgl::SampleDataStorage *)sampleStorageB;
    return gSampleStorageA->merge(*gSampleStorageB);
}

extern "C" OPENPGL_DLLEXPORT bool pglSampleStorageCompare(PGLSampleStorage sampleStorageA, PGLSampleStorage sampleStorageB)
{
    auto *gSampleStorageA = (openpgl::SampleDataStorage *)sampleStorageA;
    auto *gSampleStorageB = (openpgl::SampleDataStorage *)sampleStorageB;
    return gSampleStorageA->operator==(*gSampleStorageB);
}

///////////////////////////////////////////////////////////////////////////////
// PathSegmentStorage /////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

extern "C" OPENPGL_DLLEXPORT PGLPathSegmentStorage pglNewPathSegmentStorage() OPENPGL_CATCH_BEGIN
{
    openpgl::PathSegmentDataStorage *pathSegmentStorage = new openpgl::PathSegmentDataStorage();
    return (PGLPathSegmentStorage)pathSegmentStorage;
}
OPENPGL_CATCH_END(nullptr)

extern "C" OPENPGL_DLLEXPORT void pglReleasePathSegmentStorage(PGLPathSegmentStorage pathSegmentStorage) OPENPGL_CATCH_BEGIN
{
    auto *gPathSegmentStorage = (openpgl::PathSegmentDataStorage *)pathSegmentStorage;
    delete gPathSegmentStorage;
}
OPENPGL_CATCH_END_VOID

extern "C" OPENPGL_DLLEXPORT void pglPathSegmentStorageReserve(PGLPathSegmentStorage pathSegmentStorage, size_t size)
{
    auto *gPathSegmentStorage = (openpgl::PathSegmentDataStorage *)pathSegmentStorage;
    gPathSegmentStorage->reserve(size);
}

extern "C" OPENPGL_DLLEXPORT void pglPathSegmentStorageClear(PGLPathSegmentStorage pathSegmentStorage)
{
    auto *gPathSegmentStorage = (openpgl::PathSegmentDataStorage *)pathSegmentStorage;
    gPathSegmentStorage->clear();
}

extern "C" OPENPGL_DLLEXPORT void pglPathSegmentSetMaxDistance(PGLPathSegmentStorage pathSegmentStorage, float maxDistance)
{
    auto *gPathSegmentStorage = (openpgl::PathSegmentDataStorage *)pathSegmentStorage;
    gPathSegmentStorage->setMaxDistance(maxDistance);
}

extern "C" OPENPGL_DLLEXPORT float pglPathSegmentGetMaxDistance(PGLPathSegmentStorage pathSegmentStorage)
{
    auto *gPathSegmentStorage = (openpgl::PathSegmentDataStorage *)pathSegmentStorage;
    return gPathSegmentStorage->getMaxDistance();
}

extern "C" OPENPGL_DLLEXPORT int pglPathSegmentGetNumSegments(PGLPathSegmentStorage pathSegmentStorage)
{
    auto *gPathSegmentStorage = (openpgl::PathSegmentDataStorage *)pathSegmentStorage;
    return gPathSegmentStorage->getNumSegments();
}

extern "C" OPENPGL_DLLEXPORT int pglPathSegmentGetNumSamples(PGLPathSegmentStorage pathSegmentStorage)
{
    auto *gPathSegmentStorage = (openpgl::PathSegmentDataStorage *)pathSegmentStorage;
    return gPathSegmentStorage->getNumSamples();
}

extern "C" OPENPGL_DLLEXPORT size_t pglPathSegmentStoragePrepareSamples(PGLPathSegmentStorage pathSegmentStorage, const bool useNEEMiWeights, const bool guideDirectLight,
                                                                        const bool rrAffectsDirectContribution)
{
    auto *gPathSegmentStorage = (openpgl::PathSegmentDataStorage *)pathSegmentStorage;
    return gPathSegmentStorage->prepareSamples(useNEEMiWeights, guideDirectLight, rrAffectsDirectContribution);
}

extern "C" OPENPGL_DLLEXPORT pgl_vec3f pglPathSegmentStorageCalculatePixelEstimate(PGLPathSegmentStorage pathSegmentStorage, const bool rrAffectsDirectContribution)
{
    auto *gPathSegmentStorage = (openpgl::PathSegmentDataStorage *)pathSegmentStorage;
    return gPathSegmentStorage->calculatePixelEstimate(rrAffectsDirectContribution);
}

extern "C" OPENPGL_DLLEXPORT const PGLSampleData *pglPathSegmentStorageGetSamples(PGLPathSegmentStorage pathSegmentStorage, size_t &nSamples)
{
    auto *gPathSegmentStorage = (openpgl::PathSegmentDataStorage *)pathSegmentStorage;
    const openpgl::SampleData *opglSamples = gPathSegmentStorage->getSamples();
    nSamples = gPathSegmentStorage->getNumSamples();
    return (PGLSampleData *)opglSamples;
}

extern "C" OPENPGL_DLLEXPORT int pglPathSegmentGetNumZeroValueSamples(PGLPathSegmentStorage pathSegmentStorage)
{
    auto *gPathSegmentStorage = (openpgl::PathSegmentDataStorage *)pathSegmentStorage;
    return gPathSegmentStorage->getNumZeroValueSamples();
}

extern "C" OPENPGL_DLLEXPORT const PGLZeroValueSampleData *pglPathSegmentStorageGetZeroValueSamples(PGLPathSegmentStorage pathSegmentStorage, size_t &nSamples)
{
    auto *gPathSegmentStorage = (openpgl::PathSegmentDataStorage *)pathSegmentStorage;
    const openpgl::ZeroValueSampleData *opglSamples = gPathSegmentStorage->getZeroValueSamples();
    nSamples = gPathSegmentStorage->getNumZeroValueSamples();
    return (PGLZeroValueSampleData *)opglSamples;
}

extern "C" OPENPGL_DLLEXPORT void pglPathSegmentStorageAddSample(PGLPathSegmentStorage pathSegmentStorage, PGLSampleData sample)
{
    auto *gPathSegmentStorage = (openpgl::PathSegmentDataStorage *)pathSegmentStorage;
    gPathSegmentStorage->addSample(sample);
}

extern "C" OPENPGL_DLLEXPORT PGLPathSegmentData *pglPathSegmentStorageNextSegment(PGLPathSegmentStorage pathSegmentStorage)
{
    auto *gPathSegmentStorage = (openpgl::PathSegmentDataStorage *)pathSegmentStorage;
    return (PGLPathSegmentData *)gPathSegmentStorage->next();
}

extern "C" OPENPGL_DLLEXPORT void pglPathSegmentStorageAddSegment(PGLPathSegmentStorage pathSegmentStorage, PGLPathSegmentData segment)
{
    auto *gPathSegmentStorage = (openpgl::PathSegmentDataStorage *)pathSegmentStorage;
    gPathSegmentStorage->addSegment(segment);
}

extern "C" OPENPGL_DLLEXPORT bool pglPathSegmentStorageValidateSamples(PGLPathSegmentStorage pathSegmentStorage)
{
    auto *gPathSegmentStorage = (openpgl::PathSegmentDataStorage *)pathSegmentStorage;
    return gPathSegmentStorage->validateSamples();
}

extern "C" OPENPGL_DLLEXPORT bool pglPathSegmentStorageValidateSegments(PGLPathSegmentStorage pathSegmentStorage)
{
    auto *gPathSegmentStorage = (openpgl::PathSegmentDataStorage *)pathSegmentStorage;
    return gPathSegmentStorage->validateSegments();
}

extern "C" OPENPGL_DLLEXPORT void pglPathSegmentStoragePropagateSamples(PGLPathSegmentStorage pathSegmentStorage, PGLSampleStorage sampleStorage, bool guideDirectLight,
                                                                        bool useNEEMiWeights, bool rrAffectsDirectContribution)
{
    auto *gPathSegmentStorage = (openpgl::PathSegmentDataStorage *)pathSegmentStorage;
    auto *gSampleStorage = (openpgl::SampleDataStorage *)sampleStorage;
    gPathSegmentStorage->propagateSamples(gSampleStorage, guideDirectLight, useNEEMiWeights, rrAffectsDirectContribution);
}

///////////////////////////////////////////////////////////////////////////////
// Distribution ///// /////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

extern "C" OPENPGL_DLLEXPORT void pglReleaseSurfaceSamplingDistribution(PGLSurfaceSamplingDistribution surfaceSamplingDistribution) OPENPGL_CATCH_BEGIN
{
    ISurfaceSamplingDistribution *gSurfaceSamplingDistribution = (ISurfaceSamplingDistribution *)surfaceSamplingDistribution;
    delete gSurfaceSamplingDistribution;
}
OPENPGL_CATCH_END_VOID

extern "C" OPENPGL_DLLEXPORT void pglSurfaceSamplingDistributionApplyCosineProduct(PGLSurfaceSamplingDistribution surfaceSamplingDistribution, pgl_vec3f normal)
{
    ISurfaceSamplingDistribution *gSurfaceSamplingDistribution = (ISurfaceSamplingDistribution *)surfaceSamplingDistribution;
    openpgl::Vector3 opglNormal(normal.x, normal.y, normal.z);
    gSurfaceSamplingDistribution->applyCosineProduct(opglNormal);
}

extern "C" OPENPGL_DLLEXPORT bool pglSurfaceSamplingDistributionSupportsApplyCosineProduct(PGLSurfaceSamplingDistribution surfaceSamplingDistribution)
{
    ISurfaceSamplingDistribution *gSurfaceSamplingDistribution = (ISurfaceSamplingDistribution *)surfaceSamplingDistribution;
    return gSurfaceSamplingDistribution->supportsApplyCosineProduct();
}

extern "C" OPENPGL_DLLEXPORT pgl_vec3f pglSurfaceSamplingDistributionSample(PGLSurfaceSamplingDistribution surfaceSamplingDistribution, pgl_point2f sample)
{
    ISurfaceSamplingDistribution *gSurfaceSamplingDistribution = (ISurfaceSamplingDistribution *)surfaceSamplingDistribution;
    openpgl::Point2 opglSample(sample.x, sample.y);
    openpgl::Vector3 opglDirection = gSurfaceSamplingDistribution->sample(opglSample);
    pgl_vec3f pglDirection;
    pglVec3f(pglDirection, opglDirection.x, opglDirection.y, opglDirection.z);
    return pglDirection;
}

extern "C" OPENPGL_DLLEXPORT float pglSurfaceSamplingDistributionPDF(PGLSurfaceSamplingDistribution surfaceSamplingDistribution, pgl_vec3f direction)
{
    ISurfaceSamplingDistribution *gSurfaceSamplingDistribution = (ISurfaceSamplingDistribution *)surfaceSamplingDistribution;
    return gSurfaceSamplingDistribution->pdf(openpgl::Vector3(direction.x, direction.y, direction.z));
}

extern "C" OPENPGL_DLLEXPORT float pglSurfaceSamplingDistributionSamplePDF(PGLSurfaceSamplingDistribution surfaceSamplingDistribution, pgl_point2f sample, pgl_vec3f &direction)
{
    ISurfaceSamplingDistribution *gSurfaceSamplingDistribution = (ISurfaceSamplingDistribution *)surfaceSamplingDistribution;
    openpgl::Vector3 dir;
    float pdf = gSurfaceSamplingDistribution->samplePdf({sample.x, sample.y}, dir);
    direction = {dir.x, dir.y, dir.z};
    return pdf;
}

extern "C" OPENPGL_DLLEXPORT float pglSurfaceSamplingDistributionIncomingRadiancePDF(PGLSurfaceSamplingDistribution surfaceSamplingDistribution, pgl_vec3f direction)
{
    ISurfaceSamplingDistribution *gSurfaceSamplingDistribution = (ISurfaceSamplingDistribution *)surfaceSamplingDistribution;
    return gSurfaceSamplingDistribution->pdfLi(openpgl::Vector3(direction.x, direction.y, direction.z));
}

extern "C" OPENPGL_DLLEXPORT uint32_t pglSurfaceSamplingDistributionGetId(PGLSurfaceSamplingDistribution surfaceSamplingDistribution)
{
    ISurfaceSamplingDistribution *gSurfaceSamplingDistribution = (ISurfaceSamplingDistribution *)surfaceSamplingDistribution;
    return gSurfaceSamplingDistribution->getId();
}

extern "C" OPENPGL_DLLEXPORT bool pglSurfaceSamplingDistributionValidate(PGLSurfaceSamplingDistribution surfaceSamplingDistribution)
{
    ISurfaceSamplingDistribution *gSurfaceSamplingDistribution = (ISurfaceSamplingDistribution *)surfaceSamplingDistribution;
    return gSurfaceSamplingDistribution->validate();
}

extern "C" OPENPGL_DLLEXPORT void pglSurfaceSamplingDistributionClear(PGLSurfaceSamplingDistribution surfaceSamplingDistribution)
{
    ISurfaceSamplingDistribution *gSurfaceSamplingDistribution = (ISurfaceSamplingDistribution *)surfaceSamplingDistribution;
    gSurfaceSamplingDistribution->clear();
}

extern "C" OPENPGL_DLLEXPORT PGLRegion pglSurfaceSamplingGetRegion(PGLSurfaceSamplingDistribution surfaceSamplingDistribution)
{
    ISurfaceSamplingDistribution *gSurfaceSamplingDistribution = (ISurfaceSamplingDistribution *)surfaceSamplingDistribution;
    const IRegion *gRegion = gSurfaceSamplingDistribution->getRegion();
    return (PGLRegion)gRegion;
}

#ifdef OPENPGL_RADIANCE_CACHES
extern "C" OPENPGL_DLLEXPORT pgl_vec3f pglSurfaceSamplingDistributionIncomingRadiance(PGLSurfaceSamplingDistribution surfaceSamplingDistribution, pgl_vec3f direction,
                                                                                      const bool directLightMIS)
{
    ISurfaceSamplingDistribution *gSurfaceSamplingDistribution = (ISurfaceSamplingDistribution *)surfaceSamplingDistribution;
    openpgl::Vector3 incomingRadiance = gSurfaceSamplingDistribution->incomingRadiance(openpgl::Vector3(direction.x, direction.y, direction.z), directLightMIS);

    pgl_vec3f pglIncomingRadiance;
    pglVec3f(pglIncomingRadiance, incomingRadiance.x, incomingRadiance.y, incomingRadiance.z);
    return pglIncomingRadiance;
}

extern "C" OPENPGL_DLLEXPORT pgl_vec3f pglSurfaceSamplingDistributionOutgoingRadiance(PGLSurfaceSamplingDistribution surfaceSamplingDistribution, pgl_vec3f direction)
{
    ISurfaceSamplingDistribution *gSurfaceSamplingDistribution = (ISurfaceSamplingDistribution *)surfaceSamplingDistribution;
    openpgl::Vector3 outgoingRadiance = gSurfaceSamplingDistribution->outgoingRadiance(openpgl::Vector3(direction.x, direction.y, direction.z));

    pgl_vec3f pglOutgoingRadiance;
    pglVec3f(pglOutgoingRadiance, outgoingRadiance.x, outgoingRadiance.y, outgoingRadiance.z);
    return pglOutgoingRadiance;
}

extern "C" OPENPGL_DLLEXPORT pgl_vec3f pglSurfaceSamplingDistributionIrradiance(PGLSurfaceSamplingDistribution surfaceSamplingDistribution, pgl_vec3f normal,
                                                                                const bool directLightMIS)
{
    ISurfaceSamplingDistribution *gSurfaceSamplingDistribution = (ISurfaceSamplingDistribution *)surfaceSamplingDistribution;
    openpgl::Vector3 irradiance = gSurfaceSamplingDistribution->irradiance(openpgl::Vector3(normal.x, normal.y, normal.z), directLightMIS);

    pgl_vec3f pglIrradiance;
    pglVec3f(pglIrradiance, irradiance.x, irradiance.y, irradiance.z);
    return pglIrradiance;
}

#endif

extern "C" OPENPGL_DLLEXPORT void pglReleaseVolumeSamplingDistribution(PGLVolumeSamplingDistribution volumeSamplingDistribution) OPENPGL_CATCH_BEGIN
{
    IVolumeSamplingDistribution *gVolumeSamplingDistribution = (IVolumeSamplingDistribution *)volumeSamplingDistribution;
    delete gVolumeSamplingDistribution;
}
OPENPGL_CATCH_END_VOID

extern "C" OPENPGL_DLLEXPORT pgl_vec3f pglVolumeSamplingDistributionSample(PGLVolumeSamplingDistribution volumeSamplingDistribution, pgl_point2f sample)
{
    IVolumeSamplingDistribution *gVolumeSamplingDistribution = (IVolumeSamplingDistribution *)volumeSamplingDistribution;
    openpgl::Point2 opglSample(sample.x, sample.y);
    openpgl::Vector3 opglDirection = gVolumeSamplingDistribution->sample(opglSample);
    pgl_vec3f pglDirection;
    pglVec3f(pglDirection, opglDirection.x, opglDirection.y, opglDirection.z);
    return pglDirection;
}

extern "C" OPENPGL_DLLEXPORT float pglVolumeSamplingDistributionPDF(PGLVolumeSamplingDistribution volumeSamplingDistribution, pgl_vec3f direction)
{
    IVolumeSamplingDistribution *gVolumeSamplingDistribution = (IVolumeSamplingDistribution *)volumeSamplingDistribution;
    return gVolumeSamplingDistribution->pdf(openpgl::Vector3(direction.x, direction.y, direction.z));
}

extern "C" OPENPGL_DLLEXPORT float pglVolumeSamplingDistributionSamplePDF(PGLVolumeSamplingDistribution volumeSamplingDistribution, pgl_point2f sample, pgl_vec3f &direction)
{
    IVolumeSamplingDistribution *gVolumeSamplingDistribution = (IVolumeSamplingDistribution *)volumeSamplingDistribution;
    openpgl::Vector3 dir;
    float pdf = gVolumeSamplingDistribution->samplePdf({sample.x, sample.y}, dir);
    direction = {dir.x, dir.y, dir.z};
    return pdf;
}

extern "C" OPENPGL_DLLEXPORT float pglVolumeSamplingDistributionIncomingRadiancePDF(PGLVolumeSamplingDistribution volumeSamplingDistribution, pgl_vec3f direction)
{
    IVolumeSamplingDistribution *gVolumeSamplingDistribution = (IVolumeSamplingDistribution *)volumeSamplingDistribution;
    return gVolumeSamplingDistribution->pdfLi(openpgl::Vector3(direction.x, direction.y, direction.z));
}

extern "C" OPENPGL_DLLEXPORT uint32_t pglVolumeSamplingDistributionGetId(PGLVolumeSamplingDistribution volumeSamplingDistribution)
{
    IVolumeSamplingDistribution *gVolumeSamplingDistribution = (IVolumeSamplingDistribution *)volumeSamplingDistribution;
    return gVolumeSamplingDistribution->getId();
}

extern "C" OPENPGL_DLLEXPORT bool pglVolumeSamplingDistributionValidate(PGLVolumeSamplingDistribution volumeSamplingDistribution)
{
    IVolumeSamplingDistribution *gVolumeSamplingDistribution = (IVolumeSamplingDistribution *)volumeSamplingDistribution;
    return gVolumeSamplingDistribution->validate();
}

extern "C" OPENPGL_DLLEXPORT void pglVolumeSamplingDistributionClear(PGLVolumeSamplingDistribution volumeSamplingDistribution)
{
    IVolumeSamplingDistribution *gVolumeSamplingDistribution = (IVolumeSamplingDistribution *)volumeSamplingDistribution;
    gVolumeSamplingDistribution->clear();
}

extern "C" OPENPGL_DLLEXPORT void pglVolumeSamplingDistributionApplySingleLobeHenyeyGreensteinProduct(PGLVolumeSamplingDistribution volumeSamplingDistribution, pgl_vec3f dir,
                                                                                                      const float meanCosine)
{
    IVolumeSamplingDistribution *gVolumeSamplingDistribution = (IVolumeSamplingDistribution *)volumeSamplingDistribution;
    openpgl::Vector3 opglDir(dir.x, dir.y, dir.z);
    gVolumeSamplingDistribution->applySingleLobeHenyeyGreensteinProduct(opglDir, meanCosine);
}

extern "C" OPENPGL_DLLEXPORT bool pglVolumeSamplingDistributionSupportsApplySingleLobeHenyeyGreensteinProduct(PGLVolumeSamplingDistribution volumeSamplingDistribution)
{
    IVolumeSamplingDistribution *gVolumeSamplingDistribution = (IVolumeSamplingDistribution *)volumeSamplingDistribution;
    return gVolumeSamplingDistribution->supportsApplySingleLobeHenyeyGreensteinProduct();
}

#ifdef OPENPGL_RADIANCE_CACHES
extern "C" OPENPGL_DLLEXPORT pgl_vec3f pglVolumeSamplingDistributionIncomingRadiance(PGLVolumeSamplingDistribution volumeSamplingDistribution, pgl_vec3f direction,
                                                                                     const bool directLightMIS)
{
    IVolumeSamplingDistribution *gVolumeSamplingDistribution = (IVolumeSamplingDistribution *)volumeSamplingDistribution;
    openpgl::Vector3 incomingRadiance = gVolumeSamplingDistribution->incomingRadiance(openpgl::Vector3(direction.x, direction.y, direction.z), directLightMIS);

    pgl_vec3f pglIncomingRadiance;
    pglVec3f(pglIncomingRadiance, incomingRadiance.x, incomingRadiance.y, incomingRadiance.z);
    return pglIncomingRadiance;
}

extern "C" OPENPGL_DLLEXPORT pgl_vec3f pglVolumeSamplingDistributionOutgoingRadiance(PGLVolumeSamplingDistribution volumeSamplingDistribution, pgl_vec3f direction)
{
    IVolumeSamplingDistribution *gVolumeSamplingDistribution = (IVolumeSamplingDistribution *)volumeSamplingDistribution;
    openpgl::Vector3 outgoingRadiance = gVolumeSamplingDistribution->outgoingRadiance(openpgl::Vector3(direction.x, direction.y, direction.z));

    pgl_vec3f pglOutgoingRadiance;
    pglVec3f(pglOutgoingRadiance, outgoingRadiance.x, outgoingRadiance.y, outgoingRadiance.z);
    return pglOutgoingRadiance;
}

extern "C" OPENPGL_DLLEXPORT pgl_vec3f pglVolumeSamplingDistributionInscatteredRadiance(PGLVolumeSamplingDistribution volumeSamplingDistribution, pgl_vec3f direction, float g,
                                                                                        const bool directLightMIS)
{
    IVolumeSamplingDistribution *gVolumeSamplingDistribution = (IVolumeSamplingDistribution *)volumeSamplingDistribution;
    openpgl::Vector3 inscatteredRadiance = gVolumeSamplingDistribution->inScatteredRadiance(openpgl::Vector3(direction.x, direction.y, direction.z), g, directLightMIS);

    pgl_vec3f pglInscatteredRadiance;
    pglVec3f(pglInscatteredRadiance, inscatteredRadiance.x, inscatteredRadiance.y, inscatteredRadiance.z);
    return pglInscatteredRadiance;
}

extern "C" OPENPGL_DLLEXPORT pgl_vec3f pglVolumeSamplingDistributionFluence(PGLVolumeSamplingDistribution volumeSamplingDistribution, const bool directLightMIS)
{
    IVolumeSamplingDistribution *gVolumeSamplingDistribution = (IVolumeSamplingDistribution *)volumeSamplingDistribution;
    openpgl::Vector3 fluence = gVolumeSamplingDistribution->fluence(directLightMIS);

    pgl_vec3f pglFluence;
    pglVec3f(pglFluence, fluence.x, fluence.y, fluence.z);
    return pglFluence;
}

#endif

extern "C" OPENPGL_DLLEXPORT void pglFieldArgumentsSetDefaults(PGLFieldArguments &fieldArguments, const PGL_SPATIAL_STRUCTURE_TYPE spatialType,
                                                               const PGL_DIRECTIONAL_DISTRIBUTION_TYPE directionalType, const bool deterministic, const size_t maxSamplesPerLeaf)
{
    switch (spatialType)
    {
        default:
        case PGL_SPATIAL_STRUCTURE_TYPE::PGL_SPATIAL_STRUCTURE_KDTREE:
            fieldArguments.spatialStructureType = PGL_SPATIAL_STRUCTURE_KDTREE;
            fieldArguments.spatialSturctureArguments = new PGLKDTreeArguments();
            reinterpret_cast<PGLKDTreeArguments *>(fieldArguments.spatialSturctureArguments)->maxSamples = maxSamplesPerLeaf;
            break;
    }

    fieldArguments.deterministic = deterministic;
    fieldArguments.debugArguments.fitRegions = true;
    fieldArguments.debugArguments.dumpUpdateDistributionData = false;
    switch (directionalType)
    {
        default:
        case PGL_DIRECTIONAL_DISTRIBUTION_TYPE::PGL_DIRECTIONAL_DISTRIBUTION_PARALLAX_AWARE_VMM:
            fieldArguments.directionalDistributionType = PGL_DIRECTIONAL_DISTRIBUTION_PARALLAX_AWARE_VMM;
            fieldArguments.directionalDistributionArguments = new PGLVMMFactoryArguments(maxSamplesPerLeaf);
            break;
        case PGL_DIRECTIONAL_DISTRIBUTION_TYPE::PGL_DIRECTIONAL_DISTRIBUTION_PARALLAX_AWARE_VMM_V2:
            fieldArguments.directionalDistributionType = PGL_DIRECTIONAL_DISTRIBUTION_PARALLAX_AWARE_VMM_V2;
            fieldArguments.directionalDistributionArguments = new PGLVMMFactoryArguments(maxSamplesPerLeaf);
            break;
        case PGL_DIRECTIONAL_DISTRIBUTION_TYPE::PGL_DIRECTIONAL_DISTRIBUTION_QUADTREE:
            fieldArguments.directionalDistributionType = PGL_DIRECTIONAL_DISTRIBUTION_QUADTREE;
            fieldArguments.directionalDistributionArguments = new PGLDQTFactoryArguments();
            break;
        case PGL_DIRECTIONAL_DISTRIBUTION_TYPE::PGL_DIRECTIONAL_DISTRIBUTION_VMM:
            fieldArguments.directionalDistributionType = PGL_DIRECTIONAL_DISTRIBUTION_VMM;
            fieldArguments.directionalDistributionArguments = new PGLVMMFactoryArguments(maxSamplesPerLeaf);
            break;
    }
}

///////////////////////////////////////////////////////////////////////////////
// FieldStatistics ////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

extern "C" OPENPGL_DLLEXPORT void pglReleaseFieldStatistics(PGLFieldStatistics fieldStatistics)
{
    auto *gFieldStatistics = (openpgl::FieldStatistics *)fieldStatistics;
    delete gFieldStatistics;
}

extern "C" OPENPGL_DLLEXPORT PGLString pglFieldStatisticsToString(PGLFieldStatistics fieldStatistics)
{
    auto *gFieldStatistics = (openpgl::FieldStatistics *)fieldStatistics;
    std::string str = gFieldStatistics->toString();
    PGLString pglStr;
    pglStr.m_size = str.length() + 1;
    pglStr.m_str = new char[pglStr.m_size];
    strcpy(pglStr.m_str, str.c_str());
    return pglStr;
}

extern "C" OPENPGL_DLLEXPORT PGLString pglFieldStatisticsHeaderCSVString(PGLFieldStatistics fieldStatistics)
{
    auto *gFieldStatistics = (openpgl::FieldStatistics *)fieldStatistics;
    std::string str = gFieldStatistics->headerCSVString();
    PGLString pglStr;
    pglStr.m_size = str.length() + 1;
    pglStr.m_str = new char[pglStr.m_size];
    strcpy(pglStr.m_str, str.c_str());
    return pglStr;
}

extern "C" OPENPGL_DLLEXPORT PGLString pglFieldStatisticsToCSVString(PGLFieldStatistics fieldStatistics)
{
    auto *gFieldStatistics = (openpgl::FieldStatistics *)fieldStatistics;
    std::string str = gFieldStatistics->toCSVString();
    PGLString pglStr;
    pglStr.m_size = str.length() + 1;
    pglStr.m_str = new char[pglStr.m_size];
    strcpy(pglStr.m_str, str.c_str());
    return pglStr;
}

///////////////////////////////////////////////////////////////////////////////
// String /////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

extern "C" OPENPGL_DLLEXPORT void pglReleaseString(PGLString str)
{
    if (str.m_str)
    {
        delete[] str.m_str;
        str.m_str = nullptr;
        str.m_size = 0;
    }
}

#if defined(OPENPGL_IMAGE_SPACE_GUIDING_BUFFER)
///////////////////////////////////////////////////////////////////////////////
// ImageSpaceGuidingBuffer  ///////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

extern "C" OPENPGL_DLLEXPORT PGLImageSpaceGuidingBuffer pglFieldNewImageSpaceGuidingBuffer(const pgl_point2i resolution)
{
    return (PGLImageSpaceGuidingBuffer) new openpgl::ImageSpaceGuidingBuffer(resolution, false);
}

extern "C" OPENPGL_DLLEXPORT PGLImageSpaceGuidingBuffer pglFieldNewImageSpaceGuidingBufferFromFile(const char *fileName)
{
    return (PGLImageSpaceGuidingBuffer) new openpgl::ImageSpaceGuidingBuffer(fileName);
}

extern "C" OPENPGL_DLLEXPORT void pglReleaseImageSpaceGuidingBuffer(PGLImageSpaceGuidingBuffer imageSpaceGuidingBuffer)
{
    auto *gImageSpaceGuidingBuffer = (openpgl::ImageSpaceGuidingBuffer *)imageSpaceGuidingBuffer;
    delete gImageSpaceGuidingBuffer;
}

extern "C" OPENPGL_DLLEXPORT void pglImageSpaceGuidingBufferUpdate(PGLImageSpaceGuidingBuffer imageSpaceGuidingBuffer)
{
    auto *gImageSpaceGuidingBuffer = (openpgl::ImageSpaceGuidingBuffer *)imageSpaceGuidingBuffer;
    gImageSpaceGuidingBuffer->update();
}

extern "C" OPENPGL_DLLEXPORT void pglImageSpaceGuidingBufferAddSample(PGLImageSpaceGuidingBuffer imageSpaceGuidingBuffer, const pgl_point2i pixel, const PGLImageSpaceSample sample)
{
    auto *gImageSpaceGuidingBuffer = (openpgl::ImageSpaceGuidingBuffer *)imageSpaceGuidingBuffer;
    gImageSpaceGuidingBuffer->addSample(pixel, sample);
}

extern "C" OPENPGL_DLLEXPORT void pglImageSpaceGuidingBufferStore(PGLImageSpaceGuidingBuffer imageSpaceGuidingBuffer, const char *fileName)
{
    auto *gImageSpaceGuidingBuffer = (openpgl::ImageSpaceGuidingBuffer *)imageSpaceGuidingBuffer;
    gImageSpaceGuidingBuffer->store(fileName);
}

extern "C" OPENPGL_DLLEXPORT pgl_vec3f pglImageSpaceGuidingBufferGetPixelContributionEstimate(PGLImageSpaceGuidingBuffer imageSpaceGuidingBuffer, const pgl_point2i pixel)
{
    auto *gImageSpaceGuidingBuffer = (openpgl::ImageSpaceGuidingBuffer *)imageSpaceGuidingBuffer;
    return gImageSpaceGuidingBuffer->getContributionEstimate(pixel);
}

extern "C" OPENPGL_DLLEXPORT bool pglImageSpaceGuidingBufferIsReady(PGLImageSpaceGuidingBuffer imageSpaceGuidingBuffer)
{
    auto *gImageSpaceGuidingBuffer = (openpgl::ImageSpaceGuidingBuffer *)imageSpaceGuidingBuffer;
    return gImageSpaceGuidingBuffer->isReady();
}

extern "C" OPENPGL_DLLEXPORT void pglImageSpaceGuidingBufferReset(PGLImageSpaceGuidingBuffer imageSpaceGuidingBuffer)
{
    auto *gImageSpaceGuidingBuffer = (openpgl::ImageSpaceGuidingBuffer *)imageSpaceGuidingBuffer;
    gImageSpaceGuidingBuffer->reset();
}

#endif