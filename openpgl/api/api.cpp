// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../include/openpgl/openpgl.h"
//#include "../openpglTypes.h"

#include "openpgl_common.h"

#include "device/Device.h"
#include "field/ISurfaceVolumeField.h"
#include "directional/ISurfaceSamplingDistribution.h"
#include "directional/IVolumeSamplingDistribution.h"

#include "data/PathSegmentDataStorage.h"
#include "data/PathSegmentData.h"
#include "data/SampleData.h"
#include "data/SampleDataStorage.h"

#include "sampler/Sampler.h"

#include <cstring>

using namespace openpgl;

#define THROW_IF_NULL(obj, name)                         \
  if (obj == nullptr)                                    \
  throw std::runtime_error(std::string("null ") + name + \
                           std::string(" provided to ") + __FUNCTION__)

// convenience macros for repeated use of the above
#define THROW_IF_NULL_OBJECT(obj) THROW_IF_NULL(obj, "handle")
#define THROW_IF_NULL_STRING(str) THROW_IF_NULL(str, "string")

#define OPENPGL_CATCH_BEGIN try {
#define OPENPGL_CATCH_END(a)                                        \
  }                                                                 \
  catch (const std::bad_alloc &)                                    \
  {                                                                 \
    std::cout <<                                                    \
             "Open PGL was unable to allocate memory" << std::endl; \
    return a;                                                       \
  }                                                                 \
  catch (const std::exception &e)                                   \
  {                                                                 \
    std::cout << e.what() << std::endl;              \
    return a;                                                       \
  }                                                                 \
  catch (...)                                                       \
  {                                                                 \
    std::cout <<                                                    \
               "an unrecognized exception was caught" << std::endl; \
    return a;                                                       \
  }

typedef ISurfaceVolumeField IGuidingField;


///////////////////////////////////////////////////////////////////////////////
// Field //////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

#define OPENPGL_FIELD_STRING "OPENPGL_" OPENPGL_VERSION_STRING "_FIELD"

extern "C" OPENPGL_DLLEXPORT PGLDevice pglNewDevice(PGL_DEVICE_TYPE deviceType)OPENPGL_CATCH_BEGIN
{
#ifdef OPENPGL_DEVICE_TYPE_CPU_4
    if (deviceType == PGL_DEVICE_TYPE_CPU_4)
        return (PGLDevice) newDeviceCPU4();
#endif
#ifdef OPENPGL_DEVICE_TYPE_CPU_8
    if (deviceType == PGL_DEVICE_TYPE_CPU_8)
        return (PGLDevice) newDeviceCPU8();
#endif
#ifdef OPENPGL_DEVICE_TYPE_CPU_16
    if (deviceType == PGL_DEVICE_TYPE_CPU_16)
        return (PGLDevice) newDeviceCPU16();
#endif

    throw std::runtime_error("invalid vectorSize parameter!");
}
OPENPGL_CATCH_END(nullptr)

extern "C" OPENPGL_DLLEXPORT void pglReleaseDevice(PGLDevice device)
{
    auto *gDevice = (IDevice *)device;
    delete gDevice;
}

extern "C" OPENPGL_DLLEXPORT PGLField pglDeviceNewField(PGLDevice device, PGLFieldArguments args)OPENPGL_CATCH_BEGIN
{
    THROW_IF_NULL_OBJECT(device);
    auto *gDevice = (IDevice *)device;
    return (PGLField) gDevice->newField(args);
}
OPENPGL_CATCH_END(nullptr)

extern "C" OPENPGL_DLLEXPORT PGLField pglDeviceNewFieldFromFile(PGLDevice device, const char* fieldFileName)OPENPGL_CATCH_BEGIN
{
    THROW_IF_NULL_OBJECT(device);
    THROW_IF_NULL_STRING(fieldFileName);
    auto *gDevice = (IDevice *)device;
    return (PGLField) gDevice->newFieldFromFile(fieldFileName);
}
OPENPGL_CATCH_END(nullptr)

extern "C" OPENPGL_DLLEXPORT void pglReleaseField(PGLField field)OPENPGL_CATCH_BEGIN
{
    auto *gField = (IGuidingField *)field;
    delete gField;
}
OPENPGL_CATCH_END()

extern "C" OPENPGL_DLLEXPORT bool pglFieldStoreToFile(PGLField field, const char* fieldFileName)OPENPGL_CATCH_BEGIN
{
    THROW_IF_NULL_OBJECT(field);
    THROW_IF_NULL_STRING(fieldFileName);
    ((IGuidingField *)field)->storeToFile(fieldFileName);
    return true;
}
OPENPGL_CATCH_END(false)

extern "C" OPENPGL_DLLEXPORT size_t pglFieldGetIteration(PGLField field)OPENPGL_CATCH_BEGIN
{
    auto *gField = (IGuidingField *)field;
    return gField->getIteration();
}
OPENPGL_CATCH_END(0)

extern "C" OPENPGL_DLLEXPORT size_t pglFieldGetTotalSPP(PGLField field)OPENPGL_CATCH_BEGIN
{
    auto *gField = (IGuidingField *)field;
    return gField->getTotalSPP();
}
OPENPGL_CATCH_END(0)

extern "C" OPENPGL_DLLEXPORT void pglFieldSetSceneBounds(PGLField field, pgl_box3f bounds)
{
    auto *gField = (IGuidingField *)field;
    openpgl::BBox sceneBounds;
    sceneBounds.lower = openpgl::Vector3(bounds.lower.x,bounds.lower.y,bounds.lower.z);
    sceneBounds.upper = openpgl::Vector3(bounds.upper.x,bounds.upper.y,bounds.upper.z);
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

/*
extern "C" OPENPGL_DLLEXPORT pgl_box3f pglFieldGetSceneBounds(PGLField field)
{
    auto *gField = (GuidingField *)field;
    openpgl::BBox sceneBounds = gField->getSceneBounds();
    pgl_box3f bounds;
    pglBox3f(bounds, sceneBounds.lower.x, sceneBounds.lower.y, sceneBounds.lower.z, sceneBounds.upper.x, sceneBounds.upper.y, sceneBounds.upper.z);
    return bounds;
    //return gField->getIteration();
}
*/

extern "C" OPENPGL_DLLEXPORT  void pglFieldUpdate(PGLField field, PGLSampleStorage sampleStorage, size_t numPerPixelSamples)OPENPGL_CATCH_BEGIN
{
    auto *gField = (IGuidingField *)field;
    auto *gSampleStorage = (openpgl::SampleDataStorage *)sampleStorage;
    if (gField->getIteration() == 0)
    {
        gField->buildField(gSampleStorage->m_surfaceContainer, gSampleStorage->m_volumeContainer);
    }
    else
    {
        gField->updateField(gSampleStorage->m_surfaceContainer, gSampleStorage->m_volumeContainer);
    }

    gField->addTrainingIteration(numPerPixelSamples);
}
OPENPGL_CATCH_END()

extern "C" OPENPGL_DLLEXPORT  void pglFieldReset(PGLField field)OPENPGL_CATCH_BEGIN
{
    auto *gField = (IGuidingField *)field;
    gField->resetField();
}
OPENPGL_CATCH_END()

/*
extern "C" OPENPGL_DLLEXPORT  PGLRegion pglFieldGetSurfaceRegion(PGLField field, pgl_point3f position, PGLSampler* sampler)OPENPGL_CATCH_BEGIN
{
    const openpgl::Point3 pos(position.x, position.y, position.z);
    SamplerC gSampler(sampler);
    float sample1D = gSampler.next1D();
    auto *gField = (GuidingField *)field;
    const IRegion* gRegion = gField->getSurfaceGuidingRegion(pos, sample1D);
    return (PGLRegion) gRegion;
}
OPENPGL_CATCH_END(nullptr)

extern "C" OPENPGL_DLLEXPORT  PGLRegion pglFieldGetVolumeRegion(PGLField field, pgl_point3f position, PGLSampler* sampler)OPENPGL_CATCH_BEGIN
{
    const openpgl::Point3 pos(position.x, position.y, position.z);
    SamplerC gSampler(sampler);
    float sample1D = gSampler.next1D();
    auto *gField = (GuidingField *)field;
    const IRegion* gRegion = gField->getVolumeGuidingRegion(pos, sample1D);
    return (PGLRegion) gRegion;
}
OPENPGL_CATCH_END(nullptr)
*/

extern "C" OPENPGL_DLLEXPORT  PGLSurfaceSamplingDistribution pglFieldNewSurfaceSamplingDistribution(PGLField field)OPENPGL_CATCH_BEGIN
{
    auto *gField = (IGuidingField *)field;
    ISurfaceSamplingDistribution* surfaceSamplingDistribution = gField->newSurfaceSamplingDistribution();
    return (PGLSurfaceSamplingDistribution) surfaceSamplingDistribution;
}
OPENPGL_CATCH_END(nullptr)

extern "C" OPENPGL_DLLEXPORT  bool pglFieldInitSurfaceSamplingDistribution(PGLField field, PGLSurfaceSamplingDistribution surfaceSamplingDistribution, pgl_point3f position, float* sample1D, const bool useParallaxComp)
{
    const openpgl::Point3 pos(position.x, position.y, position.z);
    auto *gField = (IGuidingField *)field;
    ISurfaceSamplingDistribution* gSurfaceSamplingDistribution = (ISurfaceSamplingDistribution*)surfaceSamplingDistribution;
    return gField->initSurfaceSamplingDistribution(gSurfaceSamplingDistribution, pos, sample1D, useParallaxComp);
}
extern "C" OPENPGL_DLLEXPORT  PGLVolumeSamplingDistribution pglFieldNewVolumeSamplingDistribution(PGLField field)OPENPGL_CATCH_BEGIN
{
    auto *gField = (IGuidingField *)field;
    IVolumeSamplingDistribution* volumeSamplingDistribution = gField->newVolumeSamplingDistribution();
    return (PGLVolumeSamplingDistribution) volumeSamplingDistribution;
}
OPENPGL_CATCH_END(nullptr)

extern "C" OPENPGL_DLLEXPORT  bool pglFieldInitVolumeSamplingDistribution(PGLField field, PGLVolumeSamplingDistribution volumeSamplingDistribution, pgl_point3f position, float* sample1D, const bool useParallaxComp)
{
    const openpgl::Point3 pos(position.x, position.y, position.z);
    auto *gField = (IGuidingField *)field;
    IVolumeSamplingDistribution* gVolumeSamplingDistribution = (IVolumeSamplingDistribution*)volumeSamplingDistribution;
    return gField->initVolumeSamplingDistribution(gVolumeSamplingDistribution, pos, sample1D, useParallaxComp);
}

extern "C" OPENPGL_DLLEXPORT  bool pglFieldValidate(PGLField field)
{
    const auto *gField = (const IGuidingField *)field;
    return gField->validate(true, true);
}

///////////////////////////////////////////////////////////////////////////////
// Region /////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
/*
extern "C" OPENPGL_DLLEXPORT bool pglRegionGetValid(PGLRegion region)
{
    auto *gRegion = (IRegion *)region;
    return gRegion->valid;
}
*/
/*
extern "C" OPENPGL_DLLEXPORT PGLDistribution pglRegionGetDistribution(PGLRegion region, pgl_point3f samplePosition, const bool &useParallaxComp)
{
    const openpgl::Point3 samplePos(samplePosition.x, samplePosition.y, samplePosition.z);
    auto *gRegion = (IRegion *)region;

    GuidingDistribution gDistribution;
    gRegion->getDistribution(gDistribution, samplePos, useParallaxComp);
    return (PGLDistribution)nullptr;
}
*/


///////////////////////////////////////////////////////////////////////////////
// SampleStorage //////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

#define OPENPGL_SAMPLE_STORAGE_STRING "OPENPGL_" OPENPGL_VERSION_STRING "_SAMPLE_STORAGE"

extern "C" OPENPGL_DLLEXPORT PGLSampleStorage pglNewSampleStorage()OPENPGL_CATCH_BEGIN
{
    return (PGLSampleStorage) SampleDataStorage::newSampleDataStorage();
}
OPENPGL_CATCH_END(nullptr)

extern "C" OPENPGL_DLLEXPORT PGLSampleStorage pglNewSampleStorageFromFile(const char* sampleStorageFileName)OPENPGL_CATCH_BEGIN
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

extern "C" OPENPGL_DLLEXPORT bool pglSampleStorageStoreToFile(PGLSampleStorage sampleStorage, const char* sampleStorageFileName)OPENPGL_CATCH_BEGIN
{
    THROW_IF_NULL_OBJECT(sampleStorage);
    THROW_IF_NULL_STRING(sampleStorageFileName);
    openpgl::SampleDataStorage::storeSampleDataStorageToFile((openpgl::SampleDataStorage *)sampleStorage, sampleStorageFileName);
    return true;
}
OPENPGL_CATCH_END(false)

extern "C" OPENPGL_DLLEXPORT void pglSampleStorageAddSample(PGLSampleStorage sampleStorage, PGLSampleData& sample)
{
    auto *gSampleStorage = (openpgl::SampleDataStorage *)sampleStorage;
    openpgl::SampleData opglSample = /**(openpgl::SampleData*)*/sample;
    gSampleStorage->addSample(opglSample);
}

extern "C" OPENPGL_DLLEXPORT void pglSampleStorageAddSamples(PGLSampleStorage sampleStorage, const PGLSampleData* samples, size_t numSamples)
{
    auto *gSampleStorage = (openpgl::SampleDataStorage *)sampleStorage;

    openpgl::SampleData* opglSamples = (openpgl::SampleData*)samples;
    gSampleStorage->addSamples(opglSamples, numSamples);    
}

extern "C" OPENPGL_DLLEXPORT void pglSampleStorageReserve(PGLSampleStorage sampleStorage, const size_t sizeSurface, const size_t sizeVolume)
{
    auto *gSampleStorage = (openpgl::SampleDataStorage *)sampleStorage;
    gSampleStorage->reserveSurface(sizeSurface);
    gSampleStorage->reserveVolume(sizeVolume);
}

extern "C" OPENPGL_DLLEXPORT void pglSampleStorageClear(PGLSampleStorage sampleStorage)
{
    auto *gSampleStorage = (openpgl::SampleDataStorage *)sampleStorage;
    gSampleStorage->clearSurface();
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


///////////////////////////////////////////////////////////////////////////////
// SampleData /////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
/*
extern "C" OPENPGL_DLLEXPORT void pglSampleDataSetPosition(PGLSampleData sampleData, const pgl_point3f pos)
{
    sampleData->position = pos;
}

extern "C" OPENPGL_DLLEXPORT void pglSampleDataSetDirection(PGLSampleData sampleData, const pgl_vec3f direction)
{
    sampleData->direction = direction;
}

extern "C" OPENPGL_DLLEXPORT void pglSampleDataSetDistance(PGLSampleData sampleData, const float distance)
{
    sampleData->distance = distance;
}

extern "C" OPENPGL_DLLEXPORT void pglSampleDataSetPDF(PGLSampleData sampleData, const float pdf)
{
    sampleData->pdf = pdf;
}

extern "C" OPENPGL_DLLEXPORT void pglSampleDataSetWeight(PGLSampleData sampleData, const float weight)
{
    sampleData->weight = weight;
}

extern "C" OPENPGL_DLLEXPORT void pglSampleDataSetFlags(PGLSampleData sampleData, const int flags)
{
    sampleData->flags = flags;
}
*/

///////////////////////////////////////////////////////////////////////////////
// PathSegmentStorage /////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

extern "C" OPENPGL_DLLEXPORT PGLPathSegmentStorage pglNewPathSegmentStorage()OPENPGL_CATCH_BEGIN
{
    openpgl::PathSegmentDataStorage* pathSegmentStorage = new openpgl::PathSegmentDataStorage();
    return (PGLPathSegmentStorage) pathSegmentStorage;
}
OPENPGL_CATCH_END(nullptr)

extern "C" OPENPGL_DLLEXPORT void pglReleasePathSegmentStorage(PGLPathSegmentStorage pathSegmentStorage)OPENPGL_CATCH_BEGIN
{
    auto *gPathSegmentStorage = (openpgl::PathSegmentDataStorage *)pathSegmentStorage;
    delete gPathSegmentStorage;
}
OPENPGL_CATCH_END()

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

extern "C" OPENPGL_DLLEXPORT size_t pglPathSegmentStoragePrepareSamples(PGLPathSegmentStorage pathSegmentStorage, const bool spaltSamples, PGLSampler* sampler,  const bool useNEEMiWeights, const bool guideDirectLight, const bool rrAffectsDirectContribution)
{
    auto *gPathSegmentStorage = (openpgl::PathSegmentDataStorage *)pathSegmentStorage;
    SamplerC gSampler(sampler);
    return gPathSegmentStorage->prepareSamples(spaltSamples, &gSampler, useNEEMiWeights, guideDirectLight, rrAffectsDirectContribution);
}

extern "C" OPENPGL_DLLEXPORT pgl_vec3f pglPathSegmentStorageCalculatePixelEstimate(PGLPathSegmentStorage pathSegmentStorage, const bool rrAffectsDirectContribution)
{
    auto *gPathSegmentStorage = (openpgl::PathSegmentDataStorage *)pathSegmentStorage;
    return gPathSegmentStorage->calculatePixelEstimate(rrAffectsDirectContribution);
}

extern "C" OPENPGL_DLLEXPORT const PGLSampleData* pglPathSegmentStorageGetSamples(PGLPathSegmentStorage pathSegmentStorage, size_t &nSamples)
{
    auto *gPathSegmentStorage = (openpgl::PathSegmentDataStorage *)pathSegmentStorage;
    const openpgl::SampleData* opglSamples = gPathSegmentStorage->getSamples();
    nSamples = gPathSegmentStorage->getNumSamples();
    return (PGLSampleData*)opglSamples;
}


/*
extern "C" OPENPGL_DLLEXPORT void pglPathSegmentStorageAddSegment(PGLPathSegmentStorage pathSegmentStorage, PGLPathSegment pathSegment)
{
    //auto *gPathSegmentStorage = (openpgl::PathSegmentDataStorage *)pathSegmentStorage;
    //gPathSegmentStorage->
}
*/
extern "C" OPENPGL_DLLEXPORT void pglPathSegmentStorageAddSample(PGLPathSegmentStorage pathSegmentStorage, PGLSampleData sample)
{
    auto *gPathSegmentStorage = (openpgl::PathSegmentDataStorage *)pathSegmentStorage;
    gPathSegmentStorage->addSample(sample);
}

extern "C" OPENPGL_DLLEXPORT PGLPathSegmentData* pglPathSegmentStorageNextSegment(PGLPathSegmentStorage pathSegmentStorage)
{
    auto *gPathSegmentStorage = (openpgl::PathSegmentDataStorage *)pathSegmentStorage;
    return (PGLPathSegmentData*)gPathSegmentStorage->next();
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

///////////////////////////////////////////////////////////////////////////////
// Distribution ///// /////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
/*
extern "C" OPENPGL_DLLEXPORT bool pglDistributionIsValid(PGLDistribution distribution)
{
    auto *gDistribution = (GuidingDistribution *)distribution;
    return gDistribution->isValid();
}
*/
/*
extern "C" OPENPGL_DLLEXPORT PGLSurfaceSamplingDistribution pglNewSurfaceSamplingDistribution()OPENPGL_CATCH_BEGIN
{
    GuidedSurfaceSamplingDistribution* gSurfaceSamplingDistribution =  new GuidedSurfaceSamplingDistribution();
    return (PGLSurfaceSamplingDistribution)gSurfaceSamplingDistribution;
}
OPENPGL_CATCH_END(nullptr)
*/

extern "C" OPENPGL_DLLEXPORT void pglReleaseSurfaceSamplingDistribution(PGLSurfaceSamplingDistribution surfaceSamplingDistribution)OPENPGL_CATCH_BEGIN
{
    ISurfaceSamplingDistribution* gSurfaceSamplingDistribution =  (ISurfaceSamplingDistribution*)surfaceSamplingDistribution;
    delete gSurfaceSamplingDistribution;
}
OPENPGL_CATCH_END()
/*
extern "C" OPENPGL_DLLEXPORT void pglSurfaceSamplingDistributionInit(PGLSurfaceSamplingDistribution surfaceSamplingDistribution, PGLRegion region, pgl_point3f samplePosition, bool useParallaxComp)
{
    ISurfaceSamplingDistribution* gSurfaceSamplingDistribution =  (ISurfaceSamplingDistribution*)surfaceSamplingDistribution;
    IRegion *gRegion = (IRegion*)region;
    openpgl::Vector3 opglSamplePosition(samplePosition.x, samplePosition.y, samplePosition.z);
    GuidingDistribution Distribution = gRegion->getDistribution(opglSamplePosition, useParallaxComp);
    gSurfaceSamplingDistribution->init(&Distribution);
}
*/
extern "C" OPENPGL_DLLEXPORT void pglSurfaceSamplingDistributionApplyCosineProduct(PGLSurfaceSamplingDistribution surfaceSamplingDistribution, pgl_vec3f normal)
{
    ISurfaceSamplingDistribution* gSurfaceSamplingDistribution =  (ISurfaceSamplingDistribution*)surfaceSamplingDistribution;
    openpgl::Vector3 opglNormal(normal.x, normal.y, normal.z);
    gSurfaceSamplingDistribution->applyCosineProduct(opglNormal);
}

extern "C" OPENPGL_DLLEXPORT bool pglSurfaceSamplingDistributionSupportsApplyCosineProduct(PGLSurfaceSamplingDistribution surfaceSamplingDistribution)
{
    ISurfaceSamplingDistribution* gSurfaceSamplingDistribution =  (ISurfaceSamplingDistribution*)surfaceSamplingDistribution;
    return gSurfaceSamplingDistribution->supportsApplyCosineProduct();
}

extern "C" OPENPGL_DLLEXPORT pgl_vec3f pglSurfaceSamplingDistributionSample(PGLSurfaceSamplingDistribution surfaceSamplingDistribution, pgl_point2f sample)
{
    ISurfaceSamplingDistribution* gSurfaceSamplingDistribution =  (ISurfaceSamplingDistribution*)surfaceSamplingDistribution;
    openpgl::Point2 opglSample(sample.x, sample.y);
    openpgl::Vector3 opglDirection = gSurfaceSamplingDistribution->sample(opglSample);
    pgl_vec3f pglDirection;
    pglVec3f(pglDirection, opglDirection.x, opglDirection.y, opglDirection.z);
    return pglDirection;
}

extern "C" OPENPGL_DLLEXPORT float pglSurfaceSamplingDistributionPDF(PGLSurfaceSamplingDistribution surfaceSamplingDistribution, pgl_vec3f direction)
{
    ISurfaceSamplingDistribution* gSurfaceSamplingDistribution =  (ISurfaceSamplingDistribution*)surfaceSamplingDistribution;
    return gSurfaceSamplingDistribution->pdf(openpgl::Vector3(direction.x, direction.y, direction.z));
}

extern "C" OPENPGL_DLLEXPORT float pglSurfaceSamplingDistributionSamplePDF(PGLSurfaceSamplingDistribution surfaceSamplingDistribution, pgl_point2f sample, pgl_vec3f &direction)
{
    ISurfaceSamplingDistribution* gSurfaceSamplingDistribution =  (ISurfaceSamplingDistribution*)surfaceSamplingDistribution;
    openpgl::Vector3 dir;
    float pdf = gSurfaceSamplingDistribution->samplePdf({sample.x, sample.y}, dir);
    direction = {dir.x, dir.y, dir.z};
    return pdf;
}

extern "C" OPENPGL_DLLEXPORT bool pglSurfaceSamplingDistributionValidate(PGLSurfaceSamplingDistribution surfaceSamplingDistribution)
{
    ISurfaceSamplingDistribution* gSurfaceSamplingDistribution =  (ISurfaceSamplingDistribution*)surfaceSamplingDistribution;
    return gSurfaceSamplingDistribution->validate();
}

extern "C" OPENPGL_DLLEXPORT void pglSurfaceSamplingDistributionClear(PGLSurfaceSamplingDistribution surfaceSamplingDistribution)
{
    ISurfaceSamplingDistribution* gSurfaceSamplingDistribution =  (ISurfaceSamplingDistribution*)surfaceSamplingDistribution;
    gSurfaceSamplingDistribution->clear();
}

extern "C" OPENPGL_DLLEXPORT PGLRegion pglSurfaceSamplingGetRegion(PGLSurfaceSamplingDistribution surfaceSamplingDistribution)
{
    ISurfaceSamplingDistribution* gSurfaceSamplingDistribution =  (ISurfaceSamplingDistribution*)surfaceSamplingDistribution;
    const IRegion* gRegion = gSurfaceSamplingDistribution->getRegion();
    return (PGLRegion) gRegion;
}

/*
extern "C" OPENPGL_DLLEXPORT PGLVolumeSamplingDistribution pglNewVolumeSamplingDistribution()OPENPGL_CATCH_BEGIN
{
    GuidedVolumeSamplingDistribution* gVolumeSamplingDistribution =  new GuidedVolumeSamplingDistribution();
    return (PGLVolumeSamplingDistribution)gVolumeSamplingDistribution;
}
OPENPGL_CATCH_END(nullptr)
*/
extern "C" OPENPGL_DLLEXPORT void pglReleaseVolumeSamplingDistribution(PGLVolumeSamplingDistribution volumeSamplingDistribution)OPENPGL_CATCH_BEGIN
{
    IVolumeSamplingDistribution* gVolumeSamplingDistribution =  (IVolumeSamplingDistribution*)volumeSamplingDistribution;
    delete gVolumeSamplingDistribution;
}
OPENPGL_CATCH_END()
/*
extern "C" OPENPGL_DLLEXPORT void pglVolumeSamplingDistributionInit(PGLVolumeSamplingDistribution volumeSamplingDistribution, PGLRegion region, pgl_point3f samplePosition, bool useParallaxComp)
{
    IVolumeSamplingDistribution* gVolumeSamplingDistribution =  (IVolumeSamplingDistribution*)volumeSamplingDistribution;
    IRegion *gRegion = (IRegion*)region;
    openpgl::Vector3 opglSamplePosition(samplePosition.x, samplePosition.y, samplePosition.z);
    GuidingDistribution Distribution = gRegion->getDistribution(opglSamplePosition, useParallaxComp);
    gVolumeSamplingDistribution->init(&Distribution);
}
*/
extern "C" OPENPGL_DLLEXPORT pgl_vec3f pglVolumeSamplingDistributionSample(PGLVolumeSamplingDistribution volumeSamplingDistribution, pgl_point2f sample)
{
    IVolumeSamplingDistribution* gVolumeSamplingDistribution =  (IVolumeSamplingDistribution*)volumeSamplingDistribution;
    openpgl::Point2 opglSample(sample.x, sample.y);
    openpgl::Vector3 opglDirection = gVolumeSamplingDistribution->sample(opglSample);
    pgl_vec3f pglDirection;
    pglVec3f(pglDirection, opglDirection.x, opglDirection.y, opglDirection.z);
    return pglDirection;
}

extern "C" OPENPGL_DLLEXPORT float pglVolumeSamplingDistributionPDF(PGLVolumeSamplingDistribution volumeSamplingDistribution, pgl_vec3f direction)
{
    IVolumeSamplingDistribution* gVolumeSamplingDistribution =  (IVolumeSamplingDistribution*)volumeSamplingDistribution;
    return gVolumeSamplingDistribution->pdf(openpgl::Vector3(direction.x, direction.y, direction.z));
}

extern "C" OPENPGL_DLLEXPORT float pglVolumeSamplingDistributionSamplePDF(PGLVolumeSamplingDistribution volumeSamplingDistribution, pgl_point2f sample, pgl_vec3f &direction)
{
    IVolumeSamplingDistribution* gVolumeSamplingDistribution =  (IVolumeSamplingDistribution*)volumeSamplingDistribution;
    openpgl::Vector3 dir;
    float pdf = gVolumeSamplingDistribution->samplePdf({sample.x, sample.y}, dir);
    direction = {dir.x, dir.y, dir.z};
    return pdf;
}

extern "C" OPENPGL_DLLEXPORT bool pglVolumeSamplingDistributionValidate(PGLVolumeSamplingDistribution volumeSamplingDistribution)
{
    IVolumeSamplingDistribution* gVolumeSamplingDistribution =  (IVolumeSamplingDistribution*)volumeSamplingDistribution;
    return gVolumeSamplingDistribution->validate();
}

extern "C" OPENPGL_DLLEXPORT void pglVolumeSamplingDistributionClear(PGLVolumeSamplingDistribution volumeSamplingDistribution)
{
    IVolumeSamplingDistribution* gVolumeSamplingDistribution =  (IVolumeSamplingDistribution*)volumeSamplingDistribution;
    gVolumeSamplingDistribution->clear();
}

extern "C" OPENPGL_DLLEXPORT void pglVolumeSamplingDistributionApplySingleLobeHenyeyGreensteinProduct(PGLVolumeSamplingDistribution volumeSamplingDistribution, pgl_vec3f dir, const float meanCosine)
{
    IVolumeSamplingDistribution* gVolumeSamplingDistribution = (IVolumeSamplingDistribution*)volumeSamplingDistribution;
    openpgl::Vector3 opglDir(dir.x, dir.y, dir.z);
    gVolumeSamplingDistribution->applySingleLobeHenyeyGreensteinProduct(opglDir, meanCosine);
}

extern "C" OPENPGL_DLLEXPORT bool pglVolumeSamplingDistributionSupportsApplySingleLobeHenyeyGreensteinProduct(PGLVolumeSamplingDistribution volumeSamplingDistribution)
{
    IVolumeSamplingDistribution* gVolumeSamplingDistribution = (IVolumeSamplingDistribution*)volumeSamplingDistribution;
    return gVolumeSamplingDistribution->supportsApplySingleLobeHenyeyGreensteinProduct();
}

extern "C" OPENPGL_DLLEXPORT void pglFieldArgumentsSetDefaults(PGLFieldArguments &fieldArguments, const PGL_SPATIAL_STRUCTURE_TYPE spatialType, const PGL_DIRECTIONAL_DISTRIBUTION_TYPE directionalType)
{
    switch (spatialType)
    {
    default:
    case PGL_SPATIAL_STRUCTURE_TYPE::PGL_SPATIAL_STRUCTURE_KDTREE:
        fieldArguments.spatialStructureType = PGL_SPATIAL_STRUCTURE_KDTREE;
        fieldArguments.spatialSturctureArguments = new PGLKDTreeArguments();
        break;
    }


    switch (directionalType)
    {
    default:
    case PGL_DIRECTIONAL_DISTRIBUTION_TYPE::PGL_DIRECTIONAL_DISTRIBUTION_PARALLAX_AWARE_VMM:
        fieldArguments.directionalDistributionType = PGL_DIRECTIONAL_DISTRIBUTION_PARALLAX_AWARE_VMM;
        fieldArguments.directionalDistributionArguments = new PGLVMMFactoryArguments(true);
        fieldArguments.useParallaxCompensation = true;
        break;
    case PGL_DIRECTIONAL_DISTRIBUTION_TYPE::PGL_DIRECTIONAL_DISTRIBUTION_QUADTREE:
        fieldArguments.directionalDistributionType = PGL_DIRECTIONAL_DISTRIBUTION_QUADTREE;
        fieldArguments.directionalDistributionArguments = new PGLDQTFactoryArguments();
        fieldArguments.useParallaxCompensation = false;
        break;
    case PGL_DIRECTIONAL_DISTRIBUTION_TYPE::PGL_DIRECTIONAL_DISTRIBUTION_VMM:
        fieldArguments.directionalDistributionType = PGL_DIRECTIONAL_DISTRIBUTION_PARALLAX_AWARE_VMM;
        fieldArguments.directionalDistributionArguments = new PGLVMMFactoryArguments(false);
        fieldArguments.useParallaxCompensation = false;
        break;
    }


}

