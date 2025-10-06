// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../data/SampleDataStorage.h"
#include "../directional/ISurfaceSamplingDistribution.h"
#include "../directional/IVolumeSamplingDistribution.h"
#include "../include/openpgl/types.h"
#include "../include/openpgl/sdump.h"

namespace openpgl
{
namespace gpu
{
struct FieldData;
struct Device;
}  // namespace gpu

struct FieldStatistics;

struct ISurfaceVolumeField
{
    using SampleContainer = SampleDataStorage::SampleContainer;

    virtual ~ISurfaceVolumeField(){};

    virtual ISurfaceSamplingDistribution *newSurfaceSamplingDistribution() const = 0;

    virtual bool initSurfaceSamplingDistribution(ISurfaceSamplingDistribution *surfaceSamplingDistribution, const Point3 &position, float *sample1D) const = 0;

    virtual IVolumeSamplingDistribution *newVolumeSamplingDistribution() const = 0;

    virtual bool initVolumeSamplingDistribution(IVolumeSamplingDistribution *volumeSamplingDistribution, const Point3 &position, float *sample1D) const = 0;

    virtual void setSceneBounds(const openpgl::BBox &sceneBounds) = 0;

    virtual openpgl::BBox getSceneBounds() const = 0;

    virtual void dumpField(const std::string dumpFileName) const = 0;

    virtual void updateField(SampleContainer &samplesSurface, SampleContainer &samplesVolume) = 0;

    virtual void updateFieldSurface(SampleContainer &samplesSurface) = 0;

    virtual void updateFieldVolume(SampleContainer &samplesVolume) = 0;

    virtual void resetField() = 0;

    virtual PGL_SPATIAL_STRUCTURE_TYPE getSpatialStructureType() const = 0;

    virtual PGL_DIRECTIONAL_DISTRIBUTION_TYPE getDirectionalDistributionType() const = 0;

    virtual size_t getIteration() const = 0;

    virtual void serialize(std::ostream &os) const = 0;

    virtual void deserialize(std::istream &is) = 0;

    virtual bool validate(const bool checkSurface, const bool checkVolume) const = 0;

    virtual void storeToFile(const std::string fieldFileName) const = 0;

    virtual bool operator==(const ISurfaceVolumeField *b) const = 0;

    virtual FieldStatistics *getSurfaceStatistics() const = 0;

    virtual FieldStatistics *getVolumeStatistics() const = 0;

    // Stefan SYCL
    virtual void FillFieldData(openpgl::gpu::FieldData *fieldData, openpgl::gpu::Device *deviceGPU) const = 0;
    virtual void ReleaseFieldData(openpgl::gpu::FieldData *fieldGPU, openpgl::gpu::Device *deviceGPU) const = 0;
    /*
    virtual int GetNumNodes(bool isSurface = true) const = 0;
    virtual void* GetNodes(bool isSurface = true) const = 0;
    virtual int GetNumDistributions(bool isSurface = true) const = 0;
    virtual void CopyDistributionsTo(void *o_distrib, bool isSurface = true) const = 0;
    */
    virtual PGLRange getSurfaceSampleRange(size_t id) const = 0;

    virtual PGLRange getVolumeSampleRange(size_t id) const = 0;

    //virtual void runUpdateDump(const std::string updateDumpFilename, const bool surface = true) const = 0;

    virtual void sDump(SDump *sDump) const = 0;
};
}  // namespace openpgl
