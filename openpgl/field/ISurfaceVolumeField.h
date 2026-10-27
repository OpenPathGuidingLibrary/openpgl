// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../data/SampleDataStorage.h"
#include "../directional/ISurfaceSamplingDistribution.h"
#include "../directional/IVolumeSamplingDistribution.h"

namespace openpgl
{

struct FieldStatistics;

struct ISurfaceVolumeField
{
    using SampleContainer = SampleDataStorage::SampleContainer;

    virtual ~ISurfaceVolumeField() {};

    virtual ISurfaceSamplingDistribution *newSurfaceSamplingDistribution() const = 0;

    virtual bool initSurfaceSamplingDistribution(ISurfaceSamplingDistribution *surfaceSamplingDistribution, const Point3 &position, float *sample1D) const = 0;

    virtual IVolumeSamplingDistribution *newVolumeSamplingDistribution() const = 0;

    virtual bool initVolumeSamplingDistribution(IVolumeSamplingDistribution *volumeSamplingDistribution, const Point3 &position, float *sample1D) const = 0;

    virtual void setSceneBounds(const openpgl::BBox &sceneBounds) = 0;

    virtual openpgl::BBox getSceneBounds() const = 0;

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
};
}  // namespace openpgl
