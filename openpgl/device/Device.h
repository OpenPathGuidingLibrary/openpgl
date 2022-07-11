#pragma once

#include "../include/openpgl/config.h"

#include "../openpgl_common.h"

#include "field/ISurfaceVolumeField.h"
#include "field/SurfaceVolumeField.h"

#include "directional/vmm/ParallaxAwareVMM.h"
#include "directional/vmm/AdaptiveSplitandMergeFactory.h"
#include "directional/vmm/VMMSurfaceSamplingDistribution.h"
#include "directional/vmm/VMMVolumeSamplingDistribution.h"
#include "directional/vmm/VMMPhaseFunctions.h"
#include "directional/dqt/DQT.h"
#include "directional/dqt/DQTFactory.h"
#include "directional/dqt/DQTSurfaceSamplingDistribution.h"
#include "directional/dqt/DQTVolumeSamplingDistribution.h"
#include "directional/dqt/SphereToSquare.h"

#include "spatial/kdtree/KDTreeBuilder.h"

namespace openpgl {

struct IDevice {
    virtual ~IDevice() {};
    virtual ISurfaceVolumeField* newField(PGLFieldArguments args) const = 0;
    virtual ISurfaceVolumeField* newFieldFromFile(const std::string fieldFileName) const = 0;
};

template<int VecSize>
struct Device: public IDevice {
    Device(){
        VMMSingleLobeHenyeyGreensteinOracle::init();
    }

    ISurfaceVolumeField* newField(PGLFieldArguments args) const override {
        ISurfaceVolumeField* gField;

        if (args.spatialStructureType == PGL_SPATIAL_STRUCTURE_KDTREE &&
            args.directionalDistributionType ==  PGL_DIRECTIONAL_DISTRIBUTION_PARALLAX_AWARE_VMM )
        {
            using DirectionalDistributionFactory = AdaptiveSplitAndMergeFactory<ParallaxAwareVonMisesFisherMixture<VecSize, 32>>;
            using GuidingField = SurfaceVolumeField<DirectionalDistributionFactory, KDTreePartitionBuilder, VMMSurfaceSamplingDistribution<typename DirectionalDistributionFactory::Distribution>, VMMVolumeSamplingDistribution<typename DirectionalDistributionFactory::Distribution>>;

            typename GuidingField::Settings gFieldSettings;
            gFieldSettings.settings.decayOnSpatialSplit   = 0.25f;
            gFieldSettings.settings.deterministic         = false;

            PGLKDTreeArguments *spatialSturctureArguments = (PGLKDTreeArguments*)args.spatialSturctureArguments;
            gFieldSettings.settings.useStochasticNNLookUp = spatialSturctureArguments->knnLookup;
            gFieldSettings.settings.spatialSubdivBuilderSettings.minSamples = spatialSturctureArguments->minSamples;
            gFieldSettings.settings.spatialSubdivBuilderSettings.maxSamples = spatialSturctureArguments->maxSamples;
            gFieldSettings.settings.spatialSubdivBuilderSettings.maxDepth   = spatialSturctureArguments->maxDepth;
            delete spatialSturctureArguments;

            PGLVMMFactoryArguments *directionalDistributionArguments = (PGLVMMFactoryArguments*)args.directionalDistributionArguments;
            gFieldSettings.distributionFactorySettings.weightedEMCfg.parallaxCompensation = directionalDistributionArguments->parallaxCompensation;

            gFieldSettings.distributionFactorySettings.weightedEMCfg.initK = directionalDistributionArguments->initK;
            gFieldSettings.distributionFactorySettings.weightedEMCfg.initKappa = directionalDistributionArguments->initKappa;
            gFieldSettings.distributionFactorySettings.weightedEMCfg.maxK = directionalDistributionArguments->maxK;
            gFieldSettings.distributionFactorySettings.weightedEMCfg.maxEMIterrations = directionalDistributionArguments->maxEMIterrations;

            gFieldSettings.distributionFactorySettings.weightedEMCfg.maxKappa = directionalDistributionArguments->maxKappa;
            gFieldSettings.distributionFactorySettings.weightedEMCfg.maxMeanCosine = openpgl::KappaToMeanCosine<float>(gFieldSettings.distributionFactorySettings.weightedEMCfg.maxKappa);
            gFieldSettings.distributionFactorySettings.weightedEMCfg.convergenceThreshold = directionalDistributionArguments->convergenceThreshold;
            gFieldSettings.distributionFactorySettings.weightedEMCfg.weightPrior = directionalDistributionArguments->weightPrior;
            gFieldSettings.distributionFactorySettings.weightedEMCfg.meanCosinePriorStrength = directionalDistributionArguments->meanCosinePriorStrength;
            gFieldSettings.distributionFactorySettings.weightedEMCfg.meanCosinePrior = directionalDistributionArguments->meanCosinePrior;

            gFieldSettings.distributionFactorySettings.splittingThreshold = directionalDistributionArguments->splittingThreshold;
            gFieldSettings.distributionFactorySettings.mergingThreshold = directionalDistributionArguments->mergingThreshold;


            gFieldSettings.distributionFactorySettings.partialReFit = directionalDistributionArguments->partialReFit;
            gFieldSettings.distributionFactorySettings.maxSplitItr = directionalDistributionArguments->maxSplitItr;

            gFieldSettings.distributionFactorySettings.useSplitAndMerge = directionalDistributionArguments->useSplitAndMerge;
            gFieldSettings.distributionFactorySettings.minSamplesForSplitting = directionalDistributionArguments->minSamplesForSplitting;
            gFieldSettings.distributionFactorySettings.minSamplesForPartialRefitting = directionalDistributionArguments->minSamplesForPartialRefitting;
            gFieldSettings.distributionFactorySettings.minSamplesForMerging = directionalDistributionArguments->minSamplesForMerging;
            delete directionalDistributionArguments;

            gFieldSettings.useParallaxCompensation = args.useParallaxCompensation;
            gField = new GuidingField(gFieldSettings);
        } else if (args.spatialStructureType == PGL_SPATIAL_STRUCTURE_KDTREE &&
                   args.directionalDistributionType ==  PGL_DIRECTIONAL_DISTRIBUTION_QUADTREE )
        {
            using DirectionalDistributionFactory = DirectionalQuadtreeFactory<DirectionalQuadtree<SphereToSquareCylindrical>>;
            using GuidingField = SurfaceVolumeField<DirectionalDistributionFactory, KDTreePartitionBuilder, DQTSurfaceSamplingDistribution<DirectionalDistributionFactory::Distribution>, DQTVolumeSamplingDistribution<DirectionalDistributionFactory::Distribution>>;

            GuidingField::Settings gFieldSettings;
            PGLKDTreeArguments *spatialSturctureArguments = (PGLKDTreeArguments*)args.spatialSturctureArguments;
            gFieldSettings.settings.useStochasticNNLookUp = spatialSturctureArguments->knnLookup;
            gFieldSettings.settings.spatialSubdivBuilderSettings.minSamples = spatialSturctureArguments->minSamples;
            gFieldSettings.settings.spatialSubdivBuilderSettings.maxSamples = spatialSturctureArguments->maxSamples;
            gFieldSettings.settings.spatialSubdivBuilderSettings.maxDepth   = spatialSturctureArguments->maxDepth;

            PGLDQTFactoryArguments *directionalDistributionArguments = (PGLDQTFactoryArguments*)args.directionalDistributionArguments;
            gFieldSettings.distributionFactorySettings.leafEstimator = (LeafEstimator)directionalDistributionArguments->leafEstimator;
            gFieldSettings.distributionFactorySettings.splitMetric = (SplitMetric)directionalDistributionArguments->splitMetric;
            gFieldSettings.distributionFactorySettings.splitThreshold = directionalDistributionArguments->splitThreshold;
            gFieldSettings.distributionFactorySettings.footprintFactor = directionalDistributionArguments->footprintFactor;
            gFieldSettings.distributionFactorySettings.maxLevels = directionalDistributionArguments->maxLevels;

            gFieldSettings.useParallaxCompensation = false;
            gField = new GuidingField(gFieldSettings);
        }else {
            throw std::runtime_error("error: unrecognized field type");
        }

        return gField;
    }

    ISurfaceVolumeField* newFieldFromFile(const std::string fieldFileName) const override {
        std::filebuf fb;
        fb.open (fieldFileName, std::ios::in | std::ios::binary);
        if (!fb.is_open()) throw std::runtime_error("error: couldn't open file");
        std::istream is(&fb);

        auto size = strlen(FIELD_FILE_HEADER_STRING) + 1;
        OPENPGL_ASSERT(size <= 256);
        char buf[256];
        is.read(&buf[0], size);
        if (!is) throw std::runtime_error("error: invalid file header");
        for (auto i = 0; i < size; i++)
        {
            if (buf[i] != FIELD_FILE_HEADER_STRING[i])
                throw std::runtime_error("error: invalid file header");
        }

        PGL_SPATIAL_STRUCTURE_TYPE spatialStructureType;
        is.read(reinterpret_cast<char*>(&spatialStructureType), sizeof(spatialStructureType));
        PGL_DIRECTIONAL_DISTRIBUTION_TYPE directionalDistributionType;
        is.read(reinterpret_cast<char*>(&directionalDistributionType), sizeof(directionalDistributionType));

        ISurfaceVolumeField* gField;

        if (spatialStructureType == PGL_SPATIAL_STRUCTURE_KDTREE && directionalDistributionType == PGL_DIRECTIONAL_DISTRIBUTION_PARALLAX_AWARE_VMM)
        {
            using DirectionalDistributionFactory = AdaptiveSplitAndMergeFactory<ParallaxAwareVonMisesFisherMixture<VecSize, 32>>;
            using GuidingField = SurfaceVolumeField<DirectionalDistributionFactory, KDTreePartitionBuilder, VMMSurfaceSamplingDistribution<typename DirectionalDistributionFactory::Distribution>, VMMVolumeSamplingDistribution<typename DirectionalDistributionFactory::Distribution>>;

            gField = (ISurfaceVolumeField *)new GuidingField();
        } else if (spatialStructureType == PGL_SPATIAL_STRUCTURE_KDTREE &&
                   directionalDistributionType == PGL_DIRECTIONAL_DISTRIBUTION_QUADTREE)
        {
            using DirectionalDistributionFactory = DirectionalQuadtreeFactory<DirectionalQuadtree<SphereToSquareCylindrical>>;
            using GuidingField = SurfaceVolumeField<DirectionalDistributionFactory, KDTreePartitionBuilder, DQTSurfaceSamplingDistribution<DirectionalDistributionFactory::Distribution>, DQTVolumeSamplingDistribution<DirectionalDistributionFactory::Distribution>>;

            gField = (ISurfaceVolumeField *)new GuidingField();
        } else {
            fb.close();
            throw std::runtime_error("error: unrecognized field type");
        }

        gField->deserialize(is);

        fb.close();

        return gField;
    }
};

#ifdef OPENPGL_DEVICE_TYPE_CPU_4
IDevice* newDeviceCPU4();
#endif
#ifdef OPENPGL_DEVICE_TYPE_CPU_8
IDevice* newDeviceCPU8();
#endif
#ifdef OPENPGL_DEVICE_TYPE_CPU_16
IDevice* newDeviceCPU16();
#endif

}

#undef FILE_HEADER_STRING
