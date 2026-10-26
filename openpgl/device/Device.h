#pragma once

#include "../include/openpgl/config.h"
#include "../openpgl_common.h"
#include "directional/dqt/DQT.h"
#include "directional/dqt/DQTFactory.h"
#include "directional/dqt/DQTSurfaceSamplingDistribution.h"
#include "directional/dqt/DQTVolumeSamplingDistribution.h"
#include "directional/dqt/SphereToSquare.h"
#include "directional/vmm/AdaptiveSplitandMergeFactory.h"
#include "directional/vmm/ParallaxAwareVonMisesFisherMixture.h"
#include "directional/vmm/VMMPhaseFunctions.h"
#include "directional/vmm/VMMSurfaceSamplingDistribution.h"
#include "directional/vmm/VMMVolumeSamplingDistribution.h"
#include "field/ISurfaceVolumeField.h"
#include "field/SurfaceVolumeField.h"
#include "spatial/kdtree/KDTreeBuilder.h"
#include "tbb/tbb.h"

#define OPENPGL_TASK_CONTROL

namespace openpgl
{

#ifdef OPENPGL_TASK_CONTROL
#if TBB_INTERFACE_VERSION >= 11005
static tbb::global_control *g_opgl_tbb_thread_control = nullptr;
#else
static tbb::task_scheduler_init g_opgl_tbb_threads(tbb::task_scheduler_init::deferred);
#endif
#endif

struct IDevice
{
    virtual ~IDevice() {};
    virtual ISurfaceVolumeField *newField(PGLFieldArguments args) const = 0;
    virtual ISurfaceVolumeField *newFieldFromFile(const std::string fieldFileName) const = 0;
};

template <int VecSize>
struct Device : public IDevice
{
   private:
#ifdef OPENPGL_TASK_CONTROL
    size_t m_numThreads{0};
#endif
   public:
    Device(size_t numThreads = 0)
    {
#ifdef OPENPGL_TASK_CONTROL
        if (numThreads == 0)
        {
#if TBB_INTERFACE_VERSION >= 9100
            m_numThreads = tbb::this_task_arena::max_concurrency();
#else
            m_numThreads = tbb::task_scheduler_init::default_num_threads();
#endif
        }
        else
        {
#if TBB_INTERFACE_VERSION >= 9100
            m_numThreads = std::min(numThreads, (size_t)tbb::this_task_arena::max_concurrency());
#else
            m_numThreads = std::min(numThreads, (size_t)tbb::task_scheduler_init::default_num_threads());
#endif
        }
#if TBB_INTERFACE_VERSION >= 11005
        g_opgl_tbb_thread_control = new tbb::global_control(tbb::global_control::max_allowed_parallelism, m_numThreads);
#else
        g_opgl_tbb_threads.initialize(int(m_numThreads));
#endif
#endif
        VMMSingleLobeHenyeyGreensteinOracle::init();
    }

    ~Device() override
    {
#ifdef OPENPGL_TASK_CONTROL
#if TBB_INTERFACE_VERSION >= 11005
        delete g_opgl_tbb_thread_control;
        g_opgl_tbb_thread_control = nullptr;
#else
        g_opgl_tbb_threads.terminate();
#endif
#endif
    }

    ISurfaceVolumeField *newField(PGLFieldArguments args) const override
    {
        ISurfaceVolumeField *gField;

        if (args.spatialStructureType == PGL_SPATIAL_STRUCTURE_KDTREE && args.directionalDistributionType == PGL_DIRECTIONAL_DISTRIBUTION_PARALLAX_AWARE_VMM)
        {
            using DirectionalDistributionFactory = AdaptiveSplitAndMergeFactory<ParallaxAwareVonMisesFisherMixture<VecSize, 32, true>>;
            using GuidingField = SurfaceVolumeField<VecSize, DirectionalDistributionFactory, KDTreePartitionBuilder,
                                                    VMMSurfaceSamplingDistribution<typename DirectionalDistributionFactory::Distribution, true>,
                                                    VMMVolumeSamplingDistribution<typename DirectionalDistributionFactory::Distribution, true>>;

            typename GuidingField::Settings gFieldSettings;
            gFieldSettings.settings.decayOnSpatialSplit = 0.25f;
            gFieldSettings.settings.deterministic = args.deterministic;
#ifdef OPENPGL_VSP_GUIDING
            gFieldSettings.settings.varianceBasedVSP = args.varianceBasedVSP;
#endif
            gFieldSettings.debugSettings.fitRegions = args.debugArguments.fitRegions;

            PGLKDTreeArguments *spatialSturctureArguments = (PGLKDTreeArguments *)args.spatialSturctureArguments;
            gFieldSettings.settings.useStochasticNNLookUp = spatialSturctureArguments->knnLookup;
            gFieldSettings.settings.useISNNLookUp = spatialSturctureArguments->isKnnLookup;
            gFieldSettings.settings.spatialSubdivBuilderSettings.minSamples = spatialSturctureArguments->minSamples;
            gFieldSettings.settings.spatialSubdivBuilderSettings.maxSamples = spatialSturctureArguments->maxSamples;
            gFieldSettings.settings.spatialSubdivBuilderSettings.maxDepth = spatialSturctureArguments->maxDepth;
            delete spatialSturctureArguments;

            PGLVMMFactoryArguments *directionalDistributionArguments = (PGLVMMFactoryArguments *)args.directionalDistributionArguments;

            gFieldSettings.distributionFactorySettings.weightedEMCfg.initK = directionalDistributionArguments->initK;
            gFieldSettings.distributionFactorySettings.weightedEMCfg.initKappa = directionalDistributionArguments->initKappa;
            gFieldSettings.distributionFactorySettings.weightedEMCfg.maxK = directionalDistributionArguments->maxK;
            gFieldSettings.distributionFactorySettings.weightedEMCfg.maxEMIterrations = directionalDistributionArguments->maxEMIterrations;

            gFieldSettings.distributionFactorySettings.weightedEMCfg.maxKappa = directionalDistributionArguments->maxKappa;
            gFieldSettings.distributionFactorySettings.weightedEMCfg.maxMeanCosine =
                openpgl::KappaToMeanCosine<float>(gFieldSettings.distributionFactorySettings.weightedEMCfg.maxKappa);
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

            gField = new GuidingField(gFieldSettings);
        }
        else if (args.spatialStructureType == PGL_SPATIAL_STRUCTURE_KDTREE && args.directionalDistributionType == PGL_DIRECTIONAL_DISTRIBUTION_VMM)
        {
            using DirectionalDistributionFactory = AdaptiveSplitAndMergeFactory<ParallaxAwareVonMisesFisherMixture<VecSize, 32, false>>;
            using GuidingField = SurfaceVolumeField<VecSize, DirectionalDistributionFactory, KDTreePartitionBuilder,
                                                    VMMSurfaceSamplingDistribution<typename DirectionalDistributionFactory::Distribution, false>,
                                                    VMMVolumeSamplingDistribution<typename DirectionalDistributionFactory::Distribution, false>>;

            typename GuidingField::Settings gFieldSettings;
            gFieldSettings.settings.decayOnSpatialSplit = 0.25f;
            gFieldSettings.settings.deterministic = args.deterministic;
#ifdef OPENPGL_VSP_GUIDING
            gFieldSettings.settings.varianceBasedVSP = args.varianceBasedVSP;
#endif
            gFieldSettings.debugSettings.fitRegions = args.debugArguments.fitRegions;

            PGLKDTreeArguments *spatialSturctureArguments = (PGLKDTreeArguments *)args.spatialSturctureArguments;
            gFieldSettings.settings.useStochasticNNLookUp = spatialSturctureArguments->knnLookup;
            gFieldSettings.settings.useISNNLookUp = spatialSturctureArguments->isKnnLookup;
            gFieldSettings.settings.spatialSubdivBuilderSettings.minSamples = spatialSturctureArguments->minSamples;
            gFieldSettings.settings.spatialSubdivBuilderSettings.maxSamples = spatialSturctureArguments->maxSamples;
            gFieldSettings.settings.spatialSubdivBuilderSettings.maxDepth = spatialSturctureArguments->maxDepth;
            delete spatialSturctureArguments;

            PGLVMMFactoryArguments *directionalDistributionArguments = (PGLVMMFactoryArguments *)args.directionalDistributionArguments;

            gFieldSettings.distributionFactorySettings.weightedEMCfg.initK = directionalDistributionArguments->initK;
            gFieldSettings.distributionFactorySettings.weightedEMCfg.initKappa = directionalDistributionArguments->initKappa;
            gFieldSettings.distributionFactorySettings.weightedEMCfg.maxK = directionalDistributionArguments->maxK;
            gFieldSettings.distributionFactorySettings.weightedEMCfg.maxEMIterrations = directionalDistributionArguments->maxEMIterrations;

            gFieldSettings.distributionFactorySettings.weightedEMCfg.maxKappa = directionalDistributionArguments->maxKappa;
            gFieldSettings.distributionFactorySettings.weightedEMCfg.maxMeanCosine =
                openpgl::KappaToMeanCosine<float>(gFieldSettings.distributionFactorySettings.weightedEMCfg.maxKappa);
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

            gField = new GuidingField(gFieldSettings);
        }
        else if (args.spatialStructureType == PGL_SPATIAL_STRUCTURE_KDTREE && args.directionalDistributionType == PGL_DIRECTIONAL_DISTRIBUTION_QUADTREE)
        {
            using DirectionalDistributionFactory = DirectionalQuadtreeFactory<DirectionalQuadtree<SphereToSquareCylindrical>>;
            using GuidingField =
                SurfaceVolumeField<VecSize, DirectionalDistributionFactory, KDTreePartitionBuilder, DQTSurfaceSamplingDistribution<DirectionalDistributionFactory::Distribution>,
                                   DQTVolumeSamplingDistribution<DirectionalDistributionFactory::Distribution>>;

            typename GuidingField::Settings gFieldSettings;
            gFieldSettings.settings.decayOnSpatialSplit = 0.25f;
            gFieldSettings.settings.deterministic = args.deterministic;
#ifdef OPENPGL_VSP_GUIDING
            gFieldSettings.settings.varianceBasedVSP = args.varianceBasedVSP;
#endif
            gFieldSettings.debugSettings.fitRegions = args.debugArguments.fitRegions;

            PGLKDTreeArguments *spatialSturctureArguments = (PGLKDTreeArguments *)args.spatialSturctureArguments;
            gFieldSettings.settings.useStochasticNNLookUp = spatialSturctureArguments->knnLookup;
            gFieldSettings.settings.useISNNLookUp = spatialSturctureArguments->isKnnLookup;
            gFieldSettings.settings.spatialSubdivBuilderSettings.minSamples = spatialSturctureArguments->minSamples;
            gFieldSettings.settings.spatialSubdivBuilderSettings.maxSamples = spatialSturctureArguments->maxSamples;
            gFieldSettings.settings.spatialSubdivBuilderSettings.maxDepth = spatialSturctureArguments->maxDepth;

            PGLDQTFactoryArguments *directionalDistributionArguments = (PGLDQTFactoryArguments *)args.directionalDistributionArguments;
            gFieldSettings.distributionFactorySettings.leafEstimator = (LeafEstimator)directionalDistributionArguments->leafEstimator;
            gFieldSettings.distributionFactorySettings.splitMetric = (SplitMetric)directionalDistributionArguments->splitMetric;
            gFieldSettings.distributionFactorySettings.splitThreshold = directionalDistributionArguments->splitThreshold;
            gFieldSettings.distributionFactorySettings.footprintFactor = directionalDistributionArguments->footprintFactor;
            gFieldSettings.distributionFactorySettings.maxLevels = directionalDistributionArguments->maxLevels;

            gField = new GuidingField(gFieldSettings);
        }
        else
        {
            throw std::runtime_error("error: unrecognized field type");
        }

        return gField;
    }

    ISurfaceVolumeField *newFieldFromFile(const std::string fieldFileName) const override
    {
        std::filebuf fb;
        fb.open(fieldFileName, std::ios::in | std::ios::binary);
        if (!fb.is_open())
            throw std::runtime_error("error: couldn't open file");
        std::istream is(&fb);

        auto size = strlen(FIELD_FILE_HEADER_STRING) + 1;
        OPENPGL_ASSERT(size <= 256);
        char buf[256];
        is.read(&buf[0], size);
        if (!is)
            throw std::runtime_error("error: invalid file header");
#ifdef OPENPGL_STRICT_IO_VERSION_CHECKING
        for (auto i = 0; i < size; i++)
        {
            if (buf[i] != FIELD_FILE_HEADER_STRING[i])
                throw std::runtime_error("error: invalid file header");
        }
#endif
        PGL_SPATIAL_STRUCTURE_TYPE spatialStructureType;
        is.read(reinterpret_cast<char *>(&spatialStructureType), sizeof(spatialStructureType));
        PGL_DIRECTIONAL_DISTRIBUTION_TYPE directionalDistributionType;
        is.read(reinterpret_cast<char *>(&directionalDistributionType), sizeof(directionalDistributionType));

        ISurfaceVolumeField *gField;

        if (spatialStructureType == PGL_SPATIAL_STRUCTURE_KDTREE && directionalDistributionType == PGL_DIRECTIONAL_DISTRIBUTION_PARALLAX_AWARE_VMM)
        {
            using DirectionalDistributionFactory = AdaptiveSplitAndMergeFactory<ParallaxAwareVonMisesFisherMixture<VecSize, 32, true>>;
            using GuidingField = SurfaceVolumeField<VecSize, DirectionalDistributionFactory, KDTreePartitionBuilder,
                                                    VMMSurfaceSamplingDistribution<typename DirectionalDistributionFactory::Distribution, true>,
                                                    VMMVolumeSamplingDistribution<typename DirectionalDistributionFactory::Distribution, true>>;

            gField = (ISurfaceVolumeField *)new GuidingField();
        }
        else if (spatialStructureType == PGL_SPATIAL_STRUCTURE_KDTREE && directionalDistributionType == PGL_DIRECTIONAL_DISTRIBUTION_VMM)
        {
            using DirectionalDistributionFactory = AdaptiveSplitAndMergeFactory<ParallaxAwareVonMisesFisherMixture<VecSize, 32, false>>;
            using GuidingField = SurfaceVolumeField<VecSize, DirectionalDistributionFactory, KDTreePartitionBuilder,
                                                    VMMSurfaceSamplingDistribution<typename DirectionalDistributionFactory::Distribution, false>,
                                                    VMMVolumeSamplingDistribution<typename DirectionalDistributionFactory::Distribution, false>>;

            gField = (ISurfaceVolumeField *)new GuidingField();
        }
        else if (spatialStructureType == PGL_SPATIAL_STRUCTURE_KDTREE && directionalDistributionType == PGL_DIRECTIONAL_DISTRIBUTION_QUADTREE)
        {
            using DirectionalDistributionFactory = DirectionalQuadtreeFactory<DirectionalQuadtree<SphereToSquareCylindrical>>;
            using GuidingField =
                SurfaceVolumeField<VecSize, DirectionalDistributionFactory, KDTreePartitionBuilder, DQTSurfaceSamplingDistribution<DirectionalDistributionFactory::Distribution>,
                                   DQTVolumeSamplingDistribution<DirectionalDistributionFactory::Distribution>>;

            gField = (ISurfaceVolumeField *)new GuidingField();
        }
        else
        {
            fb.close();
            throw std::runtime_error("error: unrecognized field type");
        }

        gField->deserialize(is);

        fb.close();

        return gField;
    }
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

}  // namespace openpgl

#undef FILE_HEADER_STRING
