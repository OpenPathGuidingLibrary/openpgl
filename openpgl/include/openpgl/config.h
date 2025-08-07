// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef __cplusplus
#include <cstdint>
#include <cstdlib>
#else
#include <stdint.h>
#include <stdlib.h>
#endif

#include "common.h"
#ifdef OPENPGL_BUILD
#ifdef OPENPGL_DEVICE_TYPE_CPU_16
#define OPENPGL_SUPPORT_DEVICE_TYPE_CPU_16
#endif
#else
#include "defines.h"
#endif

#ifdef __cplusplus
extern "C"
{
#endif

#include "types.h"

#define PGL_TREE_MAX_SAMPLE_PER_LEAF 32000

    enum PGL_DEVICE_TYPE
    {
        PGL_DEVICE_TYPE_CPU_4,
        PGL_DEVICE_TYPE_CPU_8,
#ifdef OPENPGL_SUPPORT_DEVICE_TYPE_CPU_16
        PGL_DEVICE_TYPE_CPU_16,
#endif
        PGL_DEVICE_TYPE_NONE,
    };

    struct PGLKDTreeArguments
    {
        bool knnLookup{true};
        bool isKnnLookup{false};
        size_t minSamples{100};
        size_t maxSamples{PGL_TREE_MAX_SAMPLE_PER_LEAF};
        size_t maxDepth{32};
    };

    struct PGLVMMFactoryArguments
    {
        PGLVMMFactoryArguments(const size_t maxSamplesPerLeaf = PGL_TREE_MAX_SAMPLE_PER_LEAF)
        {
            minSamplesForSplitting = maxSamplesPerLeaf / 8;
            minSamplesForPartialRefitting = maxSamplesPerLeaf / 8;
            minSamplesForMerging = maxSamplesPerLeaf / 4;
        }

        // weighted EM arguments
        size_t initK{PGL_VMM_MAX_COMPONENTS / 2};
        float initKappa{5.0f};

        size_t maxK{PGL_VMM_MAX_COMPONENTS};
        size_t maxEMIterrations{100};

        float maxKappa{PGL_VMM_MAX_KAPPA};
        // float maxMeanCosine { KappaToMeanCosine<float>(OPENPGL_MAX_KAPPA)};
        float convergenceThreshold{0.005f};

        // MAP prior parameters
        // weight prior
        float weightPrior{0.01f};

        // concentration/meanCosine prior
        float meanCosinePriorStrength{0.2f};
        float meanCosinePrior{0.0f};

        // adaptive split and merge arguments
        bool useSplitAndMerge{true};

        float splittingThreshold{0.5f};
        float mergingThreshold{0.025f};

        bool partialReFit{true};
        int maxSplitItr{1};

        int minSamplesForSplitting{PGL_TREE_MAX_SAMPLE_PER_LEAF / 8};
        int minSamplesForPartialRefitting{PGL_TREE_MAX_SAMPLE_PER_LEAF / 8};
        int minSamplesForMerging{PGL_TREE_MAX_SAMPLE_PER_LEAF / 4};
    };

    enum PGLDQTLeafEstimator
    {
        REJECTION_SAMPLING = 0,
        PER_LEAF
    };

    enum PGLDQTSplitMetric
    {
        MEAN = 0,
        SECOND_MOMENT
    };

    struct PGLDQTFactoryArguments
    {
        PGLDQTLeafEstimator leafEstimator{PGLDQTLeafEstimator::REJECTION_SAMPLING};
        PGLDQTSplitMetric splitMetric{PGLDQTSplitMetric::MEAN};
        float splitThreshold{0.01f};
        float footprintFactor{1};
        uint32_t maxLevels{12};
    };

    struct PGLDebugArguments
    {
        bool fitRegions{true};
        bool dumpUpdateDistributionData{false};

        bool dumpCacheCellData{false};
        pgl_point3f dumpCacheCellPosition;
        char dumpCacheCellLocation[512];
    };

    struct PGLFieldArguments
    {
        PGL_SPATIAL_STRUCTURE_TYPE spatialStructureType;
        void *spatialSturctureArguments;
        PGL_DIRECTIONAL_DISTRIBUTION_TYPE directionalDistributionType;
        void *directionalDistributionArguments;
        // for debugging
        bool deterministic{false};
        PGLDebugArguments debugArguments;
    };

    OPENPGL_CORE_INTERFACE void pglFieldArgumentsSetDefaults(PGLFieldArguments &fieldArguments, const PGL_SPATIAL_STRUCTURE_TYPE spatialType,
                                                             const PGL_DIRECTIONAL_DISTRIBUTION_TYPE directionalType, const bool deterministic, const size_t maxSamplesPerLeaf);

#ifdef __cplusplus
}  // extern "C"
#endif