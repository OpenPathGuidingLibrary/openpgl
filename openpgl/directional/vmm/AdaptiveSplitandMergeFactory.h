// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <fstream>
#include <iostream>

#include "../../data/SampleData.h"
#include "../../include/openpgl/types.h"
#include "../../openpgl_common.h"
#include "ParallaxAwareVonMisesFisherWeightedEMFactory.h"
#include "VMMChiSquareComponentMerger.h"
#include "VMMChiSquareComponentSplitter.h"

namespace openpgl
{
namespace OPENPGL_KERNEL_NS
{

template <class TVMMDistribution>
struct AdaptiveSplitAndMergeFactory
{
   public:
    const static PGL_DIRECTIONAL_DISTRIBUTION_TYPE DIRECTIONAL_DISTRIBUTION_TYPE =
        TVMMDistribution::ParallaxCompensation ? PGL_DIRECTIONAL_DISTRIBUTION_PARALLAX_AWARE_VMM : PGL_DIRECTIONAL_DISTRIBUTION_VMM;

    typedef TVMMDistribution Distribution;
    typedef TVMMDistribution VMM;

    // typedef WeightedEMVonMisesFisherFactory<VMM> WeightedEMFactory;
    typedef ParallaxAwareVonMisesFisherWeightedEMFactory<VMM> WeightedEMFactory;
    typedef VonMisesFisherChiSquareComponentSplitter<WeightedEMFactory> Splitter;
    typedef VonMisesFisherChiSquareComponentMerger<WeightedEMFactory, Splitter> Merger;

    struct Configuration
    {
        typename WeightedEMFactory::Configuration weightedEMCfg;

        // The min. Chi^2 threshold for splitting a mixture component
        float splittingThreshold{0.75f};
        // The max. Chi^2 threshold for merging a two mixture components
        float mergingThreshold{0.00625f};

        bool useSplitAndMerge{true};

        bool partialReFit{false};
        int maxSplitItr{1};

        // The min. number of samples processed after the last merge step required to trigger a splitting step
        int minSamplesForSplitting{0};
        // The min. number of samples processed after the last merge step required to trigger partial refitting after splitting
        int minSamplesForPartialRefitting{0};
        // The min. number of samples processed after the last merge step required to trigger a merge step
        int minSamplesForMerging{0};

        KERNEL_FUNCTION void serialize(std::ostream &stream) const;

        KERNEL_FUNCTION void deserialize(std::istream &stream);

        KERNEL_FUNCTION std::string toString() const;

        KERNEL_FUNCTION bool operator==(const Configuration &b) const
        {
            bool equal = true;
            if (splittingThreshold != b.splittingThreshold || mergingThreshold != b.mergingThreshold || useSplitAndMerge != b.useSplitAndMerge || partialReFit != b.partialReFit ||
                maxSplitItr != b.maxSplitItr || minSamplesForSplitting != b.minSamplesForSplitting || minSamplesForPartialRefitting != b.minSamplesForPartialRefitting ||
                minSamplesForMerging != b.minSamplesForMerging || !weightedEMCfg.operator==(b.weightedEMCfg))
            {
                equal = false;
            }
            return equal;
        }
    };

    struct Statistics
    {
        typename WeightedEMFactory::SufficientStatistics sufficientStatistics;
        typename Splitter::ComponentSplitStatistics splittingStatistics;

        size_t numSamplesAfterLastSplit{0};
        size_t numSamplesAfterLastMerge{0};

        //Statistics() = default;

        KERNEL_FUNCTION void clear(const size_t &_numComponents);
        KERNEL_FUNCTION void clearAll();

        KERNEL_FUNCTION void decay(const float &alpha);

        KERNEL_FUNCTION void serialize(std::ostream &stream) const;

        KERNEL_FUNCTION void deserialize(std::istream &stream);

        KERNEL_FUNCTION bool isValid() const;

        KERNEL_FUNCTION inline size_t getNumComponents() const
        {
            OPENPGL_ASSERT(sufficientStatistics.getNumComponents() == splittingStatistics.getNumComponents());
            return sufficientStatistics.getNumComponents();
        }

        KERNEL_FUNCTION std::string toString() const;

        KERNEL_FUNCTION bool operator==(const Statistics &b) const;
    };

    struct FittingStatistics
    {
        size_t numSamples{0};
        size_t numSplits{0};
        size_t numMerges{0};

        size_t numComponents{0};

        size_t numUpdateWEMIterations{0};
        size_t numPartialUpdateWEMIterations{0};

        KERNEL_FUNCTION std::string toString() const;
    };

    KERNEL_FUNCTION void prepareSamples(SampleData *samples, const size_t numSamples, const SampleStatistics &sampleStatistics, const Configuration &cfg) const;

    KERNEL_FUNCTION void fit(VMM &vmm, Statistics &stats, const SampleData *samples, const size_t numSamples, const Configuration &cfg, FittingStatistics &fitStats) const;

    KERNEL_FUNCTION void update(VMM &vmm, Statistics &stats, const SampleData *samples, const size_t numSamples, const Configuration &cfg, FittingStatistics &fitStats) const;

    KERNEL_FUNCTION void updateFluenceEstimate(VMM &vmm, const SampleData *samples, const size_t numSamples, const size_t numZeroValueSamples, const SampleStatistics &sampleStatistics) const;

    KERNEL_FUNCTION std::string toString() const
    {
        std::ostringstream oss;
        WeightedEMFactory vmmFactory;
        oss << "AdaptiveSplitAndMergeFactory[\n";
        oss << "  VMMFactory: " << vmmFactory.toString() << '\n';
        oss << ']';

        return oss.str();
    }
};

template <class TVMMDistribution>
KERNEL_FUNCTION void AdaptiveSplitAndMergeFactory<TVMMDistribution>::Statistics::serialize(std::ostream &stream) const
{
    sufficientStatistics.serialize(stream);
    splittingStatistics.serialize(stream);

    // stream.write(reinterpret_cast<const char*>(&numComponents), sizeof(size_t));
    stream.write(reinterpret_cast<const char *>(&numSamplesAfterLastSplit), sizeof(size_t));
    stream.write(reinterpret_cast<const char *>(&numSamplesAfterLastMerge), sizeof(size_t));
}

template <class TVMMDistribution>
KERNEL_FUNCTION void AdaptiveSplitAndMergeFactory<TVMMDistribution>::Statistics::deserialize(std::istream &stream)
{
    sufficientStatistics.deserialize(stream);
    splittingStatistics.deserialize(stream);

    // stream.read(reinterpret_cast<char*>(&numComponents), sizeof(size_t));
    stream.read(reinterpret_cast<char *>(&numSamplesAfterLastSplit), sizeof(size_t));
    stream.read(reinterpret_cast<char *>(&numSamplesAfterLastMerge), sizeof(size_t));
}

template <class TVMMDistribution>
KERNEL_FUNCTION void AdaptiveSplitAndMergeFactory<TVMMDistribution>::Statistics::decay(const float &alpha)
{
    sufficientStatistics.decay(alpha);
    splittingStatistics.decay(alpha);
}

template <class TVMMDistribution>
KERNEL_FUNCTION bool AdaptiveSplitAndMergeFactory<TVMMDistribution>::Statistics::isValid() const
{
    bool valid = true;
    valid = valid && sufficientStatistics.isValid();
    OPENPGL_ASSERT(valid);
    valid = valid && splittingStatistics.isValid();
    OPENPGL_ASSERT(valid);
    valid = valid && embree::isvalid(numSamplesAfterLastSplit);
    OPENPGL_ASSERT(valid);
    valid = valid && embree::isvalid(numSamplesAfterLastMerge);
    OPENPGL_ASSERT(valid);

    return valid;
}

template <class TVMMDistribution>
KERNEL_FUNCTION std::string AdaptiveSplitAndMergeFactory<TVMMDistribution>::Statistics::toString() const
{
    std::stringstream ss;
    ss << "Statistics:" << std::endl;
    ss << "\tsufficientStatistics:" << sufficientStatistics.toString() << std::endl;
    ss << "\tsplittingStatistics:" << splittingStatistics.toString() << std::endl;
    ss << "\tnumSamplesAfterLastSplit = " << numSamplesAfterLastSplit << std::endl;
    ss << "\tnumSamplesAfterLastMerge = " << numSamplesAfterLastMerge << std::endl;
    return ss.str();
}

template <class TVMMDistribution>
KERNEL_FUNCTION std::string AdaptiveSplitAndMergeFactory<TVMMDistribution>::FittingStatistics::toString() const
{
    std::stringstream ss;
    ss << "FittingStatistics:" << std::endl;
    ss << "\tnumSamples:" << numSamples << std::endl;
    ss << "\tnumSplits:" << numSplits << std::endl;
    ss << "\tnumMerges:" << numMerges << std::endl;
    ss << "\tnumComponents:" << numComponents << std::endl;
    ss << "\tnumUpdateWEMIterations:" << numUpdateWEMIterations << std::endl;
    ss << "\tnumPartialUpdateWEMIterations:" << numPartialUpdateWEMIterations << std::endl;
    return ss.str();
}

template <class TVMMDistribution>
KERNEL_FUNCTION void AdaptiveSplitAndMergeFactory<TVMMDistribution>::Statistics::clear(const size_t &_numComponents)
{
    sufficientStatistics.clear(_numComponents);
    splittingStatistics.clear(_numComponents);

    // numComponents = _numComponents;
    numSamplesAfterLastSplit = 0;
    numSamplesAfterLastMerge = 0;
}

template <class TVMMDistribution>
KERNEL_FUNCTION void AdaptiveSplitAndMergeFactory<TVMMDistribution>::Statistics::clearAll()
{
    clear(VMM::MaxComponents);
}

template <class TVMMDistribution>
KERNEL_FUNCTION bool AdaptiveSplitAndMergeFactory<TVMMDistribution>::Statistics::operator==(const Statistics &b) const
{
    bool equal = true;
    if (numSamplesAfterLastSplit != b.numSamplesAfterLastSplit || numSamplesAfterLastMerge != b.numSamplesAfterLastMerge)
    {
        equal = false;
    }

    if (!sufficientStatistics.operator==(b.sufficientStatistics) || !splittingStatistics.operator==(b.splittingStatistics))
    {
        equal = false;
    }
    return equal;
}

template <class TVMMDistribution>
KERNEL_FUNCTION void AdaptiveSplitAndMergeFactory<TVMMDistribution>::Configuration::serialize(std::ostream &stream) const
{
    weightedEMCfg.serialize(stream);

    stream.write(reinterpret_cast<const char *>(&splittingThreshold), sizeof(float));
    stream.write(reinterpret_cast<const char *>(&mergingThreshold), sizeof(float));

    stream.write(reinterpret_cast<const char *>(&partialReFit), sizeof(bool));
    stream.write(reinterpret_cast<const char *>(&maxSplitItr), sizeof(int));

    stream.write(reinterpret_cast<const char *>(&minSamplesForSplitting), sizeof(int));
    stream.write(reinterpret_cast<const char *>(&minSamplesForMerging), sizeof(int));
    stream.write(reinterpret_cast<const char *>(&minSamplesForPartialRefitting), sizeof(int));
}

template <class TVMMDistribution>
KERNEL_FUNCTION void AdaptiveSplitAndMergeFactory<TVMMDistribution>::Configuration::deserialize(std::istream &stream)
{
    weightedEMCfg.deserialize(stream);

    stream.read(reinterpret_cast<char *>(&splittingThreshold), sizeof(float));
    stream.read(reinterpret_cast<char *>(&mergingThreshold), sizeof(float));

    stream.read(reinterpret_cast<char *>(&partialReFit), sizeof(bool));
    stream.read(reinterpret_cast<char *>(&maxSplitItr), sizeof(int));

    stream.read(reinterpret_cast<char *>(&minSamplesForSplitting), sizeof(int));
    stream.read(reinterpret_cast<char *>(&minSamplesForMerging), sizeof(int));
    stream.read(reinterpret_cast<char *>(&minSamplesForPartialRefitting), sizeof(int));
}

template <class TVMMDistribution>
KERNEL_FUNCTION std::string AdaptiveSplitAndMergeFactory<TVMMDistribution>::Configuration::toString() const
{
    std::stringstream ss;
    ss << "Configuration:" << std::endl;
    ss << "\tweightedEMCfg = " << weightedEMCfg.toString() << std::endl;
    ss << "\tsplittingThreshold = " << splittingThreshold << std::endl;
    ss << "\tmergingThreshold = " << mergingThreshold << std::endl;
    ss << "\tuseSplitAndMerge = " << useSplitAndMerge << std::endl;
    ss << "\tpartialReFit = " << partialReFit << std::endl;
    ss << "\tmaxSplitItr = " << maxSplitItr << std::endl;
    ss << "\tminSamplesForSplitting = " << minSamplesForSplitting << std::endl;
    ss << "\tminSamplesForPartialRefitting = " << minSamplesForPartialRefitting << std::endl;
    ss << "\tminSamplesForMerging = " << minSamplesForMerging << std::endl;
    return ss.str();
}

template <class TVMMDistribution>
KERNEL_FUNCTION void AdaptiveSplitAndMergeFactory<TVMMDistribution>::prepareSamples(SampleData *samples, const size_t numSamples, const SampleStatistics &sampleStatistics,
                                                                    const Configuration &cfg) const
{
    WeightedEMFactory factory = WeightedEMFactory();
    factory.prepareSamples(samples, numSamples, sampleStatistics, cfg.weightedEMCfg);
}

template <class TVMMDistribution>
KERNEL_FUNCTION void AdaptiveSplitAndMergeFactory<TVMMDistribution>::fit(VMM &vmm, Statistics &stats, const SampleData *samples, const size_t numSamples, const Configuration &cfg,
                                                         FittingStatistics &fitStats) const
{
    const size_t numComponents = cfg.weightedEMCfg.initK;
    SINGLE stats.clear(numComponents);
    SINGLE OPENPGL_ASSERT(stats.isValid());
    // Initial fitting of the mixture using standard weighted EM
    WeightedEMFactory factory = WeightedEMFactory();
    SHARED typename WeightedEMFactory::FittingStatistics wemFitStats;
    factory.fitMixture(vmm, stats.sufficientStatistics, samples, numSamples, cfg.weightedEMCfg, wemFitStats);
    SINGLE OPENPGL_ASSERT(stats.isValid());
    factory.initComponentDistances(vmm, stats.sufficientStatistics, samples, numSamples);
    SINGLE {
    OPENPGL_ASSERT(vmm.isValid());
    OPENPGL_ASSERT(vmm.getNumComponents() == stats.sufficientStatistics.getNumComponents());
    OPENPGL_ASSERT(stats.isValid());
    }

    // We use split and merge to optimze the fitting result of the standard weighted EM algorithm which can get stuck in local
    // maximas (e.g., one component trying to represent a multi-modal distribution or a distribtion containing a firely signal).
    if (cfg.useSplitAndMerge)
    {
        // Calculate the estimate of the integral of the function (e.g. radiance or importance) represented by the VMM
        float mcEstimate = stats.sufficientStatistics.getSumWeights() / stats.sufficientStatistics.getNumSamples();

        // Perfroming recursive splitting so that the mixture components best represent the observed data
        Splitter splitter = Splitter();
        splitter.PerformRecursiveSplitting(vmm, stats.sufficientStatistics, cfg.splittingThreshold, mcEstimate, samples, numSamples, cfg.weightedEMCfg);

        // After the recursive splitting process is finished we recaclulate the split statistics for current mixture from scratch
        splitter.CalculateSplitStatistics(vmm, stats.splittingStatistics, mcEstimate, samples, numSamples);

        SINGLE {
        OPENPGL_ASSERT(vmm.getNumComponents() == stats.getNumComponents());
        OPENPGL_ASSERT(vmm.isValid());
#ifdef OPENPGL_DEBUG_SAM
        if (vmm.getNumComponents() == VMM::MaxComponents)
        {
            std::cout << "SaM: Reach Component Limit" << std::endl;
        }
#endif
        }

        //////////////////////////////////////////////////////
        // Merging
        //////////////////////////////////////////////////////
        SINGLE {
        Merger merger = Merger();
        merger.PerformMerging(vmm, cfg.mergingThreshold, cfg.splittingThreshold, false, stats.sufficientStatistics, stats.splittingStatistics);
        SINGLE OPENPGL_ASSERT(vmm.isValid());
        }
    }

    SINGLE {
    stats.numSamplesAfterLastSplit = 0.0f;
    stats.numSamplesAfterLastMerge = 0.0f;
    }
    factory.initComponentDistances(vmm, stats.sufficientStatistics, samples, numSamples);
    SINGLE {
    OPENPGL_ASSERT(stats.sufficientStatistics.isValid());
    OPENPGL_ASSERT(vmm.isValid());
    }
}

template <class TVMMDistribution>
KERNEL_FUNCTION void AdaptiveSplitAndMergeFactory<TVMMDistribution>::update(VMM &vmm, Statistics &stats, const SampleData *samples, const size_t numSamples, const Configuration &cfg,
                                                            FittingStatistics &fitStats) const
{
    SINGLE {
    OPENPGL_ASSERT(vmm.isValid());
    OPENPGL_ASSERT(vmm.getNumComponents() == stats.getNumComponents());
    OPENPGL_ASSERT(stats.isValid());
    }

    // Update the mixture using standard weighted EM
    WeightedEMFactory factory = WeightedEMFactory();
    typename WeightedEMFactory::FittingStatistics wemFitStats;
    factory.updateMixture(vmm, stats.sufficientStatistics, samples, numSamples, cfg.weightedEMCfg, wemFitStats);

    SINGLE {
    OPENPGL_ASSERT(vmm.isValid());

    // Check if the update step added a new component.
    // This happens if samples are not covered by any existing component and we need to extend the splittingStats.
    const size_t prevNumberOfComponents = vmm._numComponents;
    if (prevNumberOfComponents < vmm._numComponents)
    {
        stats.splittingStatistics.setNumComponents(vmm._numComponents);
    }
    OPENPGL_ASSERT(stats.sufficientStatistics.isValid());
    }

    // We use split and merge to optimize the fitting/update of the mixture to better reflect the observed data.
    if (cfg.useSplitAndMerge)
    {
        // Calculate the estimate of the integral of the function (e.g. radiance or importance) represented by the VMM
        float mcEstimate = stats.sufficientStatistics.getSumWeights() / stats.sufficientStatistics.getNumSamples();

        SINGLE {
        fitStats.numSamples = numSamples;
        fitStats.numUpdateWEMIterations = wemFitStats.numIterations;

        stats.numSamplesAfterLastSplit += numSamples;
        stats.numSamplesAfterLastMerge += numSamples;
        }

        //////////////////////////////////////////////////////
        // Splitting
        //////////////////////////////////////////////////////
        Splitter splitter = Splitter();
        // Updating split statistics
        splitter.UpdateSplitStatistics(vmm, stats.splittingStatistics, mcEstimate, samples, numSamples);
        SINGLE OPENPGL_ASSERT(stats.splittingStatistics.isValid());

        // We only perform splitting if we have observed enough samples after the last splitting round.
        // TODO broadcast
        if (broadcast(stats.numSamplesAfterLastSplit >= cfg.minSamplesForSplitting))
        {
            int totalSplitCount = 0;
            // The binary mask tagging the split components which need to be refitted
            typename WeightedEMFactory::PartialFittingMask mask;
            // The binary mask identifying if the previous components stats of the split components should be used as prior or not.
            // In this version it is alsways set to false.
            typename WeightedEMFactory::PartialFittingMask previousAsPriorMask;
            SINGLE {
            mask.resetToFalse();
            previousAsPriorMask.resetToFalse();

            // Getting the list of split candidates sorted by their chi^2 values
            auto [splitComps, size] = stats.splittingStatistics.getSplitCandidates();
            // For each split cadidate we check if its chi^2 value is above our split threshold and if we still have free components in our mixture.
            for (size_t k = 0; k < size; k++)
            {
                if (splitComps[k].chiSquareEst > cfg.splittingThreshold && vmm._numComponents < VMM::MaxComponents)
                {
                    // We split the component.
                    bool success = splitter.SplitComponent(vmm, stats.splittingStatistics, stats.sufficientStatistics, splitComps[k].componentIndex);
                    // If the split was successfull mark the new components for partial refitting.
                    if (success)
                    {
                        mask.setToTrue(splitComps[k].componentIndex);
                        mask.setToTrue(vmm._numComponents - 1);
                        totalSplitCount++;
                    }
                }
            }

            OPENPGL_ASSERT(vmm.isValid());
            OPENPGL_ASSERT(vmm.getNumComponents() == stats.getNumComponents());
            OPENPGL_ASSERT(stats.isValid());
            }

            // We perform a partial refit, if we performed any split and the number of samples is above our minimal sample threshold
            // TODO broadcast
            if (broadcast(totalSplitCount > 0 && cfg.partialReFit && numSamples >= cfg.minSamplesForPartialRefitting))
            {
                // For this partial refitting we are NOT using the previous component stats as prior
                factory.partialUpdateMixture(vmm, mask, false, previousAsPriorMask, stats.sufficientStatistics, samples, numSamples, cfg.weightedEMCfg, wemFitStats);
                // Updating the number of component in the splitting stats to account for the new split components
                SINGLE {
                stats.splittingStatistics.setNumComponents(vmm._numComponents);
                fitStats.numPartialUpdateWEMIterations = wemFitStats.numIterations;
                OPENPGL_ASSERT(vmm.isValid());
                OPENPGL_ASSERT(vmm.getNumComponents() == stats.getNumComponents());
                OPENPGL_ASSERT(stats.isValid());
                }
            }

            SINGLE{
            fitStats.numSplits = totalSplitCount;
            stats.numSamplesAfterLastSplit = 0.0f;
            }

#ifdef OPENPGL_SHOW_PRINT_OUTS
            std::cout << "update: totalSplitCount = " << totalSplitCount << "\t splitThreshold: " << cfg.splittingThreshold << std::endl;
#endif
        }
        // OLD
        SINGLE {
        OPENPGL_ASSERT(vmm.isValid());
        OPENPGL_ASSERT(vmm.getNumComponents() == stats.getNumComponents());
        OPENPGL_ASSERT(stats.isValid());
#ifdef OPENPGL_DEBUG_SAM
        if (vmm.getNumComponents() == VMM::MaxComponents)
        {
            std::cout << "SaM: Reach Component Limit" << std::endl;
        }
#endif
        //////////////////////////////////////////////////////
        // Merging
        //////////////////////////////////////////////////////
        if (stats.numSamplesAfterLastMerge >= cfg.minSamplesForMerging)
        {
            Merger merger = Merger();
            size_t numMerges = merger.PerformMerging(vmm, cfg.mergingThreshold, cfg.splittingThreshold, false, stats.sufficientStatistics, stats.splittingStatistics);
            fitStats.numMerges = numMerges;
            stats.numSamplesAfterLastMerge = 0.0f;

            OPENPGL_ASSERT(vmm.isValid());
            OPENPGL_ASSERT(vmm.getNumComponents() == stats.getNumComponents());
            OPENPGL_ASSERT(stats.isValid());
        }

        fitStats.numComponents = vmm._numComponents;
        }
    }

    SINGLE OPENPGL_ASSERT(vmm.isValid());
    factory.updateComponentDistances(vmm, stats.sufficientStatistics, samples, numSamples);
    SINGLE {
    OPENPGL_ASSERT(vmm.getNumComponents() == stats.sufficientStatistics.getNumComponents());
    OPENPGL_ASSERT(vmm.isValid());
    OPENPGL_ASSERT(stats.sufficientStatistics.isValid());
    }
}

template <class TVMMDistribution>
KERNEL_FUNCTION void AdaptiveSplitAndMergeFactory<TVMMDistribution>::updateFluenceEstimate(VMM &vmm, const SampleData *samples, const size_t numSamples, const size_t numZeroValueSamples,
                                                                           const SampleStatistics &sampleStatistics) const
{
    WeightedEMFactory factory = WeightedEMFactory();
    factory.updateFluenceEstimate(vmm, samples, numSamples, numZeroValueSamples, sampleStatistics);
}

}
}  // namespace openpgl
