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
#include "VMMChiSquareComponentSplitterV2.h"

namespace openpgl
{

template <class TVMMDistribution>
struct AdaptiveSplitAndMergeFactoryV2
{
   public:
    const static PGL_DIRECTIONAL_DISTRIBUTION_TYPE DIRECTIONAL_DISTRIBUTION_TYPE =
        TVMMDistribution::ParallaxCompensation ? PGL_DIRECTIONAL_DISTRIBUTION_PARALLAX_AWARE_VMM_V2 : PGL_DIRECTIONAL_DISTRIBUTION_VMM_V2;

    typedef TVMMDistribution Distribution;
    typedef TVMMDistribution VMM;

    // typedef WeightedEMVonMisesFisherFactory<VMM> WeightedEMFactory;
    typedef ParallaxAwareVonMisesFisherWeightedEMFactory<VMM> WeightedEMFactory;
    typedef VonMisesFisherChiSquareComponentSplitterV2<WeightedEMFactory> Splitter;
    typedef VonMisesFisherChiSquareComponentMerger<WeightedEMFactory, Splitter> Merger;

    typedef typename VonMisesFisherChiSquareComponentSplitterV2<WeightedEMFactory>::SplitCandidate SplitCandidate;

    struct Configuration
    {
        typename WeightedEMFactory::Configuration weightedEMCfg;

        // The min. Chi^2 threshold for splitting a mixture component
        float splittingThreshold{0.75f};
        // The max. Chi^2 threshold for merging a two mixture components
        float mergingThreshold{0.00625f};

        bool useSplitAndMerge{true};

        // The min. number of samples processed after the last merge step required to trigger a merge step
        int minSamplesForMerging{250};

        void serialize(std::ostream &stream) const;

        void deserialize(std::istream &stream);

        std::string toString() const;

        bool operator==(const Configuration &b) const
        {
            bool equal = true;
            if (splittingThreshold != b.splittingThreshold || mergingThreshold != b.mergingThreshold || useSplitAndMerge != b.useSplitAndMerge ||
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

        // Count the sumber of samples processed after the last merge pass happend
        // the goal is to get rid of this parameter at some point
        size_t numSamplesAfterLastMerge{0};

        Statistics() = default;

        void clear(const size_t &_numComponents);
        void clearAll();

        void decay(const float &alpha);

        void serialize(std::ostream &stream) const;

        void deserialize(std::istream &stream);

        bool isValid() const;

        inline size_t getNumComponents() const
        {
            OPENPGL_ASSERT(sufficientStatistics.getNumComponents() == splittingStatistics.getNumComponents());
            return sufficientStatistics.getNumComponents();
        }

        std::string toString() const;

        bool operator==(const Statistics &b) const;
    };

    struct FittingStatistics
    {
        size_t numSamples{0};
        size_t numSplits{0};
        size_t numMerges{0};

        size_t numComponents{0};

        size_t numUpdateWEMIterations{0};
        size_t numPartialUpdateWEMIterations{0};

        std::string toString() const;
    };

    void prepareSamples(SampleData *samples, const size_t numSamples, const SampleStatistics &sampleStatistics, const Configuration &cfg) const;

    void fit(VMM &vmm, Statistics &stats, const SampleData *samples, const size_t numSamples, const Configuration &cfg, FittingStatistics &fitStats) const;

    void update(VMM &vmm, Statistics &stats, const SampleData *samples, const size_t numSamples, const Configuration &cfg, FittingStatistics &fitStats) const;

    void updateFluenceEstimate(VMM &vmm, const SampleData *samples, const size_t numSamples, const size_t numZeroValueSamples, const SampleStatistics &sampleStatistics) const;

    std::string toString() const
    {
        std::ostringstream oss;
        WeightedEMFactory vmmFactory;
        oss << "AdaptiveSplitAndMergeFactoryV2[\n";
        oss << "  VMMFactory: " << vmmFactory.toString() << '\n';
        oss << ']';

        return oss.str();
    }
};

template <class TVMMDistribution>
void AdaptiveSplitAndMergeFactoryV2<TVMMDistribution>::Statistics::serialize(std::ostream &stream) const
{
    sufficientStatistics.serialize(stream);
    splittingStatistics.serialize(stream);
    stream.write(reinterpret_cast<const char *>(&numSamplesAfterLastMerge), sizeof(size_t));
}

template <class TVMMDistribution>
void AdaptiveSplitAndMergeFactoryV2<TVMMDistribution>::Statistics::deserialize(std::istream &stream)
{
    sufficientStatistics.deserialize(stream);
    splittingStatistics.deserialize(stream);
    stream.read(reinterpret_cast<char *>(&numSamplesAfterLastMerge), sizeof(size_t));
}

template <class TVMMDistribution>
void AdaptiveSplitAndMergeFactoryV2<TVMMDistribution>::Statistics::decay(const float &alpha)
{
    sufficientStatistics.decay(alpha);
    splittingStatistics.decay(alpha);
}

template <class TVMMDistribution>
bool AdaptiveSplitAndMergeFactoryV2<TVMMDistribution>::Statistics::isValid() const
{
    bool valid = true;
    valid = valid && sufficientStatistics.isValid();
    OPENPGL_ASSERT(valid);
    valid = valid && splittingStatistics.isValid();
    OPENPGL_ASSERT(valid);
    valid = valid && embree::isvalid(numSamplesAfterLastMerge);
    OPENPGL_ASSERT(valid);

    return valid;
}

template <class TVMMDistribution>
std::string AdaptiveSplitAndMergeFactoryV2<TVMMDistribution>::Statistics::toString() const
{
    std::stringstream ss;
    ss << "Statistics:" << std::endl;

    for (int ii = 0; ii < splittingStatistics.numComponents; ii++)
    {
        std::cout << "weightsEst0 = " << splittingStatistics.getWeightsEst(ii) << "\t weightsEst2nd0 = " << splittingStatistics.getWeights2ndMomentEst(ii);
        std::cout << "\t var = " << sqrt(std::abs(splittingStatistics.getWeights2ndMomentEst(ii) - splittingStatistics.getWeightsEst(ii) * splittingStatistics.getWeightsEst(ii)))
                  << "\t relVar = "
                  << sqrt(std::abs(splittingStatistics.getWeights2ndMomentEst(ii) - splittingStatistics.getWeightsEst(ii) * splittingStatistics.getWeightsEst(ii))) /
                         splittingStatistics.getWeightsEst(ii);
        std::cout << "\t var2 = " << sqrt(splittingStatistics.getWeightsVarianceEst(ii))
                  << "\t relVar = " << sqrt(splittingStatistics.getWeightsVarianceEst(ii)) / splittingStatistics.getWeightsEst(ii) << std::endl;

        std::cout << std::endl;
    }

    return ss.str();
}

template <class TVMMDistribution>
std::string AdaptiveSplitAndMergeFactoryV2<TVMMDistribution>::FittingStatistics::toString() const
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
void AdaptiveSplitAndMergeFactoryV2<TVMMDistribution>::Statistics::clear(const size_t &_numComponents)
{
    sufficientStatistics.clear(_numComponents);
    splittingStatistics.clear(_numComponents);

    numSamplesAfterLastMerge = 0;
}

template <class TVMMDistribution>
void AdaptiveSplitAndMergeFactoryV2<TVMMDistribution>::Statistics::clearAll()
{
    clear(VMM::MaxComponents);
}

template <class TVMMDistribution>
bool AdaptiveSplitAndMergeFactoryV2<TVMMDistribution>::Statistics::operator==(const Statistics &b) const
{
    bool equal = true;
    if (numSamplesAfterLastMerge != b.numSamplesAfterLastMerge)
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
void AdaptiveSplitAndMergeFactoryV2<TVMMDistribution>::Configuration::serialize(std::ostream &stream) const
{
    weightedEMCfg.serialize(stream);

    stream.write(reinterpret_cast<const char *>(&splittingThreshold), sizeof(float));
    stream.write(reinterpret_cast<const char *>(&mergingThreshold), sizeof(float));
    stream.write(reinterpret_cast<const char *>(&minSamplesForMerging), sizeof(int));
}

template <class TVMMDistribution>
void AdaptiveSplitAndMergeFactoryV2<TVMMDistribution>::Configuration::deserialize(std::istream &stream)
{
    weightedEMCfg.deserialize(stream);

    stream.read(reinterpret_cast<char *>(&splittingThreshold), sizeof(float));
    stream.read(reinterpret_cast<char *>(&mergingThreshold), sizeof(float));
    stream.read(reinterpret_cast<char *>(&minSamplesForMerging), sizeof(int));
}

template <class TVMMDistribution>
std::string AdaptiveSplitAndMergeFactoryV2<TVMMDistribution>::Configuration::toString() const
{
    std::stringstream ss;
    ss << "Configuration:" << std::endl;
    ss << "\tweightedEMCfg = " << weightedEMCfg.toString() << std::endl;
    ss << "\tsplittingThreshold = " << splittingThreshold << std::endl;
    ss << "\tmergingThreshold = " << mergingThreshold << std::endl;
    ss << "\tuseSplitAndMerge = " << useSplitAndMerge << std::endl;
    ss << "\tminSamplesForMerging = " << minSamplesForMerging << std::endl;
    return ss.str();
}

template <class TVMMDistribution>
void AdaptiveSplitAndMergeFactoryV2<TVMMDistribution>::prepareSamples(SampleData *samples, const size_t numSamples, const SampleStatistics &sampleStatistics,
                                                                      const Configuration &cfg) const
{
    WeightedEMFactory factory = WeightedEMFactory();
    factory.prepareSamples(samples, numSamples, sampleStatistics, cfg.weightedEMCfg);
}

template <class TVMMDistribution>
void AdaptiveSplitAndMergeFactoryV2<TVMMDistribution>::fit(VMM &vmm, Statistics &stats, const SampleData *samples, const size_t numSamples, const Configuration &cfg,
                                                           FittingStatistics &fitStats) const
{
    const size_t numComponents = cfg.weightedEMCfg.initK;
    stats.clear(numComponents);
    // Initial fitting of the mixture using standard weighted EM
    WeightedEMFactory factory = WeightedEMFactory();
    typename WeightedEMFactory::FittingStatistics wemFitStats;
    factory.fitMixture(vmm, stats.sufficientStatistics, samples, numSamples, cfg.weightedEMCfg, wemFitStats);
    factory.initComponentDistances(vmm, stats.sufficientStatistics, samples, numSamples);
    OPENPGL_ASSERT(vmm.isValid());
    OPENPGL_ASSERT(vmm.getNumComponents() == stats.sufficientStatistics.getNumComponents());
    OPENPGL_ASSERT(stats.isValid());

    // We use split and merge to optimze the fitting result of the standard weighted EM algorithm which can get stuck in local
    // maximas (e.g., one component trying to represent a multi-modal distribution or a distribtion containing a firely signal).
    if (cfg.useSplitAndMerge)
    {
        // Calculate the estimate of the integral of the function (e.g. radiance or importance) represented by the VMM
        float mcEstimate = stats.sufficientStatistics.getSumWeights() / stats.sufficientStatistics.getNumSamples();

        //////////////////////////////////////////////////////
        // Recursive Splitting
        //////////////////////////////////////////////////////
        Splitter splitter = Splitter();

        // The bit mask for the componets that are split and needs to be refitted
        typename Splitter::PartialFittingMask mask;
        typename Splitter::PartialFittingMask previousAsPriorMask;
        previousAsPriorMask.resetToFalse();
        typename WeightedEMFactory::FittingStatistics partialWEMFitStats;
        int numSplits = -1;
        // We perform recursive splitting until we do not find a valid split candidate or we run out of mixture components
        while (vmm._numComponents < VMM::MaxComponents && numSplits != 0)
        {
            // Resetting the number of splits for this iteration
            numSplits = 0;
            // Recalcuating the split staticics for the current mixture
            stats.splittingStatistics.clearAll();
            splitter.CalculateSplitStatistics(vmm, stats.splittingStatistics, mcEstimate, samples, numSamples);

            // Getting a list of all split candidates sorteb by their Chi^2 value
            std::vector<SplitCandidate> splitComps = stats.splittingStatistics.getSplitCandidates(cfg.splittingThreshold, false);
            mask.resetToFalse();
            // For each split candidate
            for (size_t k = 0; k < splitComps.size(); k++)
            {
                if (splitComps[k].chiSquareEst > cfg.splittingThreshold && vmm._numComponents < VMM::MaxComponents)
                {
#ifndef OPENPGL_USE_THREE_SPLIT
                    typename Splitter::SplitType splitType = splitComps[k].splitType;
                    // If a proposed split was successful or not. A split might not be successfull when the concentration limit of the component is reached or proposed
                    // new split mean directions are at the current mean (i.e., split stats have an eigen value close to zero).
                    bool splitSuccess = true;
                    if (splitType == Splitter::EFirefly)
                    {
                        splitSuccess = splitter.SplitComponentFireFly(vmm, stats.splittingStatistics, stats.sufficientStatistics, splitComps[k].componentIndex, cfg.weightedEMCfg);
                    }
                    else
                    {
                        splitSuccess = splitter.SplitComponentMultiModal(vmm, stats.splittingStatistics, stats.sufficientStatistics, splitComps[k].componentIndex);
                    }

                    if (splitSuccess)
                    {
                        mask.setToTrue(splitComps[k].componentIndex);
                        mask.setToTrue(vmm._numComponents - 1);
                    }
#else
                    bool splitSuccess = splitter.SplitComponentIntoThree(vmm, stats.splittingStatistics, stats.sufficientStatistics, splitComps[k].componentIndex);

                    if (splitSuccess)
                    {
                        mask.setToTrue(splitComps[k].componentIndex);
                        mask.setToTrue(vmm._numComponents - 1);
                        mask.setToTrue(vmm._numComponents - 2);
                    }
#endif
                    // Increase the number of performed splits if the split was successfull
                    if (splitSuccess)
                    {
                        numSplits++;
                    }
                }
                else
                {
                    continue;
                }
            }
            // Partially update/refit all split mixture components. Since we are in the initial fitting stage the current sufficient statistics of the mixture are not used as
            // priors.
            if (numSplits > 0)
            {
                factory.partialUpdateMixture(vmm, mask, false, previousAsPriorMask, stats.sufficientStatistics, samples, numSamples, cfg.weightedEMCfg, partialWEMFitStats);
            }
        }

        // After the recursive splitting process is finished we recaclulate the split statistics for current mixture from scratch
        splitter.CalculateSplitStatistics(vmm, stats.splittingStatistics, mcEstimate, samples, numSamples);

        OPENPGL_ASSERT(vmm.getNumComponents() == stats.getNumComponents());
        OPENPGL_ASSERT(vmm.isValid());
#ifdef OPENPGL_DEBUG_SAM
        if (vmm.getNumComponents() == VMM::MaxComponents)
        {
            std::cout << "SaM: Reach Component Limit" << std::endl;
        }
#endif
        //////////////////////////////////////////////////////
        // Merging
        //////////////////////////////////////////////////////
        Merger merger = Merger();
        merger.PerformMerging(vmm, cfg.mergingThreshold, cfg.splittingThreshold, false, stats.sufficientStatistics, stats.splittingStatistics);
        OPENPGL_ASSERT(vmm.isValid());
    }

    stats.numSamplesAfterLastMerge = 0.0f;

    factory.initComponentDistances(vmm, stats.sufficientStatistics, samples, numSamples);
    OPENPGL_ASSERT(stats.sufficientStatistics.isValid());
    OPENPGL_ASSERT(vmm.isValid());
}

template <class TVMMDistribution>
void AdaptiveSplitAndMergeFactoryV2<TVMMDistribution>::update(VMM &vmm, Statistics &stats, const SampleData *samples, const size_t numSamples, const Configuration &cfg,
                                                              FittingStatistics &fitStats) const
{
    OPENPGL_ASSERT(vmm.isValid());
    OPENPGL_ASSERT(vmm.getNumComponents() == stats.getNumComponents());
    OPENPGL_ASSERT(stats.isValid());

    // Update the mixture using standard weighted EM
    WeightedEMFactory factory = WeightedEMFactory();
    typename WeightedEMFactory::FittingStatistics wemFitStats;
    const size_t previousNumberOfComponents = vmm._numComponents;
    factory.updateMixture(vmm, stats.sufficientStatistics, samples, numSamples, cfg.weightedEMCfg, wemFitStats);
    OPENPGL_ASSERT(vmm.isValid());
    fitStats.numSamples = numSamples;
    fitStats.numUpdateWEMIterations = wemFitStats.numIterations;

    // Check if the update step added a new component.
    // This happens if samples are not covered by any existing component and we need to extend the splittingStats.
    if (previousNumberOfComponents < vmm._numComponents)
    {
        stats.splittingStatistics.setNumComponents(vmm._numComponents);
    }
    OPENPGL_ASSERT(stats.sufficientStatistics.isValid());

    // We use split and merge to optimize the fitting/update of the mixture to better reflect the observed data.
    if (cfg.useSplitAndMerge)
    {
        // Calculate the estimate of the integral of the function (e.g. radiance or importance) represented by the VMM
        float mcEstimate = stats.sufficientStatistics.getSumWeights() / stats.sufficientStatistics.getNumSamples();

        //////////////////////////////////////////////////////
        // Splitting
        //////////////////////////////////////////////////////
        Splitter splitter = Splitter();

        auto splitStatsBefore = stats.splittingStatistics;
        // Updating split statistics
        splitter.UpdateSplitStatistics(vmm, stats.splittingStatistics, mcEstimate, samples, numSamples, true, false);
        auto splitStatsAfter = stats.splittingStatistics;

        OPENPGL_ASSERT(stats.splittingStatistics.isValid());
        // Array of flags if a mixture component was already split or not. During the update procedure we only split a component once.
        bool alreadySplitted[VMM::MaxComponents];
        for (int i = 0; i < VMM::MaxComponents; i++)
        {
            alreadySplitted[i] = false;
        }

        bool continueSplitting = true;
        // Continue splitting until no valid split candidate (e.g., candidate is bellow the split threshold) is found or the maximun number of mixture components is reached.
        while (continueSplitting && vmm.getNumComponents() < VMM::MaxComponents)
        {
            OPENPGL_ASSERT(vmm._numComponents == stats.splittingStatistics.numComponents);
            bool splitSuccess = false;
            SplitCandidate splitCandidate;
            if (stats.splittingStatistics.getHighestValidSplitComponent(splitCandidate, cfg.splittingThreshold, vmm, alreadySplitted, true))
            {
                // Split the component of the split candidate and do a partial re-fit of the resulting split components.
                splitSuccess =
                    splitter.SplitAndUpdate(vmm, mcEstimate, splitCandidate, stats.splittingStatistics, stats.sufficientStatistics, samples, numSamples, cfg.weightedEMCfg, true);
                // Wether or not the split was succesfull we mark the component as splitted to a void
                // splitting it again
                alreadySplitted[splitCandidate.componentIndex] = true;
                // if the split was succesfull avoid splitting the new component recursivly
                if (splitSuccess)
                {
                    alreadySplitted[vmm.getNumComponents() - 1] = true;
                }
            }
            else
            {
                // We stop the splitting process, if we did not find a split candidate or if the number of VMM components reached its limit
                continueSplitting = false;
            }
            OPENPGL_ASSERT(vmm._numComponents == stats.splittingStatistics.numComponents);
        }

        stats.numSamplesAfterLastMerge += numSamples;

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
            size_t numMerges = merger.PerformMerging(vmm, cfg.mergingThreshold, cfg.splittingThreshold, true, stats.sufficientStatistics, stats.splittingStatistics);
            fitStats.numMerges = numMerges;
            stats.numSamplesAfterLastMerge = 0.0f;

            OPENPGL_ASSERT(vmm.isValid());
            OPENPGL_ASSERT(vmm.getNumComponents() == stats.getNumComponents());
            OPENPGL_ASSERT(stats.isValid());
        }
        OPENPGL_ASSERT(vmm._numComponents == stats.splittingStatistics.numComponents);
        // fitStats.numSplits = totalSplitCount;
    }
    OPENPGL_ASSERT(vmm.isValid());
    factory.updateComponentDistances(vmm, stats.sufficientStatistics, samples, numSamples);

    OPENPGL_ASSERT(vmm.getNumComponents() == stats.sufficientStatistics.getNumComponents());
    OPENPGL_ASSERT(vmm.isValid());
    OPENPGL_ASSERT(stats.sufficientStatistics.isValid());
}

template <class TVMMDistribution>
void AdaptiveSplitAndMergeFactoryV2<TVMMDistribution>::updateFluenceEstimate(VMM &vmm, const SampleData *samples, const size_t numSamples, const size_t numZeroValueSamples,
                                                                             const SampleStatistics &sampleStatistics) const
{
    WeightedEMFactory factory = WeightedEMFactory();
    factory.updateFluenceEstimate(vmm, samples, numSamples, numZeroValueSamples, sampleStatistics);
}

}  // namespace openpgl
