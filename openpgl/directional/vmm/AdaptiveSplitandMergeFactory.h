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

/**
 * @brief Class that implements the variance-aware, Chi^2-based, split and merge fitting algorithm for parametric mixture models.
 *
 * This AdaptiveSplitAndMergeFactory implements the variance-aware Split-and-Merge (SaM) algorithm, for parametric mixture models
 * based on the work of Ruppert et al. "Robust Fitting of Parallax-Aware Mixtures for Path Guiding". This class implements the
 * basic split and merge fitting and updating structure and offloads the calculation of the splitting statistics and the actual
 * splitting and merging to the VMMChiSquareComponentSplitter and VMMChiSquareComponentMerger helper classes. At the moment the
 * implementation supports parallax-aware von Mises-Fisher mixtures but can in principle be extended to any parametric mixture
 * model.
 *
 */
template <class TVMMDistribution>
struct AdaptiveSplitAndMergeFactory
{
   public:
    const static PGL_DIRECTIONAL_DISTRIBUTION_TYPE DIRECTIONAL_DISTRIBUTION_TYPE =
        TVMMDistribution::ParallaxCompensation ? PGL_DIRECTIONAL_DISTRIBUTION_PARALLAX_AWARE_VMM : PGL_DIRECTIONAL_DISTRIBUTION_VMM;

    typedef TVMMDistribution Distribution;
    typedef TVMMDistribution VMM;

    typedef ParallaxAwareVonMisesFisherWeightedEMFactory<VMM> WeightedEMFactory;
    typedef VonMisesFisherChiSquareComponentSplitter<WeightedEMFactory> Splitter;
    typedef VonMisesFisherChiSquareComponentMerger<WeightedEMFactory, Splitter> Merger;

    /**
     * @brief The configuration struct containing all fitting relevant parameters.
     *
     * This struct contains all parameters for fitting parametric mixture models
     * using the split and merge (SaM) approach. These settings include the individual
     * SaM parameters such as the Chi^2 splitting and merging thresholds as well as the
     * fitting parameter sets for the standard incremental weighted MAP-EM fitting of the mixtures.
     */
    struct Configuration
    {
        // The fitting parameter set for the weighted MAP EM fitting.
        typename WeightedEMFactory::Configuration weightedEMCfg;

        // The min. Chi^2 threshold for splitting a mixture component
        float splittingThreshold{0.5f};
        // The max. Chi^2 threshold for merging a two mixture components
        float mergingThreshold{0.025f};
        // If SaM should be active or not. Mainly for debugging purpose.
        bool useSplitAndMerge{true};

        bool partialReFit{false};
        // How many split iteration should be performed per update.
        uint32_t maxSplitItr{1};

        // The min. number of samples processed after the last merge step required to trigger a splitting step
        uint32_t minSamplesForSplitting{0};
        // The min. number of samples processed after the last merge step required to trigger partial refitting after splitting
        uint32_t minSamplesForPartialRefitting{0};
        // The min. number of samples processed after the last merge step required to trigger a merge step
        uint32_t minSamplesForMerging{0};

        void serialize(std::ostream &stream) const;

        void deserialize(std::istream &stream);

        std::string toString() const;

        bool operator==(const Configuration &b) const
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

    /**
     * @brief The sufficient statistics needed to fit a parametric mixture using the split and merge approach.
     *
     * This struct contains all sufficient statistics needed to fit the parametric mixture using the split and merge (SaM) approach.
     * These statistics include the sufficient statistics of the weighted MAP EM from previous training iterations as well as the splitting
     * statistics for the variance-aware component splitting.
     */
    struct Statistics
    {
        typename WeightedEMFactory::SufficientStatistics sufficientStatistics;
        typename Splitter::ComponentSplitStatistics splittingStatistics;

        size_t numSamplesAfterLastSplit{0};
        size_t numSamplesAfterLastMerge{0};

        Statistics() = default;

        // Clears the statistics for the _numComponents of mixture components.
        void clear(const size_t &_numComponents);
        // Clears the statistics for all mixture components.
        void clearAll();
        // Decays the strength of the sufficient statistics by a constant factor.
        // Through this decay the sufficient statistics have less influence on upcoming updates.
        void decay(const float &alpha);

        void serialize(std::ostream &stream) const;

        void deserialize(std::istream &stream);
        // Checks if all values in the sufficient statistic are valid (e.g., finite or in the correct ranges).
        bool isValid() const;
        // Returns the number of mixture components the sufficient statistics cover.
        inline size_t getNumComponents() const
        {
            OPENPGL_ASSERT(sufficientStatistics.getNumComponents() == splittingStatistics.getNumComponents());
            return sufficientStatistics.getNumComponents();
        }

        // Returns the number of overall samples represented by the sufficient statistics.
        // NOTE: Due to decay this number does not need to represent the actual/overall number of samples.
        inline float getNumSamples() const
        {
            return sufficientStatistics.getNumSamples();
        };
        std::string toString() const;

        bool operator==(const Statistics &b) const;
    };

    /**
     * @brief A set of statistics of a split and merge fitting/update pass.
     *
     */
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

    /**************************************************************
     * Public functions of the AdaptiveSplitAndMergeFactory
     ***************************************************************/

   public:
    /**
     * @brief Pre-processing function that will be called to prepare the samples for the fit or update step.
     *
     * This pre-processing function need to be called before the fit or update function to preapare the training samples.
     * Based on the used model (e.g., parallax-aware von Mises-Fisher mixtures) the function can make adjustments to the
     * sample data (e.g., pivot point re-projection).
     *
     */
    void prepareSamples(SampleData *samples, const size_t numSamples, const SampleStatistics &sampleStatistics, const Configuration &cfg) const;

    /**
     * @brief The function for the initial fitting of the parametric mixture from a set of weighted samples using the split and merge approach.
     *
     * The function fits a mixture model from a set of training samples (i.e., weighted directional samples). It starts by fitting the mixture
     * using classical weighted MAP EM followed by a variance-aware split and merge step. During splitting the splitted mixture components are
     * partially re-fitted to the observed data. During this initial fit the components are split and refitted until no component meets the split
     * criteria or if the number of maximum components is reached. After the splitting one merge pass is performed.
     */
    void fit(VMM &vmm, Statistics &stats, const SampleData *samples, const size_t numSamples, const Configuration &cfg, FittingStatistics &fitStats) const;

    /**
     * @brief The function to update an already fit mixture using an new set of training samples.
     *
     * This function updates the a previously fitted mixture using newly observed samples (i.e., weighted directional sample). The update
     * procedure starts by doing and incremental weighted MAP EM fit that uses the sufficient statistics form previous iterations as additional
     * prior. After the incremental fitting step the split statistics are updated and a split and merge step is preformed.
     *
     */
    void update(VMM &vmm, Statistics &stats, const SampleData *samples, const size_t numSamples, const Configuration &cfg, FittingStatistics &fitStats) const;

#ifdef OPENPGL_RADIANCE_CACHES
    /**
     * @brief Updates the estimates of the radiance quantities (e.g., fluence or incoming radiance) represented by the mixture.
     *
     * Using the radiance estimates from the training samples the radiance estimates for the fluence or incoming are updated or initially calculated.
     * The fluence estimate is a Monte Carlo estimate calculated by summing the sample weights of training samples. For the incoming radiance estimate
     * these weights are distributed over the mixture components based on their soft assignment. To make these estimates unbiased the zero-value samples
     * are also considered while calculating/updating the estimates.
     */
    void updateFluenceEstimate(VMM &vmm, const SampleData *samples, const size_t numSamples, const size_t numZeroValueSamples, const SampleStatistics &sampleStatistics) const;
#endif
#ifdef OPENPGL_VSP_GUIDING
    /**
     * @brief Updates the estimate of the volume scatter probability used for volume scatter probability guiding (VSP).
     *
     * Using the directions, radiance weights, and the info if the incoming radiance of a samples originates from a volume or
     * surface scattering event, estimates of the volume scattering probability disribution are updated. These estimates are used during
     * rendering to guide the probability of generating a volume or surface event next. Based on the VSPG mode this probability is either based
     * on the first or second moment. Details of how to calculate these estimates can be found in the work of Xu et al. "Volume Scattering Probability Guiding".
     */
    void updateVolumeScatterProbability(VMM &vmm, Statistics &stats, const SampleData *samples, const size_t numSamples, const ZeroValueSampleData *zeroValueSamples,
                                        const size_t numZeroValueSamples, const bool varianceBased) const;
#endif

    std::string toString() const
    {
        std::ostringstream oss;
        WeightedEMFactory vmmFactory;
        oss << "AdaptiveSplitAndMergeFactory[\n";
        oss << "  VMMFactory: " << vmmFactory.toString() << '\n';
        oss << ']';

        return oss.str();
    }
};

/********************************************************************************
 * Implementation of the AdaptiveSplitAndMergeFactor::Statistics functions.
 ********************************************************************************/

template <class TVMMDistribution>
void AdaptiveSplitAndMergeFactory<TVMMDistribution>::Statistics::serialize(std::ostream &stream) const
{
    sufficientStatistics.serialize(stream);
    splittingStatistics.serialize(stream);

    stream.write(reinterpret_cast<const char *>(&numSamplesAfterLastSplit), sizeof(size_t));
    stream.write(reinterpret_cast<const char *>(&numSamplesAfterLastMerge), sizeof(size_t));
}

template <class TVMMDistribution>
void AdaptiveSplitAndMergeFactory<TVMMDistribution>::Statistics::deserialize(std::istream &stream)
{
    sufficientStatistics.deserialize(stream);
    splittingStatistics.deserialize(stream);

    stream.read(reinterpret_cast<char *>(&numSamplesAfterLastSplit), sizeof(size_t));
    stream.read(reinterpret_cast<char *>(&numSamplesAfterLastMerge), sizeof(size_t));
}

template <class TVMMDistribution>
void AdaptiveSplitAndMergeFactory<TVMMDistribution>::Statistics::decay(const float &alpha)
{
    sufficientStatistics.decay(alpha);
    splittingStatistics.decay(alpha);
}

template <class TVMMDistribution>
bool AdaptiveSplitAndMergeFactory<TVMMDistribution>::Statistics::isValid() const
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
std::string AdaptiveSplitAndMergeFactory<TVMMDistribution>::Statistics::toString() const
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
void AdaptiveSplitAndMergeFactory<TVMMDistribution>::Statistics::clear(const size_t &_numComponents)
{
    sufficientStatistics.clear(_numComponents);
    splittingStatistics.clear(_numComponents);

    numSamplesAfterLastSplit = 0;
    numSamplesAfterLastMerge = 0;
}

template <class TVMMDistribution>
void AdaptiveSplitAndMergeFactory<TVMMDistribution>::Statistics::clearAll()
{
    clear(VMM::MaxComponents);
}

template <class TVMMDistribution>
bool AdaptiveSplitAndMergeFactory<TVMMDistribution>::Statistics::operator==(const Statistics &b) const
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

/********************************************************************************
 * Implementation of the AdaptiveSplitAndMergeFactor::FittingStatistics functions.
 ********************************************************************************/

template <class TVMMDistribution>
std::string AdaptiveSplitAndMergeFactory<TVMMDistribution>::FittingStatistics::toString() const
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

/********************************************************************************
 * Implementation of the AdaptiveSplitAndMergeFactor::Configuration functions.
 ********************************************************************************/

template <class TVMMDistribution>
void AdaptiveSplitAndMergeFactory<TVMMDistribution>::Configuration::serialize(std::ostream &stream) const
{
    weightedEMCfg.serialize(stream);

    stream.write(reinterpret_cast<const char *>(&splittingThreshold), sizeof(float));
    stream.write(reinterpret_cast<const char *>(&mergingThreshold), sizeof(float));

    stream.write(reinterpret_cast<const char *>(&partialReFit), sizeof(bool));
    stream.write(reinterpret_cast<const char *>(&maxSplitItr), sizeof(uint32_t));

    stream.write(reinterpret_cast<const char *>(&minSamplesForSplitting), sizeof(uint32_t));
    stream.write(reinterpret_cast<const char *>(&minSamplesForMerging), sizeof(uint32_t));
    stream.write(reinterpret_cast<const char *>(&minSamplesForPartialRefitting), sizeof(uint32_t));
}

template <class TVMMDistribution>
void AdaptiveSplitAndMergeFactory<TVMMDistribution>::Configuration::deserialize(std::istream &stream)
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
std::string AdaptiveSplitAndMergeFactory<TVMMDistribution>::Configuration::toString() const
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

/********************************************************************************
 * Implementation of the AdaptiveSplitAndMergeFactor functions.
 ********************************************************************************/

template <class TVMMDistribution>
void AdaptiveSplitAndMergeFactory<TVMMDistribution>::prepareSamples(SampleData *samples, const size_t numSamples, const SampleStatistics &sampleStatistics,
                                                                    const Configuration &cfg) const
{
    WeightedEMFactory factory = WeightedEMFactory();
    factory.prepareSamples(samples, numSamples, sampleStatistics, cfg.weightedEMCfg);
}

template <class TVMMDistribution>
void AdaptiveSplitAndMergeFactory<TVMMDistribution>::fit(VMM &vmm, Statistics &stats, const SampleData *samples, const size_t numSamples, const Configuration &cfg,
                                                         [[maybe_unused]] FittingStatistics &fitStats) const
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

        // Perfroming recursive splitting so that the mixture components best represent the observed data
        Splitter splitter = Splitter();
        splitter.PerformRecursiveSplitting(vmm, stats.sufficientStatistics, cfg.splittingThreshold, mcEstimate, samples, numSamples, cfg.weightedEMCfg);

        // After the recursive splitting process is finished we recaclulate the split statistics for current mixture from scratch
        splitter.CalculateSplitStatistics(vmm, stats.splittingStatistics, mcEstimate, samples, numSamples);

        OPENPGL_ASSERT(vmm.getNumComponents() == stats.getNumComponents());
        OPENPGL_ASSERT(vmm.isValid());

        //////////////////////////////////////////////////////
        // Merging
        //////////////////////////////////////////////////////
        Merger merger = Merger();
        merger.PerformMerging(vmm, cfg.mergingThreshold, cfg.splittingThreshold, false, stats.sufficientStatistics, stats.splittingStatistics);
        OPENPGL_ASSERT(vmm.isValid());
    }

    stats.numSamplesAfterLastSplit = 0.0f;
    stats.numSamplesAfterLastMerge = 0.0f;

    factory.initComponentDistances(vmm, stats.sufficientStatistics, samples, numSamples);
    OPENPGL_ASSERT(stats.sufficientStatistics.isValid());
    OPENPGL_ASSERT(vmm.isValid());
}

template <class TVMMDistribution>
void AdaptiveSplitAndMergeFactory<TVMMDistribution>::update(VMM &vmm, Statistics &stats, const SampleData *samples, const size_t numSamples, const Configuration &cfg,
                                                            FittingStatistics &fitStats) const
{
    OPENPGL_ASSERT(vmm.isValid());
    OPENPGL_ASSERT(vmm.getNumComponents() == stats.getNumComponents());
    OPENPGL_ASSERT(stats.isValid());

    // Update the mixture using standard weighted EM
    WeightedEMFactory factory = WeightedEMFactory();
    typename WeightedEMFactory::FittingStatistics wemFitStats;
    const size_t prevNumberOfComponents = vmm._numComponents;
    factory.updateMixture(vmm, stats.sufficientStatistics, samples, numSamples, cfg.weightedEMCfg, wemFitStats);
    OPENPGL_ASSERT(vmm.isValid());

    // Check if the update step added a new component.
    // This happens if samples are not covered by any existing component and we need to extend the splittingStats.
    if (prevNumberOfComponents < vmm._numComponents)
    {
        stats.splittingStatistics.setNumComponents(vmm._numComponents);
    }
    OPENPGL_ASSERT(stats.sufficientStatistics.isValid());

    // We use split and merge to optimize the fitting/update of the mixture to better reflect the observed data.
    if (cfg.useSplitAndMerge)
    {
        // Calculate the estimate of the integral of the function (e.g. radiance or importance) represented by the VMM
        float mcEstimate = stats.sufficientStatistics.getSumWeights() / stats.sufficientStatistics.getNumSamples();

        fitStats.numSamples = numSamples;
        fitStats.numUpdateWEMIterations = wemFitStats.numIterations;

        stats.numSamplesAfterLastSplit += numSamples;
        stats.numSamplesAfterLastMerge += numSamples;

        //////////////////////////////////////////////////////
        // Splitting
        //////////////////////////////////////////////////////
        Splitter splitter = Splitter();
        // Updating split statistics
        splitter.UpdateSplitStatistics(vmm, stats.splittingStatistics, mcEstimate, samples, numSamples);
        OPENPGL_ASSERT(stats.splittingStatistics.isValid());

        // We only perform splitting if we have observed enough samples after the last splitting round.
        if (stats.numSamplesAfterLastSplit >= cfg.minSamplesForSplitting)
        {
            // The binary mask tagging the split components which need to be refitted
            typename WeightedEMFactory::PartialFittingMask mask;
            mask.resetToFalse();
            // The binary mask identifying if the previous components stats of the split components should be used as prior or not.
            // In this version it is alsways set to false.
            typename WeightedEMFactory::PartialFittingMask previousAsPriorMask;
            previousAsPriorMask.resetToFalse();

            // Getting the list of split candidates sorted by their chi^2 values
            std::vector<typename Splitter::SplitCandidate> splitComps = stats.splittingStatistics.getSplitCandidates();
            int totalSplitCount = 0;
            // For each split cadidate we check if its chi^2 value is above our split threshold and if we still have free components in our mixture.
            for (size_t k = 0; k < splitComps.size(); k++)
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

            // We perform a partial refit, if we performed any split and the number of samples is above our minimal sample threshold
            if (totalSplitCount > 0 && cfg.partialReFit && numSamples >= cfg.minSamplesForPartialRefitting)
            {
                // For this partial refitting we are NOT using the previous component stats as prior
                factory.partialUpdateMixture(vmm, mask, false, previousAsPriorMask, stats.sufficientStatistics, samples, numSamples, cfg.weightedEMCfg, wemFitStats);
                // Updating the number of component in the splitting stats to account for the new split components
                stats.splittingStatistics.setNumComponents(vmm._numComponents);
                fitStats.numPartialUpdateWEMIterations = wemFitStats.numIterations;
                OPENPGL_ASSERT(vmm.isValid());
                OPENPGL_ASSERT(vmm.getNumComponents() == stats.getNumComponents());
                OPENPGL_ASSERT(stats.isValid());
            }

            fitStats.numSplits = totalSplitCount;
            stats.numSamplesAfterLastSplit = 0.0f;

#ifdef OPENPGL_SHOW_PRINT_OUTS
            std::cout << "update: totalSplitCount = " << totalSplitCount << "\t splitThreshold: " << cfg.splittingThreshold << std::endl;
#endif
        }
        // OLD
        OPENPGL_ASSERT(vmm.isValid());
        OPENPGL_ASSERT(vmm.getNumComponents() == stats.getNumComponents());
        OPENPGL_ASSERT(stats.isValid());

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
    OPENPGL_ASSERT(vmm.isValid());
    factory.updateComponentDistances(vmm, stats.sufficientStatistics, samples, numSamples);

    OPENPGL_ASSERT(vmm.getNumComponents() == stats.sufficientStatistics.getNumComponents());
    OPENPGL_ASSERT(vmm.isValid());
    OPENPGL_ASSERT(stats.sufficientStatistics.isValid());
}

#ifdef OPENPGL_RADIANCE_CACHES
template <class TVMMDistribution>
void AdaptiveSplitAndMergeFactory<TVMMDistribution>::updateFluenceEstimate(VMM &vmm, const SampleData *samples, const size_t numSamples, const size_t numZeroValueSamples,
                                                                           const SampleStatistics &sampleStatistics) const
{
    WeightedEMFactory factory = WeightedEMFactory();
    factory.updateFluenceEstimate(vmm, samples, numSamples, numZeroValueSamples, sampleStatistics);
}
#endif
#ifdef OPENPGL_VSP_GUIDING
template <class TVMMDistribution>
void AdaptiveSplitAndMergeFactory<TVMMDistribution>::updateVolumeScatterProbability(VMM &vmm, Statistics &stats, const SampleData *samples, const size_t numSamples,
                                                                                    const ZeroValueSampleData *zeroValueSamples, const size_t numZeroValueSamples,
                                                                                    const bool varianceBased) const
{
    WeightedEMFactory factory = WeightedEMFactory();
    factory.updateVolumeScatterProbability(vmm, stats.sufficientStatistics, samples, numSamples, zeroValueSamples, numZeroValueSamples, varianceBased);
}
#endif
}  // namespace openpgl
