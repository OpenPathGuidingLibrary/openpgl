// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <fstream>
#include <iostream>

#include "../../data/SampleData.h"
#include "../../openpgl_common.h"
#include "ParallaxAwareVonMisesFisherMixture.h"

// #define OPENPGL_MAX_KAPPA 1000000.0f
#define OPENPGL_MAX_KAPPA 32000.0f

#define USE_HARMONIC_MEAN

#define MC_ESTIMATE_INCOMING_RADIANCE
// using namespace embree;

namespace openpgl
{
namespace OPENPGL_KERNEL_NS
{

template <class TVMMDistribution>
struct ParallaxAwareVonMisesFisherWeightedEMFactory
{
   public:
    typedef TVMMDistribution Distribution;
    using VMM = TVMMDistribution;

    struct Configuration
    {
        size_t initK{8};
        float initKappa{5.0f};

        size_t maxK{VMM::MaxComponents};
        size_t maxEMIterrations{100};

        float maxKappa{OPENPGL_MAX_KAPPA};
        float maxMeanCosine{KappaToMeanCosine<float>(OPENPGL_MAX_KAPPA)};
        float convergenceThreshold{0.0025f};

        // MAP prior parameters
        // weight prior
        float weightPrior{0.1f};

        // concentration/meanCosine prior
        float meanCosinePriorStrength{0.1f};
        float meanCosinePrior{0.0f};

        KERNEL_FUNCTION void init();

        KERNEL_FUNCTION void serialize(std::ostream &stream) const;

        KERNEL_FUNCTION void deserialize(std::istream &stream);

        KERNEL_FUNCTION std::string toString() const;

        KERNEL_FUNCTION bool operator==(const Configuration &b) const
        {
            bool equal = true;
            if (initK != b.initK || initKappa != b.initKappa || maxK != b.maxK || maxEMIterrations != b.maxEMIterrations || maxKappa != b.maxKappa ||
                maxMeanCosine != b.maxMeanCosine || convergenceThreshold != b.convergenceThreshold || weightPrior != b.weightPrior ||
                meanCosinePriorStrength != b.meanCosinePriorStrength || meanCosinePrior != b.meanCosinePrior)
            {
                equal = false;
            }
            return equal;
        }
    };

    struct FittingStatistics
    {
        size_t numSamples{0};
        size_t numIterations{0};
        float summedWeightedLogLikelihood{0.0f};
    };

    struct PartialFittingMask
    {
        vbool mask[VMM::NumVectors];

        PartialFittingMask() = default;

        KERNEL_FUNCTION void resetToFalse();
        KERNEL_FUNCTION void resetToTrue(const size_t &numComponents);
        KERNEL_FUNCTION void setToTrue(const size_t &idx);
        KERNEL_FUNCTION void setToFalse(const size_t &idx);
        KERNEL_FUNCTION bool get(const size_t &idx) const;
        KERNEL_FUNCTION std::string toString() const;
    };

    struct SufficientStatistics  //: public WEMVMMFactory::SufficientStatistics
    {
        // FittingStatistics
        embree::Vec3<vfloat > sumOfWeightedDirections[VMM::NumVectors];
        vfloat sumOfWeightedStats[VMM::NumVectors];

        float sumWeights{0.f};
        float numSamples{0.f};
        float overallNumSamples{0.f};
        uint32_t numComponents{VMM::MaxComponents};
        bool normalized{false};
        // only used to ensure consistent hashes for debugging
        bool pad[3]{false, false, false};

        vfloat sumOfDistanceWeightes[VMM::NumVectors];

        //SufficientStatistics() = default;

        KERNEL_FUNCTION SufficientStatistics &operator+=(const SufficientStatistics &stats);

        KERNEL_FUNCTION void serialize(std::ostream &stream) const;

        KERNEL_FUNCTION void deserialize(std::istream &stream);

        KERNEL_FUNCTION void clear(size_t _numComponents);

        KERNEL_FUNCTION void clearAll();

        KERNEL_FUNCTION void normalize(const float &_numSamples);

        KERNEL_FUNCTION inline bool isNormalized() const
        {
            return normalized;
        };

        KERNEL_FUNCTION void mergeComponentStats(const size_t &idx0, const size_t &idx1);

        KERNEL_FUNCTION void splitComponentsStats(const size_t &idx0, const size_t &idx1, const Vector3 &meanDirection0, const Vector3 &meanDirection1, const float &meanCosine0,
                                  const float &meanCosine1);

        KERNEL_FUNCTION void swapComponentStats(const size_t &idx0, const size_t &idx1);

        KERNEL_FUNCTION void maskedReplace(const PartialFittingMask &mask, const SufficientStatistics &stats);

        KERNEL_FUNCTION void decay(const float &alpha);

        KERNEL_FUNCTION void applyParallaxShift(const VMM &vmm, const Vector3 shift);

        KERNEL_FUNCTION inline float getNumSamples() const
        {
            return numSamples;
        }

        KERNEL_FUNCTION inline float getSumWeights() const
        {
            return sumWeights;
        }

        KERNEL_FUNCTION inline void setNumComponents(const size_t &numComponents)
        {
            this->numComponents = numComponents;
        }

        KERNEL_FUNCTION inline size_t getNumComponents() const
        {
            return this->numComponents;
        }

        KERNEL_FUNCTION std::string toString() const;

        KERNEL_FUNCTION bool isValid() const;

        KERNEL_FUNCTION bool operator==(const SufficientStatistics &b) const;
    };

    struct UnassignedSamplesStatistics
    {
        float sumOfUnassignedWeights{0.0f};
        Vector3 sumUnassignedWeightedDirections{0.0f, 0.0f, 0.0f};
        KERNEL_FUNCTION void clear();
        KERNEL_FUNCTION bool isValid() const;
    };

   public:
    KERNEL_FUNCTION ParallaxAwareVonMisesFisherWeightedEMFactory();

    KERNEL_FUNCTION void InitUniformVMM(VMM &vmm, const int &numComponents, const float &kappa) const;

    KERNEL_FUNCTION void prepareSamples(SampleData *samples, const size_t numSamples, const SampleStatistics &sampleStatistics, const Configuration &cfg) const;

    KERNEL_FUNCTION void fitMixture(VMM &vmm, SufficientStatistics &stats, const SampleData *samples, const size_t numSamples, const Configuration &cfg, FittingStatistics &fitStats) const;

    KERNEL_FUNCTION void updateMixture(VMM &vmm, SufficientStatistics &previousStats, const SampleData *samples, const size_t numSamples, const Configuration &cfg,
                       FittingStatistics &fitStats) const;

    KERNEL_FUNCTION void partialUpdateMixture(VMM &vmm, PartialFittingMask &mask, SufficientStatistics &previousStats, const SampleData *samples, const size_t numSamples, const Configuration &cfg,
                              FittingStatistics &fitStats) const;

    KERNEL_FUNCTION void updateOutgoingRadiance(VMM &vmm, const SampleData *samples, const size_t numSamples);

#ifdef OPENPGL_RADIANCE_CACHES
    KERNEL_FUNCTION void updateFluenceEstimate(VMM &vmm, const SampleData *samples, const size_t numSamples, const size_t numZeroValueSamples, const SampleStatistics &sampleStatistics) const;
#endif

    KERNEL_FUNCTION VMM VMMfromSufficientStatistics(const SufficientStatistics &suffStats, const Configuration &cfg) const;

    KERNEL_FUNCTION std::string toString() const
    {
        return "ParallaxAwareVonMisesFisherWeightedEMFactory";
    };

    KERNEL_FUNCTION void initComponentDistances(VMM &vmm, SufficientStatistics &sufficientStats, const SampleData *samples, const size_t numSamples) const;

    KERNEL_FUNCTION void updateComponentDistances(VMM &vmm, SufficientStatistics &sufficientStats, const SampleData *samples, const size_t numSamples) const;

   private:
    KERNEL_FUNCTION void _initUniformDirections();

    KERNEL_FUNCTION float weightedExpectationStep(VMM &vmm, SufficientStatistics &stats, UnassignedSamplesStatistics &unassignedStats, const SampleData *samples, const size_t numSamples) const;

    KERNEL_FUNCTION void weightedMaximumAPosteriorStep(VMM &vmm, const SufficientStatistics &previousStats, const SufficientStatistics &currentStats, const Configuration &cfg) const;

    KERNEL_FUNCTION void estimateMAPWeights(VMM &vmm, const SufficientStatistics &currentStats, const SufficientStatistics &previousStats, const float &_weightPrior) const;

    KERNEL_FUNCTION void estimateMAPMeanDirectionAndConcentration(VMM &vmm, const SufficientStatistics &currentStats, const SufficientStatistics &previousStats, const Configuration &cfg) const;

    KERNEL_FUNCTION void partialWeightedMaximumAPosteriorStep(VMM &vmm, const PartialFittingMask &mask, SufficientStatistics &previousStats, SufficientStatistics &currentStats,
                                              const Configuration &cfg) const;

    KERNEL_FUNCTION void estimatePartialMAPWeights(VMM &vmm, const PartialFittingMask &mask, SufficientStatistics &currentStats, SufficientStatistics &previousStats,
                                   const float &_weightPrior) const;

    KERNEL_FUNCTION void estimatePartialMAPMeanDirectionAndConcentration(VMM &vmm, const PartialFittingMask &mask, SufficientStatistics &currentStats, SufficientStatistics &previousStats,
                                                         const Configuration &cfg) const;

    KERNEL_FUNCTION void handleUnassignedSampleStats(UnassignedSamplesStatistics &unassignedStats, VMM &vmm, SufficientStatistics &currentStats, SufficientStatistics &previousStats) const;

    KERNEL_FUNCTION void reprojectSample(openpgl::SampleData &sample, const openpgl::Point3 &pivotPoint, const float minDistance) const;

   private:
    embree::Vec3<vfloat > _uniformDirections[VMM::MaxComponents][VMM::NumVectors];
};

////////////////////////////////////////////////////////////
/////////            UnassignedSamplesStatistics
////////////////////////////////////////////////////////////

template <class TVMMDistribution>
KERNEL_FUNCTION void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::UnassignedSamplesStatistics::clear()
{
    sumOfUnassignedWeights = 0.0f;
    sumUnassignedWeightedDirections = Vector3(0.0f);
}

template <class TVMMDistribution>
KERNEL_FUNCTION bool ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::UnassignedSamplesStatistics::isValid() const
{
    bool valid = true;
    valid = valid && embree::isvalid(sumOfUnassignedWeights);
    OPENPGL_ASSERT(valid);
    valid = valid && sumOfUnassignedWeights >= 0.f;
    OPENPGL_ASSERT(valid);

    valid = valid && embree::isvalid(sumUnassignedWeightedDirections.x);
    OPENPGL_ASSERT(valid);
    valid = valid && embree::isvalid(sumUnassignedWeightedDirections.y);
    OPENPGL_ASSERT(valid);
    valid = valid && embree::isvalid(sumUnassignedWeightedDirections.z);
    OPENPGL_ASSERT(valid);
    return valid;
}

////////////////////////////////////////////////////////////
/////////            SufficientStatistics
////////////////////////////////////////////////////////////

template <class TVMMDistribution>
KERNEL_FUNCTION void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::SufficientStatistics::applyParallaxShift(const VMM &vmm, const Vector3 shift)
{
    if (embree::length(shift) < FLT_EPSILON)
    {
        return;
    }

    const int cnt = (vmm._numComponents + VectorSize - 1) / VectorSize;
    // const int rem = vmm._numComponents % VectorSize;

    for (uint32_t k = 0; k < cnt; k++)
    {
        embree::Vec3<vfloat > suffDirections = sumOfWeightedDirections[k];
        vfloat suffMeanCosines = embree::length(suffDirections);
        suffDirections /= suffMeanCosines;
        suffDirections *= vmm._distances[k];
        suffDirections += embree::Vec3<vfloat >(shift);
        const vfloat length = embree::length(suffDirections);
        suffDirections /= length;
        suffDirections *= suffMeanCosines;
        sumOfWeightedDirections[k] = select((vmm._distances[k] > 0.0f) & (suffMeanCosines > 0.0f) & (length > FLT_EPSILON), suffDirections, sumOfWeightedDirections[k]);
    }
}

template <class TVMMDistribution>
KERNEL_FUNCTION bool ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::SufficientStatistics::isValid() const
{
    bool valid = true;

    for (size_t k = 0; k < numComponents; k++)
    {
        const div_t tmpK = div_(k, VectorSize);
        valid = valid && embree::isvalid(get(sumOfWeightedDirections[tmpK.quot].x, tmpK.rem));
        // valid = valid && get(sumOfWeightedDirections[tmpK.quot], tmpK.rem) >= 0.0f;
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(get(sumOfWeightedDirections[tmpK.quot].y, tmpK.rem));
        // valid = valid && get(sumOfWeightedDirections[tmpK.quot], tmpK.rem) >= 0.0f;
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(get(sumOfWeightedDirections[tmpK.quot].z, tmpK.rem));
        // valid = valid && get(sumOfWeightedDirections[tmpK.quot], tmpK.rem) >= 0.0f;
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(get(sumOfWeightedStats[tmpK.quot], tmpK.rem));
        valid = valid && get(sumOfWeightedStats[tmpK.quot], tmpK.rem) >= 0.0f;
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(get(sumOfDistanceWeightes[tmpK.quot], tmpK.rem));
        valid = valid && get(sumOfDistanceWeightes[tmpK.quot], tmpK.rem) >= 0.0f;
        OPENPGL_ASSERT(valid);
    }

    for (size_t k = numComponents; k < VMM::MaxComponents; k++)
    {
        const div_t tmpK = div_(k, VectorSize);
        valid = valid && get(sumOfWeightedDirections[tmpK.quot].x, tmpK.rem) == 0.0f;
        OPENPGL_ASSERT(valid);

        valid = valid && get(sumOfWeightedDirections[tmpK.quot].y, tmpK.rem) == 0.0f;
        OPENPGL_ASSERT(valid);

        valid = valid && get(sumOfWeightedDirections[tmpK.quot].z, tmpK.rem) == 0.0f;
        OPENPGL_ASSERT(valid);

        valid = valid && get(sumOfWeightedStats[tmpK.quot], tmpK.rem) == 0.0f;
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(get(sumOfDistanceWeightes[tmpK.quot], tmpK.rem));
        valid = valid && get(sumOfDistanceWeightes[tmpK.quot], tmpK.rem) == 0.0f;
        OPENPGL_ASSERT(valid);
    }

    valid = valid && embree::isvalid(numSamples);
    valid = valid && numSamples >= 0.0f;
    OPENPGL_ASSERT(valid);

    valid = valid && embree::isvalid(sumWeights);
    valid = valid && sumWeights >= 0.0f;
    OPENPGL_ASSERT(valid);

    return valid;
}

template <class TVMMDistribution>
KERNEL_FUNCTION void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::SufficientStatistics::serialize(std::ostream &stream) const
{
    serializeVec3Vectors<VMM::NumVectors>(stream, sumOfWeightedDirections);
    serializeFloatVectors<VMM::NumVectors>(stream, sumOfWeightedStats);
    serializeFloatVectors<VMM::NumVectors>(stream, sumOfDistanceWeightes);
    stream.write(reinterpret_cast<const char *>(&sumWeights), sizeof(float));
    stream.write(reinterpret_cast<const char *>(&numSamples), sizeof(float));
    stream.write(reinterpret_cast<const char *>(&overallNumSamples), sizeof(float));
    stream.write(reinterpret_cast<const char *>(&numComponents), sizeof(size_t));
    stream.write(reinterpret_cast<const char *>(&normalized), sizeof(bool));
}

template <class TVMMDistribution>
KERNEL_FUNCTION void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::SufficientStatistics::deserialize(std::istream &stream)
{
    deserializeVec3Vectors<VMM::NumVectors>(stream, sumOfWeightedDirections);
    deserializeFloatVectors<VMM::NumVectors>(stream, sumOfWeightedStats);
    deserializeFloatVectors<VMM::NumVectors>(stream, sumOfDistanceWeightes);
    stream.read(reinterpret_cast<char *>(&sumWeights), sizeof(float));
    stream.read(reinterpret_cast<char *>(&numSamples), sizeof(float));
    stream.read(reinterpret_cast<char *>(&overallNumSamples), sizeof(float));
    stream.read(reinterpret_cast<char *>(&numComponents), sizeof(size_t));
    stream.read(reinterpret_cast<char *>(&normalized), sizeof(bool));
}

template <class TVMMDistribution>
KERNEL_FUNCTION void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::SufficientStatistics::clear(size_t _numComponents)
{
    const embree::Vec3<vfloat > vecZeros(0.0f);
    const vfloat zeros(0.0f);

    numComponents = _numComponents;
    const int cnt = (numComponents + VectorSize - 1) / VectorSize;

    for (int k = 0; k < cnt; k++)
    {
        sumOfWeightedDirections[k] = vecZeros;
        sumOfWeightedStats[k] = zeros;

        sumOfDistanceWeightes[k] = zeros;
    }

    sumWeights = 0.0f;
    numSamples = 0;
    normalized = false;
    for (int i = 0; i < 3; i++) pad[i] = false;
}

template <class TVMMDistribution>
KERNEL_FUNCTION void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::SufficientStatistics::clearAll()
{
    clear(VMM::MaxComponents);
}

template <class TVMMDistribution>
KERNEL_FUNCTION void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::SufficientStatistics::decay(const float &alpha)
{
    for (int k = 0; k < VMM::NumVectors; k++)
    {
        sumOfWeightedDirections[k].x *= alpha;
        sumOfWeightedDirections[k].y *= alpha;
        sumOfWeightedDirections[k].z *= alpha;
        sumOfWeightedStats[k] *= alpha;

        sumOfDistanceWeightes[k] *= alpha;
    }

    numSamples *= alpha;
    sumWeights *= alpha;
}

template <class TVMMDistribution>
KERNEL_FUNCTION std::string ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::SufficientStatistics::toString() const
{
    std::stringstream ss;
    ss << "SufficientStatistics:" << std::endl;
    ss << "\tsumWeights = " << sumWeights << std::endl;
    ss << "\tnumSamples = " << numSamples << std::endl;
    ss << "\toverallNumSamples = " << overallNumSamples << std::endl;
    ss << "\tnumComponents = " << numComponents << std::endl;
    ss << "\tisNormalized = " << normalized << std::endl;
    // for (size_t k = 0; k < numComponents ; k++)
    for (size_t k = 0; k < VMM::MaxComponents; k++)
    {
        int i = k / VectorSize;
        int j = k % VectorSize;
        ss << "\tstat[" << k << "]:" << "\tsumWeightedStats = " << sumOfWeightedStats[i][j] << "\tsumWeightedDirections = [" << sumOfWeightedDirections[i].x[j] << ",\t"
           << sumOfWeightedDirections[i].y[j] << ",\t" << sumOfWeightedDirections[i].z[j] << "]" << "\tsumWeightedDistanceWeights = " << sumOfDistanceWeightes[i][j] << std::endl;
    }
    return ss.str();
}

template <class TVMMDistribution>
KERNEL_FUNCTION void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::SufficientStatistics::maskedReplace(const PartialFittingMask &mask, const SufficientStatistics &stats)
{
    vfloat newSumWeights{0.0f};

    for (size_t k = 0; k < ((VMM::MaxComponents + (VectorSize - 1)) / VectorSize); k++)
    {
        sumOfWeightedDirections[k] = select(mask.mask[k], stats.sumOfWeightedDirections[k], sumOfWeightedDirections[k]);
        sumOfWeightedStats[k] = select(mask.mask[k], stats.sumOfWeightedStats[k], sumOfWeightedStats[k]);
        newSumWeights += sumOfWeightedStats[k];

        sumOfDistanceWeightes[k] = select(mask.mask[k], stats.sumOfDistanceWeightes[k], sumOfDistanceWeightes[k]);
    }
    if (normalized)
    {
        numSamples = embree::reduce_add(newSumWeights);
    }
    else
    {
        sumWeights = embree::reduce_add(newSumWeights);
    }
}

template <class TVMMDistribution>
KERNEL_FUNCTION void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::SufficientStatistics::swapComponentStats(const size_t &idx0, const size_t &idx1)
{
    const div_t tmpIdx0 = div_(idx0, VectorSize);
    const div_t tmpIdx1 = div_(idx1, VectorSize);

    swap_(get(sumOfWeightedDirections[tmpIdx0.quot].x, tmpIdx0.rem), get(sumOfWeightedDirections[tmpIdx1.quot].x, tmpIdx1.rem));
    swap_(get(sumOfWeightedDirections[tmpIdx0.quot].y, tmpIdx0.rem), get(sumOfWeightedDirections[tmpIdx1.quot].y, tmpIdx1.rem));
    swap_(get(sumOfWeightedDirections[tmpIdx0.quot].z, tmpIdx0.rem), get(sumOfWeightedDirections[tmpIdx1.quot].z, tmpIdx1.rem));
    swap_(get(sumOfWeightedStats[tmpIdx0.quot], tmpIdx0.rem), get(sumOfWeightedStats[tmpIdx1.quot], tmpIdx1.rem));
    swap_(get(sumOfDistanceWeightes[tmpIdx0.quot], tmpIdx0.rem), get(sumOfDistanceWeightes[tmpIdx1.quot], tmpIdx1.rem));
}

template <class TVMMDistribution>
KERNEL_FUNCTION void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::SufficientStatistics::mergeComponentStats(const size_t &idx0, const size_t &idx1)
{
    const div_t tmpIdx0 = div_(idx0, VectorSize);
    const div_t tmpIdx1 = div_(idx1, VectorSize);
    const div_t tmpIdx2 = div_(numComponents - 1, VectorSize);

    // merging the statistics of the component 0 and 1
    get(sumOfWeightedDirections[tmpIdx0.quot].x, tmpIdx0.rem) += get(sumOfWeightedDirections[tmpIdx1.quot].x, tmpIdx1.rem);
    get(sumOfWeightedDirections[tmpIdx0.quot].y, tmpIdx0.rem) += get(sumOfWeightedDirections[tmpIdx1.quot].y, tmpIdx1.rem);
    get(sumOfWeightedDirections[tmpIdx0.quot].z, tmpIdx0.rem) += get(sumOfWeightedDirections[tmpIdx1.quot].z, tmpIdx1.rem);
    get(sumOfWeightedStats[tmpIdx0.quot], tmpIdx0.rem) += get(sumOfWeightedStats[tmpIdx1.quot], tmpIdx1.rem);
    get(sumOfDistanceWeightes[tmpIdx0.quot], tmpIdx0.rem) += get(sumOfDistanceWeightes[tmpIdx1.quot], tmpIdx1.rem);

    // copying the statistics of the last component to the position of component 1
    get(sumOfWeightedDirections[tmpIdx1.quot].x, tmpIdx1.rem) = get(sumOfWeightedDirections[tmpIdx2.quot].x, tmpIdx2.rem);
    get(sumOfWeightedDirections[tmpIdx1.quot].y, tmpIdx1.rem) = get(sumOfWeightedDirections[tmpIdx2.quot].y, tmpIdx2.rem);
    get(sumOfWeightedDirections[tmpIdx1.quot].z, tmpIdx1.rem) = get(sumOfWeightedDirections[tmpIdx2.quot].z, tmpIdx2.rem);
    get(sumOfWeightedStats[tmpIdx1.quot], tmpIdx1.rem) = get(sumOfWeightedStats[tmpIdx2.quot], tmpIdx2.rem);
    get(sumOfDistanceWeightes[tmpIdx1.quot], tmpIdx1.rem) = get(sumOfDistanceWeightes[tmpIdx2.quot], tmpIdx2.rem);

    // reseting the statistics of the last component
    get(sumOfWeightedDirections[tmpIdx2.quot].x, tmpIdx2.rem) = 0.0f;
    get(sumOfWeightedDirections[tmpIdx2.quot].y, tmpIdx2.rem) = 0.0f;
    get(sumOfWeightedDirections[tmpIdx2.quot].z, tmpIdx2.rem) = 0.0f;
    get(sumOfWeightedStats[tmpIdx2.quot], tmpIdx2.rem) = 0.0f;
    get(sumOfDistanceWeightes[tmpIdx2.quot], tmpIdx2.rem) = 0.0f;

    numComponents--;
}

template <class TVMMDistribution>
KERNEL_FUNCTION void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::SufficientStatistics::splitComponentsStats(const size_t &idx0, const size_t &idx1,
                                                                                                                const Vector3 &meanDirection0, const Vector3 &meanDirection1,
                                                                                                                const float &meanCosine0, const float &meanCosine1)
{
    // OPENPGL_ASSERT(meanCosine0 > 0.f && meanCosine0 <= 1.0f);
    // OPENPGL_ASSERT(meanCosine1 > 0.f && meanCosine1 <= 1.0f);

    const div_t tmpI = div_(idx0, static_cast<int>(VectorSize));
    const div_t tmpJ = div_(idx1, static_cast<int>(VectorSize));

    float sumStatsWeight = get(sumOfWeightedStats[tmpI.quot], tmpI.rem);
    sumStatsWeight /= 2.0f;

    OPENPGL_ASSERT(sumStatsWeight > 0.f);

    get(sumOfWeightedStats[tmpI.quot], tmpI.rem) = sumStatsWeight;
    get(sumOfWeightedDirections[tmpI.quot].x, tmpI.rem) = meanDirection0.x * meanCosine0 * sumStatsWeight;
    get(sumOfWeightedDirections[tmpI.quot].y, tmpI.rem) = meanDirection0.y * meanCosine0 * sumStatsWeight;
    get(sumOfWeightedDirections[tmpI.quot].z, tmpI.rem) = meanDirection0.z * meanCosine0 * sumStatsWeight;

    get(sumOfWeightedStats[tmpJ.quot], tmpJ.rem) = sumStatsWeight;
    get(sumOfWeightedDirections[tmpJ.quot].x, tmpJ.rem) = meanDirection1.x * meanCosine1 * sumStatsWeight;
    get(sumOfWeightedDirections[tmpJ.quot].y, tmpJ.rem) = meanDirection1.y * meanCosine1 * sumStatsWeight;
    get(sumOfWeightedDirections[tmpJ.quot].z, tmpJ.rem) = meanDirection1.z * meanCosine1 * sumStatsWeight;

    float tmp = get(sumOfDistanceWeightes[tmpI.quot], tmpI.rem) * 0.5f;
    get(sumOfDistanceWeightes[tmpI.quot], tmpI.rem) = tmp;
    get(sumOfDistanceWeightes[tmpJ.quot], tmpJ.rem) = tmp;

    numComponents += 1;

    OPENPGL_ASSERT(!std::isnan(get(sumOfWeightedDirections[tmpI.quot].x, tmpI.rem)) && std::isfinite(get(sumOfWeightedDirections[tmpI.quot].x, tmpI.rem)));
    OPENPGL_ASSERT(!std::isnan(get(sumOfWeightedDirections[tmpI.quot].y, tmpI.rem)) && std::isfinite(get(sumOfWeightedDirections[tmpI.quot].y, tmpI.rem)));
    OPENPGL_ASSERT(!std::isnan(get(sumOfWeightedDirections[tmpI.quot].z, tmpI.rem)) && std::isfinite(get(sumOfWeightedDirections[tmpI.quot].z, tmpI.rem)));
}

template <class TVMMDistribution>
KERNEL_FUNCTION void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::SufficientStatistics::normalize(const float &_numSamples)
{
    const int cnt = (numComponents + VectorSize - 1) / VectorSize;
    numSamples = _numSamples;
    vfloat sumWeightedStatsVec(0.0f);

    for (int k = 0; k < cnt; k++)
    {
        sumWeightedStatsVec += sumOfWeightedStats[k];
    }
    sumWeights = embree::reduce_add(sumWeightedStatsVec);
    vfloat norm(sumWeights > FLT_EPSILON ? _numSamples / sumWeights : 1.0f);

    for (int k = 0; k < cnt; k++)
    {
        sumOfWeightedDirections[k] *= norm;
        sumOfWeightedStats[k] *= norm;
    }
    normalized = true;
}

template <class TVMMDistribution>
KERNEL_FUNCTION typename ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::SufficientStatistics &
ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::SufficientStatistics::operator+=(
    const typename ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::SufficientStatistics &stats)
{
    // TODO: check for normalization

const int cnt = (numComponents + VectorSize - 1) / VectorSize;

    this->sumWeights += stats.sumWeights;
    this->numSamples += stats.numSamples;
    this->overallNumSamples += stats.numSamples;
    for (int k = 0; k < cnt; k++)
    {
        this->sumOfWeightedDirections[k] += stats.sumOfWeightedDirections[k];
        this->sumOfWeightedStats[k] += stats.sumOfWeightedStats[k];
        this->sumOfDistanceWeightes[k] += stats.sumOfDistanceWeightes[k];
    }

    return *this;
}

template <class TVMMDistribution>
KERNEL_FUNCTION bool ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::SufficientStatistics::operator==(const SufficientStatistics &b) const
{
    bool equal = true;
    if (sumWeights != b.sumWeights || numSamples != b.numSamples || normalized != b.normalized || overallNumSamples != b.overallNumSamples || numComponents != b.numComponents)
    {
        equal = false;
    }

    for (int k = 0; k < VMM::NumVectors; k++)
    {
        if (embree::any(sumOfWeightedDirections[k].x != b.sumOfWeightedDirections[k].x) || embree::any(sumOfWeightedDirections[k].y != b.sumOfWeightedDirections[k].y) ||
            embree::any(sumOfWeightedDirections[k].z != b.sumOfWeightedDirections[k].z) || embree::any(sumOfWeightedStats[k] != b.sumOfWeightedStats[k]) ||
            embree::any(sumOfDistanceWeightes[k] != b.sumOfDistanceWeightes[k]))
        {
            equal = false;
        }
    }
    return equal;
}

////////////////////////////////////////////////////////////
/////////            ParallaxAwareVonMisesFisherWeightedEMFactory
////////////////////////////////////////////////////////////

template <class TVMMDistribution>
KERNEL_FUNCTION ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::ParallaxAwareVonMisesFisherWeightedEMFactory()
{
    _initUniformDirections();
}

template <class TVMMDistribution>
KERNEL_FUNCTION void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::InitUniformVMM(VMM &vmm, const int &numComponents, const float &kappa) const
{
    vmm._numComponents = numComponents;
    const size_t nComp = vmm._numComponents;
    const float weight = 1.f / float(vmm._numComponents);

    size_t n = 0;
    for (int i = 0; i < VMM::NumVectors; i++)
    {
        vmm._meanDirections[i] = _uniformDirections[nComp - 1][i];
        for (int j = 0; j < VectorSize; j++)
        {
            if (n < nComp)
            {
                get(vmm._kappas[i], j) = kappa;
                get(vmm._weights[i], j) = weight;
            }
            else
            {
                get(vmm._kappas[i], j) = 0.0f;
                get(vmm._weights[i], j) = 0.0f;
                get(vmm._normalizations[i], j) = ONE_OVER_FOUR_PI;
                get(vmm._eMinus2Kappa[i], j) = 1.0f;
                get(vmm._meanCosines[i], j) = 0.0f;
            }
            n++;
        }
    }

    vmm._calculateNormalization();
    vmm._calculateMeanCosines();
}

template <class TVMMDistribution>
KERNEL_FUNCTION void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::_initUniformDirections()
{
    const float gr = 1.618033988749895f;

    for (uint32_t l = 0; l < VMM::MaxComponents; l++)
    {
        /// distributes samples l+1 uniform samples over the sphere
        /// based on "Spherical Fibonacci Point Sets for Illumination Integrals"
        uint32_t n = 0;
        for (uint32_t k = 0; k < VMM::NumVectors; k++)
        {
            for (uint32_t i = 0; i < VectorSize; i++)
            {
                if (n < l + 1)
                {
                    float phi = 2.0f * M_PI_F * ((float)n / gr);
                    float z = 1.0f - ((2.0f * n + 1.0f) / float(l + 1));
                    float theta = std::acos(z);

                    Vector3 mu = sphericalDirection(theta, phi);
                    get(_uniformDirections[l][k].x, i) = mu[0];
                    get(_uniformDirections[l][k].y, i) = mu[1];
                    get(_uniformDirections[l][k].z, i) = mu[2];
                }
                else
                {
                    get(_uniformDirections[l][k].x, i) = 0.0f;
                    get(_uniformDirections[l][k].y, i) = 0.0f;
                    get(_uniformDirections[l][k].z, i) = 1.0f;
                }
                n++;
            }
        }
    }
}

template <class TVMMDistribution>
KERNEL_FUNCTION typename ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::VMM ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::VMMfromSufficientStatistics(
    const SufficientStatistics &suffStats, const Configuration &cfg) const
{
    SufficientStatistics previousStats;
    previousStats.clear(suffStats.numComponents);
    VMM vmm;
    vmm._numComponents = suffStats.numComponents;
    weightedMaximumAPosteriorStep(vmm, previousStats, suffStats, cfg);

    return vmm;
}

template <class TVMMDistribution>
KERNEL_FUNCTION void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::fitMixture(VMM &vmm, SufficientStatistics &stats, const SampleData *samples, const size_t numSamples,
                                                                                const Configuration &cfg, FittingStatistics &fitStats) const
{
    SINGLE {
    const size_t numComponents = cfg.initK;
    // VonMisesFisherFactory< TVMMDistribution>::InitUniformVMM( vmm, numComponents, 5.0f);
    this->InitUniformVMM(vmm, numComponents, cfg.initKappa);
    // OPENPGL_ASSERT(vmm.isValid());
    // SufficientStatistics stats;
    stats.clear(numComponents);
    stats.normalized = true;
    }
    updateMixture(vmm, stats, samples, numSamples, cfg, fitStats);
}

template <class TVMMDistribution>
KERNEL_FUNCTION void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::handleUnassignedSampleStats(UnassignedSamplesStatistics &unassignedStats, VMM &vmm,
                                                                                                 SufficientStatistics &currentStats, SufficientStatistics &previousStats) const
{
    OPENPGL_ASSERT(embree::isvalid(unassignedStats.sumOfUnassignedWeights));
    OPENPGL_ASSERT(embree::isvalid(unassignedStats.sumUnassignedWeightedDirections.x));
    OPENPGL_ASSERT(embree::isvalid(unassignedStats.sumUnassignedWeightedDirections.y));
    OPENPGL_ASSERT(embree::isvalid(unassignedStats.sumUnassignedWeightedDirections.z));

    const div_t tmpK = div_(currentStats.numComponents, VectorSize);
    currentStats.numComponents++;
    get(currentStats.sumOfWeightedStats[tmpK.quot], tmpK.rem) = unassignedStats.sumOfUnassignedWeights;
    get(currentStats.sumOfWeightedDirections[tmpK.quot].x, tmpK.rem) = unassignedStats.sumUnassignedWeightedDirections.x;
    get(currentStats.sumOfWeightedDirections[tmpK.quot].y, tmpK.rem) = unassignedStats.sumUnassignedWeightedDirections.y;
    get(currentStats.sumOfWeightedDirections[tmpK.quot].z, tmpK.rem) = unassignedStats.sumUnassignedWeightedDirections.z;

    previousStats.numComponents++;
    get(previousStats.sumOfWeightedStats[tmpK.quot], tmpK.rem) = 0.0f;
    get(previousStats.sumOfWeightedDirections[tmpK.quot].x, tmpK.rem) = 0.0f;
    get(previousStats.sumOfWeightedDirections[tmpK.quot].y, tmpK.rem) = 0.0f;
    get(previousStats.sumOfWeightedDirections[tmpK.quot].z, tmpK.rem) = 0.0f;

    vmm._numComponents++;
}

template <class TVMMDistribution>
KERNEL_FUNCTION void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::updateMixture(VMM &vmm, SufficientStatistics &previousStats, const SampleData *samples,
                                                                                   const size_t numSamples, const Configuration &cfg, FittingStatistics &fitStats) const
{
    SHARED SufficientStatistics currentStats;
    // initially clear all stats
    SINGLE currentStats.clearAll();

    SHARED size_t currentEMIteration;
    SHARED bool converged;
    SHARED float previousLogLikelihood;
    SHARED float inv_previousLogLikelihood;
    SHARED UnassignedSamplesStatistics unassignedStats;

    SINGLE {
        currentEMIteration = 0;
        converged = false;
        previousLogLikelihood = 0.0f;
        inv_previousLogLikelihood = 1.0f;
    }

    SYNC; // wait for shared memory writes of thread 0

    // Running multiple EM iterations until the mixture is converged or a number of max iterations is reached
    while (broadcast(!converged && currentEMIteration < cfg.maxEMIterrations))
    {
        SINGLE OPENPGL_ASSERT(currentStats.isValid());
        // Running the E-step to calculate the sufficient statistics and estimate the current log likelihood
        float logLikelihood = weightedExpectationStep(vmm, currentStats, unassignedStats, samples, numSamples);
        
        SINGLE {
        OPENPGL_ASSERT(unassignedStats.isValid());
        OPENPGL_ASSERT(currentStats.isValid());
        // Special handling of samples which are not covered by any mixture component (i.e., adding an additional/special component)
        if (unassignedStats.sumOfUnassignedWeights > 0.0f && currentStats.numComponents < TVMMDistribution::MaxComponents)
        {
            handleUnassignedSampleStats(unassignedStats, vmm, currentStats, previousStats);
        }

        OPENPGL_ASSERT(!currentStats.isNormalized());
        // Normalizing sufficient statistics so that the weighted stats per component can be re-interpreded by number of samples
        OPENPGL_ASSERT(currentStats.isValid());
        currentStats.normalize(currentStats.numSamples);
        OPENPGL_ASSERT(currentStats.isValid());
        weightedMaximumAPosteriorStep(vmm, currentStats, previousStats, cfg);
        currentEMIteration++;

        // TODO: Add convergence check
        if (currentEMIteration > 1)
        {
            float relLogLikelihoodDifference = std::fabs(logLikelihood - previousLogLikelihood) * inv_previousLogLikelihood;
            if (relLogLikelihoodDifference < cfg.convergenceThreshold)
            {
                converged = true;
            }
            // std::cout << "logLikelihood:" <<  logLikelihood << "\t previousLogLikelihood: "<< previousLogLikelihood  << "\t relLogLikelihoodDifference: " <<
            // relLogLikelihoodDifference << std::endl;
            previousLogLikelihood = logLikelihood;
            inv_previousLogLikelihood = 1.0f / std::fabs(logLikelihood);
        }
        }
    }

    SYNC; // wait for all threads to complete. TODO necessary?

    SINGLE {
    // The merged sufficient stats from the last iteration are now the new previous/prior stats
    previousStats += currentStats;

    fitStats.numSamples = numSamples;
    fitStats.numIterations = currentEMIteration;
    fitStats.summedWeightedLogLikelihood = previousLogLikelihood;
    } 
}

template <class TVMMDistribution>
KERNEL_FUNCTION void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::partialUpdateMixture(VMM &vmm, PartialFittingMask &mask, SufficientStatistics &previousStats,
                                                                                          const SampleData *samples, const size_t numSamples, const Configuration &cfg,
                                                                                          FittingStatistics &fitStats) const
{
    SHARED SufficientStatistics currentStats;
    // initially clear all stats
    SINGLE currentStats.clearAll();

    SHARED size_t currentEMIteration;
    SHARED bool converged;
    SHARED float previousLogLikelihood;
    SHARED float inv_previousLogLikelihood;
    SHARED UnassignedSamplesStatistics unassignedStats;

    SINGLE {
        currentEMIteration = 0;
        converged = false;
        previousLogLikelihood = 0.0f;
        inv_previousLogLikelihood = 1.0f;
    }

    SYNC; // wait for shared memory writes of thread 0

    // Running multiple EM iterations until the mixture is converged or a number of max iterations is reached.
    // During these iterations only the masked mixture components are updated
    while (broadcast(!converged && currentEMIteration < cfg.maxEMIterrations))
    {
        float logLikelihood = weightedExpectationStep(vmm, currentStats, unassignedStats, samples, numSamples);

        SINGLE {
        // Special handling of samples which are not covered by any mixture component (i.e., adding an additional/special component)
        if (unassignedStats.sumOfUnassignedWeights > 0.0f && currentStats.numComponents < TVMMDistribution::MaxComponents)
        {
            handleUnassignedSampleStats(unassignedStats, vmm, currentStats, previousStats);
            mask.setToTrue(vmm._numComponents - 1);
        }

        OPENPGL_ASSERT(currentStats.isValid());
        OPENPGL_ASSERT(!currentStats.isNormalized());
        currentStats.normalize(currentStats.numSamples);
        OPENPGL_ASSERT(currentStats.isValid());

        partialWeightedMaximumAPosteriorStep(vmm, mask, currentStats, previousStats, cfg);
        currentEMIteration++;
        // TODO: Add convergence check
        if (currentEMIteration > 1)
        {
            float relLogLikelihoodDifference = std::fabs(logLikelihood - previousLogLikelihood) * inv_previousLogLikelihood;
            if (relLogLikelihoodDifference < cfg.convergenceThreshold)
            {
                converged = true;
            }
            // std::cout << "logLikelihood:" <<  logLikelihood << "\t previousLogLikelihood: "<< previousLogLikelihood  << "\t relLogLikelihoodDifference: " <<
            // relLogLikelihoodDifference << std::endl;
            previousLogLikelihood = logLikelihood;
            inv_previousLogLikelihood = 1.0f / std::fabs(logLikelihood);
        }
    }
    }

    SYNC; // wait for all threads to complete. TODO necessary?

    SINGLE {
    // The merged sufficient stats from the last iteration are now the new previous/prior stats
    previousStats += currentStats;

    fitStats.numSamples = numSamples;
    fitStats.numIterations = currentEMIteration;
    fitStats.summedWeightedLogLikelihood = previousLogLikelihood;
    }
    // std::cout << "converged:" <<  currentEMIteration << std::endl;
}

template <class TVMMDistribution>
KERNEL_FUNCTION float ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::weightedExpectationStep(VMM &vmm, SufficientStatistics &stats, UnassignedSamplesStatistics &unassignedStats,
                                                                                              const SampleData *samples, const size_t numSamples) const
{
    SHARED float summedWeightedLogLikelihood;
    SINGLE {
    unassignedStats.clear();
    stats.clear(vmm._numComponents);
    stats.numComponents = vmm._numComponents;
    stats.numSamples = numSamples;
    summedWeightedLogLikelihood = 0.f;
    }

    SYNC; // wait for shared memory writes

    const int cnt = (stats.numComponents + VectorSize - 1) / VectorSize;

    const vfloat zero(0.f);
    constexpr static int BlockDim = VMM::Kernel::BlockDim;
    Accumulator<0,               float,                BlockDim> accUW(unassignedStats.sumOfUnassignedWeights, 0.f);
    Accumulator<accUW.OffsetEnd, Vector3,              BlockDim> accUD(unassignedStats.sumUnassignedWeightedDirections, Vector3(0.f));
    Accumulator<accUD.OffsetEnd, float,                BlockDim> accLL(summedWeightedLogLikelihood, 0.f);
    Accumulator<accLL.OffsetEnd, vfloat,               BlockDim, VMM::NumVectors> accAW(stats.sumOfWeightedStats, zero);
    Accumulator<accAW.OffsetEnd, embree::Vec3<vfloat>, BlockDim, VMM::NumVectors> accAD(stats.sumOfWeightedDirections, {zero, zero, zero});

    #ifdef __CUDACC__
    //PrintConst<accAD.OffsetEnd> p;
    #endif

    // TODO evaluate softAssignment on demand to avoid local memory / register pressure
    typename VMM::SoftAssignment softAssign;

    FOREACH_COALESCED(n, numSamples)
    {
        const bool valid = n < numSamples;
        SampleData sampleData = {};
        if (valid) sampleData = samples[n];
        const vfloat sampleWeight = sampleData.weight;
        pgl_vec3f direction = sampleData.direction;
        const Vector3 sampleDirection(direction.x, direction.y, direction.z);
        const embree::Vec3<vfloat > sampleDirectionSIMD(sampleDirection);

        const bool assign = valid && vmm.softAssignment(sampleDirection, softAssign);
        const bool validAssign = valid && assign;
        const bool validUnassign = valid && !assign;

        // Calculating the soft assignment of the current sample direction for all mixture components.
        // We collect the sufficient statistics for all sample directions not covered by any mixture component.
        accUW.accumulate(validUnassign ? sampleData.weight : 0);
        accUD.accumulate(validUnassign ? sampleDirection * sampleData.weight : Vector3(0));

#ifndef __CUDACC__
        if (!assign) continue;
#endif

        // Updating the sumed loglikelihood
        accLL.accumulate(validAssign ? sampleData.weight * embree::log(softAssign.pdf) : 0);

        for (size_t k = 0; k < cnt; k++)
        {
            auto zero = embree::Vec3<vfloat>({0.f}, vfloat(0.f), vfloat(0.f));
            accAD.accumulate(k, validAssign ? sampleDirectionSIMD * softAssign.assignments[k] * sampleWeight : zero);
            accAW.accumulate(k, validAssign ? softAssign.assignments[k] * sampleWeight : 0);

            //OPENPGL_ASSERT(embree::isvalid(stats.sumOfWeightedDirections[k].x));
            //OPENPGL_ASSERT(embree::isvalid(stats.sumOfWeightedDirections[k].y));
            //OPENPGL_ASSERT(embree::isvalid(stats.sumOfWeightedDirections[k].z));
            //OPENPGL_ASSERT(embree::isvalid(stats.sumOfWeightedStats[k]));
        }
    }

    SYNC; // wait for all threads to finish before resolving variables

    accUW.resolve();
    accUD.resolve();
    accLL.resolve();
    accAW.resolve();
    accAD.resolve();

    SINGLE OPENPGL_ASSERT(stats.isValid());

    return summedWeightedLogLikelihood;
}

template <class TVMMDistribution>
KERNEL_FUNCTION void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::estimateMAPWeights(VMM &vmm, const SufficientStatistics &currentStats,
                                                                                        const SufficientStatistics &previousStats, const float &_weightPrior) const
{
    const int cnt = (vmm._numComponents + VectorSize - 1) / VectorSize;

    const size_t numComponents = vmm._numComponents;

    const vfloat weightPrior(_weightPrior);

    const vfloat numSamples = currentStats.numSamples + previousStats.numSamples;
    // const vfloat<VectorSize> numSamples = currentStats.numSamples + previousStats.overallNumSamples;

    for (size_t k = 0; k < cnt; k++)
    {
        //_sumWeights += currentStats.sumOfWeightedStats[k];
        vfloat weight = (currentStats.sumOfWeightedStats[k] + previousStats.sumOfWeightedStats[k]);
        weight = (weightPrior + (weight)) / ((weightPrior * numComponents) + numSamples);
        // vfloat<VectorSize>  weight = ( currentStats.sumOfWeightedStats[k]/* + previousStats.sumOfWeightedStats[k]*/ ) / ( sumWeights );
        // weight = ( weightPrior + ( weight * numSamples ) ) / (( weightPrior * numComponents ) + numSamples );
        vmm._weights[k] = weight;
    }

    // TODO: find better more efficient way
    // Ensuring that the weights for unused SIMD vector entries are zero
    if (vmm._numComponents % VectorSize > 0)
    {
        for (size_t i = vmm._numComponents % VectorSize; i < VectorSize; i++)
        {
            get(vmm._weights[cnt - 1], i) = 0.0f;
        }
    }
}

template <class TVMMDistribution>
KERNEL_FUNCTION void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::estimateMAPMeanDirectionAndConcentration(VMM &vmm, const SufficientStatistics &currentStats,
                                                                                                              const SufficientStatistics &previousStats,
                                                                                                              const Configuration &cfg) const
{
    const vfloat currentNumSamples = currentStats.numSamples;
    const vfloat previousNumSamples = previousStats.numSamples;
    const vfloat numSamples = currentNumSamples + previousNumSamples;
    const vfloat overallNumSamples = currentStats.numSamples + previousStats.overallNumSamples;

    const vfloat currentEstimationWeight = currentNumSamples / numSamples;
    const vfloat previousEstimationWeight = 1.0f - currentEstimationWeight;

    const vfloat meanCosinePrior = cfg.meanCosinePrior;
    const vfloat meanCosinePriorStrength = cfg.meanCosinePriorStrength;
    const vfloat maxMeanCosine = cfg.maxMeanCosine;
    const int cnt = (vmm._numComponents + VectorSize - 1) / VectorSize;
    const int rem = vmm._numComponents % VectorSize;

    for (size_t k = 0; k < cnt; k++)
    {
        // const vfloat<VectorSize> partialNumSamples = vmm._weights[k] * numSamples;
        const vfloat partialNumSamples = vmm._weights[k] * overallNumSamples;
        embree::Vec3<vfloat > currentMeanDirection;
        currentMeanDirection.x = select(currentStats.sumOfWeightedStats[k] > 0.0f, currentStats.sumOfWeightedDirections[k].x / currentStats.sumOfWeightedStats[k], 0.0f);
        currentMeanDirection.y = select(currentStats.sumOfWeightedStats[k] > 0.0f, currentStats.sumOfWeightedDirections[k].y / currentStats.sumOfWeightedStats[k], 0.0f);
        currentMeanDirection.z = select(currentStats.sumOfWeightedStats[k] > 0.0f, currentStats.sumOfWeightedDirections[k].z / currentStats.sumOfWeightedStats[k], 0.0f);

        // TODO: find a better design to precompute the previousMeanDirection
        embree::Vec3<vfloat > previousMeanDirection;
        previousMeanDirection.x = select(previousStats.sumOfWeightedStats[k] > 0.0f, previousStats.sumOfWeightedDirections[k].x / previousStats.sumOfWeightedStats[k], 0.0f);
        previousMeanDirection.y = select(previousStats.sumOfWeightedStats[k] > 0.0f, previousStats.sumOfWeightedDirections[k].y / previousStats.sumOfWeightedStats[k], 0.0f);
        previousMeanDirection.z = select(previousStats.sumOfWeightedStats[k] > 0.0f, previousStats.sumOfWeightedDirections[k].z / previousStats.sumOfWeightedStats[k], 0.0f);

        embree::Vec3<vfloat > meanDirection = currentMeanDirection * currentEstimationWeight + previousMeanDirection * previousEstimationWeight;

        vfloat meanCosine = length(meanDirection);

        vmm._meanDirections[k].x = select(meanCosine > 0.0f, meanDirection.x / meanCosine, vmm._meanDirections[k].x);
        vmm._meanDirections[k].y = select(meanCosine > 0.0f, meanDirection.y / meanCosine, vmm._meanDirections[k].y);
        vmm._meanDirections[k].z = select(meanCosine > 0.0f, meanDirection.z / meanCosine, vmm._meanDirections[k].z);

        meanCosine = (meanCosinePrior * meanCosinePriorStrength + meanCosine * partialNumSamples) / (meanCosinePriorStrength + partialNumSamples);

        meanCosine = embree::min(maxMeanCosine, meanCosine);
        vmm._meanCosines[k] = meanCosine;
        vmm._kappas[k] = MeanCosineToKappa<vfloat >(meanCosine);
    }

    // TODO: find better more efficient way
    if (rem > 0)
    {
        for (size_t i = rem; i < VectorSize; i++)
        {
            get(vmm._meanDirections[cnt - 1].x, i) = 0.0f;
            get(vmm._meanDirections[cnt - 1].y, i) = 0.0f;
            get(vmm._meanDirections[cnt - 1].z, i) = 1.0f;

            get(vmm._meanCosines[cnt - 1], i) = 0.0f;
            get(vmm._kappas[cnt - 1], i) = 0.0f;

            get(vmm._normalizations[cnt - 1], i) = ONE_OVER_FOUR_PI;
            get(vmm._eMinus2Kappa[cnt - 1], i) = 1.0f;
        }
    }

    vmm._calculateNormalization();
}

template <class TVMMDistribution>
KERNEL_FUNCTION void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::weightedMaximumAPosteriorStep(VMM &vmm,
                                                                                                   const SufficientStatistics &currentStats, const SufficientStatistics &previousStats,
                                                                                                   const Configuration &cfg) const
{
    // Estimating components weights
    estimateMAPWeights(vmm, currentStats, previousStats, cfg.weightPrior);

    // Estimating mean and concentration
    estimateMAPMeanDirectionAndConcentration(vmm, currentStats, previousStats, cfg);
}

template <class TVMMDistribution>
KERNEL_FUNCTION void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::partialWeightedMaximumAPosteriorStep(VMM &vmm, const PartialFittingMask &mask,
                                                                                                          SufficientStatistics &currentStats, SufficientStatistics &previousStats,
                                                                                                          const Configuration &cfg) const
{
    // Estimating components weights
    estimatePartialMAPWeights(vmm, mask, currentStats, previousStats, cfg.weightPrior);

    // Estimating mean and concentration
    estimatePartialMAPMeanDirectionAndConcentration(vmm, mask, currentStats, previousStats, cfg);
}

template <class TVMMDistribution>
KERNEL_FUNCTION void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::estimatePartialMAPWeights(VMM &vmm, const PartialFittingMask &mask, SufficientStatistics &currentStats,
                                                                                               SufficientStatistics &previousStats, const float &_weightPrior) const
{
    const vfloat zeros(0.0f);
    const int cnt = (vmm._numComponents + VectorSize - 1) / VectorSize;

    const size_t numComponents = vmm._numComponents;

    const vfloat weightPrior(_weightPrior);

    const vfloat numSamples = currentStats.numSamples + previousStats.numSamples;
    // const vfloat<VectorSize> numSamples = currentStats.numSamples + previousStats.overallNumSamples;

    vfloat sumWeights(0.0f);
    vfloat sumPartialWeights(0.0f);

    for (size_t k = 0; k < cnt; k++)
    {
        //_sumWeights += currentStats.sumOfWeightedStats[k];
        vfloat weight = (currentStats.sumOfWeightedStats[k] + previousStats.sumOfWeightedStats[k]);
        weight = (weightPrior + (weight)) / ((weightPrior * numComponents) + numSamples);
        // vfloat<VectorSize>  weight = ( currentStats.sumOfWeightedStats[k]/* + previousStats.sumOfWeightedStats[k]*/ ) / ( sumWeights );
        // weight = ( weightPrior + ( weight * numSamples ) ) / (( weightPrior * numComponents ) + numSamples );

        sumPartialWeights += select(mask.mask[k], weight, zeros);
        sumWeights += select(mask.mask[k], zeros, vmm._weights[k]);

        vmm._weights[k] = select(mask.mask[k], weight, vmm._weights[k]);
    }

    vfloat inv_sumPartialWeights = 1.0f / embree::reduce_add(sumPartialWeights);
    inv_sumPartialWeights *= 1.0f - embree::reduce_add(sumWeights);
    for (size_t k = 0; k < cnt; k++)
    {
        vmm._weights[k] = select(mask.mask[k], vmm._weights[k] * inv_sumPartialWeights, vmm._weights[k]);
    }

    // TODO: find better more efficient way
    if (vmm._numComponents % VectorSize > 0)
    {
        for (size_t i = vmm._numComponents % VectorSize; i < VectorSize; i++)
        {
            get(vmm._weights[cnt - 1], i) = 0.0f;
        }
    }
}

template <class TVMMDistribution>
KERNEL_FUNCTION void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::estimatePartialMAPMeanDirectionAndConcentration(VMM &vmm, const PartialFittingMask &mask,
                                                                                                                     SufficientStatistics &currentStats,
                                                                                                                     SufficientStatistics &previousStats,
                                                                                                                     const Configuration &cfg) const
{
    const vfloat currentNumSamples = currentStats.numSamples;
    const vfloat previousNumSamples = previousStats.numSamples;
    const vfloat numSamples = currentNumSamples + previousNumSamples;
    const vfloat overallNumSamples = currentStats.numSamples + previousStats.overallNumSamples;

    const vfloat currentEstimationWeight = currentNumSamples / numSamples;
    const vfloat previousEstimationWeight = 1.0f - currentEstimationWeight;

    const vfloat meanCosinePrior = cfg.meanCosinePrior;
    const vfloat meanCosinePriorStrength = cfg.meanCosinePriorStrength;
    const vfloat maxMeanCosine = cfg.maxMeanCosine;
    const int cnt = (vmm._numComponents + VectorSize - 1) / VectorSize;
    const int rem = vmm._numComponents % VectorSize;

    for (size_t k = 0; k < cnt; k++)
    {
        // const vfloat<VectorSize> partialNumSamples = vmm._weights[k] * numSamples;
        const vfloat partialNumSamples = vmm._weights[k] * overallNumSamples;
        embree::Vec3<vfloat > currentMeanDirection;
        currentMeanDirection.x = select(currentStats.sumOfWeightedStats[k] > 0.0f, currentStats.sumOfWeightedDirections[k].x / currentStats.sumOfWeightedStats[k], 0.0f);
        currentMeanDirection.y = select(currentStats.sumOfWeightedStats[k] > 0.0f, currentStats.sumOfWeightedDirections[k].y / currentStats.sumOfWeightedStats[k], 0.0f);
        currentMeanDirection.z = select(currentStats.sumOfWeightedStats[k] > 0.0f, currentStats.sumOfWeightedDirections[k].z / currentStats.sumOfWeightedStats[k], 0.0f);

        // TODO: find a better design to precompute the previousMeanDirection
        embree::Vec3<vfloat > previousMeanDirection;
        previousMeanDirection.x = select(previousStats.sumOfWeightedStats[k] > 0.0f, previousStats.sumOfWeightedDirections[k].x / previousStats.sumOfWeightedStats[k], 0.0f);
        previousMeanDirection.y = select(previousStats.sumOfWeightedStats[k] > 0.0f, previousStats.sumOfWeightedDirections[k].y / previousStats.sumOfWeightedStats[k], 0.0f);
        previousMeanDirection.z = select(previousStats.sumOfWeightedStats[k] > 0.0f, previousStats.sumOfWeightedDirections[k].z / previousStats.sumOfWeightedStats[k], 0.0f);

        embree::Vec3<vfloat > meanDirection = currentMeanDirection * currentEstimationWeight + previousMeanDirection * previousEstimationWeight;

        vfloat meanCosine = embree::length(meanDirection);

        vmm._meanDirections[k].x = select(mask.mask[k], select(meanCosine > 0.0f, meanDirection.x / meanCosine, vmm._meanDirections[k].x), vmm._meanDirections[k].x);
        vmm._meanDirections[k].y = select(mask.mask[k], select(meanCosine > 0.0f, meanDirection.y / meanCosine, vmm._meanDirections[k].y), vmm._meanDirections[k].y);
        vmm._meanDirections[k].z = select(mask.mask[k], select(meanCosine > 0.0f, meanDirection.z / meanCosine, vmm._meanDirections[k].z), vmm._meanDirections[k].z);

        meanCosine = (meanCosinePrior * meanCosinePriorStrength + meanCosine * partialNumSamples) / (meanCosinePriorStrength + partialNumSamples);

        meanCosine = embree::min(maxMeanCosine, meanCosine);
        vmm._meanCosines[k] = select(mask.mask[k], meanCosine, vmm._meanCosines[k]);
        vmm._kappas[k] = select(mask.mask[k], MeanCosineToKappa<vfloat >(meanCosine), vmm._kappas[k]);
    }

    // TODO: find better more efficient way
    if (rem > 0)
    {
        for (size_t i = rem; i < VectorSize; i++)
        {
            get(vmm._meanDirections[cnt - 1].x, i) = 0.0f;
            get(vmm._meanDirections[cnt - 1].y, i) = 0.0f;
            get(vmm._meanDirections[cnt - 1].z, i) = 1.0f;

            get(vmm._meanCosines[cnt - 1], i) = 0.0f;
            get(vmm._kappas[cnt - 1], i) = 0.0f;
        }
    }

    vmm._calculateNormalization();
}

template <class TVMMDistribution>
KERNEL_FUNCTION void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::reprojectSample(openpgl::SampleData &sample, const openpgl::Point3 &pivotPoint, const float minDistance) const
{
    if (std::isinf(sample.distance))
    {
        sample.position.x = pivotPoint[0];
        sample.position.y = pivotPoint[1];
        sample.position.z = pivotPoint[2];
        return;
    }
    else if (!(sample.distance > 0.0f))
    {
        return;
    }

    const float distance = fmaxf(minDistance, sample.distance);
    const openpgl::Point3 samplePosition(sample.position.x, sample.position.y, sample.position.z);
    pgl_vec3f direction = sample.direction;
    const openpgl::Vector3 sampleDirection(direction.x, direction.y, direction.z);
    const openpgl::Point3 originPosition = samplePosition + sampleDirection * distance;
    openpgl::Vector3 newDirection = originPosition - pivotPoint;
    const float newDistance = embree::length(newDirection);
    sample.position.x = pivotPoint[0];
    sample.position.y = pivotPoint[1];
    sample.position.z = pivotPoint[2];
    newDirection = newDistance > FLT_EPSILON ? newDirection / newDistance : sampleDirection;
    sample.distance = newDistance > FLT_EPSILON ? newDistance : distance;
    pgl_vec3f qdirection = {newDirection[0], newDirection[1], newDirection[2]};
    sample.direction = qdirection;
}

template <class TVMMDistribution>
KERNEL_FUNCTION void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::prepareSamples(SampleData *samples, const size_t numSamples, const SampleStatistics &sampleStatistics,
                                                                                    const Configuration &cfg) const
{
    if (TVMMDistribution::ParallaxCompensation)
    {
        openpgl::Vector3 sampleVariance = sampleStatistics.getVariance();
        float norm = sampleVariance.x * sampleVariance.x + sampleVariance.y * sampleVariance.y + sampleVariance.z * sampleVariance.z;
        norm = std::max(FLT_EPSILON, norm);
        float minDistance = std::sqrt(norm);
        minDistance = 3.f * 3.f * std::sqrt(minDistance);
        OPENPGL_ASSERT(embree::isvalid(sampleVariance));
        OPENPGL_ASSERT(embree::isvalid(minDistance));
        FOREACH(n, 0, numSamples)
        {
            OPENPGL_ASSERT(openpgl::isValid(samples[n]));
            reprojectSample(samples[n], sampleStatistics.getMean(), minDistance);
        }
    }
}

template <class TVMMDistribution>
KERNEL_FUNCTION void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::initComponentDistances(VMM &vmm, SufficientStatistics &sufficientStats, const SampleData *samples,
                                                                                            const size_t numSamples) const
{
    SYNC;

    OPENPGL_ASSERT(vmm.getNumComponents() == sufficientStats.getNumComponents());

    SHARED vfloat batchDistances[VMM::NumVectors];
    SHARED vfloat batchSumWeights[VMM::NumVectors];

    const vfloat zeros(0.0f);

    const int cnt = (vmm._numComponents + VectorSize - 1) / VectorSize;
    const int rem = vmm._numComponents % VectorSize;

    SINGLE {
    for (size_t k = 0; k < cnt; k++)
    {
        batchDistances[k] = zeros;
        batchSumWeights[k] = zeros;
    }
    }

    typename VMM::SoftAssignment softAssign;
    float sampleDistance;
    vfloat weights;

    const vfloat zero(0.f);
    constexpr static int BlockDim = VMM::Kernel::BlockDim;
    Accumulator<0,              vfloat, BlockDim, VMM::NumVectors> accD(batchDistances,  zero);
    Accumulator<accD.OffsetEnd, vfloat, BlockDim, VMM::NumVectors> accW(batchSumWeights, zero);

    FOREACH_COALESCED(n, numSamples)
    {
        bool valid = n < numSamples;
        SampleData sample = {};
        if (valid) sample = samples[n];

#ifdef USE_HARMONIC_MEAN
        sampleDistance = embree::rcp(sample.distance);
#else
        sampleDistance = sample.distance;
#endif
        pgl_vec3f direction = sample.direction;
        const Vector3 sampleDirection(direction.x, direction.y, direction.z);

        valid = valid && vmm.softAssignment(sampleDirection, softAssign);
        for (size_t k = 0; k < cnt; k++)
        {
            weights =
                select(vmm._weights[k] > FLT_EPSILON, sample.weight * softAssign.assignments[k] * ((softAssign.assignments[k] * softAssign.pdf) / vmm._weights[k]), zeros);
            accD.accumulate(k, valid ? weights * sampleDistance : zero);
            accW.accumulate(k, valid ? weights : zero);
        }
    }

    SYNC;

    accD.resolve();
    accW.resolve();

    SINGLE {

    //OPENPGL_ASSERT(embree::isvalid(weights));
    for (size_t k = 0; k < cnt; k++)
    {
        OPENPGL_ASSERT(embree::isvalid(batchDistances[k]));
        OPENPGL_ASSERT(embree::isvalid(batchSumWeights[k]));
    }

    for (size_t k = 0; k < cnt; k++)
    {
#ifdef USE_HARMONIC_MEAN
        sufficientStats.sumOfDistanceWeightes[k] = batchSumWeights[k];
        vmm._distances[k] = select(batchDistances[k] > FLT_EPSILON, sufficientStats.sumOfDistanceWeightes[k] / batchDistances[k], zeros);
#else
        sufficientStats.sumOfDistanceWeightes[k] = batchSumWeights[k];
        vmm._distances[k] = batchDistances[k] / sufficientStats.sumOfDistanceWeightes[k];
#endif
    }

    if (rem > 0)
    {
        for (size_t i = rem; i < VectorSize; i++)
        {
            get(vmm._distances[cnt - 1], i) = 0.0f;
            get(sufficientStats.sumOfDistanceWeightes[cnt - 1], i) = 0.0f;
        }
    }

}
}

template <class TVMMDistribution>
KERNEL_FUNCTION void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::updateComponentDistances(VMM &vmm, SufficientStatistics &sufficientStats, const SampleData *samples,
                                                                                              const size_t numSamples) const
{
    SYNC;

    SINGLE OPENPGL_ASSERT(vmm.getNumComponents() == sufficientStats.getNumComponents());

    SHARED vfloat batchDistances[VMM::NumVectors];
    SHARED vfloat batchSumWeights[VMM::NumVectors];

    const vfloat zeros(0.0f);
    const int cnt = (vmm._numComponents + VectorSize - 1) / VectorSize;
    const int rem = vmm._numComponents % VectorSize;

    SINGLE {
    for (size_t k = 0; k < cnt; k++)
    {
        batchDistances[k] = zeros;
        batchSumWeights[k] = zeros;
    }
    }

    const vfloat zero(0.f);
    constexpr static int BlockDim = VMM::Kernel::BlockDim;
    Accumulator<0,              vfloat, BlockDim, VMM::NumVectors> accD(batchDistances,  zero);
    Accumulator<accD.OffsetEnd, vfloat, BlockDim, VMM::NumVectors> accW(batchSumWeights, zero);

    typename VMM::SoftAssignment softAssign;
    float sampleDistance;
    vfloat weights;

    FOREACH_COALESCED(n, numSamples)
    {
        bool valid = n < numSamples;
        SampleData sample = {};
        if (valid) sample = samples[n];

        if (valid) OPENPGL_ASSERT(embree::isvalid(sample.distance));
        if (valid) OPENPGL_ASSERT(sample.distance > 0);
#ifdef USE_HARMONIC_MEAN
        sampleDistance = embree::rcp(sample.distance);
#else
        sampleDistance = sample.distance;
#endif
        pgl_vec3f direction = samples[n].direction;
        const Vector3 sampleDirection(direction.x, direction.y, direction.z);
        valid = valid && vmm.softAssignment(sampleDirection, softAssign);
        for (size_t k = 0; k < cnt; k++)
        {
            weights = samples[n].weight * softAssign.assignments[k] * ((softAssign.assignments[k] * softAssign.pdf) / vmm._weights[k]);
            accD.accumulate(k, valid ? weights * sampleDistance : 0);
            accW.accumulate(k, valid ? weights : 0);
        }
    }

    SYNC;

    accD.resolve();
    accW.resolve();

    SINGLE {

    for (size_t k = 0; k < cnt; k++)
    {
#ifdef USE_HARMONIC_MEAN
        // vfloat sumInverseDistances = (sufficientStats.sumOfDistanceWeightes[k] / vmm._distances[k]) + batchDistances[k];
        vfloat sumInverseDistances = batchDistances[k];
        sumInverseDistances += select(vmm._distances[k] > 0.0f, (sufficientStats.sumOfDistanceWeightes[k] / vmm._distances[k]), vfloat(0.0f));
        sufficientStats.sumOfDistanceWeightes[k] += batchSumWeights[k];
        vmm._distances[k] = sufficientStats.sumOfDistanceWeightes[k] / sumInverseDistances;
#else
        const vfloat sumInverseDistances = (sufficientStats.sumOfDistanceWeightes[k] * vmm._distances[k]) + batchDistances[k];
        sufficientStats.sumOfDistanceWeightes[k] += batchSumWeights[k];
        vmm._distances[k] = sumInverseDistances / sufficientStats.sumOfDistanceWeightes[k];
#endif
    }

    if (rem > 0)
    {
        for (size_t i = rem; i < VectorSize; i++)
        {
            get(vmm._distances[cnt - 1], i) = 0.0f;
            get(sufficientStats.sumOfDistanceWeightes[cnt - 1], i) = 0.0f;
        }
    }
    }
}

template <class TVMMDistribution>
KERNEL_FUNCTION void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::updateOutgoingRadiance(
    VMM &vmm, const SampleData *samples, const size_t numSamples) {
    SYNC;

    constexpr static int BlockDim = VMM::Kernel::BlockDim;
    Accumulator<0,                 float, BlockDim> accSORX(vmm.sumOutgoingRadiance.x, 0);
    Accumulator<accSORX.OffsetEnd, float, BlockDim> accSORY(vmm.sumOutgoingRadiance.y, 0);
    Accumulator<accSORY.OffsetEnd, float, BlockDim> accSORZ(vmm.sumOutgoingRadiance.z, 0);

    FOREACH_COALESCED(n, numSamples)
    {
        bool valid = n < numSamples;
        SampleData sample = {};
        if (valid) sample = samples[n];

        accSORX.accumulate(valid ? sample.outgoing.x : 0);
        accSORY.accumulate(valid ? sample.outgoing.y : 0);
        accSORZ.accumulate(valid ? sample.outgoing.z : 0);
    }

    SYNC;

    accSORX.resolve();
    accSORY.resolve();
    accSORZ.resolve();

    SINGLE vmm.numOutgoingRadiance += numSamples;
}


#ifdef OPENPGL_RADIANCE_CACHES
template <class TVMMDistribution>
KERNEL_FUNCTION void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::updateFluenceEstimate(VMM &vmm, const SampleData *samples, const size_t numSamples,
                                                                                           const size_t numZeroValueSamples, const SampleStatistics &sampleStatistics) const
{
#ifdef MC_ESTIMATE_INCOMING_RADIANCE  // calcualting fluence and the RGB per lob estiamtions using the MC samples
    if (numSamples == 0)
    {
        return;
    }

    const int cnt = (vmm._numComponents + VectorSize - 1) / VectorSize;
    const int rem = vmm._numComponents % VectorSize;

    const vfloat zeros(0.0f);

    // float sumFluence {0.f};
    Vector3 sumFluenceRGB{0.f, 0.f, 0.f};
    Vector3 sumFluenceRGBWithMIS{0.f, 0.f, 0.f};

    embree::Vec3<vfloat > sumFluenceRGBWeights[VMM::NumVectors];
    embree::Vec3<vfloat > sumFluenceRGBWeightsWithMIS[VMM::NumVectors];
    typename VMM::SoftAssignment softAssign;

    for (size_t k = 0; k < cnt; k++)
    {
        sumFluenceRGBWeights[k].x = zeros;
        sumFluenceRGBWeights[k].y = zeros;
        sumFluenceRGBWeights[k].z = zeros;

        sumFluenceRGBWeightsWithMIS[k].x = zeros;
        sumFluenceRGBWeightsWithMIS[k].y = zeros;
        sumFluenceRGBWeightsWithMIS[k].z = zeros;
    }

    for (size_t n = 0; n < numSamples; n++)
    {
        // sumFluence += samples[n].weight;
        pgl_vec3f direction = samples[n].direction;
        const Vector3 sampleDirection(direction.x, direction.y, direction.z);
        pgl_vec3f color = samples[n].radianceIn;
        Vector3 radianceIn(color.x, color.y, color.z);
        radianceIn /= samples[n].pdf;

        Vector3 radianceInNoMIS = isDirectLight(samples[n]) ? radianceIn / samples[n].radianceInMISWeight : radianceIn;

        sumFluenceRGB += radianceInNoMIS;
        sumFluenceRGBWithMIS += radianceIn;
        if (vmm.softAssignment(sampleDirection, softAssign))
        {
            for (size_t k = 0; k < cnt; k++)
            {
                sumFluenceRGBWeights[k].x += radianceInNoMIS.x * softAssign.assignments[k];
                sumFluenceRGBWeights[k].y += radianceInNoMIS.y * softAssign.assignments[k];
                sumFluenceRGBWeights[k].z += radianceInNoMIS.z * softAssign.assignments[k];

                sumFluenceRGBWeightsWithMIS[k].x += radianceIn.x * softAssign.assignments[k];
                sumFluenceRGBWeightsWithMIS[k].y += radianceIn.y * softAssign.assignments[k];
                sumFluenceRGBWeightsWithMIS[k].z += radianceIn.z * softAssign.assignments[k];
            }
        }
    }

    const float oldNumFluenceSamples = vmm._numFluenceSamples;
    const float newNumFluenceSamples = (oldNumFluenceSamples + float(numSamples + numZeroValueSamples));

    if (rem > 0)
    {
        for (size_t i = rem; i < VectorSize; i++)
        {
            sumFluenceRGBWeights[cnt - 1].x[i] = 0.0f;
            sumFluenceRGBWeights[cnt - 1].y[i] = 0.0f;
            sumFluenceRGBWeights[cnt - 1].z[i] = 0.0f;

            sumFluenceRGBWeightsWithMIS[cnt - 1].x[i] = 0.0f;
            sumFluenceRGBWeightsWithMIS[cnt - 1].y[i] = 0.0f;
            sumFluenceRGBWeightsWithMIS[cnt - 1].z[i] = 0.0f;
        }
    }

    // TODO: switch to numerical more stable version
    for (size_t k = 0; k < cnt; k++)
    {
        vmm._fluenceRGBWeights[k].x = ((vmm._fluenceRGBWeights[k].x * oldNumFluenceSamples) + sumFluenceRGBWeights[k].x) / newNumFluenceSamples;
        vmm._fluenceRGBWeights[k].y = ((vmm._fluenceRGBWeights[k].y * oldNumFluenceSamples) + sumFluenceRGBWeights[k].y) / newNumFluenceSamples;
        vmm._fluenceRGBWeights[k].z = ((vmm._fluenceRGBWeights[k].z * oldNumFluenceSamples) + sumFluenceRGBWeights[k].z) / newNumFluenceSamples;

        vmm._fluenceRGBWeightsWithMIS[k].x = ((vmm._fluenceRGBWeightsWithMIS[k].x * oldNumFluenceSamples) + sumFluenceRGBWeightsWithMIS[k].x) / newNumFluenceSamples;
        vmm._fluenceRGBWeightsWithMIS[k].y = ((vmm._fluenceRGBWeightsWithMIS[k].y * oldNumFluenceSamples) + sumFluenceRGBWeightsWithMIS[k].y) / newNumFluenceSamples;
        vmm._fluenceRGBWeightsWithMIS[k].z = ((vmm._fluenceRGBWeightsWithMIS[k].z * oldNumFluenceSamples) + sumFluenceRGBWeightsWithMIS[k].z) / newNumFluenceSamples;
    }

    vmm._fluenceRGB = ((vmm._fluenceRGB * oldNumFluenceSamples) + sumFluenceRGB) / newNumFluenceSamples;
    vmm._fluenceRGBWithMIS = ((vmm._fluenceRGBWithMIS * oldNumFluenceSamples) + sumFluenceRGBWithMIS) / newNumFluenceSamples;
    // vmm._fluence = ((vmm._fluence * oldNumFluenceSamples) + sumFluence) / newNumFluenceSamples;
    vmm._numFluenceSamples = newNumFluenceSamples;
#else  // calcualting fluence and the RGB per lob estiamtions using the soft assigns counter to average the incoming radiance per lobe (getting rid of the PDF dependency) TODO:
       // maybe drop this code
    const vfloat zeros(0.0f);
    const vfloat ones(1.0f);
    const int cnt = (vmm._numComponents + VectorSize - 1) / VectorSize;

    if (numSamples == 0)
    {
        return;
    }

    vfloat sumPdfs[VMM::NumVectors];
    vfloat pdfs(1.0f);
    embree::Vec3<vfloat > sumFluenceRGBWeights[VMM::NumVectors];
    float sumFluence{0.f};
    Vector3 sumFluenceRGB{0.f, 0.f, 0.f};
    Vector3 sumFluenceRGBMC{0.f, 0.f, 0.f};
    typename VMM::SoftAssignment softAssign;

    for (int k = 0; k < cnt; k++)
    {
        sumPdfs[k] = zeros;
        sumFluenceRGBWeights[k].x = zeros;
        sumFluenceRGBWeights[k].y = zeros;
        sumFluenceRGBWeights[k].z = zeros;
    }

    for (size_t n = 0; n < numSamples; n++)
    {
        const Vector3 sampleDirection(samples[n].direction.x, samples[n].direction.y, samples[n].direction.z);
        embree::Vec3<vfloat > sampleDirectionVec(sampleDirection[0], sampleDirection[1], sampleDirection[2]);

        Vector3 radianceIn(samples[n].radianceIn.x, samples[n].radianceIn.y, samples[n].radianceIn.z);
        sumFluence += samples[n].weight;
        sumFluenceRGBMC += radianceIn / samples[n].pdf;

        if (vmm.softAssignment(sampleDirection, softAssign))
        {
            for (size_t k = 0; k < cnt; k++)
            {
                const vfloat cosTheta = dot(sampleDirectionVec, vmm._meanDirections[k]);
                const vfloat cosThetaMinusOne = embree::min(cosTheta - ones, zeros);
                OPENPGL_ASSERT(embree::isvalid(pdfs));
                sumFluenceRGBWeights[k].x += radianceIn.x * softAssign.assignments[k] * pdfs;
                OPENPGL_ASSERT(embree::isvalid(sumFluenceRGBWeights[k].x));
                sumFluenceRGBWeights[k].y += radianceIn.y * softAssign.assignments[k] * pdfs;
                OPENPGL_ASSERT(embree::isvalid(sumFluenceRGBWeights[k].y));
                sumFluenceRGBWeights[k].z += radianceIn.z * softAssign.assignments[k] * pdfs;
                OPENPGL_ASSERT(embree::isvalid(sumFluenceRGBWeights[k].z));
                sumPdfs[k] += pdfs;
                OPENPGL_ASSERT(embree::isvalid(sumPdfs[k]));
            }
        }
    }

    for (int k = 0; k < cnt; k++)
    {
        sumFluenceRGBWeights[k].x = select(sumPdfs[k] > 0.0f, sumFluenceRGBWeights[k].x / sumPdfs[k], zeros) * (4.0f * M_PI_F);
        OPENPGL_ASSERT(embree::isvalid(sumFluenceRGBWeights[k].x));
        sumFluenceRGBWeights[k].y = select(sumPdfs[k] > 0.0f, sumFluenceRGBWeights[k].y / sumPdfs[k], zeros) * (4.0f * M_PI_F);
        OPENPGL_ASSERT(embree::isvalid(sumFluenceRGBWeights[k].y));
        sumFluenceRGBWeights[k].z = select(sumPdfs[k] > 0.0f, sumFluenceRGBWeights[k].z / sumPdfs[k], zeros) * (4.0f * M_PI_F);
        OPENPGL_ASSERT(embree::isvalid(sumFluenceRGBWeights[k].z));
        sumFluenceRGB.x += embree::reduce_add(sumFluenceRGBWeights[k].x);
        sumFluenceRGB.y += embree::reduce_add(sumFluenceRGBWeights[k].y);
        sumFluenceRGB.z += embree::reduce_add(sumFluenceRGBWeights[k].z);
    }

    const float oldNumFluenceSamples = vmm._numFluenceSamples;
    const float newNumFluenceSamples = (oldNumFluenceSamples + float(numSamples));

    float alpha = float(numSamples) / (vmm._numFluenceSamples + numSamples);
    for (size_t k = 0; k < cnt; k++)
    {
        vmm._fluenceRGBWeightsWithMIS[k].x = (vmm._fluenceRGBWeightsWithMIS[k].x * (1.f - alpha)) + alpha * sumFluenceRGBWeights[k].x;
        vmm._fluenceRGBWeightsWithMIS[k].y = (vmm._fluenceRGBWeightsWithMIS[k].y * (1.f - alpha)) + alpha * sumFluenceRGBWeights[k].y;
        vmm._fluenceRGBWeightsWithMIS[k].z = (vmm._fluenceRGBWeightsWithMIS[k].z * (1.f - alpha)) + alpha * sumFluenceRGBWeights[k].z;
    }

    vmm._fluenceRGB = (1.f - alpha) * vmm._fluenceRGB + alpha * sumFluenceRGB;
    vmm._fluence = (1.f - alpha) * vmm._fluence + alpha * sumFluence;
    vmm._numFluenceSamples = newNumFluenceSamples;

#endif
}
#endif

template <class TVMMDistribution>
KERNEL_FUNCTION void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::Configuration::init()
{
    maxMeanCosine = KappaToMeanCosine<float>(maxKappa);
}

template <class TVMMDistribution>
KERNEL_FUNCTION void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::Configuration::serialize(std::ostream &stream) const
{
    stream.write(reinterpret_cast<const char *>(&initK), sizeof(size_t));
    stream.write(reinterpret_cast<const char *>(&initKappa), sizeof(float));
    stream.write(reinterpret_cast<const char *>(&maxK), sizeof(size_t));
    stream.write(reinterpret_cast<const char *>(&maxEMIterrations), sizeof(size_t));

    stream.write(reinterpret_cast<const char *>(&maxKappa), sizeof(float));
    stream.write(reinterpret_cast<const char *>(&maxMeanCosine), sizeof(float));
    stream.write(reinterpret_cast<const char *>(&convergenceThreshold), sizeof(float));

    stream.write(reinterpret_cast<const char *>(&weightPrior), sizeof(float));

    stream.write(reinterpret_cast<const char *>(&meanCosinePriorStrength), sizeof(float));
    stream.write(reinterpret_cast<const char *>(&meanCosinePrior), sizeof(float));
}

template <class TVMMDistribution>
KERNEL_FUNCTION void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::Configuration::deserialize(std::istream &stream)
{
    stream.read(reinterpret_cast<char *>(&initK), sizeof(size_t));
    stream.read(reinterpret_cast<char *>(&initKappa), sizeof(float));
    stream.read(reinterpret_cast<char *>(&maxK), sizeof(size_t));
    stream.read(reinterpret_cast<char *>(&maxEMIterrations), sizeof(size_t));

    stream.read(reinterpret_cast<char *>(&maxKappa), sizeof(float));
    stream.read(reinterpret_cast<char *>(&maxMeanCosine), sizeof(float));
    stream.read(reinterpret_cast<char *>(&convergenceThreshold), sizeof(float));

    stream.read(reinterpret_cast<char *>(&weightPrior), sizeof(float));

    stream.read(reinterpret_cast<char *>(&meanCosinePriorStrength), sizeof(float));
    stream.read(reinterpret_cast<char *>(&meanCosinePrior), sizeof(float));
}

template <class TVMMDistribution>
KERNEL_FUNCTION std::string ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::Configuration::toString() const
{
    std::stringstream ss;
    ss << "Configuration:" << std::endl;
    ss << "\tinitKappa = " << initKappa << std::endl;
    ss << "\tmaxComponents = " << maxK << std::endl;
    ss << "\tinitNumComponents = " << initK << std::endl;
    ss << "\tmaxEMIterrations = " << maxEMIterrations << std::endl;
    ss << "\tmaxKappa = " << maxKappa << std::endl;
    ss << "\tmaxMeanCosine = " << maxMeanCosine << std::endl;
    ss << "\tconvergenceThreshold = " << convergenceThreshold << std::endl;
    ss << "\tweightPrior = " << weightPrior << std::endl;
    ss << "\tmeanCosinePriorStrength = " << meanCosinePriorStrength << std::endl;
    ss << "\tmeanCosinePrior = " << meanCosinePrior << std::endl;
    ss << "\tparallaxCompensation = " << TVMMDistribution::ParallaxCompensation << std::endl;
    return ss.str();
}

template <class TVMMDistribution>
KERNEL_FUNCTION void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::PartialFittingMask::resetToFalse()
{
    const vbool vFalse(false);
    for (size_t k = 0; k < ((VMM::MaxComponents + (VectorSize - 1)) / VectorSize); k++)
    {
        mask[k] = vFalse;
    }
}

template <class TVMMDistribution>
KERNEL_FUNCTION void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::PartialFittingMask::resetToTrue(const size_t &numComponents)
{
    const vbool vTrue(true);
    const int cnt = (numComponents + VectorSize - 1) / VectorSize;
    for (size_t k = 0; k < cnt; k++)
    {
        mask[k] = vTrue;
    }

    const div_t tmp = div_(numComponents, VectorSize);
    for (size_t k = tmp.rem; k < VectorSize; k++)
    {
        clear(mask[tmp.quot], k);
    }
}

template <class TVMMDistribution>
KERNEL_FUNCTION void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::PartialFittingMask::setToTrue(const size_t &idx)
{
    const div_t tmp = div_(idx, VectorSize);
    embree::set(mask[tmp.quot], tmp.rem);
}

template <class TVMMDistribution>
KERNEL_FUNCTION void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::PartialFittingMask::setToFalse(const size_t &idx)
{
    const div_t tmp = div_(idx, VectorSize);
    embree::clear(mask[tmp.quot], tmp.rem);
}

template <class TVMMDistribution>
KERNEL_FUNCTION bool ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::PartialFittingMask::get(const size_t &idx) const
{
    const div_t tmp = div_(idx, VectorSize);
    return embree::get(mask[tmp.quot], tmp.rem);
}

template <class TVMMDistribution>
KERNEL_FUNCTION std::string ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::PartialFittingMask::toString() const
{
    std::stringstream ss;
    ss << "PartialFittingMask:" << std::endl;
    for (size_t k = 0; k < VMM::MaxComponents; k++)
    {
        const div_t tmp = div_(k, VectorSize);
        ss << "mask[" << k << "]: " << get(mask[tmp.quot], tmp.rem) << std::endl;
    }
    return ss.str();
}

}
}  // namespace openpgl
