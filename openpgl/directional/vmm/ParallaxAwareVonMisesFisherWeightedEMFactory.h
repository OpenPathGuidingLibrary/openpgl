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

template <class TVMMDistribution>
struct ParallaxAwareVonMisesFisherWeightedEMFactory
{
   public:
    typedef TVMMDistribution Distribution;
    using VMM = TVMMDistribution;

    struct Configuration
    {
        size_t initK{VMM::VectorSize};
        float initKappa{5.0f};

        size_t maxK{VMM::MaxComponents};
        size_t maxEMIterrations{100};

        float maxKappa{OPENPGL_MAX_KAPPA};
        float maxMeanCosine{KappaToMeanCosine<float>(OPENPGL_MAX_KAPPA)};
        float convergenceThreshold{0.005f};

        // MAP prior parameters
        // weight prior
        float weightPrior{0.01f};

        // concentration/meanCosine prior
        float meanCosinePriorStrength{0.2f};
        float meanCosinePrior{0.0f};

        void init();

        void serialize(std::ostream &stream) const;

        void deserialize(std::istream &stream);

        std::string toString() const;

        bool operator==(const Configuration &b) const
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
        embree::vbool<VMM::VectorSize> mask[VMM::NumVectors];

        PartialFittingMask() = default;

        void resetToFalse();
        void resetToTrue(const size_t &numComponents);
        void setToTrue(const size_t &idx);
        void setToFalse(const size_t &idx);
        bool get(const size_t &idx) const;
        std::string toString() const;
    };

    struct SufficientStatistics  //: public WEMVMMFactory::SufficientStatistics
    {
        // FittingStatistics
       public:
        embree::Vec3<embree::vfloat<VMM::VectorSize> > sumOfWeightedDirections[VMM::NumVectors];
        embree::vfloat<VMM::VectorSize> sumOfWeightedStats[VMM::NumVectors];
#ifdef OPENPGL_VSP_GUIDING
        embree::vfloat<VMM::VectorSize> volumeContributionWeights[VMM::NumVectors];
        embree::vfloat<VMM::VectorSize> surfaceContributionWeights[VMM::NumVectors];
        embree::vfloat<VMM::VectorSize> volumeSampleNumberWeights[VMM::NumVectors];
        embree::vfloat<VMM::VectorSize> surfaceSampleNumberWeights[VMM::NumVectors];
#endif
        float sumWeights{0.f};
        float numSamples{0.f};
        float overallNumSamples{0.f};
        size_t numComponents{VMM::MaxComponents};
        bool normalized{false};

        embree::vfloat<VMM::VectorSize> sumOfDistanceWeightes[VMM::NumVectors];

        SufficientStatistics() = default;

        SufficientStatistics &operator+=(const SufficientStatistics &stats);

        void serialize(std::ostream &stream) const;

        void deserialize(std::istream &stream);

        void clear(size_t _numComponents);

        void clearComponentStats(const size_t &idx);

        void clearAll();

        virtual void normalize(const float &_numSamples);

        inline bool isNormalized() const
        {
            return normalized;
        };

        void mergeComponentStats(const size_t &idx0, const size_t &idx1);

        void splitComponentsStats(const size_t &idx0, const size_t &idx1, const Vector3 &meanDirection0, const Vector3 &meanDirection1, const float &meanCosine0,
                                  const float &meanCosine1);

        void swapComponentStats(const size_t &idx0, const size_t &idx1);

        void maskedReplace(const PartialFittingMask &mask, const SufficientStatistics &stats);

        void decay(const float &alpha);

        void applyParallaxShift(const VMM &vmm, const Vector3 shift);

        inline float getNumSamples() const
        {
            return numSamples;
        }

        inline float getSumWeights() const
        {
            return sumWeights;
        }

        inline void setNumComponents(const size_t &numComponents)
        {
            this->numComponents = numComponents;
        }

        inline size_t getNumComponents() const
        {
            return this->numComponents;
        }

        std::string toString() const;

        bool isValid() const;

        bool operator==(const SufficientStatistics &b) const;
    };

    struct UnassignedSamplesStatistics
    {
        float sumOfUnassignedWeights{0.0f};
        Vector3 sumUnassignedWeightedDirections{0.0f, 0.0f, 0.0f};
        void clear();
        bool isValid() const;
    };

   public:
    ParallaxAwareVonMisesFisherWeightedEMFactory();

    void InitUniformVMM(VMM &vmm, const int &numComponents, const float &kappa) const;

    void prepareSamples(SampleData *samples, const size_t numSamples, const SampleStatistics &sampleStatistics, const Configuration &cfg) const;

    void fitMixture(VMM &vmm, SufficientStatistics &stats, const SampleData *samples, const size_t numSamples, const Configuration &cfg, FittingStatistics &fitStats) const;

    void updateMixture(VMM &vmm, SufficientStatistics &previousStats, const SampleData *samples, const size_t numSamples, const Configuration &cfg,
                       FittingStatistics &fitStats) const;

    void partialUpdateMixture(VMM &vmm, PartialFittingMask &mask, SufficientStatistics &previousStats, const SampleData *samples, const size_t numSamples, const Configuration &cfg,
                              FittingStatistics &fitStats) const;

#ifdef OPENPGL_RADIANCE_CACHES
    void updateFluenceEstimate(VMM &vmm, const SampleData *samples, const size_t numSamples, const size_t numZeroValueSamples, const SampleStatistics &sampleStatistics) const;
#endif

    VMM VMMfromSufficientStatistics(const SufficientStatistics &suffStats, const Configuration &cfg) const;

    std::string toString() const
    {
        return "ParallaxAwareVonMisesFisherWeightedEMFactory";
    };

    void initComponentDistances(VMM &vmm, SufficientStatistics &sufficientStats, const SampleData *samples, const size_t numSamples) const;

    void updateComponentDistances(VMM &vmm, SufficientStatistics &sufficientStats, const SampleData *samples, const size_t numSamples) const;

#ifdef OPENPGL_VSP_GUIDING
    void updateVolumeScatterProbability(VMM &vmm, SufficientStatistics &sufficientStats, const SampleData *samples, const size_t numSamples,
                                        const ZeroValueSampleData *zeroValueSamples, const size_t numZeroValueSamples, const bool varianceBased) const;
#endif
   private:
    void _initUniformDirections();

    float weightedExpectationStep(VMM &vmm, SufficientStatistics &stats, UnassignedSamplesStatistics &unassignedStats, const SampleData *samples, const size_t numSamples) const;

    void weightedMaximumAPosteriorStep(VMM &vmm, const SufficientStatistics &previousStats, const SufficientStatistics &currentStats, const Configuration &cfg) const;

    void estimateMAPWeights(VMM &vmm, const SufficientStatistics &currentStats, const SufficientStatistics &previousStats, const float &_weightPrior) const;

    void estimateMAPMeanDirectionAndConcentration(VMM &vmm, const SufficientStatistics &currentStats, const SufficientStatistics &previousStats, const Configuration &cfg) const;

    void partialWeightedMaximumAPosteriorStep(VMM &vmm, const PartialFittingMask &mask, SufficientStatistics &previousStats, SufficientStatistics &currentStats,
                                              const Configuration &cfg) const;

    void estimatePartialMAPWeights(VMM &vmm, const PartialFittingMask &mask, SufficientStatistics &currentStats, SufficientStatistics &previousStats,
                                   const float &_weightPrior) const;

    void estimatePartialMAPMeanDirectionAndConcentration(VMM &vmm, const PartialFittingMask &mask, SufficientStatistics &currentStats, SufficientStatistics &previousStats,
                                                         const Configuration &cfg) const;

    void handleUnassignedSampleStats(UnassignedSamplesStatistics &unassignedStats, VMM &vmm, SufficientStatistics &currentStats, SufficientStatistics &previousStats) const;

    void reprojectSample(openpgl::SampleData &sample, const openpgl::Point3 &pivotPoint, const float minDistance) const;

   private:
    embree::Vec3<embree::vfloat<VMM::VectorSize> > _uniformDirections[VMM::MaxComponents][VMM::NumVectors];
};

////////////////////////////////////////////////////////////
/////////            UnassignedSamplesStatistics
////////////////////////////////////////////////////////////

template <class TVMMDistribution>
void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::UnassignedSamplesStatistics::clear()
{
    sumOfUnassignedWeights = 0.0f;
    sumUnassignedWeightedDirections = Vector3(0.0f);
}

template <class TVMMDistribution>
bool ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::UnassignedSamplesStatistics::isValid() const
{
    bool valid = true;
    valid = valid && embree::isvalid(sumOfUnassignedWeights);
    valid = valid && sumOfUnassignedWeights >= 0.f;
    OPENPGL_ASSERT(valid);

    valid = valid && embree::isvalid(sumUnassignedWeightedDirections.x);
    valid = valid && embree::isvalid(sumUnassignedWeightedDirections.y);
    valid = valid && embree::isvalid(sumUnassignedWeightedDirections.z);
    OPENPGL_ASSERT(valid);
    return valid;
}

////////////////////////////////////////////////////////////
/////////            SufficientStatistics
////////////////////////////////////////////////////////////

template <class TVMMDistribution>
void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::SufficientStatistics::applyParallaxShift(const VMM &vmm, const Vector3 shift)
{
    if (embree::length(shift) < FLT_EPSILON)
    {
        return;
    }

    const int cnt = (vmm._numComponents + VMM::VectorSize - 1) / VMM::VectorSize;
    // const int rem = vmm._numComponents % VMM::VectorSize;

    for (uint32_t k = 0; k < cnt; k++)
    {
        embree::Vec3<embree::vfloat<VMM::VectorSize> > suffDirections = sumOfWeightedDirections[k];
        embree::vfloat<VMM::VectorSize> suffMeanCosines = embree::length(suffDirections);
        suffDirections /= suffMeanCosines;
        suffDirections *= vmm._distances[k];
        suffDirections += embree::Vec3<embree::vfloat<VMM::VectorSize> >(shift);
        const embree::vfloat<VMM::VectorSize> length = embree::length(suffDirections);
        suffDirections /= length;
        suffDirections *= suffMeanCosines;
        sumOfWeightedDirections[k] = select((vmm._distances[k] > 0.0f) & (suffMeanCosines > 0.0f) & (length > FLT_EPSILON), suffDirections, sumOfWeightedDirections[k]);
    }
}

template <class TVMMDistribution>
bool ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::SufficientStatistics::isValid() const
{
    bool valid = true;

    for (size_t k = 0; k < numComponents; k++)
    {
        const div_t tmpK = div(k, VMM::VectorSize);
        valid = valid && embree::isvalid(sumOfWeightedDirections[tmpK.quot].x[tmpK.rem]);
        // valid = valid && sumOfWeightedDirections[tmpK.quot][tmpK.rem] >= 0.0f;
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(sumOfWeightedDirections[tmpK.quot].y[tmpK.rem]);
        // valid = valid && sumOfWeightedDirections[tmpK.quot][tmpK.rem] >= 0.0f;
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(sumOfWeightedDirections[tmpK.quot].z[tmpK.rem]);
        // valid = valid && sumOfWeightedDirections[tmpK.quot][tmpK.rem] >= 0.0f;
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(sumOfWeightedStats[tmpK.quot][tmpK.rem]);
        valid = valid && sumOfWeightedStats[tmpK.quot][tmpK.rem] >= 0.0f;
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(sumOfDistanceWeightes[tmpK.quot][tmpK.rem]);
        valid = valid && sumOfDistanceWeightes[tmpK.quot][tmpK.rem] >= 0.0f;
        OPENPGL_ASSERT(valid);

#ifdef OPENPGL_VSP_GUIDING
        valid = valid && embree::isvalid(volumeContributionWeights[tmpK.quot][tmpK.rem]);
        valid = valid && volumeContributionWeights[tmpK.quot][tmpK.rem] >= 0.0f;
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(surfaceContributionWeights[tmpK.quot][tmpK.rem]);
        valid = valid && surfaceContributionWeights[tmpK.quot][tmpK.rem] >= 0.0f;
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(volumeSampleNumberWeights[tmpK.quot][tmpK.rem]);
        valid = valid && volumeSampleNumberWeights[tmpK.quot][tmpK.rem] >= 0.0f;
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(surfaceSampleNumberWeights[tmpK.quot][tmpK.rem]);
        valid = valid && surfaceSampleNumberWeights[tmpK.quot][tmpK.rem] >= 0.0f;
        OPENPGL_ASSERT(valid);
#endif
    }

    for (size_t k = numComponents; k < VMM::MaxComponents; k++)
    {
        const div_t tmpK = div(k, VMM::VectorSize);
        valid = valid && sumOfWeightedDirections[tmpK.quot].x[tmpK.rem] == 0.0f;
        OPENPGL_ASSERT(valid);

        valid = valid && sumOfWeightedDirections[tmpK.quot].y[tmpK.rem] == 0.0f;
        OPENPGL_ASSERT(valid);

        valid = valid && sumOfWeightedDirections[tmpK.quot].z[tmpK.rem] == 0.0f;
        OPENPGL_ASSERT(valid);

        valid = valid && sumOfWeightedStats[tmpK.quot][tmpK.rem] == 0.0f;
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(sumOfDistanceWeightes[tmpK.quot][tmpK.rem]);
        valid = valid && sumOfDistanceWeightes[tmpK.quot][tmpK.rem] == 0.0f;
        OPENPGL_ASSERT(valid);

#ifdef OPENPGL_VSP_GUIDING
        valid = valid && embree::isvalid(volumeContributionWeights[tmpK.quot][tmpK.rem]);
        valid = valid && volumeContributionWeights[tmpK.quot][tmpK.rem] == 0.0f;
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(surfaceContributionWeights[tmpK.quot][tmpK.rem]);
        valid = valid && surfaceContributionWeights[tmpK.quot][tmpK.rem] == 0.0f;
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(volumeSampleNumberWeights[tmpK.quot][tmpK.rem]);
        valid = valid && volumeSampleNumberWeights[tmpK.quot][tmpK.rem] == 0.0f;
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(surfaceSampleNumberWeights[tmpK.quot][tmpK.rem]);
        valid = valid && surfaceSampleNumberWeights[tmpK.quot][tmpK.rem] == 0.0f;
        OPENPGL_ASSERT(valid);
#endif
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
void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::SufficientStatistics::serialize(std::ostream &stream) const
{
    serializeVec3Vectors<VMM::NumVectors, VMM::VectorSize>(stream, sumOfWeightedDirections);
    serializeFloatVectors<VMM::NumVectors, VMM::VectorSize>(stream, sumOfWeightedStats);
    serializeFloatVectors<VMM::NumVectors, VMM::VectorSize>(stream, sumOfDistanceWeightes);
    stream.write(reinterpret_cast<const char *>(&sumWeights), sizeof(float));
    stream.write(reinterpret_cast<const char *>(&numSamples), sizeof(float));
    stream.write(reinterpret_cast<const char *>(&overallNumSamples), sizeof(float));
    stream.write(reinterpret_cast<const char *>(&numComponents), sizeof(size_t));
    stream.write(reinterpret_cast<const char *>(&normalized), sizeof(bool));
#ifdef OPENPGL_VSP_GUIDING
    serializeFloatVectors<VMM::NumVectors, VMM::VectorSize>(stream, volumeContributionWeights);
    serializeFloatVectors<VMM::NumVectors, VMM::VectorSize>(stream, surfaceContributionWeights);
    serializeFloatVectors<VMM::NumVectors, VMM::VectorSize>(stream, volumeSampleNumberWeights);
    serializeFloatVectors<VMM::NumVectors, VMM::VectorSize>(stream, surfaceSampleNumberWeights);
#endif
}

template <class TVMMDistribution>
void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::SufficientStatistics::deserialize(std::istream &stream)
{
    deserializeVec3Vectors<VMM::NumVectors, VMM::VectorSize>(stream, sumOfWeightedDirections);
    deserializeFloatVectors<VMM::NumVectors, VMM::VectorSize>(stream, sumOfWeightedStats);
    deserializeFloatVectors<VMM::NumVectors, VMM::VectorSize>(stream, sumOfDistanceWeightes);
    stream.read(reinterpret_cast<char *>(&sumWeights), sizeof(float));
    stream.read(reinterpret_cast<char *>(&numSamples), sizeof(float));
    stream.read(reinterpret_cast<char *>(&overallNumSamples), sizeof(float));
    stream.read(reinterpret_cast<char *>(&numComponents), sizeof(size_t));
    stream.read(reinterpret_cast<char *>(&normalized), sizeof(bool));
#ifdef OPENPGL_VSP_GUIDING
    deserializeFloatVectors<VMM::NumVectors, VMM::VectorSize>(stream, volumeContributionWeights);
    deserializeFloatVectors<VMM::NumVectors, VMM::VectorSize>(stream, surfaceContributionWeights);
    deserializeFloatVectors<VMM::NumVectors, VMM::VectorSize>(stream, volumeSampleNumberWeights);
    deserializeFloatVectors<VMM::NumVectors, VMM::VectorSize>(stream, surfaceSampleNumberWeights);
#endif
}

template <class TVMMDistribution>
void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::SufficientStatistics::clear(size_t _numComponents)
{
    const embree::Vec3<embree::vfloat<VMM::VectorSize> > vecZeros(0.0f);
    const embree::vfloat<VMM::VectorSize> zeros(0.0f);

    numComponents = _numComponents;
    const int cnt = (numComponents + VMM::VectorSize - 1) / VMM::VectorSize;

    for (int k = 0; k < cnt; k++)
    {
        sumOfWeightedDirections[k] = vecZeros;
        sumOfWeightedStats[k] = zeros;

        sumOfDistanceWeightes[k] = zeros;
#ifdef OPENPGL_VSP_GUIDING
        volumeContributionWeights[k] = 0.0f;
        surfaceContributionWeights[k] = 0.0f;
        volumeSampleNumberWeights[k] = 0.0f;
        surfaceSampleNumberWeights[k] = 0.0f;
#endif
    }

    sumWeights = 0.0f;
    numSamples = 0;
    normalized = false;
}

template <class TVMMDistribution>
void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::SufficientStatistics::clearComponentStats(const size_t &idx)
{
    const div_t tmpIdx = div(idx, VMM::VectorSize);

    sumOfWeightedDirections[tmpIdx.quot].x[tmpIdx.rem] = 0.f;
    sumOfWeightedDirections[tmpIdx.quot].y[tmpIdx.rem] = 0.f;
    sumOfWeightedDirections[tmpIdx.quot].z[tmpIdx.rem] = 0.f;
    sumOfWeightedStats[tmpIdx.quot][tmpIdx.rem] = 0.f;

    sumOfDistanceWeightes[tmpIdx.quot][tmpIdx.rem] = 0.f;
#ifdef OPENPGL_VSP_GUIDING
    volumeContributionWeights[tmpIdx.quot][tmpIdx.rem] = 0.f;
    surfaceContributionWeights[tmpIdx.quot][tmpIdx.rem] = 0.f;
    volumeSampleNumberWeights[tmpIdx.quot][tmpIdx.rem] = 0.f;
    surfaceSampleNumberWeights[tmpIdx.quot][tmpIdx.rem] = 0.f;
#endif
}

template <class TVMMDistribution>
void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::SufficientStatistics::clearAll()
{
    clear(VMM::MaxComponents);
}

template <class TVMMDistribution>
void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::SufficientStatistics::decay(const float &alpha)
{
    for (int k = 0; k < VMM::NumVectors; k++)
    {
        sumOfWeightedDirections[k].x *= alpha;
        sumOfWeightedDirections[k].y *= alpha;
        sumOfWeightedDirections[k].z *= alpha;
        sumOfWeightedStats[k] *= alpha;

        sumOfDistanceWeightes[k] *= alpha;
#ifdef OPENPGL_VSP_GUIDING
        volumeContributionWeights[k] *= alpha;
        surfaceContributionWeights[k] *= alpha;
        volumeSampleNumberWeights[k] *= alpha;
        surfaceSampleNumberWeights[k] *= alpha;
#endif
    }

    numSamples *= alpha;
    sumWeights *= alpha;
}

template <class TVMMDistribution>
std::string ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::SufficientStatistics::toString() const
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
        int i = k / VMM::VectorSize;
        int j = k % VMM::VectorSize;
        ss << "\tstat[" << k << "]:" << "\tsumWeightedStats = " << sumOfWeightedStats[i][j] << "\tsumWeightedDirections = [" << sumOfWeightedDirections[i].x[j] << ",\t"
           << sumOfWeightedDirections[i].y[j] << ",\t" << sumOfWeightedDirections[i].z[j] << "]" << "\tsumWeightedDistanceWeights = " << sumOfDistanceWeightes[i][j] << std::endl;
    }
    return ss.str();
}

template <class TVMMDistribution>
void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::SufficientStatistics::maskedReplace(const PartialFittingMask &mask, const SufficientStatistics &stats)
{
    embree::vfloat<VMM::VectorSize> newSumWeights{0.0f};

    for (size_t k = 0; k < ((VMM::MaxComponents + (VMM::VectorSize - 1)) / VMM::VectorSize); k++)
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
void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::SufficientStatistics::swapComponentStats(const size_t &idx0, const size_t &idx1)
{
    const div_t tmpIdx0 = div(idx0, VMM::VectorSize);
    const div_t tmpIdx1 = div(idx1, VMM::VectorSize);

    std::swap(sumOfWeightedDirections[tmpIdx0.quot].x[tmpIdx0.rem], sumOfWeightedDirections[tmpIdx1.quot].x[tmpIdx1.rem]);
    std::swap(sumOfWeightedDirections[tmpIdx0.quot].y[tmpIdx0.rem], sumOfWeightedDirections[tmpIdx1.quot].y[tmpIdx1.rem]);
    std::swap(sumOfWeightedDirections[tmpIdx0.quot].z[tmpIdx0.rem], sumOfWeightedDirections[tmpIdx1.quot].z[tmpIdx1.rem]);
    std::swap(sumOfWeightedStats[tmpIdx0.quot][tmpIdx0.rem], sumOfWeightedStats[tmpIdx1.quot][tmpIdx1.rem]);
    std::swap(sumOfDistanceWeightes[tmpIdx0.quot][tmpIdx0.rem], sumOfDistanceWeightes[tmpIdx1.quot][tmpIdx1.rem]);
#ifdef OPENPGL_VSP_GUIDING
    std::swap(volumeContributionWeights[tmpIdx0.quot][tmpIdx0.rem], volumeContributionWeights[tmpIdx1.quot][tmpIdx1.rem]);
    std::swap(surfaceContributionWeights[tmpIdx0.quot][tmpIdx0.rem], surfaceContributionWeights[tmpIdx1.quot][tmpIdx1.rem]);
    std::swap(volumeSampleNumberWeights[tmpIdx0.quot][tmpIdx0.rem], volumeSampleNumberWeights[tmpIdx1.quot][tmpIdx1.rem]);
    std::swap(surfaceSampleNumberWeights[tmpIdx0.quot][tmpIdx0.rem], surfaceSampleNumberWeights[tmpIdx1.quot][tmpIdx1.rem]);
#endif
}

template <class TVMMDistribution>
void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::SufficientStatistics::mergeComponentStats(const size_t &idx0, const size_t &idx1)
{
    const div_t tmpIdx0 = div(idx0, VMM::VectorSize);
    const div_t tmpIdx1 = div(idx1, VMM::VectorSize);

    // merging the statistics of the component 0 and 1
    sumOfWeightedDirections[tmpIdx0.quot].x[tmpIdx0.rem] += sumOfWeightedDirections[tmpIdx1.quot].x[tmpIdx1.rem];
    sumOfWeightedDirections[tmpIdx0.quot].y[tmpIdx0.rem] += sumOfWeightedDirections[tmpIdx1.quot].y[tmpIdx1.rem];
    sumOfWeightedDirections[tmpIdx0.quot].z[tmpIdx0.rem] += sumOfWeightedDirections[tmpIdx1.quot].z[tmpIdx1.rem];
    sumOfWeightedStats[tmpIdx0.quot][tmpIdx0.rem] += sumOfWeightedStats[tmpIdx1.quot][tmpIdx1.rem];
    sumOfDistanceWeightes[tmpIdx0.quot][tmpIdx0.rem] += sumOfDistanceWeightes[tmpIdx1.quot][tmpIdx1.rem];

#ifdef OPENPGL_VSP_GUIDING
    volumeContributionWeights[tmpIdx0.quot][tmpIdx0.rem] += volumeContributionWeights[tmpIdx1.quot][tmpIdx1.rem];
    surfaceContributionWeights[tmpIdx0.quot][tmpIdx0.rem] += surfaceContributionWeights[tmpIdx1.quot][tmpIdx1.rem];
    volumeSampleNumberWeights[tmpIdx0.quot][tmpIdx0.rem] += volumeSampleNumberWeights[tmpIdx1.quot][tmpIdx1.rem];
    surfaceSampleNumberWeights[tmpIdx0.quot][tmpIdx0.rem] += surfaceSampleNumberWeights[tmpIdx1.quot][tmpIdx1.rem];
#endif
    swapComponentStats(idx1, numComponents - 1);
    clearComponentStats(numComponents - 1);

    numComponents--;
}

template <class TVMMDistribution>
void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::SufficientStatistics::splitComponentsStats(const size_t &idx0, const size_t &idx1,
                                                                                                                const Vector3 &meanDirection0, const Vector3 &meanDirection1,
                                                                                                                const float &meanCosine0, const float &meanCosine1)
{
    // OPENPGL_ASSERT(meanCosine0 > 0.f && meanCosine0 <= 1.0f);
    // OPENPGL_ASSERT(meanCosine1 > 0.f && meanCosine1 <= 1.0f);

    const div_t tmpI = div(idx0, static_cast<int>(VMM::VectorSize));
    const div_t tmpJ = div(idx1, static_cast<int>(VMM::VectorSize));

    float sumStatsWeight = sumOfWeightedStats[tmpI.quot][tmpI.rem];
    sumStatsWeight /= 2.0f;

    OPENPGL_ASSERT(sumStatsWeight > 0.f);

    sumOfWeightedStats[tmpI.quot][tmpI.rem] = sumStatsWeight;
    sumOfWeightedDirections[tmpI.quot].x[tmpI.rem] = meanDirection0.x * meanCosine0 * sumStatsWeight;
    sumOfWeightedDirections[tmpI.quot].y[tmpI.rem] = meanDirection0.y * meanCosine0 * sumStatsWeight;
    sumOfWeightedDirections[tmpI.quot].z[tmpI.rem] = meanDirection0.z * meanCosine0 * sumStatsWeight;

    sumOfWeightedStats[tmpJ.quot][tmpJ.rem] = sumStatsWeight;
    sumOfWeightedDirections[tmpJ.quot].x[tmpJ.rem] = meanDirection1.x * meanCosine1 * sumStatsWeight;
    sumOfWeightedDirections[tmpJ.quot].y[tmpJ.rem] = meanDirection1.y * meanCosine1 * sumStatsWeight;
    sumOfWeightedDirections[tmpJ.quot].z[tmpJ.rem] = meanDirection1.z * meanCosine1 * sumStatsWeight;

    float tmp = sumOfDistanceWeightes[tmpI.quot][tmpI.rem] * 0.5f;
    sumOfDistanceWeightes[tmpI.quot][tmpI.rem] = tmp;
    sumOfDistanceWeightes[tmpJ.quot][tmpJ.rem] = tmp;

#ifdef OPENPGL_VSP_GUIDING
    float weight0 = 0.5f;
    float weight1 = 0.5f;
    volumeContributionWeights[tmpJ.quot][tmpJ.rem] = weight1 * volumeContributionWeights[tmpI.quot][tmpI.rem];
    volumeContributionWeights[tmpI.quot][tmpI.rem] *= weight0;
    surfaceContributionWeights[tmpJ.quot][tmpJ.rem] = weight1 * surfaceContributionWeights[tmpI.quot][tmpI.rem];
    surfaceContributionWeights[tmpI.quot][tmpI.rem] *= weight0;
    volumeSampleNumberWeights[tmpJ.quot][tmpJ.rem] = weight1 * volumeSampleNumberWeights[tmpI.quot][tmpI.rem];
    volumeSampleNumberWeights[tmpI.quot][tmpI.rem] *= weight0;
    surfaceSampleNumberWeights[tmpJ.quot][tmpJ.rem] = weight1 * surfaceSampleNumberWeights[tmpI.quot][tmpI.rem];
    surfaceSampleNumberWeights[tmpI.quot][tmpI.rem] *= weight0;
#endif
    numComponents += 1;

    OPENPGL_ASSERT(!std::isnan(sumOfWeightedDirections[tmpI.quot].x[tmpI.rem]) && std::isfinite(sumOfWeightedDirections[tmpI.quot].x[tmpI.rem]));
    OPENPGL_ASSERT(!std::isnan(sumOfWeightedDirections[tmpI.quot].y[tmpI.rem]) && std::isfinite(sumOfWeightedDirections[tmpI.quot].y[tmpI.rem]));
    OPENPGL_ASSERT(!std::isnan(sumOfWeightedDirections[tmpI.quot].z[tmpI.rem]) && std::isfinite(sumOfWeightedDirections[tmpI.quot].z[tmpI.rem]));
}

template <class TVMMDistribution>
void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::SufficientStatistics::normalize(const float &_numSamples)
{
    const int cnt = (numComponents + VMM::VectorSize - 1) / VMM::VectorSize;
    numSamples = _numSamples;
    embree::vfloat<VMM::VectorSize> sumWeightedStatsVec(0.0f);

    for (int k = 0; k < cnt; k++)
    {
        sumWeightedStatsVec += sumOfWeightedStats[k];
    }
    sumWeights = reduce_add(sumWeightedStatsVec);
    embree::vfloat<VMM::VectorSize> norm(sumWeights > FLT_EPSILON ? _numSamples / sumWeights : 1.0f);

    for (int k = 0; k < cnt; k++)
    {
        sumOfWeightedDirections[k] *= norm;
        sumOfWeightedStats[k] *= norm;
    }
    normalized = true;
}

template <class TVMMDistribution>
typename ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::SufficientStatistics &
ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::SufficientStatistics::operator+=(
    const typename ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::SufficientStatistics &stats)
{
    // TODO: check for normalization

    const int cnt = (numComponents + VMM::VectorSize - 1) / VMM::VectorSize;

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
bool ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::SufficientStatistics::operator==(const SufficientStatistics &b) const
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
ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::ParallaxAwareVonMisesFisherWeightedEMFactory()
{
    _initUniformDirections();
}

template <class TVMMDistribution>
void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::InitUniformVMM(VMM &vmm, const int &numComponents, const float &kappa) const
{
    vmm._numComponents = numComponents;
    const size_t nComp = vmm._numComponents;
    const float weight = 1.f / float(vmm._numComponents);

    size_t n = 0;
    for (int i = 0; i < VMM::NumVectors; i++)
    {
        vmm._meanDirections[i] = _uniformDirections[nComp - 1][i];
        for (int j = 0; j < VMM::VectorSize; j++)
        {
            if (n < nComp)
            {
                vmm._kappas[i][j] = kappa;
                vmm._weights[i][j] = weight;
            }
            else
            {
                vmm._kappas[i][j] = 0.0f;
                vmm._weights[i][j] = 0.0f;
                vmm._normalizations[i][j] = ONE_OVER_FOUR_PI;
                vmm._eMinus2Kappa[i][j] = 1.0f;
                vmm._meanCosines[i][j] = 0.0f;
            }
            n++;
        }
    }

    vmm._calculateNormalization();
    vmm._calculateMeanCosines();
}

template <class TVMMDistribution>
void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::_initUniformDirections()
{
    const float gr = 1.618033988749895f;

    for (uint32_t l = 0; l < VMM::MaxComponents; l++)
    {
        /// distributes samples l+1 uniform samples over the sphere
        /// based on "Spherical Fibonacci Point Sets for Illumination Integrals"
        uint32_t n = 0;
        for (uint32_t k = 0; k < VMM::NumVectors; k++)
        {
            for (uint32_t i = 0; i < VMM::VectorSize; i++)
            {
                if (n < l + 1)
                {
                    float phi = 2.0f * M_PI_F * ((float)n / gr);
                    float z = 1.0f - ((2.0f * n + 1.0f) / float(l + 1));
                    float theta = std::acos(z);

                    Vector3 mu = sphericalDirection(theta, phi);
                    _uniformDirections[l][k].x[i] = mu[0];
                    _uniformDirections[l][k].y[i] = mu[1];
                    _uniformDirections[l][k].z[i] = mu[2];
                }
                else
                {
                    _uniformDirections[l][k].x[i] = 0.0f;
                    _uniformDirections[l][k].y[i] = 0.0f;
                    _uniformDirections[l][k].z[i] = 1.0f;
                }
                n++;
            }
        }
    }
}

template <class TVMMDistribution>
typename ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::VMM ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::VMMfromSufficientStatistics(
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
void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::fitMixture(VMM &vmm, SufficientStatistics &stats, const SampleData *samples, const size_t numSamples,
                                                                                const Configuration &cfg, FittingStatistics &fitStats) const
{
    const size_t numComponents = cfg.initK;
    // VonMisesFisherFactory< TVMMDistribution>::InitUniformVMM( vmm, numComponents, 5.0f);
    this->InitUniformVMM(vmm, numComponents, cfg.initKappa);
    // OPENPGL_ASSERT(vmm.isValid());
    // SufficientStatistics stats;
    stats.clear(numComponents);
    stats.normalized = true;
    // stats.clearAll();
    updateMixture(vmm, stats, samples, numSamples, cfg, fitStats);
}

template <class TVMMDistribution>
void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::handleUnassignedSampleStats(UnassignedSamplesStatistics &unassignedStats, VMM &vmm,
                                                                                                 SufficientStatistics &currentStats, SufficientStatistics &previousStats) const
{
    const div_t tmpK = div(currentStats.numComponents, TVMMDistribution::VectorSize);
    currentStats.numComponents++;
    currentStats.sumOfWeightedStats[tmpK.quot][tmpK.rem] = unassignedStats.sumOfUnassignedWeights;
    currentStats.sumOfWeightedDirections[tmpK.quot].x[tmpK.rem] = unassignedStats.sumUnassignedWeightedDirections.x;
    currentStats.sumOfWeightedDirections[tmpK.quot].y[tmpK.rem] = unassignedStats.sumUnassignedWeightedDirections.y;
    currentStats.sumOfWeightedDirections[tmpK.quot].z[tmpK.rem] = unassignedStats.sumUnassignedWeightedDirections.z;

    previousStats.numComponents++;
    previousStats.sumOfWeightedStats[tmpK.quot][tmpK.rem] = 0.0f;
    previousStats.sumOfWeightedDirections[tmpK.quot].x[tmpK.rem] = 0.0f;
    previousStats.sumOfWeightedDirections[tmpK.quot].y[tmpK.rem] = 0.0f;
    previousStats.sumOfWeightedDirections[tmpK.quot].z[tmpK.rem] = 0.0f;

    vmm._numComponents++;
}

template <class TVMMDistribution>
void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::updateMixture(VMM &vmm, SufficientStatistics &previousStats, const SampleData *samples,
                                                                                   const size_t numSamples, const Configuration &cfg, FittingStatistics &fitStats) const
{
    SufficientStatistics currentStats;
    // initially clear all stats
    currentStats.clearAll();

    size_t currentEMIteration = 0;
    bool converged = false;
    float previousLogLikelihood = 0.0f;
    float inv_previousLogLikelihood = 1.0f;
    UnassignedSamplesStatistics unassignedStats;
    while (!converged && currentEMIteration < cfg.maxEMIterrations)
    {
        float logLikelihood = weightedExpectationStep(vmm, currentStats, unassignedStats, samples, numSamples);
        if (unassignedStats.sumOfUnassignedWeights > 0.0f && currentStats.numComponents < TVMMDistribution::MaxComponents)
        {
            handleUnassignedSampleStats(unassignedStats, vmm, currentStats, previousStats);
        }

        OPENPGL_ASSERT(!currentStats.isNormalized());
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
    previousStats += currentStats;

    fitStats.numSamples = numSamples;
    fitStats.numIterations = currentEMIteration;
    fitStats.summedWeightedLogLikelihood = previousLogLikelihood;
    // td::cout << "converged:" <<  currentEMIteration << std::endl;
}

template <class TVMMDistribution>
void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::partialUpdateMixture(VMM &vmm, PartialFittingMask &mask, SufficientStatistics &previousStats,
                                                                                          const SampleData *samples, const size_t numSamples, const Configuration &cfg,
                                                                                          FittingStatistics &fitStats) const
{
    SufficientStatistics currentStats;
    // initially clear all stats
    currentStats.clearAll();

    size_t currentEMIteration = 0;
    bool converged = false;
    float previousLogLikelihood = 0.0f;
    float inv_previousLogLikelihood = 1.0f;

    UnassignedSamplesStatistics unassignedStats;
    while (!converged && currentEMIteration < cfg.maxEMIterrations)
    {
        float logLikelihood = weightedExpectationStep(vmm, currentStats, unassignedStats, samples, numSamples);
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
    previousStats += currentStats;

    fitStats.numSamples = numSamples;
    fitStats.numIterations = currentEMIteration;
    fitStats.summedWeightedLogLikelihood = previousLogLikelihood;
    // std::cout << "converged:" <<  currentEMIteration << std::endl;
}

template <class TVMMDistribution>
float ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::weightedExpectationStep(VMM &vmm, SufficientStatistics &stats, UnassignedSamplesStatistics &unassignedStats,
                                                                                              const SampleData *samples, const size_t numSamples) const
{
    unassignedStats.clear();
    stats.clear(vmm._numComponents);
    stats.numComponents = vmm._numComponents;
    stats.numSamples = numSamples;

    const int cnt = (stats.numComponents + VMM::VectorSize - 1) / VMM::VectorSize;

    float summedWeightedLogLikelihood{0.f};

    typename VMM::SoftAssignment softAssign;

    for (size_t n = 0; n < numSamples; n++)
    {
        const SampleData sampleData = samples[n];
        const embree::vfloat<VMM::VectorSize> sampleWeight = sampleData.weight;
        pgl_vec3f direction = sampleData.direction;
        const Vector3 sampleDirection(direction.x, direction.y, direction.z);
        const embree::Vec3<embree::vfloat<VMM::VectorSize> > sampleDirectionSIMD(sampleDirection);

        // check if the samples is covered by any of the components
        if (!vmm.softAssignment(sampleDirection, softAssign))
        {
            unassignedStats.sumOfUnassignedWeights += sampleData.weight;
            unassignedStats.sumUnassignedWeightedDirections += sampleDirection * sampleData.weight;
#ifdef OPENPGL_DEBUG_MODE
            std::cout << "continue" << std::endl;
#endif
            continue;
        }

        summedWeightedLogLikelihood += sampleData.weight * embree::log(softAssign.pdf);

        for (size_t k = 0; k < cnt; k++)
        {
            stats.sumOfWeightedDirections[k] += sampleDirectionSIMD * softAssign.assignments[k] * sampleWeight;
            stats.sumOfWeightedStats[k] += softAssign.assignments[k] * sampleWeight;
        }
    }
    return summedWeightedLogLikelihood;
}

template <class TVMMDistribution>
void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::estimateMAPWeights(VMM &vmm, const SufficientStatistics &currentStats,
                                                                                        const SufficientStatistics &previousStats, const float &_weightPrior) const
{
    const int cnt = (vmm._numComponents + VMM::VectorSize - 1) / VMM::VectorSize;

    const size_t numComponents = vmm._numComponents;

    const embree::vfloat<VMM::VectorSize> weightPrior(_weightPrior);

    const embree::vfloat<VMM::VectorSize> numSamples = currentStats.numSamples + previousStats.numSamples;
    // const vfloat<VMM::VectorSize> numSamples = currentStats.numSamples + previousStats.overallNumSamples;

    for (size_t k = 0; k < cnt; k++)
    {
        //_sumWeights += currentStats.sumOfWeightedStats[k];
        embree::vfloat<VMM::VectorSize> weight = (currentStats.sumOfWeightedStats[k] + previousStats.sumOfWeightedStats[k]);
        weight = (weightPrior + (weight)) / ((weightPrior * numComponents) + numSamples);
        // vfloat<VMM::VectorSize>  weight = ( currentStats.sumOfWeightedStats[k]/* + previousStats.sumOfWeightedStats[k]*/ ) / ( sumWeights );
        // weight = ( weightPrior + ( weight * numSamples ) ) / (( weightPrior * numComponents ) + numSamples );
        vmm._weights[k] = weight;
    }

    // TODO: find better more efficient way
    if (vmm._numComponents % VMM::VectorSize > 0)
    {
        for (size_t i = vmm._numComponents % VMM::VectorSize; i < VMM::VectorSize; i++)
        {
            vmm._weights[cnt - 1][i] = 0.0f;
        }
    }
}

template <class TVMMDistribution>
void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::estimateMAPMeanDirectionAndConcentration(VMM &vmm, const SufficientStatistics &currentStats,
                                                                                                              const SufficientStatistics &previousStats,
                                                                                                              const Configuration &cfg) const
{
    const embree::vfloat<VMM::VectorSize> currentNumSamples = currentStats.numSamples;
    const embree::vfloat<VMM::VectorSize> previousNumSamples = previousStats.numSamples;
    const embree::vfloat<VMM::VectorSize> numSamples = currentNumSamples + previousNumSamples;
    const embree::vfloat<VMM::VectorSize> overallNumSamples = currentStats.numSamples + previousStats.overallNumSamples;

    const embree::vfloat<VMM::VectorSize> currentEstimationWeight = currentNumSamples / numSamples;
    const embree::vfloat<VMM::VectorSize> previousEstimationWeight = 1.0f - currentEstimationWeight;

    const embree::vfloat<VMM::VectorSize> meanCosinePrior = cfg.meanCosinePrior;
    const embree::vfloat<VMM::VectorSize> meanCosinePriorStrength = cfg.meanCosinePriorStrength;
    const embree::vfloat<VMM::VectorSize> maxMeanCosine = cfg.maxMeanCosine;
    const int cnt = (vmm._numComponents + VMM::VectorSize - 1) / VMM::VectorSize;
    const int rem = vmm._numComponents % VMM::VectorSize;

    for (size_t k = 0; k < cnt; k++)
    {
        // const vfloat<VMM::VectorSize> partialNumSamples = vmm._weights[k] * numSamples;
        const embree::vfloat<VMM::VectorSize> partialNumSamples = vmm._weights[k] * overallNumSamples;
        embree::Vec3<embree::vfloat<VMM::VectorSize> > currentMeanDirection;
        currentMeanDirection.x = select(currentStats.sumOfWeightedStats[k] > 0.0f, currentStats.sumOfWeightedDirections[k].x / currentStats.sumOfWeightedStats[k], 0.0f);
        currentMeanDirection.y = select(currentStats.sumOfWeightedStats[k] > 0.0f, currentStats.sumOfWeightedDirections[k].y / currentStats.sumOfWeightedStats[k], 0.0f);
        currentMeanDirection.z = select(currentStats.sumOfWeightedStats[k] > 0.0f, currentStats.sumOfWeightedDirections[k].z / currentStats.sumOfWeightedStats[k], 0.0f);

        // TODO: find a better design to precompute the previousMeanDirection
        embree::Vec3<embree::vfloat<VMM::VectorSize> > previousMeanDirection;
        previousMeanDirection.x = select(previousStats.sumOfWeightedStats[k] > 0.0f, previousStats.sumOfWeightedDirections[k].x / previousStats.sumOfWeightedStats[k], 0.0f);
        previousMeanDirection.y = select(previousStats.sumOfWeightedStats[k] > 0.0f, previousStats.sumOfWeightedDirections[k].y / previousStats.sumOfWeightedStats[k], 0.0f);
        previousMeanDirection.z = select(previousStats.sumOfWeightedStats[k] > 0.0f, previousStats.sumOfWeightedDirections[k].z / previousStats.sumOfWeightedStats[k], 0.0f);

        embree::Vec3<embree::vfloat<VMM::VectorSize> > meanDirection = currentMeanDirection * currentEstimationWeight + previousMeanDirection * previousEstimationWeight;

        embree::vfloat<VMM::VectorSize> meanCosine = length(meanDirection);

        vmm._meanDirections[k].x = select(meanCosine > 0.0f, meanDirection.x / meanCosine, vmm._meanDirections[k].x);
        vmm._meanDirections[k].y = select(meanCosine > 0.0f, meanDirection.y / meanCosine, vmm._meanDirections[k].y);
        vmm._meanDirections[k].z = select(meanCosine > 0.0f, meanDirection.z / meanCosine, vmm._meanDirections[k].z);

        meanCosine = (meanCosinePrior * meanCosinePriorStrength + meanCosine * partialNumSamples) / (meanCosinePriorStrength + partialNumSamples);

        meanCosine = embree::min(maxMeanCosine, meanCosine);
        vmm._meanCosines[k] = meanCosine;
        vmm._kappas[k] = MeanCosineToKappa<embree::vfloat<VMM::VectorSize> >(meanCosine);
    }

    // TODO: find better more efficient way
    if (rem > 0)
    {
        for (size_t i = rem; i < VMM::VectorSize; i++)
        {
            vmm._meanDirections[cnt - 1].x[i] = 0.0f;
            vmm._meanDirections[cnt - 1].y[i] = 0.0f;
            vmm._meanDirections[cnt - 1].z[i] = 1.0f;

            vmm._meanCosines[cnt - 1][i] = 0.0f;
            vmm._kappas[cnt - 1][i] = 0.0f;

            vmm._normalizations[cnt - 1][i] = ONE_OVER_FOUR_PI;
            vmm._eMinus2Kappa[cnt - 1][i] = 1.0f;
        }
    }

    vmm._calculateNormalization();
}

template <class TVMMDistribution>
void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::weightedMaximumAPosteriorStep(VMM &vmm, const SufficientStatistics &currentStats,
                                                                                                   const SufficientStatistics &previousStats, const Configuration &cfg) const
{
    // Estimating components weights
    estimateMAPWeights(vmm, currentStats, previousStats, cfg.weightPrior);

    // Estimating mean and concentration
    estimateMAPMeanDirectionAndConcentration(vmm, currentStats, previousStats, cfg);
}

template <class TVMMDistribution>
void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::partialWeightedMaximumAPosteriorStep(VMM &vmm, const PartialFittingMask &mask,
                                                                                                          SufficientStatistics &currentStats, SufficientStatistics &previousStats,
                                                                                                          const Configuration &cfg) const
{
    // Estimating components weights
    estimatePartialMAPWeights(vmm, mask, currentStats, previousStats, cfg.weightPrior);

    // Estimating mean and concentration
    estimatePartialMAPMeanDirectionAndConcentration(vmm, mask, currentStats, previousStats, cfg);
}

template <class TVMMDistribution>
void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::estimatePartialMAPWeights(VMM &vmm, const PartialFittingMask &mask, SufficientStatistics &currentStats,
                                                                                               SufficientStatistics &previousStats, const float &_weightPrior) const
{
    const embree::vfloat<VMM::VectorSize> zeros(0.0f);
    const int cnt = (vmm._numComponents + VMM::VectorSize - 1) / VMM::VectorSize;

    const size_t numComponents = vmm._numComponents;

    const embree::vfloat<VMM::VectorSize> weightPrior(_weightPrior);

    const embree::vfloat<VMM::VectorSize> numSamples = currentStats.numSamples + previousStats.numSamples;
    // const vfloat<VMM::VectorSize> numSamples = currentStats.numSamples + previousStats.overallNumSamples;

    embree::vfloat<VMM::VectorSize> sumWeights(0.0f);
    embree::vfloat<VMM::VectorSize> sumPartialWeights(0.0f);

    for (size_t k = 0; k < cnt; k++)
    {
        //_sumWeights += currentStats.sumOfWeightedStats[k];
        embree::vfloat<VMM::VectorSize> weight = (currentStats.sumOfWeightedStats[k] + previousStats.sumOfWeightedStats[k]);
        weight = (weightPrior + (weight)) / ((weightPrior * numComponents) + numSamples);
        // vfloat<VMM::VectorSize>  weight = ( currentStats.sumOfWeightedStats[k]/* + previousStats.sumOfWeightedStats[k]*/ ) / ( sumWeights );
        // weight = ( weightPrior + ( weight * numSamples ) ) / (( weightPrior * numComponents ) + numSamples );

        sumPartialWeights += select(mask.mask[k], weight, zeros);
        sumWeights += select(mask.mask[k], zeros, vmm._weights[k]);

        vmm._weights[k] = select(mask.mask[k], weight, vmm._weights[k]);
    }

    embree::vfloat<VMM::VectorSize> inv_sumPartialWeights = 1.0f / reduce_add(sumPartialWeights);
    inv_sumPartialWeights *= 1.0f - reduce_add(sumWeights);
    for (size_t k = 0; k < cnt; k++)
    {
        vmm._weights[k] = select(mask.mask[k], vmm._weights[k] * inv_sumPartialWeights, vmm._weights[k]);
    }

    // TODO: find better more efficient way
    if (vmm._numComponents % VMM::VectorSize > 0)
    {
        for (size_t i = vmm._numComponents % VMM::VectorSize; i < VMM::VectorSize; i++)
        {
            vmm._weights[cnt - 1][i] = 0.0f;
        }
    }
}

template <class TVMMDistribution>
void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::estimatePartialMAPMeanDirectionAndConcentration(VMM &vmm, const PartialFittingMask &mask,
                                                                                                                     SufficientStatistics &currentStats,
                                                                                                                     SufficientStatistics &previousStats,
                                                                                                                     const Configuration &cfg) const
{
    const embree::vfloat<VMM::VectorSize> currentNumSamples = currentStats.numSamples;
    const embree::vfloat<VMM::VectorSize> previousNumSamples = previousStats.numSamples;
    const embree::vfloat<VMM::VectorSize> numSamples = currentNumSamples + previousNumSamples;
    const embree::vfloat<VMM::VectorSize> overallNumSamples = currentStats.numSamples + previousStats.overallNumSamples;

    const embree::vfloat<VMM::VectorSize> currentEstimationWeight = currentNumSamples / numSamples;
    const embree::vfloat<VMM::VectorSize> previousEstimationWeight = 1.0f - currentEstimationWeight;

    const embree::vfloat<VMM::VectorSize> meanCosinePrior = cfg.meanCosinePrior;
    const embree::vfloat<VMM::VectorSize> meanCosinePriorStrength = cfg.meanCosinePriorStrength;
    const embree::vfloat<VMM::VectorSize> maxMeanCosine = cfg.maxMeanCosine;
    const int cnt = (vmm._numComponents + VMM::VectorSize - 1) / VMM::VectorSize;
    const int rem = vmm._numComponents % VMM::VectorSize;

    for (size_t k = 0; k < cnt; k++)
    {
        // const vfloat<VMM::VectorSize> partialNumSamples = vmm._weights[k] * numSamples;
        const embree::vfloat<VMM::VectorSize> partialNumSamples = vmm._weights[k] * overallNumSamples;
        embree::Vec3<embree::vfloat<VMM::VectorSize> > currentMeanDirection;
        currentMeanDirection.x = select(currentStats.sumOfWeightedStats[k] > 0.0f, currentStats.sumOfWeightedDirections[k].x / currentStats.sumOfWeightedStats[k], 0.0f);
        currentMeanDirection.y = select(currentStats.sumOfWeightedStats[k] > 0.0f, currentStats.sumOfWeightedDirections[k].y / currentStats.sumOfWeightedStats[k], 0.0f);
        currentMeanDirection.z = select(currentStats.sumOfWeightedStats[k] > 0.0f, currentStats.sumOfWeightedDirections[k].z / currentStats.sumOfWeightedStats[k], 0.0f);

        // TODO: find a better design to precompute the previousMeanDirection
        embree::Vec3<embree::vfloat<VMM::VectorSize> > previousMeanDirection;
        previousMeanDirection.x = select(previousStats.sumOfWeightedStats[k] > 0.0f, previousStats.sumOfWeightedDirections[k].x / previousStats.sumOfWeightedStats[k], 0.0f);
        previousMeanDirection.y = select(previousStats.sumOfWeightedStats[k] > 0.0f, previousStats.sumOfWeightedDirections[k].y / previousStats.sumOfWeightedStats[k], 0.0f);
        previousMeanDirection.z = select(previousStats.sumOfWeightedStats[k] > 0.0f, previousStats.sumOfWeightedDirections[k].z / previousStats.sumOfWeightedStats[k], 0.0f);

        embree::Vec3<embree::vfloat<VMM::VectorSize> > meanDirection = currentMeanDirection * currentEstimationWeight + previousMeanDirection * previousEstimationWeight;

        embree::vfloat<VMM::VectorSize> meanCosine = embree::length(meanDirection);

        vmm._meanDirections[k].x = select(mask.mask[k], select(meanCosine > 0.0f, meanDirection.x / meanCosine, vmm._meanDirections[k].x), vmm._meanDirections[k].x);
        vmm._meanDirections[k].y = select(mask.mask[k], select(meanCosine > 0.0f, meanDirection.y / meanCosine, vmm._meanDirections[k].y), vmm._meanDirections[k].y);
        vmm._meanDirections[k].z = select(mask.mask[k], select(meanCosine > 0.0f, meanDirection.z / meanCosine, vmm._meanDirections[k].z), vmm._meanDirections[k].z);

        meanCosine = (meanCosinePrior * meanCosinePriorStrength + meanCosine * partialNumSamples) / (meanCosinePriorStrength + partialNumSamples);

        meanCosine = embree::min(maxMeanCosine, meanCosine);
        vmm._meanCosines[k] = select(mask.mask[k], meanCosine, vmm._meanCosines[k]);
        vmm._kappas[k] = select(mask.mask[k], MeanCosineToKappa<embree::vfloat<VMM::VectorSize> >(meanCosine), vmm._kappas[k]);
    }

    // TODO: find better more efficient way
    if (rem > 0)
    {
        for (size_t i = rem; i < VMM::VectorSize; i++)
        {
            vmm._meanDirections[cnt - 1].x[i] = 0.0f;
            vmm._meanDirections[cnt - 1].y[i] = 0.0f;
            vmm._meanDirections[cnt - 1].z[i] = 1.0f;

            vmm._meanCosines[cnt - 1][i] = 0.0f;
            vmm._kappas[cnt - 1][i] = 0.0f;
        }
    }

    vmm._calculateNormalization();
}

template <class TVMMDistribution>
void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::reprojectSample(openpgl::SampleData &sample, const openpgl::Point3 &pivotPoint, const float minDistance) const
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
void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::prepareSamples(SampleData *samples, const size_t numSamples, const SampleStatistics &sampleStatistics,
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
        for (size_t n = 0; n < numSamples; n++)
        {
            reprojectSample(samples[n], sampleStatistics.getMean(), minDistance);
        }
    }
}

template <class TVMMDistribution>
void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::initComponentDistances(VMM &vmm, SufficientStatistics &sufficientStats, const SampleData *samples,
                                                                                            const size_t numSamples) const
{
    OPENPGL_ASSERT(vmm.getNumComponents() == sufficientStats.getNumComponents());

    embree::vfloat<VMM::VectorSize> batchDistances[VMM::NumVectors];
    embree::vfloat<VMM::VectorSize> batchSumWeights[VMM::NumVectors];

    const embree::vfloat<VMM::VectorSize> zeros(0.0f);

    const int cnt = (vmm._numComponents + VMM::VectorSize - 1) / VMM::VectorSize;
    const int rem = vmm._numComponents % VMM::VectorSize;

    for (size_t k = 0; k < cnt; k++)
    {
        batchDistances[k] = zeros;
        batchSumWeights[k] = zeros;
    }

    typename VMM::SoftAssignment softAssign;
    float sampleDistance;
    embree::vfloat<VMM::VectorSize> weights;
    for (size_t n = 0; n < numSamples; n++)
    {
#ifdef USE_HARMONIC_MEAN
        sampleDistance = embree::rcp(samples[n].distance);
#else
        sampleDistance = samples[n].distance;
#endif
        pgl_vec3f direction = samples[n].direction;
        const Vector3 sampleDirection(direction.x, direction.y, direction.z);
        if (vmm.softAssignment(sampleDirection, softAssign))
        {
            for (size_t k = 0; k < cnt; k++)
            {
                weights =
                    select(vmm._weights[k] > FLT_EPSILON, samples[n].weight * softAssign.assignments[k] * ((softAssign.assignments[k] * softAssign.pdf) / vmm._weights[k]), zeros);
                batchDistances[k] += weights * sampleDistance;
                batchSumWeights[k] += weights;
                OPENPGL_ASSERT(embree::isvalid(weights));
                OPENPGL_ASSERT(embree::isvalid(batchDistances[k]));
                OPENPGL_ASSERT(embree::isvalid(batchSumWeights[k]));
            }
        }
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
        for (size_t i = rem; i < VMM::VectorSize; i++)
        {
            vmm._distances[cnt - 1][i] = 0.0f;
            sufficientStats.sumOfDistanceWeightes[cnt - 1][i] = 0.0f;
        }
    }
}

template <class TVMMDistribution>
void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::updateComponentDistances(VMM &vmm, SufficientStatistics &sufficientStats, const SampleData *samples,
                                                                                              const size_t numSamples) const
{
    OPENPGL_ASSERT(vmm.getNumComponents() == sufficientStats.getNumComponents());

    embree::vfloat<VMM::VectorSize> batchDistances[VMM::NumVectors];
    embree::vfloat<VMM::VectorSize> batchSumWeights[VMM::NumVectors];

    const embree::vfloat<VMM::VectorSize> zeros(0.0f);
    const int cnt = (vmm._numComponents + VMM::VectorSize - 1) / VMM::VectorSize;
    const int rem = vmm._numComponents % VMM::VectorSize;

    for (size_t k = 0; k < cnt; k++)
    {
        batchDistances[k] = zeros;
        batchSumWeights[k] = zeros;
    }

    typename VMM::SoftAssignment softAssign;
    float sampleDistance;
    embree::vfloat<VMM::VectorSize> weights;
    for (size_t n = 0; n < numSamples; n++)
    {
        OPENPGL_ASSERT(samples[n].distance > 0);
        OPENPGL_ASSERT(embree::isvalid(samples[n].distance));
#ifdef USE_HARMONIC_MEAN
        sampleDistance = embree::rcp(samples[n].distance);
#else
        sampleDistance = samples[n].distance;
#endif
        pgl_vec3f direction = samples[n].direction;
        const Vector3 sampleDirection(direction.x, direction.y, direction.z);
        if (vmm.softAssignment(sampleDirection, softAssign))
        {
            for (size_t k = 0; k < cnt; k++)
            {
                weights = samples[n].weight * softAssign.assignments[k] * ((softAssign.assignments[k] * softAssign.pdf) / vmm._weights[k]);
                batchDistances[k] += weights * sampleDistance;
                batchSumWeights[k] += weights;
            }
        }
    }

    for (size_t k = 0; k < cnt; k++)
    {
#ifdef USE_HARMONIC_MEAN
        // embree::vfloat<VMM::VectorSize> sumInverseDistances = (sufficientStats.sumOfDistanceWeightes[k] / vmm._distances[k]) + batchDistances[k];
        embree::vfloat<VMM::VectorSize> sumInverseDistances = batchDistances[k];
        sumInverseDistances += select(vmm._distances[k] > 0.0f, (sufficientStats.sumOfDistanceWeightes[k] / vmm._distances[k]), embree::vfloat<VMM::VectorSize>(0.0f));
        sufficientStats.sumOfDistanceWeightes[k] += batchSumWeights[k];
        vmm._distances[k] = sufficientStats.sumOfDistanceWeightes[k] / sumInverseDistances;
#else
        const embree::vfloat<VMM::VectorSize> sumInverseDistances = (sufficientStats.sumOfDistanceWeightes[k] * vmm._distances[k]) + batchDistances[k];
        sufficientStats.sumOfDistanceWeightes[k] += batchSumWeights[k];
        vmm._distances[k] = sumInverseDistances / sufficientStats.sumOfDistanceWeightes[k];
#endif
    }

    if (rem > 0)
    {
        for (size_t i = rem; i < VMM::VectorSize; i++)
        {
            vmm._distances[cnt - 1][i] = 0.0f;
            sufficientStats.sumOfDistanceWeightes[cnt - 1][i] = 0.0f;
        }
    }
}

#ifdef OPENPGL_RADIANCE_CACHES
template <class TVMMDistribution>
void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::updateFluenceEstimate(VMM &vmm, const SampleData *samples, const size_t numSamples,
                                                                                           const size_t numZeroValueSamples, const SampleStatistics &sampleStatistics) const
{
#ifdef MC_ESTIMATE_INCOMING_RADIANCE  // calcualting fluence and the RGB per lob estiamtions using the MC samples
    if (numSamples == 0)
    {
        return;
    }

    const int cnt = (vmm._numComponents + VMM::VectorSize - 1) / VMM::VectorSize;
    const int rem = vmm._numComponents % VMM::VectorSize;

    const embree::vfloat<VMM::VectorSize> zeros(0.0f);

    // float sumFluence {0.f};
    Vector3 sumFluenceRGB{0.f, 0.f, 0.f};
    Vector3 sumFluenceRGBWithMIS{0.f, 0.f, 0.f};

    embree::Vec3<embree::vfloat<VMM::VectorSize> > sumFluenceRGBWeights[VMM::NumVectors];
    embree::Vec3<embree::vfloat<VMM::VectorSize> > sumFluenceRGBWeightsWithMIS[VMM::NumVectors];
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
        for (size_t i = rem; i < VMM::VectorSize; i++)
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
    const embree::vfloat<VMM::VectorSize> zeros(0.0f);
    const embree::vfloat<VMM::VectorSize> ones(1.0f);
    const int cnt = (vmm._numComponents + VMM::VectorSize - 1) / VMM::VectorSize;

    if (numSamples == 0)
    {
        return;
    }

    embree::vfloat<VMM::VectorSize> sumPdfs[VMM::NumVectors];
    embree::vfloat<VMM::VectorSize> pdfs(1.0f);
    embree::Vec3<embree::vfloat<VMM::VectorSize> > sumFluenceRGBWeights[VMM::NumVectors];
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
        embree::Vec3<embree::vfloat<VMM::VectorSize> > sampleDirectionVec(sampleDirection[0], sampleDirection[1], sampleDirection[2]);

        Vector3 radianceIn(samples[n].radianceIn.x, samples[n].radianceIn.y, samples[n].radianceIn.z);
        sumFluence += samples[n].weight;
        sumFluenceRGBMC += radianceIn / samples[n].pdf;

        if (vmm.softAssignment(sampleDirection, softAssign))
        {
            for (size_t k = 0; k < cnt; k++)
            {
                const embree::vfloat<VMM::VectorSize> cosTheta = embree::dot(sampleDirectionVec, vmm._meanDirections[k]);
                const embree::vfloat<VMM::VectorSize> cosThetaMinusOne = embree::min(cosTheta - ones, zeros);
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
void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::Configuration::init()
{
    maxMeanCosine = KappaToMeanCosine<float>(maxKappa);
}

template <class TVMMDistribution>
void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::Configuration::serialize(std::ostream &stream) const
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
void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::Configuration::deserialize(std::istream &stream)
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
std::string ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::Configuration::toString() const
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
void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::PartialFittingMask::resetToFalse()
{
    const embree::vbool<VMM::VectorSize> vFalse(false);
    for (size_t k = 0; k < ((VMM::MaxComponents + (VMM::VectorSize - 1)) / VMM::VectorSize); k++)
    {
        mask[k] = vFalse;
    }
}

template <class TVMMDistribution>
void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::PartialFittingMask::resetToTrue(const size_t &numComponents)
{
    const embree::vbool<VMM::VectorSize> vTrue(true);
    const int cnt = (numComponents + VMM::VectorSize - 1) / VMM::VectorSize;
    for (size_t k = 0; k < cnt; k++)
    {
        mask[k] = vTrue;
    }

    const div_t tmp = div(numComponents, VMM::VectorSize);
    for (size_t k = tmp.rem; k < VMM::VectorSize; k++)
    {
        clear(mask[tmp.quot], k);
    }
}

template <class TVMMDistribution>
void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::PartialFittingMask::setToTrue(const size_t &idx)
{
    const div_t tmp = div(idx, VMM::VectorSize);
    embree::set(mask[tmp.quot], tmp.rem);
}

template <class TVMMDistribution>
void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::PartialFittingMask::setToFalse(const size_t &idx)
{
    const div_t tmp = div(idx, VMM::VectorSize);
    embree::clear(mask[tmp.quot], tmp.rem);
}

template <class TVMMDistribution>
bool ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::PartialFittingMask::get(const size_t &idx) const
{
    const div_t tmp = div(idx, VMM::VectorSize);
    return embree::get(mask[tmp.quot], tmp.rem);
}

template <class TVMMDistribution>
std::string ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::PartialFittingMask::toString() const
{
    std::stringstream ss;
    ss << "PartialFittingMask:" << std::endl;
    for (size_t k = 0; k < VMM::MaxComponents; k++)
    {
        const div_t tmp = div(k, VMM::VectorSize);
        ss << "mask[" << k << "]: " << mask[tmp.quot][tmp.rem] << std::endl;
    }
    return ss.str();
}

#ifdef OPENPGL_VSP_GUIDING
template <class TVMMDistribution>
void ParallaxAwareVonMisesFisherWeightedEMFactory<TVMMDistribution>::updateVolumeScatterProbability(VMM &vmm, SufficientStatistics &sufficientStats, const SampleData *samples,
                                                                                                    const size_t numSamples, const ZeroValueSampleData *zeroValueSamples,
                                                                                                    const size_t numZeroValueSamples, const bool varianceBased) const
{
    // OPENPGL_ASSERT(vmm.isValid());

    if (numSamples + numZeroValueSamples == 0)
    {
        return;
    }

    const int cnt = (vmm._numComponents + VMM::VectorSize - 1) / VMM::VectorSize;
    const int rem = vmm._numComponents % VMM::VectorSize;

    const embree::vfloat<VMM::VectorSize> zeros(0.0f);
    const embree::vfloat<VMM::VectorSize> ones(1.0f);

    typename VMM::SoftAssignment softAssign;

    for (size_t n = 0; n < numSamples; n++)
    {
        const float weight = samples[n].weight;
        pgl_vec3f direction = samples[n].direction;
        const Vector3 sampleDirection(direction.x, direction.y, direction.z);

        if (vmm.softAssignment(sampleDirection, softAssign))
        {
            for (size_t k = 0; k < cnt; k++)
            {
                if (isNextEventVolume(samples[n]))
                {
                    if (!varianceBased)
                    {
                        sufficientStats.volumeContributionWeights[k] += weight * softAssign.assignments[k];
                        OPENPGL_ASSERT(embree::isvalid(sufficientStats.volumeContributionWeights[k]));
                    }
                    else
                    {
                        sufficientStats.volumeContributionWeights[k] += weight * weight * softAssign.assignments[k];
                        OPENPGL_ASSERT(embree::isvalid(sufficientStats.volumeContributionWeights[k]));
                    }
                    sufficientStats.volumeSampleNumberWeights[k] += softAssign.assignments[k];
                    OPENPGL_ASSERT(embree::isvalid(sufficientStats.volumeSampleNumberWeights[k]));
                }
                else
                {
                    if (!varianceBased)
                    {
                        sufficientStats.surfaceContributionWeights[k] += weight * softAssign.assignments[k];
                        OPENPGL_ASSERT(embree::isvalid(sufficientStats.surfaceContributionWeights[k]));
                    }
                    else
                    {
                        sufficientStats.surfaceContributionWeights[k] += weight * weight * softAssign.assignments[k];
                        OPENPGL_ASSERT(embree::isvalid(sufficientStats.surfaceContributionWeights[k]));
                    }
                    sufficientStats.surfaceSampleNumberWeights[k] += softAssign.assignments[k];
                    OPENPGL_ASSERT(embree::isvalid(sufficientStats.surfaceSampleNumberWeights[k]));
                }
            }
        }
    }

    for (size_t n = 0; n < numZeroValueSamples; n++)
    {
        pgl_vec3f direction = zeroValueSamples[n].direction;
        const Vector3 sampleDirection(direction.x, direction.y, direction.z);

        if (vmm.softAssignment(sampleDirection, softAssign))
        {
            for (size_t k = 0; k < cnt; k++)
            {
                if (isNextEventVolume(zeroValueSamples[n]))
                {
                    sufficientStats.volumeSampleNumberWeights[k] += softAssign.assignments[k];
                    OPENPGL_ASSERT(embree::isvalid(sufficientStats.volumeSampleNumberWeights[k]));
                }
                else
                {
                    sufficientStats.surfaceSampleNumberWeights[k] += softAssign.assignments[k];
                    OPENPGL_ASSERT(embree::isvalid(sufficientStats.surfaceSampleNumberWeights[k]));
                }
            }
        }
    }

    if (rem > 0)
    {
        for (size_t i = rem; i < VMM::VectorSize; i++)
        {
            vmm._volumeScatterProbabilityWeights[cnt - 1][i] = 0.0f;
            sufficientStats.volumeContributionWeights[cnt - 1][i] = 0.0f;
            sufficientStats.surfaceContributionWeights[cnt - 1][i] = 0.0f;
            sufficientStats.volumeSampleNumberWeights[cnt - 1][i] = 0.0f;
            sufficientStats.surfaceSampleNumberWeights[cnt - 1][i] = 0.0f;
        }
    }

    for (size_t k = 0; k < cnt; k++)
    {
        embree::vfloat<VMM::VectorSize> numVolSamples = sufficientStats.volumeSampleNumberWeights[k];
        embree::vfloat<VMM::VectorSize> numSurfSamples = sufficientStats.surfaceSampleNumberWeights[k];
        embree::vfloat<VMM::VectorSize> numSurfVolSamples = numVolSamples + numSurfSamples;
        embree::vfloat<VMM::VectorSize> estPVol = select(numSurfVolSamples > FLT_EPSILON, numVolSamples / numSurfVolSamples, zeros);

        if (!varianceBased)
        {
            embree::vfloat<VMM::VectorSize> volumeContributionWeight = select(numVolSamples > FLT_EPSILON, sufficientStats.volumeContributionWeights[k] / numVolSamples, zeros);
            embree::vfloat<VMM::VectorSize> surfaceContributionWeight = select(numSurfSamples > FLT_EPSILON, sufficientStats.surfaceContributionWeights[k] / numSurfSamples, zeros);
            embree::vfloat<VMM::VectorSize> sumContributionWeight = (surfaceContributionWeight * (ones - estPVol)) + (volumeContributionWeight * estPVol);
            vmm._volumeScatterProbabilityWeights[k] = select(sumContributionWeight > FLT_EPSILON, (volumeContributionWeight * estPVol) / sumContributionWeight, zeros);
        }
        else
        {
            embree::vfloat<VMM::VectorSize> volumeContributionSecondMomentWeight =
                select(numVolSamples > FLT_EPSILON, sufficientStats.volumeContributionWeights[k] / numVolSamples, zeros);
            volumeContributionSecondMomentWeight *= (estPVol * estPVol);
            volumeContributionSecondMomentWeight = select(volumeContributionSecondMomentWeight > FLT_EPSILON, embree::sqrt(volumeContributionSecondMomentWeight), zeros);

            embree::vfloat<VMM::VectorSize> surfaceContributionSecondMomentWeight =
                select(numSurfSamples > FLT_EPSILON, sufficientStats.surfaceContributionWeights[k] / numSurfSamples, zeros);
            surfaceContributionSecondMomentWeight *= (ones - estPVol) * (ones - estPVol);
            surfaceContributionSecondMomentWeight = select(surfaceContributionSecondMomentWeight > FLT_EPSILON, embree::sqrt(surfaceContributionSecondMomentWeight), zeros);

            embree::vfloat<VMM::VectorSize> sumContributionSecondMomentWeight = surfaceContributionSecondMomentWeight + volumeContributionSecondMomentWeight;
            vmm._volumeScatterProbabilityWeights[k] =
                select(sumContributionSecondMomentWeight > FLT_EPSILON, volumeContributionSecondMomentWeight / sumContributionSecondMomentWeight, zeros);
        }
    }

    if (rem > 0)
    {
        for (size_t i = rem; i < VMM::VectorSize; i++)
        {
            vmm._volumeScatterProbabilityWeights[cnt - 1][i] = 0.0f;
            sufficientStats.volumeContributionWeights[cnt - 1][i] = 0.0f;
            sufficientStats.surfaceContributionWeights[cnt - 1][i] = 0.0f;
            sufficientStats.volumeSampleNumberWeights[cnt - 1][i] = 0.0f;
            sufficientStats.surfaceSampleNumberWeights[cnt - 1][i] = 0.0f;
        }
    }
}
#endif
}  // namespace openpgl
