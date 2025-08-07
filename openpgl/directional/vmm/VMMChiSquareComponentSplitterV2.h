// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <embreeSrc/common/math/vec2.h>
#include <embreeSrc/common/math/vec3.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>

#include "../../data/SampleData.h"
#include "../../openpgl_common.h"
#include "ParallaxAwareVonMisesFisherMixture.h"

#define OPENPGL_USE_LOGMAP
#define OPENPGL_ZERO_MEAN
// #define OPENPGL_USE_THREE_SPLIT

namespace openpgl
{

struct ComponentSplitinfoV2
{
    Vector2 mean{0.0f};
    Vector3 covariance{0.0f};

    float eigenValue0{0.0f};
    float eigenValue1{0.0f};

    Vector2 eigenVector0{0.0f};
    Vector2 eigenVector1{0.0f};

    std::string toString() const;
};

template <class TVMMFactory>
struct VonMisesFisherChiSquareComponentSplitterV2
{
   public:
    typedef typename TVMMFactory::Distribution VMM;
    typedef TVMMFactory VMMFactory;

    typedef typename VMMFactory::Configuration Configuration;
    typedef typename VMMFactory::SufficientStatistics SufficientStatistics;
    typedef typename VMMFactory::PartialFittingMask PartialFittingMask;

    enum SplitType
    {
        EFirefly = 0,
        EMultiModal,
        ENone
    };

    struct SplitCandidate
    {
        size_t componentIndex;
        float chiSquareEst;

        SplitType splitType;

        bool operator<(const SplitCandidate &sc) const
        {
            return chiSquareEst < sc.chiSquareEst;
        }

        bool operator>(const SplitCandidate &sc) const
        {
            return chiSquareEst > sc.chiSquareEst;
        }
    };

    struct ComponentSplitStatistics
    {
        ComponentSplitStatistics() = default;

        embree::vfloat<VMM::VectorSize> chiSquareMCEstimates[VMM::NumVectors];
        embree::vfloat<VMM::VectorSize> chiSquareMCEstimate2ndMoments[VMM::NumVectors];
        embree::Vec2<embree::vfloat<VMM::VectorSize> > splitMeans[VMM::NumVectors];
        embree::Vec3<embree::vfloat<VMM::VectorSize> > splitWeightedSampleCovariances[VMM::NumVectors];

        embree::vfloat<VMM::VectorSize> weightsEstimates[VMM::NumVectors];
        // TODO: check if we really need both variance and 2nd moment
        // TODO: find a better name dborFireFly....
        embree::vfloat<VMM::VectorSize> weights2ndmomentEstimates[VMM::NumVectors];
        embree::vfloat<VMM::VectorSize> weightsVarianceEstimates[VMM::NumVectors];
        embree::vfloat<VMM::VectorSize> numWeightsEstimatesSamples[VMM::NumVectors];

        embree::vint<VMM::VectorSize> splitType[VMM::NumVectors];

        embree::vfloat<VMM::VectorSize> numSamples[VMM::NumVectors];
        embree::vfloat<VMM::VectorSize> sumWeights[VMM::NumVectors];

        embree::vfloat<VMM::VectorSize> sumAssignedSamples[VMM::NumVectors];

        size_t numComponents{0};

        // sufficient stats for a single (firefly) component of the last update step
        embree::vfloat<VMM::VectorSize> weights[VMM::NumVectors];
        embree::Vec3<embree::vfloat<VMM::VectorSize> > weightedMeans[VMM::NumVectors];

        void clear(const size_t &_numComponents);
        void clearMasked(const size_t &_numComponents, const PartialFittingMask &mask);
        void clearAll();

        void replaceReset(const div_t k, const div_t l);
        void reset(const div_t k);

        float getChiSquareEst(const size_t &idx) const;
        float getChiSquare2ndMomentEst(const size_t &idx) const;

        float getWeightsEst(const size_t &idx) const;
        float getRelVarianceEst(const size_t &idx) const;
        float getVarianceEst(const size_t &idx) const;
        float getWeights2ndMomentEst(const size_t &idx) const;
        float getWeightsVarianceEst(const size_t &idx) const;

        float getSumChiSquareEst() const;
        size_t getHighestChiSquareIdx() const;

        bool getHighestValidSplitComponent(SplitCandidate &splitCandiate, const float splitChiSquareThreshold, const VMM &vmm, const bool *alreadySplitted,
                                           const bool useConfidence) const;

        void mergeComponentStats(const size_t &idxI, const size_t &idxJ, const float &weightI, const Vector3 &meanDirectionI, const float &weightJ, const Vector3 &meanDirectionJ,
                                 const float &weightK, const Vector3 &meanDirectionK);

        Vector2 getSplitMean(const size_t &idx) const;

        Vector3 getSplitCovariance(const size_t &idx) const;

        SplitType getSplitType(const size_t &idx) const;

        std::vector<SplitCandidate> getSplitCandidates(const float splitThreshold, const bool useConfidence) const;

        void decay(const float &alpha);

        bool isValid() const;

        void serialize(std::ostream &stream) const;

        void deserialize(std::istream &stream);

        inline size_t getNumComponents() const
        {
            return numComponents;
        }

        void setNumComponents(const size_t &n)
        {
            numComponents = n;
        }

        std::string toString() const;

        bool operator==(const ComponentSplitStatistics &b) const;
    };

    bool SplitAndUpdate(VMM &vmm, const float &mcEstimate, const SplitCandidate &candidate, ComponentSplitStatistics &splitStatistics, SufficientStatistics &suffStatistics,
                        const SampleData *data, const size_t &numData, const typename VMMFactory::Configuration factoryCfg, const bool &doPartialRefit) const;

    void CalucalteWeightsEstimates(const VMM &vmm, ComponentSplitStatistics &splitStats, const SampleData *data, const size_t &numData) const;

    void CalculateSplitStatistics(const VMM &vmm, ComponentSplitStatistics &splitStats, const float &mcEstimate, const SampleData *data, const size_t &numData) const;

    void PartialCalculateSplitStatistics(const VMM &vmm, const PartialFittingMask &mask, ComponentSplitStatistics &splitStats, const float &mcEstimate, const SampleData *data,
                                         const size_t &numData) const;

    void UpdateSplitStatistics(const VMM &vmm, ComponentSplitStatistics &splitStats, const float &mcEstimate, const SampleData *data, const size_t &numData,
                               bool updateWeightsEstimates, bool onlyConsiderFireflySamples) const;

    void PartialUpdateSplitStatistics(const VMM &vmm, const PartialFittingMask &mask, ComponentSplitStatistics &splitStats, const float &mcEstimate, const SampleData *data,
                                      const size_t &numData) const;

    bool SplitComponentMultiModal(VMM &vmm, ComponentSplitStatistics &splitStats, SufficientStatistics &suffStats, const size_t idx) const;

    bool SplitComponentFireFly(VMM &vmm, ComponentSplitStatistics &splitStats, SufficientStatistics &suffStats, const size_t idx, const Configuration &cfg) const;
#ifdef OPENPGL_USE_THREE_SPLIT
    bool SplitComponentIntoThree(VMM &vmm, ComponentSplitStatistics &splitStats, SufficientStatistics &suffStats, const size_t idx) const;
#endif
    ComponentSplitinfoV2 GetProjectedLocalDirections(const VMM &vmm, const size_t &idx, const SampleData *data, const size_t &numData, Vector3 *local2D) const;
};
/*
#ifndef OPENPGL_USE_LOGMAP

template <typename Vec3Type, typename Vec2Type, typename ScalarType>
inline Vec2Type Map3DTo2D(const Vec3Type &vec3D)
{
    return Vec2Type(vec3D.x, vec3D.y);
}

template <typename Vec3Type, typename Vec2Type, typename ScalarType>
inline Vec3Type Map2DTo3D(const Vec2Type &vec2D)
{
    Vec3Type vec3D = Vec3Type(0.0f);
    vec3D.x = vec2D.x;
    vec3D.y = vec2D.y;
    vec3D.z = embree::sqrt(1.0f - vec2D.x * vec2D.x - vec2D.y * vec2D.y);
    return vec3D;
}

#else

// logMapping https://ronnybergmann.net/mvirt/manifolds/Sn/log.html
template <typename Vec3Type, typename Vec2Type, typename ScalarType>
inline Vec2Type Map3DTo2D(const Vec3Type &vec3D)
{
    Vec2Type vec2D(0.0f);

    // OPENPGL_ASSERT((vec3D.z <= 1.0f &&  vec3D.z >= -1.0f));
    ScalarType alpha = embree::fastapprox::acos(vec3D.z);
    ScalarType inv_sinc = alpha / embree::fastapprox::sin(alpha);
    // TODO: Needs to be implmented

    vec2D.x = embree::select(alpha > 0.0f, vec3D.x * inv_sinc, vec2D.x);
    vec2D.y = embree::select(alpha > 0.0f, vec3D.y * inv_sinc, vec2D.y);
    return vec2D;
}

// expMapping https://ronnybergmann.net/mvirt/manifolds/Sn/exp.html
template <typename Vec3Type, typename Vec2Type, typename ScalarType>
inline Vec3Type Map2DTo3D(const Vec2Type &vec2D)
{
    Vec3Type vec3D = Vec3Type(0.0f);
    ScalarType length = embree::sqrt(vec2D.x * vec2D.x + vec2D.y * vec2D.y);
    OPENPGL_ASSERT(length < M_PI_F);
    ScalarType sinc = embree::fastapprox::sin(length) / length;

    vec3D.x = embree::select(length > 0.0f, vec2D.x * sinc, vec3D.x);
    vec3D.y = embree::select(length > 0.0f, vec2D.y * sinc, vec3D.y);
    vec3D.z = embree::cos(length);

    return vec3D;
}

#endif
*/
inline std::string ComponentSplitinfoV2::toString() const
{
    std::stringstream ss;
    ss << "ComponentSplitinfoV2:" << std::endl;
    // ss << "mean: " << mean << std::endl;
    // ss << "covariance: " << covariance << std::endl;
    ss << "eigenValue0: " << eigenValue0 << std::endl;
    ss << "eigenValue1: " << eigenValue1 << std::endl;
    // ss << "eigenVector0: " << eigenVector0 << std::endl;
    // ss << "eigenVector1: " << eigenVector1 << std::endl;
    return ss.str();
}

template <class TVMMFactory>
void VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::CalculateSplitStatistics(const VMM &vmm, ComponentSplitStatistics &splitStats, const float &mcEstimate,
                                                                                       const SampleData *data, const size_t &numData) const
{
    splitStats.clear(vmm._numComponents);
    // Calculating the estimatates for components' mean weights and their second moments which are used for the firefly detection
    this->CalucalteWeightsEstimates(vmm, splitStats, data, numData);
    OPENPGL_ASSERT(splitStats.isValid());
    // Calcualte/Update the split statistics
    this->UpdateSplitStatistics(vmm, splitStats, mcEstimate, data, numData, false, true);
    OPENPGL_ASSERT(splitStats.isValid());
}

template <class TVMMFactory>
void VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::PartialCalculateSplitStatistics(const VMM &vmm, const PartialFittingMask &mask, ComponentSplitStatistics &splitStats,
                                                                                              const float &mcEstimate, const SampleData *data, const size_t &numData) const
{
    // Reset the split statistics for the components the stats should be caclulated
    splitStats.clearMasked(vmm._numComponents, mask);
    // Perform a partial updated of the split statistics. Due to the previous reset this is similar to a recalculation of the stats.
    this->PartialUpdateSplitStatistics(vmm, mask, splitStats, mcEstimate, data, numData);
}

template <class TVMMFactory>
bool VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::SplitAndUpdate(VMM &vmm, const float &mcEstimate, const SplitCandidate &candidate,
                                                                             ComponentSplitStatistics &splitStatistics, SufficientStatistics &suffStatistics,
                                                                             const SampleData *data, const size_t &numData, const typename VMMFactory::Configuration factoryCfg,
                                                                             const bool &doPartialRefit) const
{
    OPENPGL_ASSERT(vmm.isValid());
    PartialFittingMask mask;
    PartialFittingMask previousAsPriorMask;

    VMMFactory vmmFactory;
    typename VMMFactory::FittingStatistics vmmFitStats;
    // If the proposed split was successfull or not.
    bool splitSuccess = false;
    // Checking if the mixture still has enough free components.
#ifndef OPENPGL_USE_THREE_SPLIT
    if (vmm._numComponents < VMM::MaxComponents)
#else
    if (vmm._numComponents < VMM::MaxComponents - 1)
#endif
    {
        previousAsPriorMask.resetToTrue();
        mask.resetToFalse();
#ifndef OPENPGL_USE_THREE_SPLIT
        size_t idx = candidate.componentIndex;
        if (candidate.splitType == EFirefly)
        {
            splitSuccess = SplitComponentFireFly(vmm, splitStatistics, suffStatistics, idx, factoryCfg);
             // For a firefly split we use both previous component stats as prior
            previousAsPriorMask.setToTrue(idx);
            previousAsPriorMask.setToFalse(vmm._numComponents - 1);
        }
        else
        {
            splitSuccess = SplitComponentMultiModal(vmm, splitStatistics, suffStatistics, idx);
            // For a multi modal split we use both previous component stats as prior 
            previousAsPriorMask.setToTrue(idx);
            previousAsPriorMask.setToTrue(vmm._numComponents - 1);
        }

        // If the split was successfull we tag both components for partial refitting/updating
        if (splitSuccess)
        {
            mask.setToTrue(idx);
            mask.setToTrue(vmm._numComponents - 1);
        }
#else
        bool splitSuccess = SplitComponentIntoThree(vmm, splitStatistics, suffStatistics, idx);
        mask.setToTrue(idx);
        mask.setToTrue(vmm._numComponents - 2);
        mask.setToTrue(vmm._numComponents - 1);
#endif

        // If the split was succesfull we partialy update the mixture and recalculate the splitting statistics
        if (true && splitSuccess)
        {
            OPENPGL_ASSERT(vmm.isValid());
            vmmFactory.partialUpdateMixture(vmm, mask, true, previousAsPriorMask, suffStatistics, data, numData, factoryCfg, vmmFitStats);
            OPENPGL_ASSERT(vmm.isValid());
            this->PartialCalculateSplitStatistics(vmm, mask, splitStatistics, mcEstimate, data, numData);
            OPENPGL_ASSERT(vmm.isValid());
        }
    }
    return splitSuccess;
}

template <class TVMMFactory>
ComponentSplitinfoV2 VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::GetProjectedLocalDirections(const VMM &vmm, const size_t &idx, const SampleData *data,
                                                                                                          const size_t &numData, Vector3 *local2D) const
{
    typename VMM::SoftAssignment softAssign;
    const embree::vfloat<VMM::VectorSize> zeros(0.f);

    ComponentSplitinfoV2 splitInfo;

    Vector2 mean(0.0f);
    Vector3 covarianceStats(0.0f);
    float sumWeights = 0.0f;

    for (size_t n = 0; n < numData; n++)
    {
        const SampleData sample = data[n];
        pgl_vec3f direction = sample.direction;
        openpgl::Vector3 sampleDirection(direction.x, direction.y, direction.z);
        if (vmm.softAssignment(sampleDirection, softAssign))
        {
            const div_t tmp = div(idx, static_cast<int>(VMM::VectorSize));

            const embree::vfloat<VMM::VectorSize> weight = sample.weight;

            const embree::Vec3<embree::vfloat<VMM::VectorSize> > localDirection =
                embree::frame(vmm._meanDirections[tmp.quot]).inverse() * embree::Vec3<embree::vfloat<VMM::VectorSize> >(sampleDirection);
            const Vector2 localDirection2D = Map3DTo2D<Vector3, Vector2, float>(Vector3(localDirection.x[tmp.rem], localDirection.y[tmp.rem], localDirection.z[tmp.rem]));

            const embree::vfloat<VMM::VectorSize> assignedWeight = softAssign.assignments[tmp.quot] * weight;
            local2D[n].x = localDirection2D.x;
            local2D[n].y = localDirection2D.y;
            local2D[n].z = assignedWeight[tmp.rem];

            sumWeights += assignedWeight[tmp.rem];
#ifdef OPENPGL_ZERO_MEAN
            mean.x += 0.0f;
            mean.y += 0.0f;
#else
            mean.x += assignedWeight[tmp.rem] * localDirection2D.x;
            mean.y += assignedWeight[tmp.rem] * localDirection2D.y;
#endif
            covarianceStats.x += assignedWeight[tmp.rem] * localDirection2D.x * localDirection2D.x;
            covarianceStats.y += assignedWeight[tmp.rem] * localDirection2D.y * localDirection2D.y;
            covarianceStats.z += assignedWeight[tmp.rem] * localDirection2D.x * localDirection2D.y;
        }
    }
    mean /= sumWeights;

    splitInfo.mean = mean;
    splitInfo.covariance.x = covarianceStats.x / sumWeights - mean.x * mean.x;
    splitInfo.covariance.y = covarianceStats.y / sumWeights - mean.y * mean.y;
    splitInfo.covariance.z = covarianceStats.z / sumWeights - mean.x * mean.y;

    float D = embree::sqrt((splitInfo.covariance.x - splitInfo.covariance.y) * (splitInfo.covariance.x - splitInfo.covariance.y) +
                           (splitInfo.covariance.z * splitInfo.covariance.z * 4.0f)) *
              0.5f;
    splitInfo.eigenValue0 = (splitInfo.covariance.x + splitInfo.covariance.y) * 0.5;
    splitInfo.eigenValue0 += D;

    splitInfo.eigenValue1 = (splitInfo.covariance.x + splitInfo.covariance.y) * 0.5;
    splitInfo.eigenValue1 -= D;

    splitInfo.eigenVector0.x = -splitInfo.covariance.z;
    splitInfo.eigenVector0.y = splitInfo.covariance.x - splitInfo.eigenValue0;

    splitInfo.eigenVector1.x = splitInfo.covariance.z;
    splitInfo.eigenVector1.y = splitInfo.covariance.x - splitInfo.eigenValue1;

    splitInfo.eigenVector0 /= embree::sqrt(splitInfo.eigenVector0.x * splitInfo.eigenVector0.x + splitInfo.eigenVector0.y * splitInfo.eigenVector0.y);
    splitInfo.eigenVector1 /= embree::sqrt(splitInfo.eigenVector1.x * splitInfo.eigenVector1.x + splitInfo.eigenVector1.y * splitInfo.eigenVector1.y);
#ifdef OPENPGL_SHOW_PRINT_OUTS
    std::cout << "split: " << "\tmean: " << splitInfo.mean.x << ", \t " << splitInfo.mean.y << "\t covariance: " << splitInfo.covariance.x << ", \t " << splitInfo.covariance.y
              << ", \t " << splitInfo.covariance.z << std::endl;
    std::cout << "eigen: " << "\tevalue0: " << splitInfo.eigenValue0 << "\teVec0: " << splitInfo.eigenVector0.x << ", \t " << splitInfo.eigenVector0.y
              << "\tevalue1: " << splitInfo.eigenValue1 << "\teVec1: " << splitInfo.eigenVector1.x << ", \t " << splitInfo.eigenVector1.y << std::endl;
#endif
    return splitInfo;
}

template <class TVMMFactory>
void VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::CalucalteWeightsEstimates(const VMM &vmm, ComponentSplitStatistics &splitStats, const SampleData *data,
                                                                                        const size_t &numData) const
{
    OPENPGL_ASSERT(vmm._numComponents == splitStats.numComponents);

    typename VMM::SoftAssignment softAssign;
    const embree::vfloat<VMM::VectorSize> zeros(0.f);
    const int cnt = (splitStats.numComponents + VMM::VectorSize - 1) / VMM::VectorSize;

    // Resetting the component weights statistics
    for (size_t k = 0; k < cnt; k++)
    {
        splitStats.weightsEstimates[k] = zeros;
        splitStats.weights2ndmomentEstimates[k] = zeros;
        splitStats.weightsVarianceEstimates[k] = zeros;
        splitStats.numWeightsEstimatesSamples[k] = zeros;
    }

    // For each sample update the weights mean, second moment, variance, and number of samples using
    // the numerical stable incremental mean and variance algorithms.
    for (size_t n = 0; n < numData; n++)
    {
        const SampleData sample = data[n];
        pgl_vec3f direction = sample.direction;
        const openpgl::Vector3 sampleDirection(direction.x, direction.y, direction.z);
        if (vmm.softAssignment(sampleDirection, softAssign))
        {
            const embree::vfloat<VMM::VectorSize> weight = sample.weight;
            for (size_t k = 0; k < cnt; k++)
            {
                const embree::vfloat<VMM::VectorSize> assignedWeight = softAssign.assignments[k] * weight;
                splitStats.numWeightsEstimatesSamples[k] += 1.0f;
                auto oldWeightsEstimates = splitStats.weightsEstimates[k];
                splitStats.weightsEstimates[k] += (assignedWeight - splitStats.weightsEstimates[k]) / splitStats.numWeightsEstimatesSamples[k];
                splitStats.weights2ndmomentEstimates[k] += (assignedWeight * assignedWeight - splitStats.weights2ndmomentEstimates[k]) / splitStats.numWeightsEstimatesSamples[k];
                splitStats.weightsVarianceEstimates[k] += (assignedWeight - splitStats.weightsEstimates[k]) * (assignedWeight - oldWeightsEstimates);
            }
        }
    }
}

template <class TVMMFactory>
void VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::UpdateSplitStatistics(const VMM &vmm, ComponentSplitStatistics &splitStats, const float &mcEstimate,
                                                                                    const SampleData *data, const size_t &numData, bool updateWeightsEstimates,
                                                                                    bool onlyConsiderFireflySamples) const
{
    OPENPGL_ASSERT(vmm._numComponents == splitStats.numComponents);

    typename VMM::SoftAssignment softAssign;
    const embree::vfloat<VMM::VectorSize> zeros(0.f);
    const int cnt = (splitStats.numComponents + VMM::VectorSize - 1) / VMM::VectorSize;

    const embree::vint<VMM::VectorSize> stFF((int32_t)EFirefly);
    const embree::vint<VMM::VectorSize> stMM((int32_t)EMultiModal);

    // Resetting stats for the fire fly componets and the setting the split type to multi modal
    for (size_t k = 0; k < cnt; k++)
    {
        splitStats.weights[k] = zeros;
        splitStats.weightedMeans[k].x = zeros;
        splitStats.weightedMeans[k].y = zeros;
        splitStats.weightedMeans[k].z = zeros;
        splitStats.splitType[k] = stMM;
    }

    // For each new sample we update the chi square estimates and the split component of each mixture component
    for (size_t n = 0; n < numData; n++)
    {
        const SampleData sample = data[n];
        pgl_vec3f direction = sample.direction;
        const openpgl::Vector3 sampleDirection(direction.x, direction.y, direction.z);

        if (vmm.softAssignment(sampleDirection, softAssign))
        {
            const embree::vfloat<VMM::VectorSize> weight = sample.weight;
            const embree::vfloat<VMM::VectorSize> samplePDF = sample.pdf;
            const embree::vfloat<VMM::VectorSize> value = weight * samplePDF;

            // For numerical reasons we ensure a max value of the samples' mixture pdf
            softAssign.pdf = std::max(softAssign.pdf, FLT_EPSILON);

            for (size_t k = 0; k < cnt; k++)
            {
                const embree::vfloat<VMM::VectorSize> weightMean = splitStats.weightsEstimates[k];
                const embree::vfloat<VMM::VectorSize> weightStd =
                    select(splitStats.weights2ndmomentEstimates[k] > 0, embree::sqrt(splitStats.weights2ndmomentEstimates[k] - weightMean * weightMean), 0.f);

                const embree::vfloat<VMM::VectorSize> assignedWeight = softAssign.assignments[k] * weight;
                OPENPGL_ASSERT(embree::all(embree::isvalid(splitStats.splitMeans[k].x)));
                OPENPGL_ASSERT(embree::all(embree::isvalid(splitStats.splitMeans[k].y)));
                OPENPGL_ASSERT(embree::all(embree::isvalid(splitStats.splitWeightedSampleCovariances[k].x)));
                OPENPGL_ASSERT(embree::all(embree::isvalid(splitStats.splitWeightedSampleCovariances[k].y)));
                OPENPGL_ASSERT(embree::all(embree::isvalid(splitStats.splitWeightedSampleCovariances[k].z)));

                // Checking based on the assigned mean and std if the current sample is a firefly or not
                // We use different splitting methods if a split is triggered by a firefly and not by a multi modal distribution
                splitStats.splitType[k] = select((assignedWeight > weightMean + 3.f * weightStd), stFF, splitStats.splitType[k]);

                embree::vfloat<VMM::VectorSize> assignedWeightTmp = assignedWeight;
                if (onlyConsiderFireflySamples)
                    assignedWeightTmp = select((assignedWeight > weightMean + 3.f * weightStd), assignedWeight, zeros);
                // Updating the firefly component statistics for the components
                splitStats.weights[k] += assignedWeightTmp;
                splitStats.weightedMeans[k].x += assignedWeightTmp * direction.x;
                splitStats.weightedMeans[k].y += assignedWeightTmp * direction.y;
                splitStats.weightedMeans[k].z += assignedWeightTmp * direction.z;

                // Calculate the chi square estimate for the current sample
                embree::vfloat<VMM::VectorSize> vmfPDF = softAssign.assignments[k] * softAssign.pdf;
                embree::vfloat<VMM::VectorSize> partialValuePDF = vmfPDF * value;
                partialValuePDF /= (mcEstimate * softAssign.pdf);

                embree::vfloat<VMM::VectorSize> chiSquareEst = value * value * vmfPDF;
                chiSquareEst /= mcEstimate * mcEstimate * softAssign.pdf * softAssign.pdf;
                chiSquareEst -= 2.0f * partialValuePDF;
                chiSquareEst += vmfPDF;
                chiSquareEst /= samplePDF;

                // For numerical reasons we check if the soft assignment is not close to zero (i.e., sample is unimportant for the current component)
                chiSquareEst = select(softAssign.assignments[k] > FLT_EPSILON, chiSquareEst, zeros);

                splitStats.sumAssignedSamples[k] += softAssign.assignments[k];
                // Incremental updated of the MC chiSquare estimate, its second moment and variance
                splitStats.numSamples[k] += 1.0f;
                splitStats.chiSquareMCEstimates[k] += (chiSquareEst - splitStats.chiSquareMCEstimates[k]) / splitStats.numSamples[k];
                splitStats.chiSquareMCEstimate2ndMoments[k] += (chiSquareEst * chiSquareEst - splitStats.chiSquareMCEstimate2ndMoments[k]) / splitStats.numSamples[k];

                // Updating the weights mean, 2nd moment and variance estimates for the DBOR-based firefly detector if requested.
                if (updateWeightsEstimates)
                {
                    splitStats.numWeightsEstimatesSamples[k] += 1.0f;
                    auto oldWeightsEstimates = splitStats.weightsEstimates[k];
                    splitStats.weightsEstimates[k] += (assignedWeight - splitStats.weightsEstimates[k]) / splitStats.numWeightsEstimatesSamples[k];
                    splitStats.weights2ndmomentEstimates[k] +=
                        (assignedWeight * assignedWeight - splitStats.weights2ndmomentEstimates[k]) / splitStats.numWeightsEstimatesSamples[k];
                    splitStats.weightsVarianceEstimates[k] += (assignedWeight - splitStats.weightsEstimates[k]) * (assignedWeight - oldWeightsEstimates);
                }

                splitStats.sumWeights[k] += assignedWeight;

                /////////////////////////////////////////////////////
                // Updating split component
                /////////////////////////////////////////////////////

                // Transforming the sample direction to the local frame of the mixture component.
                const embree::Vec3<embree::vfloat<VMM::VectorSize> > localDirection =
                    embree::frame(vmm._meanDirections[k]).inverse() * embree::Vec3<embree::vfloat<VMM::VectorSize> >(sampleDirection);
                const embree::Vec2<embree::vfloat<VMM::VectorSize> > localDirection2D(localDirection.x, localDirection.y);

                // Updating the mean and covariance for the split compoment.
#ifdef OPENPGL_ZERO_MEAN
                splitStats.splitMeans[k] += embree::Vec2<embree::vfloat<VMM::VectorSize> >(0.0f);
                splitStats.splitWeightedSampleCovariances[k].x += assignedWeight * (localDirection2D.x * localDirection2D.x);
                splitStats.splitWeightedSampleCovariances[k].y += assignedWeight * (localDirection2D.y * localDirection2D.y);
                splitStats.splitWeightedSampleCovariances[k].z += assignedWeight * (localDirection2D.x * localDirection2D.y);
#else
                const Vec2<vfloat<VMM::VectorSize> > previousSplitMeans = splitStats.splitMeans[k];
                splitStats.splitMeans[k] += incWeight * (localDirection2D - splitStats.splitMeans[k]);
                splitStats.splitWeightedSampleCovariances[k].x +=
                    assignedWeight * ((localDirection2D.x - previousSplitMeans.x) * (localDirection2D.x - splitStats.splitMeans[k].x));
                splitStats.splitWeightedSampleCovariances[k].y +=
                    assignedWeight * ((localDirection2D.y - previousSplitMeans.y) * (localDirection2D.y - splitStats.splitMeans[k].y));
                splitStats.splitWeightedSampleCovariances[k].z +=
                    assignedWeight * ((localDirection2D.x - previousSplitMeans.x) * (localDirection2D.y - splitStats.splitMeans[k].y));
#endif
                OPENPGL_ASSERT(embree::all(embree::isvalid(assignedWeight)));
                OPENPGL_ASSERT(embree::all(embree::isvalid(splitStats.splitMeans[k].x)));
                OPENPGL_ASSERT(embree::all(embree::isvalid(splitStats.splitMeans[k].y)));
                OPENPGL_ASSERT(embree::all(embree::isvalid(splitStats.splitWeightedSampleCovariances[k].x)));
                OPENPGL_ASSERT(embree::all(embree::isvalid(splitStats.splitWeightedSampleCovariances[k].y)));
                OPENPGL_ASSERT(embree::all(embree::isvalid(splitStats.splitWeightedSampleCovariances[k].z)));
            }
        }
    }
}

template <class TVMMFactory>
void VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::PartialUpdateSplitStatistics(const VMM &vmm, const PartialFittingMask &mask, ComponentSplitStatistics &splitStats,
                                                                                           const float &mcEstimate, const SampleData *data, const size_t &numData) const
{
    OPENPGL_ASSERT(vmm._numComponents == splitStats.numComponents);

    typename VMM::SoftAssignment softAssign;
    const embree::vfloat<VMM::VectorSize> zeros(0.f);
    const int cnt = (splitStats.numComponents + VMM::VectorSize - 1) / VMM::VectorSize;

    const embree::vint<VMM::VectorSize> stFF((int32_t)EFirefly);
    const embree::vint<VMM::VectorSize> stMM((int32_t)EMultiModal);

    // Resetting stats, for selected component, for the fire fly componets and the setting the split type to multi modal
    for (size_t k = 0; k < cnt; k++)
    {
        splitStats.weights[k] = select(mask.mask[k], zeros, splitStats.weights[k]);
        splitStats.weightedMeans[k].x = select(mask.mask[k], zeros, splitStats.weightedMeans[k].x);
        splitStats.weightedMeans[k].y = select(mask.mask[k], zeros, splitStats.weightedMeans[k].y);
        splitStats.weightedMeans[k].z = select(mask.mask[k], zeros, splitStats.weightedMeans[k].z);
        splitStats.splitType[k] = select(mask.mask[k], stMM, splitStats.splitType[k]);
    }

    // For each new sample we update the chi square estimates and the split component of each selected mixture component
    for (size_t n = 0; n < numData; n++)
    {
        const SampleData sample = data[n];
        const pgl_vec3f direction = sample.direction;
        const Vector3 sampleDirection(direction.x, direction.y, direction.z);

        if (vmm.softAssignment(sampleDirection, softAssign))
        {
            const embree::vfloat<VMM::VectorSize> weight = sample.weight;
            const embree::vfloat<VMM::VectorSize> samplePDF = sample.pdf;
            const embree::vfloat<VMM::VectorSize> value = weight * samplePDF;

            // For numerical reasons we ensure a max value of the samples' mixture pdf
            softAssign.pdf = std::max(softAssign.pdf, FLT_EPSILON);

            for (size_t k = 0; k < cnt; k++)
            {
                OPENPGL_ASSERT(embree::all(embree::isvalid(splitStats.splitMeans[k].x)));
                OPENPGL_ASSERT(embree::all(embree::isvalid(splitStats.splitMeans[k].y)));
                OPENPGL_ASSERT(embree::all(embree::isvalid(splitStats.splitWeightedSampleCovariances[k].x)));
                OPENPGL_ASSERT(embree::all(embree::isvalid(splitStats.splitWeightedSampleCovariances[k].y)));
                OPENPGL_ASSERT(embree::all(embree::isvalid(splitStats.splitWeightedSampleCovariances[k].z)));

                const embree::vfloat<VMM::VectorSize> assignedWeight = softAssign.assignments[k] * weight;

                // Updating the firefly component statistics for the selected components
                splitStats.weights[k] = select(mask.mask[k], splitStats.weights[k] + assignedWeight, splitStats.weights[k]);
                splitStats.weightedMeans[k].x = select(mask.mask[k], splitStats.weightedMeans[k].x + assignedWeight * direction.x, splitStats.weightedMeans[k].x);
                splitStats.weightedMeans[k].y = select(mask.mask[k], splitStats.weightedMeans[k].y + assignedWeight * direction.y, splitStats.weightedMeans[k].y);
                splitStats.weightedMeans[k].z = select(mask.mask[k], splitStats.weightedMeans[k].z + assignedWeight * direction.z, splitStats.weightedMeans[k].z);

                // Calculate the chi square estimate for the current sample and the selected mixture components
                embree::vfloat<VMM::VectorSize> vmfPDF = softAssign.assignments[k] * softAssign.pdf;
                embree::vfloat<VMM::VectorSize> partialValuePDF = vmfPDF * value;
                partialValuePDF /= (mcEstimate * softAssign.pdf);

                embree::vfloat<VMM::VectorSize> chiSquareEst = value * value * vmfPDF;
                chiSquareEst /= mcEstimate * mcEstimate * softAssign.pdf * softAssign.pdf;
                chiSquareEst -= 2.0f * partialValuePDF;
                chiSquareEst += vmfPDF;
                chiSquareEst /= samplePDF;
                // Resetting the chi square estimates for the non-selected mixture components
                chiSquareEst = select(softAssign.assignments[k] > FLT_EPSILON, chiSquareEst, zeros);

                splitStats.sumAssignedSamples[k] = select(mask.mask[k], splitStats.sumAssignedSamples[k] + softAssign.assignments[k], splitStats.sumAssignedSamples[k]);
                // Incremental updated of the MC chiSquare estimate, its second moment and variance for the selected mixture components
                splitStats.numSamples[k] = select(mask.mask[k], splitStats.numSamples[k] + 1.0f, splitStats.numSamples[k]);
                splitStats.chiSquareMCEstimates[k] =
                    select(mask.mask[k], splitStats.chiSquareMCEstimates[k] + (chiSquareEst - splitStats.chiSquareMCEstimates[k]) / splitStats.numSamples[k],
                           splitStats.chiSquareMCEstimates[k]);
                splitStats.chiSquareMCEstimate2ndMoments[k] =
                    select(mask.mask[k],
                           splitStats.chiSquareMCEstimate2ndMoments[k] + (chiSquareEst * chiSquareEst - splitStats.chiSquareMCEstimate2ndMoments[k]) / splitStats.numSamples[k],
                           splitStats.chiSquareMCEstimate2ndMoments[k]);

                // Updating the weights mean, 2nd moment and variance estimates for the DBOR-based firefly detector if requested.
                splitStats.numWeightsEstimatesSamples[k] = select(mask.mask[k], splitStats.numWeightsEstimatesSamples[k] + 1.0f, splitStats.numWeightsEstimatesSamples[k]);
                auto oldWeightsEstimates = splitStats.weightsEstimates[k];
                splitStats.weightsEstimates[k] =
                    select(mask.mask[k], splitStats.weightsEstimates[k] + (assignedWeight - splitStats.weightsEstimates[k]) / splitStats.numWeightsEstimatesSamples[k],
                           splitStats.weightsEstimates[k]);
                splitStats.weights2ndmomentEstimates[k] =
                    select(mask.mask[k],
                           splitStats.chiSquareMCEstimate2ndMoments[k] +
                               (assignedWeight * assignedWeight - splitStats.chiSquareMCEstimate2ndMoments[k]) / splitStats.numWeightsEstimatesSamples[k],
                           splitStats.chiSquareMCEstimate2ndMoments[k]);

                splitStats.weightsVarianceEstimates[k] =
                    select(mask.mask[k], splitStats.weightsVarianceEstimates[k] + ((assignedWeight - splitStats.weightsEstimates[k]) * (assignedWeight - oldWeightsEstimates)),
                           splitStats.weightsVarianceEstimates[k]);

                splitStats.sumWeights[k] += assignedWeight;

                /////////////////////////////////////////////////////
                // Updating split component
                /////////////////////////////////////////////////////

                // Transforming the sample direction to the local frame of the mixture component.
                const embree::Vec3<embree::vfloat<VMM::VectorSize> > localDirection =
                    embree::frame(vmm._meanDirections[k]).inverse() * embree::Vec3<embree::vfloat<VMM::VectorSize> >(sampleDirection);
                const embree::Vec2<embree::vfloat<VMM::VectorSize> > localDirection2D(localDirection.x, localDirection.y);

                // Updating the mean and covariance for the split compoment.
#ifdef OPENPGL_ZERO_MEAN
                splitStats.splitMeans[k] += embree::Vec2<embree::vfloat<VMM::VectorSize> >(0.0f);
                splitStats.splitWeightedSampleCovariances[k].x =
                    select(mask.mask[k], splitStats.splitWeightedSampleCovariances[k].x + assignedWeight * (localDirection2D.x * localDirection2D.x),
                           splitStats.splitWeightedSampleCovariances[k].x);
                splitStats.splitWeightedSampleCovariances[k].y =
                    select(mask.mask[k], splitStats.splitWeightedSampleCovariances[k].y + assignedWeight * (localDirection2D.y * localDirection2D.y),
                           splitStats.splitWeightedSampleCovariances[k].y);
                splitStats.splitWeightedSampleCovariances[k].z =
                    select(mask.mask[k], splitStats.splitWeightedSampleCovariances[k].z + assignedWeight * (localDirection2D.x * localDirection2D.y),
                           splitStats.splitWeightedSampleCovariances[k].z);
#else
                const Vec2<vfloat<VMM::VectorSize> > previousSplitMeans = splitStats.splitMeans[k];
                splitStats.splitMeans[k] = select(mask.mask[k], splitStats.splitMeans[k] + incWeight * (localDirection2D - splitStats.splitMeans[k]), splitStats.splitMeans[k]);
                splitStats.splitWeightedSampleCovariances[k].x =
                    select(mask.mask[k],
                           splitStats.splitWeightedSampleCovariances[k].x +
                               assignedWeight * ((localDirection2D.x - previousSplitMeans.x) * (localDirection2D.x - splitStats.splitMeans[k].x)),
                           splitStats.splitWeightedSampleCovariances[k].x);
                splitStats.splitWeightedSampleCovariances[k].y =
                    select(mask.mask[k],
                           splitStats.splitWeightedSampleCovariances[k].y +
                               assignedWeight * ((localDirection2D.y - previousSplitMeans.y) * (localDirection2D.y - splitStats.splitMeans[k].y)),
                           splitStats.splitWeightedSampleCovariances[k].y);
                splitStats.splitWeightedSampleCovariances[k].z =
                    select(mask.mask[k],
                           splitStats.splitWeightedSampleCovariances[k].z +
                               assignedWeight * ((localDirection2D.x - previousSplitMeans.x) * (localDirection2D.y - splitStats.splitMeans[k].y)),
                           splitStats.splitWeightedSampleCovariances[k].z);
#endif
                OPENPGL_ASSERT(embree::all(embree::isvalid(assignedWeight)));
                OPENPGL_ASSERT(embree::all(embree::isvalid(splitStats.splitMeans[k].x)));
                OPENPGL_ASSERT(embree::all(embree::isvalid(splitStats.splitMeans[k].y)));
                OPENPGL_ASSERT(embree::all(embree::isvalid(splitStats.splitWeightedSampleCovariances[k].x)));
                OPENPGL_ASSERT(embree::all(embree::isvalid(splitStats.splitWeightedSampleCovariances[k].y)));
                OPENPGL_ASSERT(embree::all(embree::isvalid(splitStats.splitWeightedSampleCovariances[k].z)));
            }
        }
    }
}

template <class TVMMFactory>
bool VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::SplitComponentMultiModal(VMM &vmm, ComponentSplitStatistics &splitStats, SufficientStatistics &suffStats,
                                                                                       const size_t idx) const
{
    ComponentSplitinfoV2 splitInfo;
    const div_t tmpK = div(idx, static_cast<int>(VMM::VectorSize));

    // The number of samples that got assinged to the split component after the last split/reset
    float numAssignedSamples = splitStats.sumAssignedSamples[tmpK.quot][tmpK.rem];

    float inv_sumWeights = embree::rcp(splitStats.sumWeights[tmpK.quot][tmpK.rem]);
    OPENPGL_ASSERT(embree::isvalid(inv_sumWeights));
    splitInfo.mean = Vector2(splitStats.splitMeans[tmpK.quot].x[tmpK.rem], splitStats.splitMeans[tmpK.quot].y[tmpK.rem]);
    // Converting the sample covariance matrix to a covariance matrix
    splitInfo.covariance.x = splitStats.splitWeightedSampleCovariances[tmpK.quot].x[tmpK.rem] * inv_sumWeights;
    splitInfo.covariance.y = splitStats.splitWeightedSampleCovariances[tmpK.quot].y[tmpK.rem] * inv_sumWeights;
    splitInfo.covariance.z = splitStats.splitWeightedSampleCovariances[tmpK.quot].z[tmpK.rem] * inv_sumWeights;

    // Perform the Eigen value decomposition
    float D = embree::sqrt((splitInfo.covariance.x - splitInfo.covariance.y) * (splitInfo.covariance.x - splitInfo.covariance.y) +
                           (splitInfo.covariance.z * splitInfo.covariance.z * 4.0f)) *
              0.5f;
    splitInfo.eigenValue0 = (splitInfo.covariance.x + splitInfo.covariance.y) * 0.5;
    splitInfo.eigenValue0 += D;

    splitInfo.eigenValue1 = (splitInfo.covariance.x + splitInfo.covariance.y) * 0.5;
    splitInfo.eigenValue1 -= D;

    splitInfo.eigenVector0.x = -splitInfo.covariance.z;
    splitInfo.eigenVector0.y = splitInfo.covariance.x - splitInfo.eigenValue0;

    splitInfo.eigenVector1.x = splitInfo.covariance.z;
    splitInfo.eigenVector1.y = splitInfo.covariance.x - splitInfo.eigenValue1;

    splitInfo.eigenVector0 /= embree::sqrt(splitInfo.eigenVector0.x * splitInfo.eigenVector0.x + splitInfo.eigenVector0.y * splitInfo.eigenVector0.y);
    splitInfo.eigenVector1 /= embree::sqrt(splitInfo.eigenVector1.x * splitInfo.eigenVector1.x + splitInfo.eigenVector1.y * splitInfo.eigenVector1.y);

    // Getting the original component parameters (e.g., weight mean cosine and kappa)
    float weight = vmm._weights[tmpK.quot][tmpK.rem];
    float meanCosine = vmm._meanCosines[tmpK.quot][tmpK.rem];
    float kappa = vmm._kappas[tmpK.quot][tmpK.rem];

    // We do not split a component which concentration (i.e., kappa) is already close to the maximum kappa.
    // This usally mean we have already reached the expressiveness of our model (e.g., the signal is smaller than our model can robustly represent).
    if (kappa >= OPENPGL_MAX_KAPPA * 0.9)
    {
        return false;
    }

    ///////////////////////////////////
    // Splitting the component
    ///////////////////////////////////
    Vector3 meanDirection = Vector3(vmm._meanDirections[tmpK.quot].x[tmpK.rem], vmm._meanDirections[tmpK.quot].y[tmpK.rem], vmm._meanDirections[tmpK.quot].z[tmpK.rem]);

    float newWeight0 = weight * 0.5f;
    float newWeight1 = newWeight0;

    // TODO: do we need the backup split or should we just cancle the split?
    // Intitialize the mean direction and mean cosine parameters for the new components with
    // backup values in case the Eigen value decomposiion was not sucessfull.
    Vector3 meanDirection0 = meanDirection;
    Vector3 meanDirection1 = meanDirection;

    float newMeanCosine0 = meanCosine;
    float newMeanCosine1 = meanCosine * meanCosine;

    // If the Eigen value decomposition was successfull we split the component according to the mean and covariance of the splitt statistics.
    if (D > 1e-8f)
    {
        // Offsetting the new mean directions in the mapped 2D space
        Vector2 meanDir2D0 = splitInfo.mean + (splitInfo.eigenVector0 * splitInfo.eigenValue0 * 0.5f);
        Vector2 meanDir2D1 = splitInfo.mean - (splitInfo.eigenVector0 * splitInfo.eigenValue0 * 0.5f);
        // Converting the new mean directions into the 3D space
        meanDirection0 = embree::frame(meanDirection) * Map2DTo3D<Vector3, Vector2, float>(meanDir2D0);
        meanDirection1 = embree::frame(meanDirection) * Map2DTo3D<Vector3, Vector2, float>(meanDir2D1);
        OPENPGL_ASSERT(std::abs(dot(meanDirection, meanDirection0)) > 0.f);
        OPENPGL_ASSERT(std::abs(dot(meanDirection, meanDirection1)) > 0.f);

        // Calcualting the new mean cosine for the split components
        newMeanCosine0 = meanCosine / std::abs(dot(meanDirection, meanDirection0));
        OPENPGL_ASSERT(newMeanCosine0 >= 0.f);
        // Ensuring that the new mean cosine is in a valid range (i.e., < 1.0 and < the mean cosine of max kappa).
        newMeanCosine0 = std::min(newMeanCosine0, KappaToMeanCosine<float>(OPENPGL_MAX_KAPPA));
        newMeanCosine1 = newMeanCosine0;

        // Checking if the new mean direction is not too close to the old one.
        // This would mean that we probably reached the expressivness of our model and a split would not improve the representation.
        // TODO: Can we identify such situation before hand?
        if (dot(meanDirection0, meanDirection1) >= 0.99f)
        {
            //  STEP ONE FIX
#ifdef OPENPGL_DEBUG_SAM
            std::cout << "Unsuccessfull split: " << "\t meanCosine = " << meanCosine << "\t D = " << D << "\t eigenValue0 = " << splitInfo.eigenValue0 << "\t eigenValue1 = " << splitInfo.eigenValue1 << std::endl;
#endif
            
            // TODO: add reset of the component
            splitStats.reset(tmpK);
            return false;
        }
#ifdef OPENPGL_SHOW_PRINT_OUTS
        std::cout << "D: " << D << "\t idx: " << idx << " \t assignedSamples: " << numAssignedSamples << std::endl;
#endif
    }
    else
    {
#ifdef OPENPGL_SHOW_PRINT_OUTS
        std::cout << "!!!!   D: " << D << "\t idx: " << idx << " \t assignedSamples: " << numAssignedSamples << std::endl;

        std::cout << "sampleCovariance: [" << splitStats.splitWeightedSampleCovariances[tmpK.quot].x[tmpK.rem] << ",\t"
                  << splitStats.splitWeightedSampleCovariances[tmpK.quot].y[tmpK.rem] << ",\t" << splitStats.splitWeightedSampleCovariances[tmpK.quot].z[tmpK.rem] << "]"
                  << std::endl;
        std::cout << "sumWeights: " << splitStats.sumWeights[tmpK.quot][tmpK.rem] << std::endl;
        std::cout << "weight: " << weight << "\t meanCosine: " << meanCosine << std::endl;
#endif
        // TODO: remove this check
        //if (numAssignedSamples < 2.0f)
        {
            splitStats.reset(tmpK);
            return false;
        }
    }

    // TODO: move this into the previous if statement.
    size_t K = vmm._numComponents;

    const div_t tmpI = tmpK;
    const div_t tmpJ = div(K, static_cast<int>(VMM::VectorSize));

    vmm.splitComponent(idx, K, newWeight0, newWeight1, meanDirection0, meanDirection1, newMeanCosine0, newMeanCosine1);
    suffStats.splitComponentsStats(idx, K, meanDirection0, meanDirection1, newMeanCosine0, newMeanCosine1);
    splitStats.reset(tmpI);
    splitStats.reset(tmpJ);

    splitStats.numComponents = K + 1;

    return true;
}

/**
 * Spilts or better adds a commenent that represents a firelfy observed during the last training iteration.
 */
template <class TVMMFactory>
bool VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::SplitComponentFireFly(VMM &vmm, ComponentSplitStatistics &splitStats, SufficientStatistics &suffStats,
                                                                                    const size_t idx, const Configuration &cfg) const
{
    OPENPGL_ASSERT(vmm.isValid());
    ComponentSplitinfoV2 splitInfo;
    const div_t tmpK = div(idx, static_cast<int>(VMM::VectorSize));

    // Abort splitting if the split component data was estimated with an insufficent number of assigned samples and if the weight is bellow a minmal value.
    if (splitStats.sumAssignedSamples[tmpK.quot][tmpK.rem] < 1.0f || splitStats.weights[tmpK.quot][tmpK.rem] < FLT_EPSILON)
    {
        return false;
    }

    // Calculating the new firely component mean direction and mean cosine using the firefly stats of the component
    Vector3 newMeanDirection =
        Vector3(splitStats.weightedMeans[tmpK.quot].x[tmpK.rem], splitStats.weightedMeans[tmpK.quot].y[tmpK.rem], splitStats.weightedMeans[tmpK.quot].z[tmpK.rem]);
    OPENPGL_ASSERT(splitStats.weights[tmpK.quot][tmpK.rem] > 0.f);
    newMeanDirection /= splitStats.weights[tmpK.quot][tmpK.rem];
    OPENPGL_ASSERT(embree::isvalid(newMeanDirection.x));
    OPENPGL_ASSERT(embree::isvalid(newMeanDirection.y));
    OPENPGL_ASSERT(embree::isvalid(newMeanDirection.z));
    float newMeanCosine = embree::length(newMeanDirection);
    OPENPGL_ASSERT(newMeanCosine > 0.f);
    newMeanDirection /= newMeanCosine;
    // Ensuring that the firefly mean cosine is bellow the max. mean cosine
    newMeanCosine = embree::min(cfg.maxMeanCosine, newMeanCosine);

    // Calculating the fraction the firefly contributes to the comonent weight using the ratio of the energy/weight changed due to adding the firefly
    float frac = splitStats.weights[tmpK.quot][tmpK.rem] / (suffStats.sumOfWeightedStats[tmpK.quot][tmpK.rem] * suffStats.inv_norm);
    // Bounding the firefly weight fraction to avoid putting all energy to the firefly component.
    frac = std::min(0.9f, frac);

    // The index to the new component
    size_t K = vmm._numComponents;

    // Splitting the mixture component
    vmm.splitFireFlyComponent(idx, K, frac, newMeanDirection, newMeanCosine);
    // Splitting the suffitient statistics for the mixture
    suffStats.splitFireFlyComponentsStats(idx, K, frac, newMeanDirection, newMeanCosine);

    // Resetting the split statistics for the two split components.
    const div_t tmpI = tmpK;
    const div_t tmpJ = div(K, static_cast<int>(VMM::VectorSize));
    splitStats.reset(tmpI);
    splitStats.reset(tmpJ);

    // Increasing to number of mixture componets by one to include the new component
    splitStats.numComponents = K + 1;
    OPENPGL_ASSERT(splitStats.isValid());
    OPENPGL_ASSERT(vmm.isValid());
    return true;
}

#ifdef OPENPGL_USE_THREE_SPLIT
template <class TVMMFactory>
bool VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::SplitComponentIntoThree(VMM &vmm, ComponentSplitStatistics &splitStats, SufficientStatistics &suffStats,
                                                                                      const size_t idx) const
{
    ComponentSplitinfoV2 splitInfo;
    const div_t tmpK = div(idx, static_cast<int>(VMM::VectorSize));

    float numAssignedSamples = splitStats.sumAssignedSamples[tmpK.quot][tmpK.rem];

    float inv_sumWeights = rcp(splitStats.sumWeights[tmpK.quot][tmpK.rem]);
    splitInfo.mean = Vector2(splitStats.splitMeans[tmpK.quot].x[tmpK.rem], splitStats.splitMeans[tmpK.quot].y[tmpK.rem]);

    splitInfo.covariance.x = splitStats.splitWeightedSampleCovariances[tmpK.quot].x[tmpK.rem] * inv_sumWeights;
    splitInfo.covariance.y = splitStats.splitWeightedSampleCovariances[tmpK.quot].y[tmpK.rem] * inv_sumWeights;
    splitInfo.covariance.z = splitStats.splitWeightedSampleCovariances[tmpK.quot].z[tmpK.rem] * inv_sumWeights;

    float D = embree::sqrt((splitInfo.covariance.x - splitInfo.covariance.y) * (splitInfo.covariance.x - splitInfo.covariance.y) +
                           (splitInfo.covariance.z * splitInfo.covariance.z * 4.0f)) *
              0.5f;
    splitInfo.eigenValue0 = (splitInfo.covariance.x + splitInfo.covariance.y) * 0.5;
    splitInfo.eigenValue0 += D;

    splitInfo.eigenValue1 = (splitInfo.covariance.x + splitInfo.covariance.y) * 0.5;
    splitInfo.eigenValue1 -= D;

    splitInfo.eigenVector0.x = -splitInfo.covariance.z;
    splitInfo.eigenVector0.y = splitInfo.covariance.x - splitInfo.eigenValue0;

    splitInfo.eigenVector1.x = splitInfo.covariance.z;
    splitInfo.eigenVector1.y = splitInfo.covariance.x - splitInfo.eigenValue1;

    splitInfo.eigenVector0 /= embree::sqrt(splitInfo.eigenVector0.x * splitInfo.eigenVector0.x + splitInfo.eigenVector0.y * splitInfo.eigenVector0.y);
    splitInfo.eigenVector1 /= embree::sqrt(splitInfo.eigenVector1.x * splitInfo.eigenVector1.x + splitInfo.eigenVector1.y * splitInfo.eigenVector1.y);
    /* */
    // std::cout << "D: " << D << std::endl;
    // std::cout << "sumWeights: " << splitStats.sumWeights[tmpK.quot][tmpK.rem] << "\t inSumWeights: " << inv_sumWeights << std::endl;
    // std::cout << "splitMean: " << splitInfo.mean << "\t splitCovariance: " << splitInfo.covariance << std::endl;

    // std::cout << "splitCovariancesRaw: " << splitStats.splitCovariances[tmpK.quot].x[tmpK.rem] << "\t" << splitStats.splitCovariances[tmpK.quot].y[tmpK.rem] << "\t" <<
    // splitStats.splitCovariances[tmpK.quot].z[tmpK.rem] << std::endl;
    //    std::cout << "eigenValue0: " << splitInfo.eigenValue0 << "\t eigenVector0: " << splitInfo.eigenVector0 << std::endl;
    //    std::cout << "eigenValue1: " << splitInfo.eigenValue1 << "\t eigenVector1: " << splitInfo.eigenVector1 << std::endl;
    /**/

    float weight = vmm._weights[tmpK.quot][tmpK.rem];
    float meanCosine = vmm._meanCosines[tmpK.quot][tmpK.rem];
    float kappa = vmm._kappas[tmpK.quot][tmpK.rem];

    if (kappa >= OPENPGL_MAX_KAPPA * 0.9)
    {
        return false;
    }

    Vector3 meanDirection = Vector3(vmm._meanDirections[tmpK.quot].x[tmpK.rem], vmm._meanDirections[tmpK.quot].y[tmpK.rem], vmm._meanDirections[tmpK.quot].z[tmpK.rem]);

    float distance = vmm._distances[tmpK.quot][tmpK.rem];

    float newWeight0 = weight * embree::rcp(3.0f);
    float newWeight1 = newWeight0;
    float newWeight2 = newWeight0;

    Vector3 meanDirection0 = meanDirection;
    Vector3 meanDirection1 = meanDirection;

    float newMeanCosine0 = meanCosine;
    float newMeanCosine1 = meanCosine * meanCosine;

    float newKkappa0 = MeanCosineToKappa<float>(newMeanCosine0);
    float newKkappa1 = MeanCosineToKappa<float>(newMeanCosine1);

    if (D > 1e-8f)
    {
        Vector2 meanDir2D0 = splitInfo.mean + (splitInfo.eigenVector0 * splitInfo.eigenValue0 * 1.0f);
        meanDirection0 = embree::frame(meanDirection) * Map2DTo3D<Vector3, Vector2, float>(meanDir2D0);
        newMeanCosine0 = meanCosine / dot(meanDirection, meanDirection0);
        // ensure that the new mean cosine is in a valid range (i.e., < 1.0 and < the mean cosine of max kappa)
        newMeanCosine0 = std::min(newMeanCosine0, KappaToMeanCosine<float>(OPENPGL_MAX_KAPPA));
        newMeanCosine1 = newMeanCosine0;
        newKkappa0 = MeanCosineToKappa<float>(newMeanCosine0);
        newKkappa1 = newKkappa0;

        Vector2 meanDir2D1 = splitInfo.mean - (splitInfo.eigenVector0 * splitInfo.eigenValue0 * 1.0f);
        meanDirection1 = embree::frame(meanDirection) * Map2DTo3D<Vector3, Vector2, float>(meanDir2D1);
#ifdef OPENPGL_SHOW_PRINT_OUTS
        std::cout << "D: " << D << "\t idx: " << idx << " \t assignedSamples: " << numAssignedSamples << std::endl;
        std::cout << "kappa: " << kappa << " \t newKkappa: " << newKkappa0 << " \t costheta0: " << dot(meanDirection, meanDirection0)
                  << "\t angle: " << std::acos(dot(meanDirection, meanDirection0)) * 180.0f / M_PI_F << std::endl;
#endif
    }
    else
    {
#ifdef OPENPGL_SHOW_PRINT_OUTS
        std::cout << "!!!!   D: " << D << "\t idx: " << idx << " \t assignedSamples: " << numAssignedSamples << std::endl;

        std::cout << "sampleCovariance: [" << splitStats.splitWeightedSampleCovariances[tmpK.quot].x[tmpK.rem] << ",\t"
                  << splitStats.splitWeightedSampleCovariances[tmpK.quot].y[tmpK.rem] << ",\t" << splitStats.splitWeightedSampleCovariances[tmpK.quot].z[tmpK.rem] << "]"
                  << std::endl;
        std::cout << "sumWeights: " << splitStats.sumWeights[tmpK.quot][tmpK.rem] << std::endl;
        std::cout << "weight: " << weight << "\t meanCosine: " << meanCosine << std::endl;
#endif
        if (numAssignedSamples < 2.0f)
        {
            return false;
        }
    }
    size_t K = vmm._numComponents;
    // vmm.swapComponents(K-1, idx);
    // suffStats.swapComponentStats(K-1, idx);
    // const div_t tmpI = div(K-1, static_cast<int>(VMM::VectorSize));
    const div_t tmpI = tmpK;
    const div_t tmpJ = div(K, static_cast<int>(VMM::VectorSize));
    const div_t tmpL = div(K + 1, static_cast<int>(VMM::VectorSize));

    vmm._weights[tmpI.quot][tmpI.rem] = newWeight0;
    vmm._meanCosines[tmpI.quot][tmpI.rem] = newMeanCosine0;
    vmm._kappas[tmpI.quot][tmpI.rem] = newKkappa0;
    vmm._meanDirections[tmpI.quot].x[tmpI.rem] = meanDirection0.x;
    vmm._meanDirections[tmpI.quot].y[tmpI.rem] = meanDirection0.y;
    vmm._meanDirections[tmpI.quot].z[tmpI.rem] = meanDirection0.z;
    vmm._distances[tmpI.quot][tmpI.rem] = distance;

    vmm._weights[tmpJ.quot][tmpJ.rem] = newWeight1;
    vmm._meanCosines[tmpJ.quot][tmpJ.rem] = newMeanCosine1;
    vmm._kappas[tmpJ.quot][tmpJ.rem] = newKkappa1;
    vmm._meanDirections[tmpJ.quot].x[tmpJ.rem] = meanDirection1.x;
    vmm._meanDirections[tmpJ.quot].y[tmpJ.rem] = meanDirection1.y;
    vmm._meanDirections[tmpJ.quot].z[tmpJ.rem] = meanDirection1.z;
    vmm._distances[tmpJ.quot][tmpJ.rem] = distance;

    vmm._weights[tmpL.quot][tmpL.rem] = newWeight2;
    vmm._meanCosines[tmpL.quot][tmpL.rem] = meanCosine;
    vmm._kappas[tmpL.quot][tmpL.rem] = kappa;
    vmm._meanDirections[tmpL.quot].x[tmpL.rem] = meanDirection.x;
    vmm._meanDirections[tmpL.quot].y[tmpL.rem] = meanDirection.y;
    vmm._meanDirections[tmpL.quot].z[tmpL.rem] = meanDirection.z;
    vmm._distances[tmpL.quot][tmpL.rem] = distance;

    vmm._numComponents = K + 2;
    vmm._calculateNormalization();

    float sumStatsWeight = suffStats.sumOfWeightedStats[tmpK.quot][tmpK.rem];
    sumStatsWeight /= 3.0f;

    suffStats.sumOfWeightedStats[tmpI.quot][tmpI.rem] = sumStatsWeight;
    suffStats.sumOfWeightedDirections[tmpI.quot].x[tmpI.rem] = meanDirection0.x * newMeanCosine0 * sumStatsWeight;
    suffStats.sumOfWeightedDirections[tmpI.quot].y[tmpI.rem] = meanDirection0.y * newMeanCosine0 * sumStatsWeight;
    suffStats.sumOfWeightedDirections[tmpI.quot].z[tmpI.rem] = meanDirection0.z * newMeanCosine0 * sumStatsWeight;

    suffStats.sumOfWeightedStats[tmpJ.quot][tmpJ.rem] = sumStatsWeight;
    suffStats.sumOfWeightedDirections[tmpJ.quot].x[tmpJ.rem] = meanDirection1.x * newMeanCosine1 * sumStatsWeight;
    suffStats.sumOfWeightedDirections[tmpJ.quot].y[tmpJ.rem] = meanDirection1.y * newMeanCosine1 * sumStatsWeight;
    suffStats.sumOfWeightedDirections[tmpJ.quot].z[tmpJ.rem] = meanDirection1.z * newMeanCosine1 * sumStatsWeight;

    suffStats.sumOfWeightedStats[tmpL.quot][tmpL.rem] = sumStatsWeight;
    suffStats.sumOfWeightedDirections[tmpL.quot].x[tmpL.rem] = meanDirection.x * meanCosine * sumStatsWeight;
    suffStats.sumOfWeightedDirections[tmpL.quot].y[tmpL.rem] = meanDirection.y * meanCosine * sumStatsWeight;
    suffStats.sumOfWeightedDirections[tmpL.quot].z[tmpL.rem] = meanDirection.z * meanCosine * sumStatsWeight;

    suffStats.numComponents = K + 2;

    OPENPGL_ASSERT(!std::isnan(suffStats.sumOfWeightedDirections[tmpI.quot].x[tmpI.rem]) && std::isfinite(suffStats.sumOfWeightedDirections[tmpI.quot].x[tmpI.rem]));
    OPENPGL_ASSERT(!std::isnan(suffStats.sumOfWeightedDirections[tmpI.quot].y[tmpI.rem]) && std::isfinite(suffStats.sumOfWeightedDirections[tmpI.quot].y[tmpI.rem]));
    OPENPGL_ASSERT(!std::isnan(suffStats.sumOfWeightedDirections[tmpI.quot].z[tmpI.rem]) && std::isfinite(suffStats.sumOfWeightedDirections[tmpI.quot].z[tmpI.rem]));

    // reseting the split statistics for the two new components
    splitStats.reset(tmpI);
    splitStats.reset(tmpJ);
    splitStats.reset(tmpL);

    splitStats.numComponents = K + 2;

    return true;
}
#endif

template <class TVMMFactory>
void VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::ComponentSplitStatistics::serialize(std::ostream &stream) const
{
    serializeFloatVectors<VMM::NumVectors, VMM::VectorSize>(stream, chiSquareMCEstimates);
    serializeFloatVectors<VMM::NumVectors, VMM::VectorSize>(stream, chiSquareMCEstimate2ndMoments);
    serializeFloatVectors<VMM::NumVectors, VMM::VectorSize>(stream, weightsEstimates);
    serializeFloatVectors<VMM::NumVectors, VMM::VectorSize>(stream, weights2ndmomentEstimates);
    serializeFloatVectors<VMM::NumVectors, VMM::VectorSize>(stream, weightsVarianceEstimates);
    serializeFloatVectors<VMM::NumVectors, VMM::VectorSize>(stream, numWeightsEstimatesSamples);
    serializeVec2Vectors<VMM::NumVectors, VMM::VectorSize>(stream, splitMeans);
    serializeVec3Vectors<VMM::NumVectors, VMM::VectorSize>(stream, splitWeightedSampleCovariances);
    serializeFloatVectors<VMM::NumVectors, VMM::VectorSize>(stream, numSamples);
    serializeFloatVectors<VMM::NumVectors, VMM::VectorSize>(stream, sumWeights);
    serializeFloatVectors<VMM::NumVectors, VMM::VectorSize>(stream, sumAssignedSamples);
    serializeIntVectors<VMM::NumVectors, VMM::VectorSize>(stream, splitType);
    stream.write(reinterpret_cast<const char *>(&numComponents), sizeof(size_t));
}

template <class TVMMFactory>
void VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::ComponentSplitStatistics::deserialize(std::istream &stream)
{
    deserializeFloatVectors<VMM::NumVectors, VMM::VectorSize>(stream, chiSquareMCEstimates);
    deserializeFloatVectors<VMM::NumVectors, VMM::VectorSize>(stream, chiSquareMCEstimate2ndMoments);
    deserializeFloatVectors<VMM::NumVectors, VMM::VectorSize>(stream, weightsEstimates);
    deserializeFloatVectors<VMM::NumVectors, VMM::VectorSize>(stream, weights2ndmomentEstimates);
    deserializeFloatVectors<VMM::NumVectors, VMM::VectorSize>(stream, weightsVarianceEstimates);
    deserializeFloatVectors<VMM::NumVectors, VMM::VectorSize>(stream, numWeightsEstimatesSamples);
    deserializeVec2Vectors<VMM::NumVectors, VMM::VectorSize>(stream, splitMeans);
    deserializeVec3Vectors<VMM::NumVectors, VMM::VectorSize>(stream, splitWeightedSampleCovariances);
    deserializeFloatVectors<VMM::NumVectors, VMM::VectorSize>(stream, numSamples);
    deserializeFloatVectors<VMM::NumVectors, VMM::VectorSize>(stream, sumWeights);
    deserializeFloatVectors<VMM::NumVectors, VMM::VectorSize>(stream, sumAssignedSamples);
    deserializeIntVectors<VMM::NumVectors, VMM::VectorSize>(stream, splitType);
    stream.read(reinterpret_cast<char *>(&numComponents), sizeof(size_t));
}

template <class TVMMFactory>
bool VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::ComponentSplitStatistics::isValid() const
{
    bool valid = true;

    embree::vbool<VMM::VectorSize> validVec(true);
    const int cnt = (VMM::MaxComponents + VMM::VectorSize - 1) / VMM::VectorSize;
    for (size_t k = 0; k < cnt; k++)
    {
        validVec &= embree::isvalid(splitMeans[k].x);
        validVec &= embree::isvalid(splitMeans[k].y);
        OPENPGL_ASSERT(embree::any(validVec));

        validVec &= embree::isvalid(splitWeightedSampleCovariances[k].x);
        validVec &= embree::isvalid(splitWeightedSampleCovariances[k].y);
        validVec &= embree::isvalid(splitWeightedSampleCovariances[k].z);
        OPENPGL_ASSERT(embree::any(validVec));

        validVec &= embree::isvalid(chiSquareMCEstimates[k]);
        validVec &= chiSquareMCEstimates[k] >= 0.0f;
        OPENPGL_ASSERT(embree::any(validVec));

        validVec &= embree::isvalid(chiSquareMCEstimate2ndMoments[k]);
        validVec &= chiSquareMCEstimate2ndMoments[k] >= 0.0f;
        OPENPGL_ASSERT(embree::any(validVec));

        validVec &= embree::isvalid(weightsEstimates[k]);
        validVec &= weightsEstimates[k] >= 0.0f;
        OPENPGL_ASSERT(embree::any(validVec));

        validVec &= embree::isvalid(weights2ndmomentEstimates[k]);
        validVec &= weights2ndmomentEstimates[k] >= 0.0f;
        OPENPGL_ASSERT(embree::any(validVec));

        validVec &= embree::isvalid(weightsVarianceEstimates[k]);
        validVec &= weightsVarianceEstimates[k] >= 0.0f;
        OPENPGL_ASSERT(embree::any(validVec));

        validVec &= embree::isvalid(numWeightsEstimatesSamples[k]);
        validVec &= numWeightsEstimatesSamples[k] >= 0.0f;
        OPENPGL_ASSERT(embree::any(validVec));

        validVec &= embree::isvalid(sumWeights[k]);
        validVec &= sumWeights[k] >= 0.0f;
        OPENPGL_ASSERT(embree::any(validVec));

        validVec &= embree::isvalid(sumAssignedSamples[k]);
        validVec &= sumAssignedSamples[k] >= 0.0f;
        OPENPGL_ASSERT(embree::any(validVec));

        validVec &= embree::isvalid(numSamples[k]);
        validVec &= numSamples[k] >= 0.0f;
        OPENPGL_ASSERT(embree::any(validVec));
    }

    valid = valid && embree::any(validVec);
    OPENPGL_ASSERT(valid);
    valid = valid && numComponents > 0;
    valid = valid && numComponents <= VMM::MaxComponents;
    OPENPGL_ASSERT(valid);

    return valid;
}

template <class TVMMFactory>
bool VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::ComponentSplitStatistics::operator==(const ComponentSplitStatistics &b) const
{
    bool equal = true;
    if (numComponents != b.numComponents)
    {
        equal = false;
    }

    for (int k = 0; k < VMM::NumVectors; k++)
    {
        if (embree::any(chiSquareMCEstimates[k] != b.chiSquareMCEstimates[k]) || embree::any(chiSquareMCEstimate2ndMoments[k] != b.chiSquareMCEstimate2ndMoments[k]) ||
            embree::any(weightsEstimates[k] != b.weightsEstimates[k]) || embree::any(weights2ndmomentEstimates[k] != b.weights2ndmomentEstimates[k]) ||
            embree::any(weightsVarianceEstimates[k] != b.weightsVarianceEstimates[k]) || embree::any(numWeightsEstimatesSamples[k] != b.numWeightsEstimatesSamples[k]) ||
            embree::any(splitMeans[k].x != b.splitMeans[k].x) || embree::any(splitMeans[k].y != b.splitMeans[k].y) ||
            embree::any(splitWeightedSampleCovariances[k].x != b.splitWeightedSampleCovariances[k].x) ||
            embree::any(splitWeightedSampleCovariances[k].y != b.splitWeightedSampleCovariances[k].y) ||
            embree::any(splitWeightedSampleCovariances[k].z != b.splitWeightedSampleCovariances[k].z) || embree::any(numSamples[k] != b.numSamples[k]) ||
            embree::any(sumWeights[k] != b.sumWeights[k]) || embree::any(sumAssignedSamples[k] != b.sumAssignedSamples[k]))
        {
            equal = false;
        }
    }

    return equal;
}

template <class TVMMFactory>
Vector2 VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::ComponentSplitStatistics::getSplitMean(const size_t &idx) const
{
    const div_t tmp = div(idx, static_cast<int>(VMM::VectorSize));
    return Vector2(splitMeans[tmp.quot].x[tmp.rem], splitMeans[tmp.quot].y[tmp.rem]);
}

template <class TVMMFactory>
Vector3 VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::ComponentSplitStatistics::getSplitCovariance(const size_t &idx) const
{
    const div_t tmp = div(idx, static_cast<int>(VMM::VectorSize));
    Vector3 covariance(splitWeightedSampleCovariances[tmp.quot].x[tmp.rem], splitWeightedSampleCovariances[tmp.quot].y[tmp.rem],
                       splitWeightedSampleCovariances[tmp.quot].z[tmp.rem]);
    covariance /= sumWeights[tmp.quot][tmp.rem];
    return covariance;
}

template <class TVMMFactory>
typename VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::SplitType VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::ComponentSplitStatistics::getSplitType(
    const size_t &idx) const
{
    const div_t tmp = div(idx, static_cast<int>(VMM::VectorSize));
    return (VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::SplitType)splitType[tmp.quot][tmp.rem];
}

template <class TVMMFactory>
void VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::ComponentSplitStatistics::mergeComponentStats(const size_t &idxI, const size_t &idxJ, const float &weightI,
                                                                                                            const Vector3 &meanDirectionI, const float &weightJ,
                                                                                                            const Vector3 &meanDirectionJ, const float &weightK,
                                                                                                            const Vector3 &meanDirectionK)
{
    // The SIMD positions of the two components which should be merged
    const div_t tmpI = div(idxI, static_cast<int>(VMM::VectorSize));
    const div_t tmpJ = div(idxJ, static_cast<int>(VMM::VectorSize));
    // The SIMD positon of the last component
    const div_t tmpL = div(numComponents - 1, VMM::VectorSize);

    // The matrix to transform from world space to the local space of the merged component
    auto transformK = embree::frame(meanDirectionK);
    auto inv_transformK = transformK.inverse();
#ifdef OPENPGL_ZERO_MEAN
    Vector3 meanDirectionI3D = meanDirectionI;
    Vector3 meanDirectionJ3D = meanDirectionJ;
#else
    auto transformI = embree::frame(meanDirectionI);
    auto transformJ = embree::frame(meanDirectionJ);
    Vector2 meanDirection2DI = Vector2(splitMeans[tmpI.quot].x[tmpI.rem], splitMeans[tmpI.quot].y[tmpI.rem]);
    Vector2 meanDirection2DJ = Vector2(splitMeans[tmpJ.quot].x[tmpJ.rem], splitMeans[tmpJ.quot].y[tmpJ.rem]);

    Vector3 meanDirectionI3D = transformI * Map2DTo3D<Vector3, Vector2, float>(meanDirection2DI);
    Vector3 meanDirectionJ3D = transformJ * Map2DTo3D<Vector3, Vector2, float>(meanDirection2DJ);

#endif

    // Mapping the mean direction of the two merge components (I and J)  into the 2D projected space of the merged component (K).
    Vector2 meanDirection2DItoK = Map3DTo2D<Vector3, Vector2, float>(inv_transformK * meanDirectionI3D);
    Vector2 meanDirection2DJtoK = Map3DTo2D<Vector3, Vector2, float>(inv_transformK * meanDirectionJ3D);

    const float inv_weightK = (weightK > FLT_EPSILON) ? embree::rcp(weightK) : 1.f;

    const float sumWeightsI = sumWeights[tmpI.quot][tmpI.rem];
    const float sumWeightsJ = sumWeights[tmpJ.quot][tmpJ.rem];
    const float sumWeightsK = sumWeightsI + sumWeightsJ;
    // Caclulating the 2D covariance matricies for the two merge components (I and J).
    const Vector3 covarianceI = (sumWeightsI > FLT_EPSILON) ? Vector3(splitWeightedSampleCovariances[tmpI.quot].x[tmpI.rem], splitWeightedSampleCovariances[tmpI.quot].y[tmpI.rem],
                                                                      splitWeightedSampleCovariances[tmpI.quot].z[tmpI.rem]) *
                                                                  embree::rcp(sumWeightsI)
                                                            : Vector3(0.f);
    const Vector3 covarianceJ = (sumWeightsJ > FLT_EPSILON) ? Vector3(splitWeightedSampleCovariances[tmpJ.quot].x[tmpJ.rem], splitWeightedSampleCovariances[tmpJ.quot].y[tmpJ.rem],
                                                                      splitWeightedSampleCovariances[tmpJ.quot].z[tmpJ.rem]) *
                                                                  embree::rcp(sumWeightsJ)
                                                            : Vector3(0.f);

#ifdef OPENPGL_ZERO_MEAN
    const Vector2 meanDirectionK2D(0.f);
#else
    const Vector2 meanDirectionK2D = inv_weightK * (weightI * meanDirection2DItoK + weightJ * meanDirection2DJtoK);
#endif
    // Merging the two 2D covariances to one assuming the mean of the merges Gaussian is at the origin (0,0).
    Vector3 meanII = Vector3(meanDirection2DItoK.x * meanDirection2DItoK.x, meanDirection2DItoK.y * meanDirection2DItoK.y, meanDirection2DItoK.x * meanDirection2DItoK.y);
    Vector3 meanJJ = Vector3(meanDirection2DJtoK.x * meanDirection2DJtoK.x, meanDirection2DJtoK.y * meanDirection2DJtoK.y, meanDirection2DJtoK.x * meanDirection2DJtoK.y);
    Vector3 covarianceK = (weightI * covarianceI + weightI * meanII + weightJ * covarianceJ + weightJ * meanJJ);
    covarianceK *= inv_weightK;
#ifndef OPENPGL_ZERO_MEAN
    Vector3 meanKK = Vector3(meanDirectionK2D.x * meanDirectionK2D.x, meanDirectionK2D.y * meanDirectionK2D.y, meanDirectionK2D.x * meanDirectionK2D.y);
    covarianceK -= meanKK;
#endif
    // Converting the merged covariance matrix into a sample variance matrix
    const Vector3 sampleCovarianceK = covarianceK * sumWeightsK;
    OPENPGL_ASSERT(embree::isvalid(sampleCovarianceK.x));
    OPENPGL_ASSERT(embree::isvalid(sampleCovarianceK.y));
    OPENPGL_ASSERT(embree::isvalid(sampleCovarianceK.z));

    // Merging the Chi square stats of the two merge components.
    const float sumAssignedSamplesK = sumAssignedSamples[tmpI.quot][tmpI.rem] + sumAssignedSamples[tmpJ.quot][tmpJ.rem];
    const float numSamplesK = inv_weightK * (weightI * numSamples[tmpI.quot][tmpI.rem] + weightJ * numSamples[tmpJ.quot][tmpJ.rem]);
    const float chiSquareMCEstimatesK = chiSquareMCEstimates[tmpI.quot][tmpI.rem] + chiSquareMCEstimates[tmpJ.quot][tmpJ.rem];
    const float chiSquareMCEstimate2ndMomentsK = chiSquareMCEstimate2ndMoments[tmpI.quot][tmpI.rem] + chiSquareMCEstimate2ndMoments[tmpJ.quot][tmpJ.rem];

    // Merging the dbor firefly stats of the two merge components.
    const float weightsEstimateK = inv_weightK * (weightI * weightsEstimates[tmpI.quot][tmpI.rem] + weightJ * weightsEstimates[tmpJ.quot][tmpJ.rem]);
    const float weights2ndmomentEstimateK = inv_weightK * (weightI * weights2ndmomentEstimates[tmpI.quot][tmpI.rem] + weightJ * weights2ndmomentEstimates[tmpJ.quot][tmpJ.rem]);
    const float weightsVarianceEstimatesK = inv_weightK * (weightI * weightsVarianceEstimates[tmpI.quot][tmpI.rem] + weightJ * weightsVarianceEstimates[tmpJ.quot][tmpJ.rem]);
    const float numWeightsEstimatesSamplesK = inv_weightK * (weightI * numWeightsEstimatesSamples[tmpI.quot][tmpI.rem] + weightJ * numWeightsEstimatesSamples[tmpJ.quot][tmpJ.rem]);

    // Inserting stats of the merged component a the ith positions (the position of the first merge component).
#ifdef OPENPGL_ZERO_MEAN
    splitMeans[tmpI.quot].x[tmpI.rem] = 0.0f;
    splitMeans[tmpI.quot].y[tmpI.rem] = 0.0f;
#else
    splitMeans[tmpI.quot].x[tmpI.rem] = meanDirectionK2D.x;
    splitMeans[tmpI.quot].y[tmpI.rem] = meanDirectionK2D.y;
#endif
    splitWeightedSampleCovariances[tmpI.quot].x[tmpI.rem] = sampleCovarianceK.x;
    splitWeightedSampleCovariances[tmpI.quot].y[tmpI.rem] = sampleCovarianceK.y;
    splitWeightedSampleCovariances[tmpI.quot].z[tmpI.rem] = sampleCovarianceK.z;

    sumWeights[tmpI.quot][tmpI.rem] = sumWeightsK;
    numSamples[tmpI.quot][tmpI.rem] = numSamplesK;
    sumAssignedSamples[tmpI.quot][tmpI.rem] = sumAssignedSamplesK;
    chiSquareMCEstimates[tmpI.quot][tmpI.rem] = chiSquareMCEstimatesK;
    chiSquareMCEstimate2ndMoments[tmpI.quot][tmpI.rem] = chiSquareMCEstimate2ndMomentsK;

    weightsEstimates[tmpI.quot][tmpI.rem] = weightsEstimateK;
    weights2ndmomentEstimates[tmpI.quot][tmpI.rem] = weights2ndmomentEstimateK;
    weightsVarianceEstimates[tmpI.quot][tmpI.rem] = weightsVarianceEstimatesK;
    numWeightsEstimatesSamples[tmpI.quot][tmpI.rem] = numWeightsEstimatesSamplesK;

    // Replacing the stats of the second, now unneeded, merge component with the ones from the last mixtur component and
    // resetting the stats at the position of the last component.
    replaceReset(tmpJ, tmpL);

    // Decreasing the number of components by one since we merge two components.
    numComponents--;
}

template <class TVMMFactory>
std::vector<typename VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::SplitCandidate>
VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::ComponentSplitStatistics::getSplitCandidates(const float splitThreshold, const bool useConfidence) const
{
    std::vector<SplitCandidate> splitCandidates;
    // For each mixture component
    for (size_t k = 0; k < numComponents; k++)
    {
        const div_t tmp = div(k, static_cast<int>(VMM::VectorSize));
        // Check if the estimated chi square value is aboth our splitting threshold
        if (chiSquareMCEstimates[tmp.quot][tmp.rem] > splitThreshold)
        {
            // TODO: add convidence estimates to check if the chi^2 is valid/stable
            /*
            if (useConfidence)
            {
                float chiSquareVar = chiSquareMCEstimate2ndMoments[tmp.quot][tmp.rem] - chiSquareMCEstimates[tmp.quot][tmp.rem] * chiSquareMCEstimates[tmp.quot][tmp.rem];
                chiSquareVar /= sumAssignedSamples[tmp.quot][tmp.rem];
                float chiSquareStd = sqrt(chiSquareVar);
                float confidence = chiSquareMCEstimates[tmp.quot][tmp.rem] / (chiSquareMCEstimates[tmp.quot][tmp.rem] + chiSquareStd);
                std::cout << "chiSquare = " << chiSquareMCEstimates[tmp.quot][tmp.rem] << "\t chiSquareVar = " << chiSquareVar << "\t chiSquareStd = " << chiSquareStd
                          << "\t sumAssignedSamples = " << sumAssignedSamples[tmp.quot][tmp.rem] << "\t confidence = " << confidence << std::endl;
            }
            */
            SplitCandidate sc;
            sc.chiSquareEst = chiSquareMCEstimates[tmp.quot][tmp.rem];
            sc.componentIndex = k;
            sc.splitType = (SplitType)splitType[tmp.quot][tmp.rem];
            splitCandidates.push_back(sc);
        }
    }

    // If we have more than one split candidate sort all candidates based on their chi square values
    if (splitCandidates.size() > 1)
    {
        std::sort(splitCandidates.begin(), splitCandidates.end(), [](SplitCandidate a, SplitCandidate b) {
            return a > b;
        });
    }
    return splitCandidates;
}

template <class TVMMFactory>
void VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::ComponentSplitStatistics::clear(const size_t &_numComponents)
{
    const embree::vfloat<VMM::VectorSize> zeros(0.f);

    this->numComponents = _numComponents;

    for (size_t k = 0; k < VMM::NumVectors; k++)
    {
        chiSquareMCEstimates[k] = zeros;
        chiSquareMCEstimate2ndMoments[k] = zeros;
        weightsEstimates[k] = zeros;
        weights2ndmomentEstimates[k] = zeros;
        weightsVarianceEstimates[k] = zeros;
        numWeightsEstimatesSamples[k] = zeros;

        splitWeightedSampleCovariances[k].x = zeros;
        splitWeightedSampleCovariances[k].y = zeros;
        splitWeightedSampleCovariances[k].z = zeros;

        splitMeans[k].x = zeros;
        splitMeans[k].y = zeros;

        numSamples[k] = zeros;
        sumWeights[k] = zeros;
        sumAssignedSamples[k] = zeros;

        weights[k] = zeros;
        weightedMeans[k].x = zeros;
        weightedMeans[k].y = zeros;
        weightedMeans[k].z = zeros;
    }
}

template <class TVMMFactory>
void VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::ComponentSplitStatistics::clearMasked(const size_t &_numComponents, const PartialFittingMask &mask)
{
    const embree::vfloat<VMM::VectorSize> zeros(0.f);

    this->numComponents = _numComponents;
    const int cnt = (this->numComponents + VMM::VectorSize - 1) / VMM::VectorSize;

    for (size_t k = 0; k < cnt; k++)
    {
        chiSquareMCEstimates[k] = select(mask.mask[k], zeros, chiSquareMCEstimates[k]);
        chiSquareMCEstimate2ndMoments[k] = select(mask.mask[k], zeros, chiSquareMCEstimate2ndMoments[k]);
        weightsEstimates[k] = select(mask.mask[k], zeros, weightsEstimates[k]);
        weights2ndmomentEstimates[k] = select(mask.mask[k], zeros, weights2ndmomentEstimates[k]);
        weightsVarianceEstimates[k] = select(mask.mask[k], zeros, weightsVarianceEstimates[k]);
        numWeightsEstimatesSamples[k] = select(mask.mask[k], zeros, numWeightsEstimatesSamples[k]);

        splitWeightedSampleCovariances[k].x = select(mask.mask[k], zeros, splitWeightedSampleCovariances[k].x);
        splitWeightedSampleCovariances[k].y = select(mask.mask[k], zeros, splitWeightedSampleCovariances[k].y);
        splitWeightedSampleCovariances[k].z = select(mask.mask[k], zeros, splitWeightedSampleCovariances[k].z);

        splitMeans[k].x = select(mask.mask[k], zeros, splitMeans[k].x);
        splitMeans[k].y = select(mask.mask[k], zeros, splitMeans[k].y);

        numSamples[k] = select(mask.mask[k], zeros, numSamples[k]);
        sumWeights[k] = select(mask.mask[k], zeros, sumWeights[k]);
        sumAssignedSamples[k] = select(mask.mask[k], zeros, sumAssignedSamples[k]);

        weights[k] = select(mask.mask[k], zeros, weights[k]);
        weightedMeans[k].x = select(mask.mask[k], zeros, weightedMeans[k].x);
        weightedMeans[k].y = select(mask.mask[k], zeros, weightedMeans[k].y);
        weightedMeans[k].z = select(mask.mask[k], zeros, weightedMeans[k].z);
    }
}

template <class TVMMFactory>
void VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::ComponentSplitStatistics::decay(const float &alpha)
{
    const int cnt = (this->numComponents + VMM::VectorSize - 1) / VMM::VectorSize;

    for (size_t k = 0; k < cnt; k++)
    {
        splitWeightedSampleCovariances[k].x *= alpha;
        splitWeightedSampleCovariances[k].y *= alpha;
        splitWeightedSampleCovariances[k].z *= alpha;

        numSamples[k] *= alpha;
        sumWeights[k] *= alpha;
        sumAssignedSamples[k] *= alpha;

        weightsVarianceEstimates[k] *= alpha;
        numWeightsEstimatesSamples[k] *= alpha;
    }
}

template <class TVMMFactory>
size_t VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::ComponentSplitStatistics::getHighestChiSquareIdx() const
{
    size_t maxIdx = 0;
    float maxChiSquareValue = chiSquareMCEstimates[0][0];
    for (size_t k = 1; k < numComponents; k++)
    {
        const div_t tmp = div(k, static_cast<int>(VMM::VectorSize));
        if (chiSquareMCEstimates[tmp.quot][tmp.rem] > maxChiSquareValue)
        {
            maxChiSquareValue = chiSquareMCEstimates[tmp.quot][tmp.rem];
            maxIdx = k;
        }
    }
    return maxIdx;
}

template <class TVMMFactory>
bool VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::ComponentSplitStatistics::getHighestValidSplitComponent(SplitCandidate &splitCandiate,
                                                                                                                      const float splitChiSquareThreshold, const VMM &vmm,
                                                                                                                      const bool *alreadySplitted, const bool useConfidence) const
{
    bool success = false;
    splitCandiate.componentIndex = VMM::MaxComponents;
    splitCandiate.splitType = ENone;
    float maxChiSquareValue = 0.f;
    for (size_t k = 0; k < numComponents; k++)
    {
        const div_t tmp = div(k, static_cast<int>(VMM::VectorSize));
        const float componentChiSquareEst = chiSquareMCEstimates[tmp.quot][tmp.rem];
        if (!alreadySplitted[k] && componentChiSquareEst > splitChiSquareThreshold && componentChiSquareEst > maxChiSquareValue &&
            vmm._kappas[tmp.quot][tmp.rem] < OPENPGL_MAX_KAPPA * 0.9)
        {
            /*
            if (useConfidence)
            {
                float chiSquareVar = chiSquareMCEstimate2ndMoments[tmp.quot][tmp.rem] - chiSquareMCEstimates[tmp.quot][tmp.rem] * chiSquareMCEstimates[tmp.quot][tmp.rem];
                chiSquareVar /= sumAssignedSamples[tmp.quot][tmp.rem];
                float chiSquareStd = sqrt(chiSquareVar);
                float confidence = chiSquareMCEstimates[tmp.quot][tmp.rem] / (chiSquareMCEstimates[tmp.quot][tmp.rem] + chiSquareStd);
                std::cout << "chiSquare = " << chiSquareMCEstimates[tmp.quot][tmp.rem] << "\t chiSquareVar = " << chiSquareVar << "\t chiSquareStd = " << chiSquareStd
                          << "\t sumAssignedSamples = " << sumAssignedSamples[tmp.quot][tmp.rem] << "\t confidence = " << confidence << std::endl;
            }
            */
            maxChiSquareValue = chiSquareMCEstimates[tmp.quot][tmp.rem];
            splitCandiate.chiSquareEst = maxChiSquareValue;
            splitCandiate.componentIndex = k;
        }
    }
    if (splitCandiate.componentIndex < VMM::MaxComponents)
    {
        splitCandiate.splitType = getSplitType(splitCandiate.componentIndex);
        // candidate.splitType = EFirefly;
        success = true;
    }
    return success;
}

template <class TVMMFactory>
void VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::ComponentSplitStatistics::clearAll()
{
    this->clear(VMM::MaxComponents);
}

template <class TVMMFactory>
void VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::ComponentSplitStatistics::replaceReset(const div_t k, const div_t l)
{
    this->chiSquareMCEstimates[k.quot][k.rem] = this->chiSquareMCEstimates[l.quot][l.rem];
    this->chiSquareMCEstimates[l.quot][l.rem] = 0.0f;
    this->chiSquareMCEstimate2ndMoments[k.quot][k.rem] = this->chiSquareMCEstimate2ndMoments[l.quot][l.rem];
    this->chiSquareMCEstimate2ndMoments[l.quot][l.rem] = 0.0f;

    this->weightsEstimates[k.quot][k.rem] = this->weightsEstimates[l.quot][l.rem];
    this->weightsEstimates[l.quot][l.rem] = 0.0f;
    this->weights2ndmomentEstimates[k.quot][k.rem] = this->weights2ndmomentEstimates[l.quot][l.rem];
    this->weights2ndmomentEstimates[l.quot][l.rem] = 0.0f;
    this->weightsVarianceEstimates[k.quot][k.rem] = this->weightsVarianceEstimates[l.quot][l.rem];
    this->weightsVarianceEstimates[l.quot][l.rem] = 0.0f;
    this->numWeightsEstimatesSamples[k.quot][k.rem] = this->numWeightsEstimatesSamples[l.quot][l.rem];
    this->numWeightsEstimatesSamples[l.quot][l.rem] = 0.0f;

    this->sumAssignedSamples[k.quot][k.rem] = this->sumAssignedSamples[l.quot][l.rem];
    this->sumAssignedSamples[l.quot][l.rem] = 0.0f;
    this->numSamples[k.quot][k.rem] = this->numSamples[l.quot][l.rem];
    this->numSamples[l.quot][l.rem] = 0.0f;
    this->sumWeights[k.quot][k.rem] = this->sumWeights[l.quot][l.rem];
    this->sumWeights[l.quot][l.rem] = 0.0f;
    this->splitMeans[k.quot].x[k.rem] = this->splitMeans[l.quot].x[l.rem];
    this->splitMeans[l.quot].x[l.rem] = 0.0f;
    this->splitMeans[k.quot].y[k.rem] = this->splitMeans[l.quot].y[l.rem];
    this->splitMeans[l.quot].y[l.rem] = 0.0f;
    this->splitWeightedSampleCovariances[k.quot].x[k.rem] = this->splitWeightedSampleCovariances[l.quot].x[l.rem];
    this->splitWeightedSampleCovariances[l.quot].x[l.rem] = 0.0f;
    this->splitWeightedSampleCovariances[k.quot].y[k.rem] = this->splitWeightedSampleCovariances[l.quot].y[l.rem];
    this->splitWeightedSampleCovariances[l.quot].y[l.rem] = 0.0f;
    this->splitWeightedSampleCovariances[k.quot].z[k.rem] = this->splitWeightedSampleCovariances[l.quot].z[l.rem];
    this->splitWeightedSampleCovariances[l.quot].z[l.rem] = 0.0f;
    this->weights[k.quot][k.rem] = this->weights[l.quot][l.rem];
    this->weights[l.quot][l.rem] = 0.0f;
    this->weightedMeans[k.quot].x[k.rem] = this->weightedMeans[l.quot].x[l.rem];
    this->weightedMeans[l.quot].x[l.rem] = 0.0f;
    this->weightedMeans[k.quot].y[k.rem] = this->weightedMeans[l.quot].y[l.rem];
    this->weightedMeans[l.quot].y[l.rem] = 0.0f;
    this->weightedMeans[k.quot].z[k.rem] = this->weightedMeans[l.quot].z[l.rem];
    this->weightedMeans[l.quot].z[l.rem] = 0.0f;
}

template <class TVMMFactory>
void VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::ComponentSplitStatistics::reset(const div_t k)
{
    this->chiSquareMCEstimates[k.quot][k.rem] = 0.0f;
    this->chiSquareMCEstimate2ndMoments[k.quot][k.rem] = 0.0f;

    this->weightsEstimates[k.quot][k.rem] = 0.0f;
    this->weights2ndmomentEstimates[k.quot][k.rem] = 0.0f;
    this->weightsVarianceEstimates[k.quot][k.rem] = 0.0f;
    this->numWeightsEstimatesSamples[k.quot][k.rem] = 0.0f;

    this->sumAssignedSamples[k.quot][k.rem] = 0.0f;
    this->numSamples[k.quot][k.rem] = 0.0f;
    this->sumWeights[k.quot][k.rem] = 0.0f;
    this->splitMeans[k.quot].x[k.rem] = 0.0f;
    this->splitMeans[k.quot].y[k.rem] = 0.0f;
    this->splitWeightedSampleCovariances[k.quot].x[k.rem] = 0.0f;
    this->splitWeightedSampleCovariances[k.quot].y[k.rem] = 0.0f;
    this->splitWeightedSampleCovariances[k.quot].z[k.rem] = 0.0f;
    this->weights[k.quot][k.rem] = 0.0f;
    this->weightedMeans[k.quot].x[k.rem] = 0.0f;
    this->weightedMeans[k.quot].y[k.rem] = 0.0f;
    this->weightedMeans[k.quot].z[k.rem] = 0.0f;
}

template <class TVMMFactory>
float VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::ComponentSplitStatistics::getChiSquareEst(const size_t &idx) const
{
    const div_t tmp = div(idx, static_cast<int>(VMM::VectorSize));
    return chiSquareMCEstimates[tmp.quot][tmp.rem];
}

template <class TVMMFactory>
float VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::ComponentSplitStatistics::getChiSquare2ndMomentEst(const size_t &idx) const
{
    const div_t tmp = div(idx, static_cast<int>(VMM::VectorSize));
    return chiSquareMCEstimate2ndMoments[tmp.quot][tmp.rem];
}

template <class TVMMFactory>
float VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::ComponentSplitStatistics::getWeightsEst(const size_t &idx) const
{
    const div_t tmp = div(idx, static_cast<int>(VMM::VectorSize));
    return weightsEstimates[tmp.quot][tmp.rem];
}
template <class TVMMFactory>
float VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::ComponentSplitStatistics::getRelVarianceEst(const size_t &idx) const
{
    const div_t tmp = div(idx, static_cast<int>(VMM::VectorSize));
    return (std::abs(weights2ndmomentEstimates[tmp.quot][tmp.rem] - weightsEstimates[tmp.quot][tmp.rem] * weightsEstimates[tmp.quot][tmp.rem])) /
           (weightsEstimates[tmp.quot][tmp.rem] * weightsEstimates[tmp.quot][tmp.rem]);
}

template <class TVMMFactory>
float VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::ComponentSplitStatistics::getVarianceEst(const size_t &idx) const
{
    const div_t tmp = div(idx, static_cast<int>(VMM::VectorSize));
    return (std::abs(weights2ndmomentEstimates[tmp.quot][tmp.rem] - weightsEstimates[tmp.quot][tmp.rem] * weightsEstimates[tmp.quot][tmp.rem]));
}

template <class TVMMFactory>
float VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::ComponentSplitStatistics::getWeights2ndMomentEst(const size_t &idx) const
{
    const div_t tmp = div(idx, static_cast<int>(VMM::VectorSize));
    return weights2ndmomentEstimates[tmp.quot][tmp.rem];
}

template <class TVMMFactory>
float VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::ComponentSplitStatistics::getWeightsVarianceEst(const size_t &idx) const
{
    const div_t tmp = div(idx, static_cast<int>(VMM::VectorSize));
    return weightsVarianceEstimates[tmp.quot][tmp.rem] / numWeightsEstimatesSamples[tmp.quot][tmp.rem];
}

template <class TVMMFactory>
float VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::ComponentSplitStatistics::getSumChiSquareEst() const
{
    float sumChiSquareEst = 0.0f;

    for (int k = 0; k < numComponents; k++)
    {
        const div_t tmp = div(k, static_cast<int>(VMM::VectorSize));
        sumChiSquareEst += chiSquareMCEstimates[tmp.quot][tmp.rem];
    }
    return sumChiSquareEst;
}

template <class TVMMFactory>
std::string VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::ComponentSplitStatistics::toString() const
{
    std::stringstream ss;
    ss << "ComponentSplitStatistics:" << std::endl;
    ss << "numComponents: " << numComponents << std::endl;
    float sumChiSquareEst = 0.0f;
    // for ( int k = 0; k < numComponents; k++)
    for (int k = 0; k < VMM::MaxComponents; k++)
    {
        const div_t tmp = div(k, static_cast<int>(VMM::VectorSize));
        ss << "\t stats[" << k << "]: " << "chiSquareEst: " << chiSquareMCEstimates[tmp.quot][tmp.rem];
        ss << std::endl;
        ss << "\t" << "mean: [" << splitMeans[tmp.quot].x[tmp.rem] << ",\t" << splitMeans[tmp.quot].y[tmp.rem] << "]";
        ss << "\t samplevar: [" << splitWeightedSampleCovariances[tmp.quot].x[tmp.rem] << ",\t" << splitWeightedSampleCovariances[tmp.quot].y[tmp.rem] << ",\t"
           << splitWeightedSampleCovariances[tmp.quot].z[tmp.rem] << "]";
        if (sumWeights[tmp.quot][tmp.rem] > 0.f)
        {
            ss << "\t covar: [" << splitWeightedSampleCovariances[tmp.quot].x[tmp.rem] / sumWeights[tmp.quot][tmp.rem] << ",\t"
               << splitWeightedSampleCovariances[tmp.quot].y[tmp.rem] / sumWeights[tmp.quot][tmp.rem] << ",\t"
               << splitWeightedSampleCovariances[tmp.quot].z[tmp.rem] / sumWeights[tmp.quot][tmp.rem] << "]";
        }
        else
        {
            ss << "\t covar: [" << 0.0f << ",\t" << 0.0f << ",\t" << 0.0f << "]";
        }
        ss << std::endl;

        ss << "\t" << "numSamples: " << numSamples[tmp.quot][tmp.rem] << "\t sumWeights: " << sumWeights[tmp.quot][tmp.rem]
           << "\t sumAssignedSamples: " << sumAssignedSamples[tmp.quot][tmp.rem];
        ss << std::endl;

        sumChiSquareEst += chiSquareMCEstimates[tmp.quot][tmp.rem];
    }
    ss << "sumChiSquareEst: " << sumChiSquareEst << std::endl;
    return ss.str();
}

}  // namespace openpgl
