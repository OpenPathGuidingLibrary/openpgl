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

struct ComponentSplitinfo
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
struct VonMisesFisherChiSquareComponentSplitter
{
   public:
    typedef typename TVMMFactory::Distribution VMM;
    typedef TVMMFactory VMMFactory;

    typedef typename VMMFactory::SufficientStatistics SufficientStatistics;
    typedef typename VMMFactory::PartialFittingMask PartialFittingMask;

    struct SplitCandidate
    {
        size_t componentIndex;
        float chiSquareEst;

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
        embree::Vec2<embree::vfloat<VMM::VectorSize> > splitMeans[VMM::NumVectors];
        embree::Vec3<embree::vfloat<VMM::VectorSize> > splitWeightedSampleCovariances[VMM::NumVectors];

        embree::vfloat<VMM::VectorSize> numSamples[VMM::NumVectors];
        embree::vfloat<VMM::VectorSize> sumWeights[VMM::NumVectors];

        embree::vfloat<VMM::VectorSize> sumAssignedSamples[VMM::NumVectors];

        size_t numComponents{0};

        void clear(const size_t &_numComponents);
        void clearAll();

        float getChiSquareEst(const size_t &idx) const;
        float getSumChiSquareEst() const;
        size_t getHighestChiSquareIdx() const;

        void mergeComponentStats(const size_t &idxI, const size_t &idxJ, const float &weightI, const Vector3 &meanDirectionI, const float &weightJ, const Vector3 &meanDirectionJ,
                                 const float &weightK, const Vector3 &meanDirectionK);

        Vector2 getSplitMean(const size_t &idx) const;

        Vector3 getSplitCovariance(const size_t &idx) const;

        std::vector<SplitCandidate> getSplitCandidates() const;

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

    void PerformSplitting(VMM &vmm, const float &splitThreshold, const float &mcEstimate, const SampleData *data, const size_t &numData,
                          const typename VMMFactory::Configuration factoryCfg, const bool &doPartialRefit, const int &maxSplittingItr = -1) const;

    void PerformRecursiveSplitting(VMM &vmm, typename VMMFactory::SufficientStatistics &suffStats, const float &splitThreshold, const float &mcEstimate, const SampleData *data,
                                   const size_t &numData, const typename VMMFactory::Configuration factoryCfg) const;

    void PerformSplittingIteration(VMM &vmm, const float &splitThreshold) const;

    void CalculateSplitStatistics(const VMM &vmm, ComponentSplitStatistics &splitStats, const float &mcEstimate, const SampleData *data, const size_t &numData) const;

    void UpdateSplitStatistics(const VMM &vmm, ComponentSplitStatistics &splitStats, const float &mcEstimate, const SampleData *data, const size_t &numData) const;

    bool SplitComponent(VMM &vmm, ComponentSplitStatistics &splitStats, SufficientStatistics &suffStats, const size_t idx) const;

    bool SplitComponentIntoThree(VMM &vmm, ComponentSplitStatistics &splitStats, SufficientStatistics &suffStats, const size_t idx) const;

    ComponentSplitinfo GetProjectedLocalDirections(const VMM &vmm, const size_t &idx, const SampleData *data, const size_t &numData, Vector3 *local2D) const;
};

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

    float cosTheta = std::max(-1.0f, std::min(1.0f, vec3D.z));
    ScalarType alpha = embree::fastapprox::acos(cosTheta);
    OPENPGL_ASSERT(embree::isvalid(alpha));
    ScalarType sinAlpha = embree::fastapprox::sin(alpha);
    OPENPGL_ASSERT(embree::isvalid(sinAlpha));
    ScalarType inv_sinc = sinAlpha != 0.f ? alpha / sinAlpha : 0.f;
    OPENPGL_ASSERT(embree::isvalid(inv_sinc));

    vec2D.x = embree::select(alpha > 0.0f, vec3D.x * inv_sinc, vec2D.x);
    vec2D.y = embree::select(alpha > 0.0f, vec3D.y * inv_sinc, vec2D.y);
    OPENPGL_ASSERT(embree::isvalid(vec2D.x));
    OPENPGL_ASSERT(embree::isvalid(vec2D.y));
    return vec2D;
}

// expMapping https://ronnybergmann.net/mvirt/manifolds/Sn/exp.html
template <typename Vec3Type, typename Vec2Type, typename ScalarType>
inline Vec3Type Map2DTo3D(const Vec2Type &vec2D)
{
    Vec3Type vec3D = Vec3Type(0.0f);
    ScalarType norm2 = vec2D.x * vec2D.x + vec2D.y * vec2D.y;
    ScalarType length = norm2 > 0.f ? embree::sqrt(norm2) : 0.f;
    OPENPGL_ASSERT(length < M_PI_F);
    ScalarType sinc = length > FLT_EPSILON ? embree::fastapprox::sin(length) / length : 0.f;

    vec3D.x = embree::select(length > 0.0f, vec2D.x * sinc, vec3D.x);
    vec3D.y = embree::select(length > 0.0f, vec2D.y * sinc, vec3D.y);
    vec3D.z = embree::cos(length);

    return vec3D;
}

#endif

inline std::string ComponentSplitinfo::toString() const
{
    std::stringstream ss;
    ss << "ComponentSplitinfo:" << std::endl;
    // ss << "mean: " << mean << std::endl;
    // ss << "covariance: " << covariance << std::endl;
    ss << "eigenValue0: " << eigenValue0 << std::endl;
    ss << "eigenValue1: " << eigenValue1 << std::endl;
    // ss << "eigenVector0: " << eigenVector0 << std::endl;
    // ss << "eigenVector1: " << eigenVector1 << std::endl;
    return ss.str();
}

template <class TVMMFactory>
void VonMisesFisherChiSquareComponentSplitter<TVMMFactory>::CalculateSplitStatistics(const VMM &vmm, ComponentSplitStatistics &splitStats, const float &mcEstimate,
                                                                                     const SampleData *data, const size_t &numData) const
{
    splitStats.clear(vmm._numComponents);
    this->UpdateSplitStatistics(vmm, splitStats, mcEstimate, data, numData);
}

template <class TVMMFactory>
void VonMisesFisherChiSquareComponentSplitter<TVMMFactory>::PerformSplitting(VMM &vmm, const float &splitThreshold, const float &mcEstimate, const SampleData *data,
                                                                             const size_t &numData, const typename VMMFactory::Configuration factoryCfg, const bool &doPartialRefit,
                                                                             const int &maxSplittingItr) const
{
    PartialFittingMask mask;
    ComponentSplitStatistics splitStatistics;
    SufficientStatistics suffStatistics;

    bool stopSplitting = false;

    size_t splitItr = 0;

    VMMFactory vmmFactory;
    typename VMMFactory::FittingStatistics vmmFitStats;

#ifndef OPENPGL_USE_THREE_SPLIT
    while (vmm._numComponents < VMM::MaxComponents && !stopSplitting)
#else
    while (vmm._numComponents < VMM::MaxComponents - 1 && !stopSplitting)
#endif
    {
        stopSplitting = true;
        splitStatistics.clearAll();
        this->CalculateSplitStatistics(vmm, splitStatistics, mcEstimate, data, numData);

        std::vector<SplitCandidate> splitComps = splitStatistics.getSplitCandidates();

        mask.resetToFalse();
        const size_t numComp = vmm._numComponents;
        for (size_t k = 0; k < numComp; k++)
        {
            if (splitComps[k].chiSquareEst > splitThreshold && vmm._numComponents < VMM::MaxComponents)
            {
                // std::cout << "split[" << k << "]: idx:" << splitComps[k].componentIndex << "\t chi2: " << splitComps[k].chiSquareEst << std::endl;
#ifndef OPENPGL_USE_THREE_SPLIT
                bool splitSuccess = SplitComponent(vmm, splitStatistics, suffStatistics, splitComps[k].componentIndex);
                mask.setToTrue(splitComps[k].componentIndex);
                mask.setToTrue(vmm._numComponents - 1);
#else
                bool splitSuccess = SplitComponentIntoThree(vmm, splitStatistics, suffStatistics, splitComps[k].componentIndex);
                mask.setToTrue(splitComps[k].componentIndex);
                mask.setToTrue(vmm._numComponents - 2);
                mask.setToTrue(vmm._numComponents - 1);
#endif
                if (splitSuccess)
                {
                    stopSplitting = false;
                }
            }
            else
            {
                continue;
            }
        }
        suffStatistics.clear(vmm._numComponents);
        // std::cout << "mask: " << mask.toString() << std::endl;
        // std::cout << "vmmSplit: " << vmm.toString() << std::endl;
        // std::cout << "factoryCfg: " << factoryCfg.toString() << std::endl;
        // std::cout << "suffStatistics: " << suffStatistics.toString() << std::endl;
        if (doPartialRefit)
        {
            vmmFactory.partialUpdateMixture(vmm, mask, suffStatistics, data, numData, factoryCfg, vmmFitStats);
            // std::cout << "vmmpartialUpdate: " << vmm.toString() << std::endl;
            splitItr++;
        }
        else
        {
            stopSplitting = true;
        }

        if (splitItr >= maxSplittingItr)
        {
            stopSplitting = true;
        }
    }
}

template <class TVMMFactory>
void VonMisesFisherChiSquareComponentSplitter<TVMMFactory>::PerformRecursiveSplitting(VMM &vmm, typename VMMFactory::SufficientStatistics &suffStatistics,
                                                                                      const float &splitThreshold, const float &mcEstimate, const SampleData *data,
                                                                                      const size_t &numData, const typename VMMFactory::Configuration factoryCfg) const
{
    PartialFittingMask mask;
    ComponentSplitStatistics splitStatistics;
    SufficientStatistics tempSuffStatistics = suffStatistics;

    // bool stopSplitting = false;
    // size_t splitItr = 0;

    VMMFactory vmmFactory;
    typename VMMFactory::FittingStatistics vmmFitStats;
    // std::cout << "vmm: " << vmm.toString() << std::endl;
    int numSplits = -1;
#ifndef OPENPGL_USE_THREE_SPLIT
    while (vmm._numComponents < VMM::MaxComponents && numSplits != 0)
#else

#endif
    // for (size_t j =0; j<1; j++)
    {
        numSplits = 0;
        splitStatistics.clearAll();
        this->CalculateSplitStatistics(vmm, splitStatistics, mcEstimate, data, numData);

        std::vector<SplitCandidate> splitComps = splitStatistics.getSplitCandidates();

        mask.resetToFalse();
        const size_t numComp = vmm._numComponents;
        for (size_t k = 0; k < numComp; k++)
        {
            if (splitComps[k].chiSquareEst > splitThreshold && vmm._numComponents < VMM::MaxComponents)
            {
#ifndef OPENPGL_USE_THREE_SPLIT
                bool splitSuccess = SplitComponent(vmm, splitStatistics, tempSuffStatistics, splitComps[k].componentIndex);
                if (splitSuccess)
                {
                    mask.setToTrue(splitComps[k].componentIndex);
                    mask.setToTrue(vmm._numComponents - 1);
                }
#else
                bool splitSuccess = SplitComponentIntoThree(vmm, splitStatistics, tempSuffStatistics, splitComps[k].componentIndex);
                if (splitSuccess)
                {
                    mask.setToTrue(splitComps[k].componentIndex);
                    mask.setToTrue(vmm._numComponents - 1);
                    mask.setToTrue(vmm._numComponents - 2);
                }
#endif
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
        if (numSplits > 0)
        {
            tempSuffStatistics.clear(vmm._numComponents);
            vmmFactory.partialUpdateMixture(vmm, mask, tempSuffStatistics, data, numData, factoryCfg, vmmFitStats);
            // std::cout << "tempSuffStatistics" << std::endl << tempSuffStatistics.toString() << std::endl;
            suffStatistics.setNumComponents(vmm._numComponents);
            suffStatistics.maskedReplace(mask, tempSuffStatistics);
        }
        // std::cout << "vmmpartialUpdate: " << vmm.toString() << std::endl;
        // splitItr++;
    }
}

template <class TVMMFactory>
ComponentSplitinfo VonMisesFisherChiSquareComponentSplitter<TVMMFactory>::GetProjectedLocalDirections(const VMM &vmm, const size_t &idx, const SampleData *data,
                                                                                                      const size_t &numData, Vector3 *local2D) const
{
    typename VMM::SoftAssignment softAssign;
    const embree::vfloat<VMM::VectorSize> zeros(0.f);
    // const int cnt = (vmm._numComponents + VMM::VectorSize-1) / VMM::VectorSize;
    // size_t validDataCount = 0.0f;

    ComponentSplitinfo splitInfo;

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
            // const embree::vfloat<VMM::VectorSize> samplePDF = sample.pdf;
            // const vfloat<VMM::VectorSize> value =  weight * samplePDF;

            const embree::Vec3<embree::vfloat<VMM::VectorSize> > localDirection =
                embree::frame(vmm._meanDirections[tmp.quot]).inverse() * embree::Vec3<embree::vfloat<VMM::VectorSize> >(sampleDirection);
            // const embree::Vec2< embree::vfloat<VMM::VectorSize> > localDirection2D = Map3DTo2D< embree::Vec3< embree::vfloat<VMM::VectorSize> >,  embree::Vec2<
            // embree::vfloat<VMM::VectorSize> >, embree::vfloat<VMM::VectorSize> >(localDirection);
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

    float norm0 = splitInfo.eigenVector0.x * splitInfo.eigenVector0.x + splitInfo.eigenVector0.y * splitInfo.eigenVector0.y;
    float norm1 = splitInfo.eigenVector1.x * splitInfo.eigenVector1.x + splitInfo.eigenVector1.y * splitInfo.eigenVector1.y;
    norm0 = norm0 > FLT_EPSILON ? embree::rsqrt(norm0) : 1.f;
    norm1 = norm1 > FLT_EPSILON ? embree::rsqrt(norm1) : 1.f;
    splitInfo.eigenVector0 *= norm0;
    splitInfo.eigenVector1 *= norm1;
#ifdef OPENPGL_SHOW_PRINT_OUTS
    std::cout << "split: " << "\tmean: " << splitInfo.mean.x << ", \t " << splitInfo.mean.y << "\t covariance: " << splitInfo.covariance.x << ", \t " << splitInfo.covariance.y
              << ", \t " << splitInfo.covariance.z << std::endl;
    std::cout << "eigen: " << "\tevalue0: " << splitInfo.eigenValue0 << "\teVec0: " << splitInfo.eigenVector0.x << ", \t " << splitInfo.eigenVector0.y
              << "\tevalue1: " << splitInfo.eigenValue1 << "\teVec1: " << splitInfo.eigenVector1.x << ", \t " << splitInfo.eigenVector1.y << std::endl;
#endif
    return splitInfo;
}

template <class TVMMFactory>
void VonMisesFisherChiSquareComponentSplitter<TVMMFactory>::UpdateSplitStatistics(const VMM &vmm, ComponentSplitStatistics &splitStats, const float &mcEstimate,
                                                                                  const SampleData *data, const size_t &numData) const
{
    // std::cout << "UpdateSplitStatistics" << std::endl;

    OPENPGL_ASSERT(vmm._numComponents == splitStats.numComponents);

    typename VMM::SoftAssignment softAssign;
    const embree::vfloat<VMM::VectorSize> zeros(0.f);
    const int cnt = (splitStats.numComponents + VMM::VectorSize - 1) / VMM::VectorSize;
    // size_t validDataCount = 0.0f;

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
            // std::cout << "data[" << n << "]: " << "value: " << value << "\t samplePDF: " << samplePDF;
            for (size_t k = 0; k < cnt; k++)
            {
                OPENPGL_ASSERT(embree::all(embree::isvalid(splitStats.splitMeans[k].x)));
                OPENPGL_ASSERT(embree::all(embree::isvalid(splitStats.splitMeans[k].y)));
                OPENPGL_ASSERT(embree::all(embree::isvalid(splitStats.splitWeightedSampleCovariances[k].x)));
                OPENPGL_ASSERT(embree::all(embree::isvalid(splitStats.splitWeightedSampleCovariances[k].y)));
                OPENPGL_ASSERT(embree::all(embree::isvalid(splitStats.splitWeightedSampleCovariances[k].z)));

                embree::vfloat<VMM::VectorSize> vmfPDF = softAssign.assignments[k] * softAssign.pdf;
                embree::vfloat<VMM::VectorSize> partialValuePDF = vmfPDF * value;
                partialValuePDF /= (mcEstimate * softAssign.pdf);
                // partialValuePDF /= vmm._weights[k] * mcEstimate;
                // std::cout << "\tweights: " << vmm._weights[k] << "\t assign: " << softAssign.assignments[k] << "\t pdf: " << softAssign.pdf << std::endl;
                // std::cout << "\tpvPDF: " << partialValuePDF << "\t vmfPDF: " << vmfPDF << std::endl;

                embree::vfloat<VMM::VectorSize> chiSquareEst = value * value * vmfPDF;
                chiSquareEst /= mcEstimate * mcEstimate * softAssign.pdf * softAssign.pdf;
                // chiSquareEst *= chiSquareEst;
                chiSquareEst -= 2.0f * partialValuePDF;
                chiSquareEst += vmfPDF;
                chiSquareEst /= samplePDF;

                chiSquareEst = select(softAssign.assignments[k] > 0.f, chiSquareEst, zeros);

                splitStats.sumAssignedSamples[k] += softAssign.assignments[k];
                // incremental updated of the MC chiSquare estimate
                splitStats.numSamples[k] += 1.0f;
                splitStats.chiSquareMCEstimates[k] += (chiSquareEst - splitStats.chiSquareMCEstimates[k]) / splitStats.numSamples[k];

                const embree::Vec3<embree::vfloat<VMM::VectorSize> > localDirection =
                    embree::frame(vmm._meanDirections[k]).inverse() * embree::Vec3<embree::vfloat<VMM::VectorSize> >(sampleDirection);
                const embree::Vec2<embree::vfloat<VMM::VectorSize> > localDirection2D(localDirection.x, localDirection.y);
                const embree::vfloat<VMM::VectorSize> assignedWeight = softAssign.assignments[k] * weight;
                // const vfloat<VMM::VectorSize> assignedWeight = softAssign.assignments[k] * weight * weight;

                splitStats.sumWeights[k] += assignedWeight;
                //                const vfloat<VMM::VectorSize> incWeight = select(splitStats.sumWeights[k] > 0.0f, assignedWeight / splitStats.sumWeights[k], zeros);

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
                // splitStats.sumWeights[k] += assignedWeight;
            }
            // validDataCount++;
            // std::cout << std::endl;
        }
    }
    // splitStats.numSamplesOld += validDataCount;
    // splitStats.mcEstimate += mcEstimate;
}

template <class TVMMFactory>
bool VonMisesFisherChiSquareComponentSplitter<TVMMFactory>::SplitComponent(VMM &vmm, ComponentSplitStatistics &splitStats, SufficientStatistics &suffStats, const size_t idx) const
{
    ComponentSplitinfo splitInfo;
    const div_t tmpK = div(idx, static_cast<int>(VMM::VectorSize));

    float numAssignedSamples = splitStats.sumAssignedSamples[tmpK.quot][tmpK.rem];

    float inv_sumWeights = embree::rcp(splitStats.sumWeights[tmpK.quot][tmpK.rem]);
    OPENPGL_ASSERT(embree::isvalid(inv_sumWeights));
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

    float norm0 = splitInfo.eigenVector0.x * splitInfo.eigenVector0.x + splitInfo.eigenVector0.y * splitInfo.eigenVector0.y;
    float norm1 = splitInfo.eigenVector1.x * splitInfo.eigenVector1.x + splitInfo.eigenVector1.y * splitInfo.eigenVector1.y;
    norm0 = norm0 > FLT_EPSILON ? embree::rsqrt(norm0) : 1.f;
    norm1 = norm1 > FLT_EPSILON ? embree::rsqrt(norm1) : 1.f;
    splitInfo.eigenVector0 *= norm0;
    splitInfo.eigenVector1 *= norm1;

    OPENPGL_ASSERT(embree::isvalid(splitInfo.eigenVector0.x));
    OPENPGL_ASSERT(embree::isvalid(splitInfo.eigenVector0.y));
    OPENPGL_ASSERT(embree::isvalid(splitInfo.eigenVector1.x));
    OPENPGL_ASSERT(embree::isvalid(splitInfo.eigenVector1.y));

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

    float newWeight0 = weight * 0.5f;
    float newWeight1 = newWeight0;

    Vector3 meanDirection0 = meanDirection;
    Vector3 meanDirection1 = meanDirection;

    float newMeanCosine0 = meanCosine;
    float newMeanCosine1 = meanCosine * meanCosine;

    if (D > 1e-8f)
    {
        Vector2 meanDir2D0 = splitInfo.mean + (splitInfo.eigenVector0 * splitInfo.eigenValue0 * 0.5f);
        meanDirection0 = embree::frame(meanDirection) * Map2DTo3D<Vector3, Vector2, float>(meanDir2D0);
        newMeanCosine0 = meanCosine / std::abs(dot(meanDirection, meanDirection0));

        // TODO: further investigate:
        // newMeanCosine0 = meanCosine / dot(meanDirection, meanDirection0);

        OPENPGL_ASSERT(meanCosine >= 0.f);
        OPENPGL_ASSERT(std::abs(dot(meanDirection, meanDirection0)) > 0.f);
        OPENPGL_ASSERT(newMeanCosine0 >= 0.f);
        // ensure that the new mean cosine is in a valid range (i.e., < 1.0 and < the mean cosine of max kappa)
        newMeanCosine0 = std::min(newMeanCosine0, KappaToMeanCosine<float>(OPENPGL_MAX_KAPPA));
        newMeanCosine1 = newMeanCosine0;

        Vector2 meanDir2D1 = splitInfo.mean - (splitInfo.eigenVector0 * splitInfo.eigenValue0 * 0.5f);
        meanDirection1 = embree::frame(meanDirection) * Map2DTo3D<Vector3, Vector2, float>(meanDir2D1);

#ifdef OPENPGL_SHOW_PRINT_OUTS
        // std::cout << "meanCosine: " << meanCosine << "\t kappa: " << kappa << "\t newMeanCosine: " << newMeanCosine0 << " \t newKkappa: " <<  newKkappa0 << std::endl;
        // std::cout << "localMeanDirection0: " << Map2DTo3D<Vector3, Vector2, float>(meanDir2D0) << "\t meanDirection0: " << meanDirection0 << "\t meanCosine: " << meanCosine << "
        // \t costheta0: " <<  dot(meanDirection, meanDirection0) << std::endl; std::cout << "localMeanDirection1: " << Map2DTo3D<Vector3, Vector2, float>(meanDir2D1) << "\t
        // meanDirection1: " << meanDirection1 << "\t meanCosine: " << meanCosine << " \t costheta1: " <<  dot(meanDirection, meanDirection1) << std::endl; std::cout <<
        // "eigenValue0: " << splitInfo.eigenValue0 << "\t eigenVector0: " << splitInfo.eigenVector0 << std::endl; std::cout << "eigenValue1: " << splitInfo.eigenValue1 << "\t
        // eigenVector1: " << splitInfo.eigenVector1 << std::endl;
        std::cout << "D: " << D << "\t idx: " << idx << " \t assignedSamples: " << numAssignedSamples << std::endl;
        // std::cout << "kappa: " << kappa <<  " \t newKkappa: " <<  newKkappa0  << " \t costheta0: " <<  dot(meanDirection, meanDirection0) << "\t angle: " <<
        // std::acos(dot(meanDirection, meanDirection0)) * 180.0f / M_PI_F<< std::endl;
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
        return false;
    }
    size_t K = vmm._numComponents;

    const div_t tmpI = tmpK;
    const div_t tmpJ = div(K, static_cast<int>(VMM::VectorSize));

    vmm.splitComponent(idx, K, newWeight0, newWeight1, meanDirection0, meanDirection1, newMeanCosine0, newMeanCosine1);
    suffStats.splitComponentsStats(idx, K, meanDirection0, meanDirection1, newMeanCosine0, newMeanCosine1);

    // reseting the split statistics for the two new components
    splitStats.chiSquareMCEstimates[tmpI.quot][tmpI.rem] = 0.0f;
    splitStats.sumAssignedSamples[tmpI.quot][tmpI.rem] = 0.0f;
    splitStats.numSamples[tmpI.quot][tmpI.rem] = 0.0f;
    splitStats.sumWeights[tmpI.quot][tmpI.rem] = 0.0f;
    splitStats.splitMeans[tmpI.quot].x[tmpI.rem] = 0.0f;
    splitStats.splitMeans[tmpI.quot].y[tmpI.rem] = 0.0f;
    splitStats.splitWeightedSampleCovariances[tmpI.quot].x[tmpI.rem] = 0.0f;
    splitStats.splitWeightedSampleCovariances[tmpI.quot].y[tmpI.rem] = 0.0f;
    splitStats.splitWeightedSampleCovariances[tmpI.quot].z[tmpI.rem] = 0.0f;

    splitStats.chiSquareMCEstimates[tmpJ.quot][tmpJ.rem] = 0.0f;
    splitStats.sumAssignedSamples[tmpJ.quot][tmpJ.rem] = 0.0f;
    splitStats.numSamples[tmpJ.quot][tmpJ.rem] = 0.0f;
    splitStats.sumWeights[tmpJ.quot][tmpJ.rem] = 0.0f;
    splitStats.splitMeans[tmpJ.quot].x[tmpJ.rem] = 0.0f;
    splitStats.splitMeans[tmpJ.quot].y[tmpJ.rem] = 0.0f;
    splitStats.splitWeightedSampleCovariances[tmpJ.quot].x[tmpJ.rem] = 0.0f;
    splitStats.splitWeightedSampleCovariances[tmpJ.quot].y[tmpJ.rem] = 0.0f;
    splitStats.splitWeightedSampleCovariances[tmpJ.quot].z[tmpJ.rem] = 0.0f;

    splitStats.numComponents = K + 1;

    return true;
}

template <class TVMMFactory>
bool VonMisesFisherChiSquareComponentSplitter<TVMMFactory>::SplitComponentIntoThree(VMM &vmm, ComponentSplitStatistics &splitStats, SufficientStatistics &suffStats,
                                                                                    const size_t idx) const
{
    ComponentSplitinfo splitInfo;
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

    float norm0 = splitInfo.eigenVector0.x * splitInfo.eigenVector0.x + splitInfo.eigenVector0.y * splitInfo.eigenVector0.y;
    float norm1 = splitInfo.eigenVector1.x * splitInfo.eigenVector1.x + splitInfo.eigenVector1.y * splitInfo.eigenVector1.y;
    norm0 = norm0 > FLT_EPSILON ? embree::rsqrt(norm0) : 1.f;
    norm1 = norm1 > FLT_EPSILON ? embree::rsqrt(norm1) : 1.f;
    splitInfo.eigenVector0 *= norm0;
    splitInfo.eigenVector1 *= norm1;
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
        // float meanCosine1 = meanCosine / meanDirection0.z;
        // float kappa1 = MeanCosineToKappa<float> (meanCosine1);
#ifdef OPENPGL_SHOW_PRINT_OUTS
        // std::cout << "meanCosine: " << meanCosine << "\t kappa: " << kappa << "\t newMeanCosine: " << newMeanCosine0 << " \t newKkappa: " <<  newKkappa0 << std::endl;
        // std::cout << "localMeanDirection0: " << Map2DTo3D<Vector3, Vector2, float>(meanDir2D0) << "\t meanDirection0: " << meanDirection0 << "\t meanCosine: " << meanCosine << "
        // \t costheta0: " <<  dot(meanDirection, meanDirection0) << std::endl; std::cout << "localMeanDirection1: " << Map2DTo3D<Vector3, Vector2, float>(meanDir2D1) << "\t
        // meanDirection1: " << meanDirection1 << "\t meanCosine: " << meanCosine << " \t costheta1: " <<  dot(meanDirection, meanDirection1) << std::endl; std::cout <<
        // "eigenValue0: " << splitInfo.eigenValue0 << "\t eigenVector0: " << splitInfo.eigenVector0 << std::endl; std::cout << "eigenValue1: " << splitInfo.eigenValue1 << "\t
        // eigenVector1: " << splitInfo.eigenVector1 << std::endl;
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
    splitStats.chiSquareMCEstimates[tmpI.quot][tmpI.rem] = 0.0f;
    splitStats.sumAssignedSamples[tmpI.quot][tmpI.rem] = 0.0f;
    splitStats.numSamples[tmpI.quot][tmpI.rem] = 0.0f;
    splitStats.sumWeights[tmpI.quot][tmpI.rem] = 0.0f;
    splitStats.splitMeans[tmpI.quot].x[tmpI.rem] = 0.0f;
    splitStats.splitMeans[tmpI.quot].y[tmpI.rem] = 0.0f;
    splitStats.splitWeightedSampleCovariances[tmpI.quot].x[tmpI.rem] = 0.0f;
    splitStats.splitWeightedSampleCovariances[tmpI.quot].y[tmpI.rem] = 0.0f;
    splitStats.splitWeightedSampleCovariances[tmpI.quot].z[tmpI.rem] = 0.0f;

    splitStats.chiSquareMCEstimates[tmpJ.quot][tmpJ.rem] = 0.0f;
    splitStats.sumAssignedSamples[tmpJ.quot][tmpJ.rem] = 0.0f;
    splitStats.numSamples[tmpJ.quot][tmpJ.rem] = 0.0f;
    splitStats.sumWeights[tmpJ.quot][tmpJ.rem] = 0.0f;
    splitStats.splitMeans[tmpJ.quot].x[tmpJ.rem] = 0.0f;
    splitStats.splitMeans[tmpJ.quot].y[tmpJ.rem] = 0.0f;
    splitStats.splitWeightedSampleCovariances[tmpJ.quot].x[tmpJ.rem] = 0.0f;
    splitStats.splitWeightedSampleCovariances[tmpJ.quot].y[tmpJ.rem] = 0.0f;
    splitStats.splitWeightedSampleCovariances[tmpJ.quot].z[tmpJ.rem] = 0.0f;

    splitStats.chiSquareMCEstimates[tmpL.quot][tmpL.rem] = 0.0f;
    splitStats.sumAssignedSamples[tmpL.quot][tmpL.rem] = 0.0f;
    splitStats.numSamples[tmpL.quot][tmpL.rem] = 0.0f;
    splitStats.sumWeights[tmpL.quot][tmpL.rem] = 0.0f;
    splitStats.splitMeans[tmpL.quot].x[tmpL.rem] = 0.0f;
    splitStats.splitMeans[tmpL.quot].y[tmpL.rem] = 0.0f;
    splitStats.splitWeightedSampleCovariances[tmpL.quot].x[tmpL.rem] = 0.0f;
    splitStats.splitWeightedSampleCovariances[tmpL.quot].y[tmpL.rem] = 0.0f;
    splitStats.splitWeightedSampleCovariances[tmpL.quot].z[tmpL.rem] = 0.0f;

    splitStats.numComponents = K + 2;

    return true;
}

template <class TVMMFactory>
void VonMisesFisherChiSquareComponentSplitter<TVMMFactory>::ComponentSplitStatistics::serialize(std::ostream &stream) const
{
    serializeFloatVectors<VMM::NumVectors, VMM::VectorSize>(stream, chiSquareMCEstimates);
    serializeVec2Vectors<VMM::NumVectors, VMM::VectorSize>(stream, splitMeans);
    serializeVec3Vectors<VMM::NumVectors, VMM::VectorSize>(stream, splitWeightedSampleCovariances);
    serializeFloatVectors<VMM::NumVectors, VMM::VectorSize>(stream, numSamples);
    serializeFloatVectors<VMM::NumVectors, VMM::VectorSize>(stream, sumWeights);
    serializeFloatVectors<VMM::NumVectors, VMM::VectorSize>(stream, sumAssignedSamples);
    stream.write(reinterpret_cast<const char *>(&numComponents), sizeof(size_t));
}

template <class TVMMFactory>
void VonMisesFisherChiSquareComponentSplitter<TVMMFactory>::ComponentSplitStatistics::deserialize(std::istream &stream)
{
    deserializeFloatVectors<VMM::NumVectors, VMM::VectorSize>(stream, chiSquareMCEstimates);
    deserializeVec2Vectors<VMM::NumVectors, VMM::VectorSize>(stream, splitMeans);
    deserializeVec3Vectors<VMM::NumVectors, VMM::VectorSize>(stream, splitWeightedSampleCovariances);
    deserializeFloatVectors<VMM::NumVectors, VMM::VectorSize>(stream, numSamples);
    deserializeFloatVectors<VMM::NumVectors, VMM::VectorSize>(stream, sumWeights);
    deserializeFloatVectors<VMM::NumVectors, VMM::VectorSize>(stream, sumAssignedSamples);
    stream.read(reinterpret_cast<char *>(&numComponents), sizeof(size_t));
}

template <class TVMMFactory>
bool VonMisesFisherChiSquareComponentSplitter<TVMMFactory>::ComponentSplitStatistics::isValid() const
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
bool VonMisesFisherChiSquareComponentSplitter<TVMMFactory>::ComponentSplitStatistics::operator==(const ComponentSplitStatistics &b) const
{
    bool equal = true;
    if (numComponents != b.numComponents)
    {
        equal = false;
    }

    for (int k = 0; k < VMM::NumVectors; k++)
    {
        if (embree::any(chiSquareMCEstimates[k] != b.chiSquareMCEstimates[k]) || embree::any(splitMeans[k].x != b.splitMeans[k].x) ||
            embree::any(splitMeans[k].y != b.splitMeans[k].y) || embree::any(splitWeightedSampleCovariances[k].x != b.splitWeightedSampleCovariances[k].x) ||
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
Vector2 VonMisesFisherChiSquareComponentSplitter<TVMMFactory>::ComponentSplitStatistics::getSplitMean(const size_t &idx) const
{
    const div_t tmp = div(idx, static_cast<int>(VMM::VectorSize));
    return Vector2(splitMeans[tmp.quot].x[tmp.rem], splitMeans[tmp.quot].y[tmp.rem]);
}

template <class TVMMFactory>
Vector3 VonMisesFisherChiSquareComponentSplitter<TVMMFactory>::ComponentSplitStatistics::getSplitCovariance(const size_t &idx) const
{
    const div_t tmp = div(idx, static_cast<int>(VMM::VectorSize));
    Vector3 covariance(splitWeightedSampleCovariances[tmp.quot].x[tmp.rem], splitWeightedSampleCovariances[tmp.quot].y[tmp.rem],
                       splitWeightedSampleCovariances[tmp.quot].z[tmp.rem]);
    covariance /= sumWeights[tmp.quot][tmp.rem];
    return covariance;
}

template <class TVMMFactory>
void VonMisesFisherChiSquareComponentSplitter<TVMMFactory>::ComponentSplitStatistics::mergeComponentStats(const size_t &idxI, const size_t &idxJ, const float &weightI,
                                                                                                          const Vector3 &meanDirectionI, const float &weightJ,
                                                                                                          const Vector3 &meanDirectionJ, const float &weightK,
                                                                                                          const Vector3 &meanDirectionK)
{
    // TODO: check if numSamples or sumWeights are zero for one of the merge components
    // std::cout << "mergeComponentStats: " << "\tidxI: " << idxI << "\tidxJ: " << idxJ << "\tweightI: " << weightI << "\tmeanDirectionI: " << meanDirectionI<< "\tweightJ: " <<
    // weightJ << "\tmeanDirectionJ: " << meanDirectionJ<< "\tweightK: " << weightK << "\tmeanDirectionK: " << meanDirectionK << std::endl;

    // EM algorithms for Gaussian mixtures with split-and-merge operation
    const div_t tmpI = div(idxI, static_cast<int>(VMM::VectorSize));
    const div_t tmpJ = div(idxJ, static_cast<int>(VMM::VectorSize));

    const div_t tmpL = div(numComponents - 1, VMM::VectorSize);

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

    Vector2 meanDirection2DItoK = Map3DTo2D<Vector3, Vector2, float>(inv_transformK * meanDirectionI3D);
    Vector2 meanDirection2DJtoK = Map3DTo2D<Vector3, Vector2, float>(inv_transformK * meanDirectionJ3D);

    const float inv_weightK = (weightK > 0.f) ? embree::rcp(weightK) : 1.f;

    const float sumWeightsI = sumWeights[tmpI.quot][tmpI.rem];
    const float sumWeightsJ = sumWeights[tmpJ.quot][tmpJ.rem];
    const float sumWeightsK = sumWeightsI + sumWeightsJ;

    // std::cout << "\tsumWeightsI: " << sumWeightsI << "\tsumWeightsJ: " << sumWeightsJ << "\tsumWeightsK: " << sumWeightsK << std::endl;
    // std::cout << "\tnumSamplesI: " << numSamples[tmpI.quot][tmpI.rem] << "\tsumWeightsJ: " << numSamples[tmpJ.quot][tmpJ.rem] << std::endl;

    const Vector3 covarianceI = (sumWeightsI > 0.f) ? Vector3(splitWeightedSampleCovariances[tmpI.quot].x[tmpI.rem], splitWeightedSampleCovariances[tmpI.quot].y[tmpI.rem],
                                                              splitWeightedSampleCovariances[tmpI.quot].z[tmpI.rem]) *
                                                          embree::rcp(sumWeightsI)
                                                    : Vector3(0.f);
    const Vector3 covarianceJ = (sumWeightsJ > 0.f) ? Vector3(splitWeightedSampleCovariances[tmpJ.quot].x[tmpJ.rem], splitWeightedSampleCovariances[tmpJ.quot].y[tmpJ.rem],
                                                              splitWeightedSampleCovariances[tmpJ.quot].z[tmpJ.rem]) *
                                                          embree::rcp(sumWeightsJ)
                                                    : Vector3(0.f);

#ifdef OPENPGL_ZERO_MEAN
    const Vector2 meanDirectionK2D(0.f);
#else
    const Vector2 meanDirectionK2D = inv_weightK * (weightI * meanDirection2DItoK + weightJ * meanDirection2DJtoK);
#endif
    Vector3 meanII = Vector3(meanDirection2DItoK.x * meanDirection2DItoK.x, meanDirection2DItoK.y * meanDirection2DItoK.y, meanDirection2DItoK.x * meanDirection2DItoK.y);
    Vector3 meanJJ = Vector3(meanDirection2DJtoK.x * meanDirection2DJtoK.x, meanDirection2DJtoK.y * meanDirection2DJtoK.y, meanDirection2DJtoK.x * meanDirection2DJtoK.y);
    Vector3 covarianceK = (weightI * covarianceI + weightI * meanII + weightJ * covarianceJ + weightJ * meanJJ);
    covarianceK *= inv_weightK;
#ifndef OPENPGL_ZERO_MEAN
    Vector3 meanKK = Vector3(meanDirectionK2D.x * meanDirectionK2D.x, meanDirectionK2D.y * meanDirectionK2D.y, meanDirectionK2D.x * meanDirectionK2D.y);
    covarianceK -= meanKK;
#endif
    const Vector3 sampleCovarianceK = covarianceK * sumWeightsK;
    OPENPGL_ASSERT(embree::isvalid(sampleCovarianceK.x));
    OPENPGL_ASSERT(embree::isvalid(sampleCovarianceK.y));
    OPENPGL_ASSERT(embree::isvalid(sampleCovarianceK.z));

    // merge additional stats
    const float sumAssignedSamplesK = sumAssignedSamples[tmpI.quot][tmpI.rem] + sumAssignedSamples[tmpJ.quot][tmpJ.rem];
    const float numSamplesK = inv_weightK * (weightI * numSamples[tmpI.quot][tmpI.rem] + weightJ * numSamples[tmpJ.quot][tmpJ.rem]);
    const float chiSquareMCEstimatesK = chiSquareMCEstimates[tmpI.quot][tmpI.rem] + chiSquareMCEstimates[tmpJ.quot][tmpJ.rem];

    // insert stats of the merged components a the ith positions
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

    // replace stats of the last and jth component
    splitMeans[tmpJ.quot].x[tmpJ.rem] = splitMeans[tmpL.quot].x[tmpL.rem];
    splitMeans[tmpJ.quot].y[tmpJ.rem] = splitMeans[tmpL.quot].y[tmpL.rem];
    splitWeightedSampleCovariances[tmpJ.quot].x[tmpJ.rem] = splitWeightedSampleCovariances[tmpL.quot].x[tmpL.rem];
    splitWeightedSampleCovariances[tmpJ.quot].y[tmpJ.rem] = splitWeightedSampleCovariances[tmpL.quot].y[tmpL.rem];
    splitWeightedSampleCovariances[tmpJ.quot].z[tmpJ.rem] = splitWeightedSampleCovariances[tmpL.quot].z[tmpL.rem];
    sumWeights[tmpJ.quot][tmpJ.rem] = sumWeights[tmpL.quot][tmpL.rem];
    numSamples[tmpJ.quot][tmpJ.rem] = numSamples[tmpL.quot][tmpL.rem];
    sumAssignedSamples[tmpJ.quot][tmpJ.rem] = sumAssignedSamples[tmpL.quot][tmpL.rem];
    chiSquareMCEstimates[tmpJ.quot][tmpJ.rem] = chiSquareMCEstimates[tmpL.quot][tmpL.rem];

    // reset stats of last component
    splitMeans[tmpL.quot].x[tmpL.rem] = 0.0f;
    splitMeans[tmpL.quot].y[tmpL.rem] = 0.0f;
    splitWeightedSampleCovariances[tmpL.quot].x[tmpL.rem] = 0.0f;
    splitWeightedSampleCovariances[tmpL.quot].y[tmpL.rem] = 0.0f;
    splitWeightedSampleCovariances[tmpL.quot].z[tmpL.rem] = 0.0f;
    sumWeights[tmpL.quot][tmpL.rem] = 0.0f;
    numSamples[tmpL.quot][tmpL.rem] = 0.0f;
    sumAssignedSamples[tmpL.quot][tmpL.rem] = 0.0f;
    chiSquareMCEstimates[tmpL.quot][tmpL.rem] = 0.0f;

    numComponents--;
}

template <class TVMMFactory>
std::vector<typename VonMisesFisherChiSquareComponentSplitter<TVMMFactory>::SplitCandidate>
VonMisesFisherChiSquareComponentSplitter<TVMMFactory>::ComponentSplitStatistics::getSplitCandidates() const
{
    std::vector<SplitCandidate> splitCandidates;
    for (size_t k = 0; k < numComponents; k++)
    {
        const div_t tmp = div(k, static_cast<int>(VMM::VectorSize));
        SplitCandidate sc;
        sc.chiSquareEst = chiSquareMCEstimates[tmp.quot][tmp.rem];
        sc.componentIndex = k;
        splitCandidates.push_back(sc);
    }

    std::sort(splitCandidates.begin(), splitCandidates.end(), [](SplitCandidate a, SplitCandidate b) {
        return a > b;
    });
    return splitCandidates;
}

template <class TVMMFactory>
void VonMisesFisherChiSquareComponentSplitter<TVMMFactory>::ComponentSplitStatistics::clear(const size_t &_numComponents)
{
    const embree::vfloat<VMM::VectorSize> zeros(0.f);

    this->numComponents = _numComponents;
    const int cnt = (this->numComponents + VMM::VectorSize - 1) / VMM::VectorSize;

    for (size_t k = 0; k < cnt; k++)
    {
        chiSquareMCEstimates[k] = zeros;
        splitWeightedSampleCovariances[k].x = zeros;
        splitWeightedSampleCovariances[k].y = zeros;
        splitWeightedSampleCovariances[k].z = zeros;

        splitMeans[k].x = zeros;
        splitMeans[k].y = zeros;

        numSamples[k] = zeros;
        sumWeights[k] = zeros;
        sumAssignedSamples[k] = zeros;
    }
}

template <class TVMMFactory>
void VonMisesFisherChiSquareComponentSplitter<TVMMFactory>::ComponentSplitStatistics::decay(const float &alpha)
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
    }
}

template <class TVMMFactory>
size_t VonMisesFisherChiSquareComponentSplitter<TVMMFactory>::ComponentSplitStatistics::getHighestChiSquareIdx() const
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
void VonMisesFisherChiSquareComponentSplitter<TVMMFactory>::ComponentSplitStatistics::clearAll()
{
    this->clear(VMM::MaxComponents);
}

template <class TVMMFactory>
float VonMisesFisherChiSquareComponentSplitter<TVMMFactory>::ComponentSplitStatistics::getChiSquareEst(const size_t &idx) const
{
    const div_t tmp = div(idx, static_cast<int>(VMM::VectorSize));
    return chiSquareMCEstimates[tmp.quot][tmp.rem];
}

template <class TVMMFactory>
float VonMisesFisherChiSquareComponentSplitter<TVMMFactory>::ComponentSplitStatistics::getSumChiSquareEst() const
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
std::string VonMisesFisherChiSquareComponentSplitter<TVMMFactory>::ComponentSplitStatistics::toString() const
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