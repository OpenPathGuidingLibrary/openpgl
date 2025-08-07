// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../../openpgl_common.h"
#include "ParallaxAwareVonMisesFisherMixture.h"
#include "VMMChiSquareComponentSplitter.h"

namespace openpgl
{

template <class TVMMFactory, class TSplitter>
struct VonMisesFisherChiSquareComponentMerger
{
   public:
    typedef typename TVMMFactory::Distribution VMM;
    typedef typename TVMMFactory::SufficientStatistics SufficientStatistics;
    typedef typename TSplitter::ComponentSplitStatistics ComponentSplitStatistics;
    // typedef std::integral_constant<size_t, (maxComponents + (VecSize -1)) / VecSize> NumVectors;

    float MergeNext(VMM &vmm) const;

    bool ThresholdedMergeNext(VMM &vmm, const float &mergeThreshold, float &mergeCost) const;

    bool ThresholdedMergeNext(VMM &vmm, const float mergeThreshold, const float splitThreshold, const bool considerSplitThreshold, float &mergeCost,
                              SufficientStatistics &suffStats, ComponentSplitStatistics &splitStats) const;

    float CalculateMergeCost(const VMM &vmm, const size_t &idx0, const size_t &idx1) const;

    size_t PerformMerging(VMM &vmm, const float &mergeThreshold) const;

    size_t PerformMerging(VMM &vmm, const float mergeThreshold, const float splitThreshold, const bool considerSplitThreshold, SufficientStatistics &suffStats,
                          ComponentSplitStatistics &splitStats) const;

   private:
    inline float _IntegratedProduct(const Vector3 &meanDirection0, const float &kappa0, const float &normalization0, const Vector3 &meanDirection1, const float &kappa1,
                                    const float &normalization1) const;

    inline float _IntegratedDivision(const Vector3 &meanDirection0, const float &kappa0, const float &normalization0, const Vector3 &meanDirection1, const float &kappa1,
                                     const float &normalization1, const float &eMinus2Kappa1) const;

    inline float _Product(const Vector3 &meanDirection0, const float &kappa0, const float &normalization0, const Vector3 &meanDirection1, const float &kappa1,
                          const float &normalization1, Vector3 &meanDirection, float &kappa, float &normalization) const;
};

template <class TVMMFactory, class TSplitter>
size_t VonMisesFisherChiSquareComponentMerger<TVMMFactory, TSplitter>::PerformMerging(VMM &vmm, const float &mergeThreshold) const
{
    bool stopMerging = false;
    size_t totalMergeCount = 0;
    // while (vmm._numComponents > 1 && !stopMerging)
    while (vmm._numComponents > VMM::VectorSize && !stopMerging)
    {
        float mergeCost = 0.0f;
        stopMerging = true;
        bool mergedComponents = ThresholdedMergeNext(vmm, mergeThreshold, mergeCost);
        stopMerging = !mergedComponents;
        // std::cout << "merge: " << "\tmergedComponents: " << mergedComponents << "\tmergeCost: " << mergeCost << std::endl;
        if (mergedComponents)
        {
            totalMergeCount++;
        }
    }
    return totalMergeCount;
}

template <class TVMMFactory, class TSplitter>
size_t VonMisesFisherChiSquareComponentMerger<TVMMFactory, TSplitter>::PerformMerging(VMM &vmm, const float mergeThreshold, const float splitThreshold,
                                                                                      const bool considerSplitThreshold, SufficientStatistics &suffStats,
                                                                                      ComponentSplitStatistics &splitStats) const
{
    bool stopMerging = false;
    size_t totalMergeCount = 0;
    // while (vmm._numComponents > 1 && !stopMerging)
    while (vmm._numComponents > VMM::VectorSize && !stopMerging)
    {
        float mergeCost = 0.0f;
        stopMerging = true;
        bool mergedComponents = ThresholdedMergeNext(vmm, mergeThreshold, splitThreshold, considerSplitThreshold, mergeCost, suffStats, splitStats);

        stopMerging = !mergedComponents;
        // std::cout << "merge: " << "\tmergedComponents: " << mergedComponents << "\tmergeCost: " << mergeCost << std::endl;
        if (mergedComponents)
        {
            totalMergeCount++;
        }
    }
#ifdef OPENPGL_SHOW_PRINT_OUTS
    std::cout << "PerformMerging: totalMergeCount = " << totalMergeCount << "\t mergeThreshold: " << mergeThreshold << std::endl;
#endif
    return totalMergeCount;
}

template <class TVMMFactory, class TSplitter>
float VonMisesFisherChiSquareComponentMerger<TVMMFactory, TSplitter>::CalculateMergeCost(const VMM &vmm, const size_t &idx0, const size_t &idx1) const
{
    const div_t div0 = div(idx0, VMM::VectorSize);
    float weight0 = vmm._weights[div0.quot][div0.rem];
    const float kappa0 = vmm._kappas[div0.quot][div0.rem];
    const Vector3 meanDirection0 = Vector3(vmm._meanDirections[div0.quot].x[div0.rem], vmm._meanDirections[div0.quot].y[div0.rem], vmm._meanDirections[div0.quot].z[div0.rem]);
    const float meanCosine0 = vmm._meanCosines[div0.quot][div0.rem];
    const float normalization0 = vmm._normalizations[div0.quot][div0.rem];

    const div_t div1 = div(idx1, VMM::VectorSize);
    float weight1 = vmm._weights[div1.quot][div1.rem];
    const float kappa1 = vmm._kappas[div1.quot][div1.rem];
    const Vector3 meanDirection1 = Vector3(vmm._meanDirections[div1.quot].x[div1.rem], vmm._meanDirections[div1.quot].y[div1.rem], vmm._meanDirections[div1.quot].z[div1.rem]);
    const float meanCosine1 = vmm._meanCosines[div1.quot][div1.rem];
    const float normalization1 = vmm._normalizations[div1.quot][div1.rem];

    if (idx0 == idx1)
    {
        weight0 /= 2.0f;
        weight1 /= 2.0f;
    }

    // merge component
    float weight = weight0 + weight1;
    float kappa = 0.0f;
    float meanCosine = 0.0f;
    float normalization = ONE_OVER_FOUR_PI;
    float eMinus2Kappa = 1.0f;

    Vector3 meanDirection = weight0 * meanCosine0 * meanDirection0 + weight1 * meanCosine1 * meanDirection1;
    meanDirection /= weight;
    meanCosine = meanDirection.x * meanDirection.x + meanDirection.y * meanDirection.y + meanDirection.z * meanDirection.z;
    if (meanCosine > 0.f)
    {
        meanCosine = std::sqrt(meanCosine);
        kappa = MeanCosineToKappa<float>(meanCosine);
        kappa = kappa < 1e-3f ? 0.f : kappa;
        // eMin2Kappa = math::fastexp( -2.0f * kappa );
        eMinus2Kappa = embree::exp(-2.0f * kappa);
        normalization = kappa / (2.0f * M_PI_F * (1.0f - eMinus2Kappa));

        meanDirection /= meanCosine;
    }
    else
    {
        meanDirection = meanDirection0;
    }

    float weight00 = weight0 * weight0;
    float kappa00;
    float normalization00;
    Vector3 meanDirection00;
    float scale00 = _Product(meanDirection0, kappa0, normalization0, meanDirection0, kappa0, normalization0, meanDirection00, kappa00, normalization00);

    float weight11 = weight1 * weight1;
    float kappa11;
    float normalization11;
    Vector3 meanDirection11;
    float scale11 = _Product(meanDirection1, kappa1, normalization1, meanDirection1, kappa1, normalization1, meanDirection11, kappa11, normalization11);

    float weight01 = weight0 * weight1;
    float kappa01;
    float normalization01;
    Vector3 meanDirection01;
    float scale01 = _Product(meanDirection0, kappa0, normalization0, meanDirection1, kappa1, normalization1, meanDirection01, kappa01, normalization01);

    float chiSquareIJ = 0.0f;
    chiSquareIJ += (weight00 / weight) * (scale00 * _IntegratedDivision(meanDirection00, kappa00, normalization00, -meanDirection, kappa, normalization, eMinus2Kappa));
    chiSquareIJ += (weight11 / weight) * (scale11 * _IntegratedDivision(meanDirection11, kappa11, normalization11, -meanDirection, kappa, normalization, eMinus2Kappa));
    chiSquareIJ += 2.0f * (weight01 / weight) * (scale01 * _IntegratedDivision(meanDirection01, kappa01, normalization01, -meanDirection, kappa, normalization, eMinus2Kappa));
    chiSquareIJ -= weight;

    return chiSquareIJ;
}

template <class TVMMFactory, class TSplitter>
bool VonMisesFisherChiSquareComponentMerger<TVMMFactory, TSplitter>::ThresholdedMergeNext(VMM &vmm, const float &mergeThreshold, float &mergeCost) const
{
    // std::cout  << vmm.toString()<<std::endl;
    int K = vmm._numComponents;
    int mergeCandidateI = 0;
    int mergeCandidateJ = 0;
    float minMergeCost = std::numeric_limits<float>::max();

    bool foundMergeCandidates = false;
    for (size_t i = 0; i < K - 1; i++)
    {
        for (size_t j = i + 1; j < K; j++)
        {
            float mergeCost = CalculateMergeCost(vmm, i, j);
            if (mergeCost < mergeThreshold && mergeCost < minMergeCost)
            {
                mergeCandidateI = i;
                mergeCandidateJ = j;
                minMergeCost = mergeCost;
                foundMergeCandidates = true;
            }
        }
    }

    if (foundMergeCandidates)
    {
        vmm.mergeComponents(mergeCandidateI, mergeCandidateJ);
        mergeCost = minMergeCost;
#ifdef OPENPGL_SHOW_PRINT_OUTS
        std::cout << "merge: " << "\tidx0: " << mergeCandidateI << "\tidx1: " << mergeCandidateJ << "\tK: " << vmm._numComponents << std::endl;
#endif
    }
    return foundMergeCandidates;
}

template <class TVMMFactory, class TSplitter>
bool VonMisesFisherChiSquareComponentMerger<TVMMFactory, TSplitter>::ThresholdedMergeNext(VMM &vmm, const float mergeThreshold, const float splitThreshold,
                                                                                          const bool considerSplitThreshold, float &mergeCost, SufficientStatistics &suffStats,
                                                                                          ComponentSplitStatistics &splitStats) const
{
    OPENPGL_ASSERT(splitStats.isValid());
    int K = vmm._numComponents;
    int mergeCandidateI = 0;
    int mergeCandidateJ = 0;
    float minMergeCost = std::numeric_limits<float>::max();

    bool foundMergeCandidates = false;
    for (size_t i = 0; i < K - 1; i++)
    {
        const div_t tmpI = div(i, static_cast<int>(VMM::VectorSize));
        for (size_t j = i + 1; j < K; j++)
        {
            const div_t tmpJ = div(j, static_cast<int>(VMM::VectorSize));
            float mergeCost = CalculateMergeCost(vmm, i, j);
            if (mergeCost < mergeThreshold && mergeCost < minMergeCost && splitStats.numSamples[tmpI.quot][tmpI.rem] > 0.0f && splitStats.numSamples[tmpJ.quot][tmpJ.rem] > 0.0f &&
                (!considerSplitThreshold ||
                 (considerSplitThreshold && (splitStats.chiSquareMCEstimates[tmpI.quot][tmpI.rem] + splitStats.chiSquareMCEstimates[tmpJ.quot][tmpJ.rem] < splitThreshold))))
            {
                mergeCandidateI = i;
                mergeCandidateJ = j;
                minMergeCost = mergeCost;
                foundMergeCandidates = true;
            }
        }
    }

    if (foundMergeCandidates)
    {
#ifdef OPENPGL_SHOW_PRINT_OUTS
        std::cout << "merge: " << "\tidx0: " << mergeCandidateI << "\tidx1: " << mergeCandidateJ << "\tK: " << vmm._numComponents << std::endl;
        std::cout << "\tweightI: " << vmm.getComponentWeight(mergeCandidateI) << "\tweightJ: " << vmm.getComponentWeight(mergeCandidateJ);
        std::cout << "\tkappaI: " << vmm.getComponentKappa(mergeCandidateI) << "\tkappaJ: " << vmm.getComponentKappa(mergeCandidateJ);
        std::cout << "\t angle: " << std::acos(dot(vmm.getComponentMeanDirection(mergeCandidateI), vmm.getComponentMeanDirection(mergeCandidateJ))) * 180.0f / M_PI_F << std::endl;
#endif
        // get old (before merge) mean directions and weights
        Vector3 meanDirectionI = vmm.getComponentMeanDirection(mergeCandidateI);
        float weightI = vmm.getComponentWeight(mergeCandidateI);
        Vector3 meanDirectionJ = vmm.getComponentMeanDirection(mergeCandidateJ);
        float weightJ = vmm.getComponentWeight(mergeCandidateJ);

        vmm.mergeComponents(mergeCandidateI, mergeCandidateJ);

        // get the merged mean direction and weight
        Vector3 meanDirectionK = vmm.getComponentMeanDirection(mergeCandidateI);
        float weightK = vmm.getComponentWeight(mergeCandidateI);
        OPENPGL_ASSERT(splitStats.isValid());
        splitStats.mergeComponentStats(mergeCandidateI, mergeCandidateJ, weightI, meanDirectionI, weightJ, meanDirectionJ, weightK, meanDirectionK);
        OPENPGL_ASSERT(splitStats.isValid());
        OPENPGL_ASSERT(suffStats.isValid());
        suffStats.mergeComponentStats(mergeCandidateI, mergeCandidateJ);
        OPENPGL_ASSERT(suffStats.isValid());
        mergeCost = minMergeCost;

        // OPENPGL_ASSERT(vmm._numComponents == suffStats.numComponents);
        OPENPGL_ASSERT(vmm._numComponents == splitStats.numComponents);
    }
    return foundMergeCandidates;
}

template <class TVMMFactory, class TSplitter>
float VonMisesFisherChiSquareComponentMerger<TVMMFactory, TSplitter>::MergeNext(VMM &vmm) const
{
    int K = vmm._numComponents;
    int mergeCandidateI = 0;
    int mergeCandidateJ = 0;
    float minMergeCost = std::numeric_limits<float>::max();

    for (size_t i = 0; i < K - 1; i++)
    {
        for (size_t j = i + 1; j < K; j++)
        {
            float mergeCost = CalculateMergeCost(vmm, i, j);
            if (mergeCost < minMergeCost)
            {
                mergeCandidateI = i;
                mergeCandidateJ = j;
                minMergeCost = mergeCost;
            }
        }
    }

    vmm.mergeComponents(mergeCandidateI, mergeCandidateJ);
    return minMergeCost;
}

template <class TVMMFactory, class TSplitter>
float VonMisesFisherChiSquareComponentMerger<TVMMFactory, TSplitter>::_IntegratedProduct(const Vector3 &meanDirection0, const float &kappa0, const float &normalization0,
                                                                                         const Vector3 &meanDirection1, const float &kappa1, const float &normalization1) const
{
    Vector3 productMeanDirection = kappa0 * meanDirection0 + kappa1 * meanDirection1;

    float productKappa = embree::sqrt(dot(productMeanDirection, productMeanDirection));

    float productNormalization = 1.0f / (4.0f * M_PI_F);
    float productEMinus2Kappa = 1.0f;
    if (productKappa > 1e-3f)
    {
        productEMinus2Kappa = std::exp(-2.0f * productKappa);
        productNormalization = productKappa / (2.0f * M_PI_F * (1.0f - productEMinus2Kappa));
        productMeanDirection /= productKappa;
    }
    else
    {
        productKappa = 0.0f;
        productMeanDirection = meanDirection0;
    }

    float scale = (normalization0 * normalization1) / productNormalization;
    float cosTheta0 = dot(meanDirection0, productMeanDirection);
    float cosTheta1 = dot(meanDirection1, productMeanDirection);
    scale *= std::exp(kappa0 * (cosTheta0 - 1.0f) + kappa1 * (cosTheta1 - 1.0f));

    return scale;
}

template <class TVMMFactory, class TSplitter>
float VonMisesFisherChiSquareComponentMerger<TVMMFactory, TSplitter>::_IntegratedDivision(const Vector3 &meanDirection0, const float &kappa0, const float &normalization0,
                                                                                          const Vector3 &meanDirection1, const float &kappa1, const float &normalization1,
                                                                                          const float &eMinus2Kappa1) const
{
    Vector3 productMeanDirection = kappa0 * meanDirection0 + kappa1 * meanDirection1;

    float productKappa = embree::sqrt(dot(productMeanDirection, productMeanDirection));

    float productNormalization = 1.0f / (4.0f * M_PI_F);
    float productEMinus2Kappa = 1.0f;
    if (productKappa > 1e-3f)
    {
        productEMinus2Kappa = std::exp(-2.0f * productKappa);
        productNormalization = productKappa / (2.0f * M_PI_F * (1.0f - productEMinus2Kappa));
        productMeanDirection /= productKappa;
    }
    else
    {
        productKappa = 0.0f;
        productMeanDirection = meanDirection0;
    }

    float scale = (normalization0 * normalization1) / productNormalization;
    float cosTheta0 = dot(meanDirection0, productMeanDirection);
    float cosTheta1 = dot(meanDirection1, productMeanDirection);
    scale *= (4.0f * M_PI_F * M_PI_F * (1.0f - eMinus2Kappa1)) / (kappa1 * kappa1);
    scale *= std::exp((kappa0 * (cosTheta0 - 1.0f) + kappa1 * (cosTheta1 - 1.0f)) + (2.0f * kappa1));

    return scale;
}

template <class TVMMFactory, class TSplitter>
float VonMisesFisherChiSquareComponentMerger<TVMMFactory, TSplitter>::_Product(const Vector3 &meanDirection0, const float &kappa0, const float &normalization0,
                                                                               const Vector3 &meanDirection1, const float &kappa1, const float &normalization1,
                                                                               Vector3 &productMeanDirection, float &productKappa, float &productNormalization) const
{
    productMeanDirection = kappa0 * meanDirection0 + kappa1 * meanDirection1;
    productKappa = embree::sqrt(dot(productMeanDirection, productMeanDirection));

    productNormalization = 1.0f / (4.0f * M_PI_F);
    float productEMinus2Kappa = 1.0f;
    if (productKappa > 1e-3f)
    {
        productEMinus2Kappa = std::exp(-2.0f * productKappa);
        productNormalization = productKappa / (2.0f * M_PI_F * (1.0f - productEMinus2Kappa));
        productMeanDirection /= productKappa;
    }
    else
    {
        productKappa = 0.0f;
        productMeanDirection = meanDirection0;
    }

    float scale = (normalization0 * normalization1) / productNormalization;
    float cosTheta0 = dot(meanDirection0, productMeanDirection);
    float cosTheta1 = dot(meanDirection1, productMeanDirection);

    scale *= std::exp(kappa0 * (cosTheta0 - 1.0f) + kappa1 * (cosTheta1 - 1.0f));

    return scale;
}

}  // namespace openpgl
