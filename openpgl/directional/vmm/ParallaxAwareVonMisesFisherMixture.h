// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <embreeSrc/common/math/linearspace3.h>
#include <embreeSrc/common/math/transcendental.h>
#include <embreeSrc/common/math/vec2.h>
#include <embreeSrc/common/math/vec3.h>
#if !defined(__CUDACC__)
#include <embreeSrc/common/simd/simd.h>
#endif
#include <math.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>

#include "../../openpgl_common.h"
#include "../../data/BlobWriter.h"

#define OPENPGL_MIN_KAPPA 1e-3f

#define USE_SIMD_CDF_SAMPLING
// #define VALIDATE_SELECT_COMPONENT_SIMD

namespace openpgl
{
namespace OPENPGL_KERNEL_NS
{

template <typename Type>
KERNEL_FUNCTION Type MeanCosineToKappa(const Type &meanCosine);

template <typename Type>
KERNEL_FUNCTION Type KappaToMeanCosine(const Type &kappa);

template <class Kernel_, int maxComponents, bool UseParallaxCompensation>
struct ParallaxAwareVonMisesFisherMixture
{
   public:
    using Kernel = Kernel_;

    enum
    {
        ParallaxCompensation = UseParallaxCompensation
    };

    enum
    {
        MaxComponents = maxComponents,
        //VectorSize = VectorSize,
        NumVectors = (maxComponents + (VectorSize - 1)) / VectorSize
    };

   public:
    struct SoftAssignment
    {
        vfloat assignments[NumVectors];
        size_t size;
        float pdf;

        KERNEL_FUNCTION std::string toString() const;
        KERNEL_FUNCTION bool isValid() const;
    };

   public:
    ParallaxAwareVonMisesFisherMixture() = default;

    // VMM attributes
    vfloat _weights[NumVectors];
    vfloat _kappas[NumVectors];
    embree::Vec3<vfloat> _meanDirections[NumVectors];

    vfloat _normalizations[NumVectors];
    vfloat _eMinus2Kappa[NumVectors];
    vfloat _meanCosines[NumVectors];

    uint32_t _numComponents{maxComponents};

    // Parallax-aware attributes
    vfloat _distances[NumVectors];
    Point3 _pivotPosition;//{0.0f, 0.0f, 0.0f};

    void dump(BlobWriter& writer) const {
        //writer << (uint64_t)(sizeof(uint64_t) + getNumComponents() * 5 * sizeof(float));
        writer << (uint64_t)getNumComponents();
        // emulate datastructure used by Thomas' PPG ipmlementation for visualizer!
        for (int k = 0; k < getNumComponents(); k++) {
            const div_t tmp = div(k, static_cast<int>(VectorSize));
            writer << _kappas[tmp.quot][tmp.rem];
            writer << _meanDirections[tmp.quot].x[tmp.rem] << _meanDirections[tmp.quot].y[tmp.rem]
                   << _meanDirections[tmp.quot].z[tmp.rem];
            writer << _distances[tmp.quot][tmp.rem];
            writer << _weights[tmp.quot][tmp.rem];
        }
    };

#ifdef OPENPGL_RADIANCE_CACHES
    // fluence attributes
    // float _fluence {0.0f};
    float _numFluenceSamples{0.f};
    Vector3 _fluenceRGB{0.0f, 0.0f, 0.0f};
    Vector3 _fluenceRGBWithMIS{0.0f, 0.0f, 0.0f};
    embree::Vec3<vfloat> _fluenceRGBWeightsWithMIS[NumVectors];
    embree::Vec3<vfloat> _fluenceRGBWeights[NumVectors];
#endif
    KERNEL_FUNCTION void serialize(std::ostream &stream) const;

    KERNEL_FUNCTION void deserialize(std::istream &stream);

    KERNEL_FUNCTION void uniformInit(float kappa);

    KERNEL_FUNCTION bool softAssignment(Vector3 direction, SoftAssignment &assignment) const;

    KERNEL_FUNCTION float pdf(Vector3 direction) const;

    KERNEL_FUNCTION Vector3 sample(const Vector2 sample) const;

#ifdef USE_SIMD_CDF_SAMPLING
    KERNEL_FUNCTION void selectComponentSIMD(uint32_t &selectedVector, uint32_t &selectedComponent, Vector2 &_sample) const;
#endif
    KERNEL_FUNCTION void selectComponent(uint32_t &selectedVector, uint32_t &selectedComponent, Vector2 &_sample) const;

    KERNEL_FUNCTION void mergeComponents(const size_t &idx0, const size_t &idx1);

    KERNEL_FUNCTION void splitComponent(const size_t &idx0, const size_t &idx1, const float &weight0, const float &weight1, const Vector3 &meanDirection0, const Vector3 &meanDirection1,
                        const float &meanCosine0, const float &meanCosine1);

    KERNEL_FUNCTION void performRelativeParallaxShift(const Vector3 &shiftDirection);

#ifdef OPENPGL_RADIANCE_CACHES
    KERNEL_FUNCTION Vector3 incomingRadiance(const Vector3 &direction, const bool directLightMIS) const;

    KERNEL_FUNCTION Vector3 irradiance(const Vector3 &normal, const bool directLightMIS) const;

    KERNEL_FUNCTION Vector3 inscatteredRadiance(const Vector3 &dir, const float meanCosine, const bool directLightMIS) const;

    KERNEL_FUNCTION Vector3 fluence(const bool directLightMIS) const;
#endif

    // Product and convolution functions
    KERNEL_FUNCTION void convole(const float &meanCosine);

    KERNEL_FUNCTION float product(const float &weight, const Vector3 &meanDirection, const float &kappa);

    KERNEL_FUNCTION float product(const float &weight, const Vector3 &meanDirection, const float &kappa, const float &normalization);

    // Mixture component methods
    KERNEL_FUNCTION void swapComponents(const size_t &idx0, const size_t &idx1);

    KERNEL_FUNCTION void clearComponent(const size_t &idx);

    // Getter methods for the PAVMM attributes
    SHARED_FUNCTION size_t getNumComponents() const;

    KERNEL_FUNCTION void setNumComponents(const size_t &numComponents);

    KERNEL_FUNCTION Vector3 getComponentMeanDirection(const size_t &idx) const;

    KERNEL_FUNCTION void setComponentMeanDirection(const size_t idx, const Vector3 &meanDirection);

    KERNEL_FUNCTION float getComponentWeight(const size_t &idx) const;

    KERNEL_FUNCTION void setComponentWeight(const size_t idx, const float &weight);

    KERNEL_FUNCTION float getComponentKappa(const size_t &idx) const;

    KERNEL_FUNCTION void setComponentKappa(const size_t idx, const float &kappa);

    KERNEL_FUNCTION float getComponentDistance(const size_t &idx) const;

    KERNEL_FUNCTION void setComponentDistance(const size_t &idx, const float &distance);

    KERNEL_FUNCTION void decay(const float alpha)
    {
#ifdef OPENPGL_RADIANCE_CACHES
        _numFluenceSamples *= alpha;
#endif
    }

    KERNEL_FUNCTION bool isValid() const;

    KERNEL_FUNCTION std::string toString() const;

    KERNEL_FUNCTION void _calculateNormalization();

    KERNEL_FUNCTION void _calculateMeanCosines();

    KERNEL_FUNCTION void _normalizeWeights();

    KERNEL_FUNCTION bool operator==(const ParallaxAwareVonMisesFisherMixture &b) const;

   private:
    KERNEL_FUNCTION vfloat _convolvePDF(const size_t k, const embree::Vec3<vfloat> &normal, const vfloat &meanCosine) const;
};

template <class Kernel, int maxComponents, bool UseParallaxCompensation>
SHARED_FUNCTION size_t ParallaxAwareVonMisesFisherMixture<Kernel, maxComponents, UseParallaxCompensation>::getNumComponents() const
{
    return _numComponents;
}

template <class Kernel, int maxComponents, bool UseParallaxCompensation>
KERNEL_FUNCTION void ParallaxAwareVonMisesFisherMixture<Kernel, maxComponents, UseParallaxCompensation>::setNumComponents(const size_t &numComponents)
{
    _numComponents = numComponents;
}

template <class Kernel, int maxComponents, bool UseParallaxCompensation>
KERNEL_FUNCTION std::string ParallaxAwareVonMisesFisherMixture<Kernel, maxComponents, UseParallaxCompensation>::SoftAssignment::toString() const
{
    std::stringstream ss;
    ss << "SoftAssignment:" << std::endl;
    ss << "size: " << size << std::endl;
    ss << "pdf: " << pdf << std::endl;
    for (int k = 0; k < size; k++)
    {
        const div_t tmp = div_(k, static_cast<int>(VectorSize));
        ss << "assign[" << k << "]: " << get(assignments[tmp.quot], tmp.rem);
        ss << std::endl;
    }
    return ss.str();
}

template <class Kernel, int maxComponents, bool UseParallaxCompensation>
KERNEL_FUNCTION bool ParallaxAwareVonMisesFisherMixture<Kernel, maxComponents, UseParallaxCompensation>::SoftAssignment::isValid() const
{
    bool valid = true;

    valid = valid && size > 0;
    valid = valid && size <= maxComponents;
    OPENPGL_ASSERT(valid);

    valid = valid && pdf >= 0;
    valid = valid && embree::isvalid(pdf);
    OPENPGL_ASSERT(valid);

    for (int k = 0; k < size; k++)
    {
        const div_t tmpK = div_(k, static_cast<int>(VectorSize));
        valid = valid && get(assignments[tmpK.quot], tmpK.rem) >= 0.0f;
        valid = valid && embree::isvalid(get(assignments[tmpK.quot], tmpK.rem));
        OPENPGL_ASSERT(valid);
    }
    OPENPGL_ASSERT(valid);
    return valid;
}

template <class Kernel, int maxComponents, bool UseParallaxCompensation>
KERNEL_FUNCTION std::string ParallaxAwareVonMisesFisherMixture<Kernel, maxComponents, UseParallaxCompensation>::toString() const
{
    std::stringstream ss;
    ss.precision(5);
    if (UseParallaxCompensation)
        ss << "ParallaxAwareVonMisesFisherMixture:" << std::endl;
    else
        ss << "VonMisesFisherMixture:" << std::endl;
    ss << "maxComponents: " << maxComponents << std::endl;
    ss << "VectorSize: " << VectorSize << std::endl;
    ss << "numVectors: " << NumVectors << std::endl;
    ss << "---------------------- " << std::endl;
    ss << "numComponents: " << this->_numComponents << std::endl;
    float sumWeights = 0.0f;
    // for ( int k = 0; k < this->_numComponents; k++)
    for (int k = 0; k < maxComponents; k++)
    {
        const div_t tmp = div_(k, static_cast<int>(VectorSize));
        ss << "vmm[" << k << "]: " << "weight: " << get(this->_weights[tmp.quot], tmp.rem);
        ss << "\t kappa: " << get(this->_kappas[tmp.quot], tmp.rem);
        ss << "\t meanDirection: [" << get(this->_meanDirections[tmp.quot].x, tmp.rem) << "\t" << get(this->_meanDirections[tmp.quot].y, tmp.rem) << "\t"
           << get(this->_meanDirections[tmp.quot].z, tmp.rem) << "]";
        ss << "\t length: "
           << embree::length(Vector3(get(this->_meanDirections[tmp.quot].x, tmp.rem), get(this->_meanDirections[tmp.quot].y, tmp.rem), get(this->_meanDirections[tmp.quot].z, tmp.rem)));
        ss << "\t normalization: " << get(this->_normalizations[tmp.quot], tmp.rem);
        ss << "\t eMinus2Kappa: " << get(this->_eMinus2Kappa[tmp.quot], tmp.rem);
        ss << "\t meanCosine: " << get(this->_meanCosines[tmp.quot], tmp.rem);
        ss << "\t distance: " << get(_distances[tmp.quot], tmp.rem);
#ifdef OPENPGL_RADIANCE_CACHES
        ss << "\t fluenceRGBWeightWithMIS: " << get(_fluenceRGBWeightsWithMIS[tmp.quot].x, tmp.rem) << "\t" << get(_fluenceRGBWeightsWithMIS[tmp.quot].y, tmp.rem) << "\t"
           << get(_fluenceRGBWeightsWithMIS[tmp.quot].z, tmp.rem);
        ss << "\t fluenceRGBWeight: " << get(_fluenceRGBWeights[tmp.quot].x, tmp.rem) << "\t" << get(_fluenceRGBWeights[tmp.quot].y, tmp.rem) << "\t"
           << get(_fluenceRGBWeights[tmp.quot].z, tmp.rem);
#endif
        ss << std::endl;
        sumWeights += get(this->_weights[tmp.quot], tmp.rem);
    }

    ss << "pivot: " << _pivotPosition << std::endl;
    ss << "sumWeights: " << sumWeights << std::endl;
#ifdef OPENPGL_RADIANCE_CACHES
    // ss << "fluence: " << _fluence << std::endl;
    ss << "fluenceRGB: " << _fluenceRGB.x << "\t" << _fluenceRGB.y << "\t" << _fluenceRGB.z << std::endl;
    ss << "numFluenceSamples: " << _numFluenceSamples << std::endl;
#endif

    return ss.str();
}

template <class Kernel, int maxComponents, bool UseParallaxCompensation>
KERNEL_FUNCTION void ParallaxAwareVonMisesFisherMixture<Kernel, maxComponents, UseParallaxCompensation>::splitComponent(const size_t &idx0, const size_t &idx1, const float &weight0,
                                                                                                         const float &weight1, const Vector3 &meanDirection0,
                                                                                                         const Vector3 &meanDirection1, const float &meanCosine0,
                                                                                                         const float &meanCosine1)
{
    OPENPGL_ASSERT(meanCosine0 > 0.0f && meanCosine0 <= 1.0f);
    OPENPGL_ASSERT(meanCosine1 > 0.0f && meanCosine1 <= 1.0f);

    const div_t tmpIdx0 = div_(idx0, static_cast<int>(VectorSize));
    const div_t tmpIdx1 = div_(idx1, static_cast<int>(VectorSize));

    // splitting VMM
    get(_weights[tmpIdx0.quot], tmpIdx0.rem) = weight0;
    get(_meanCosines[tmpIdx0.quot], tmpIdx0.rem) = meanCosine0;
    get(_kappas[tmpIdx0.quot], tmpIdx0.rem) = MeanCosineToKappa<float>(meanCosine0);
    get(_meanDirections[tmpIdx0.quot].x, tmpIdx0.rem) = meanDirection0.x;
    get(_meanDirections[tmpIdx0.quot].y, tmpIdx0.rem) = meanDirection0.y;
    get(_meanDirections[tmpIdx0.quot].z, tmpIdx0.rem) = meanDirection0.z;

    get(_weights[tmpIdx1.quot], tmpIdx1.rem) = weight1;
    get(_meanCosines[tmpIdx1.quot], tmpIdx1.rem) = meanCosine1;
    get(_kappas[tmpIdx1.quot], tmpIdx1.rem) = MeanCosineToKappa<float>(meanCosine1);
    get(_meanDirections[tmpIdx1.quot].x, tmpIdx1.rem) = meanDirection1.x;
    get(_meanDirections[tmpIdx1.quot].y, tmpIdx1.rem) = meanDirection1.y;
    get(_meanDirections[tmpIdx1.quot].z, tmpIdx1.rem) = meanDirection1.z;

    // splitting PAVMM
    get(_distances[tmpIdx1.quot], tmpIdx1.rem) = get(_distances[tmpIdx0.quot], tmpIdx0.rem);
#ifdef OPENPGL_RADIANCE_CACHES
    const float nWeight0 = weight0 / (weight0 + weight1);
    const float nWeight1 = weight1 / (weight0 + weight1);
    get(_fluenceRGBWeightsWithMIS[tmpIdx1.quot].x, tmpIdx1.rem) = get(_fluenceRGBWeightsWithMIS[tmpIdx0.quot].x, tmpIdx0.rem) * nWeight1;
    get(_fluenceRGBWeightsWithMIS[tmpIdx1.quot].y, tmpIdx1.rem) = get(_fluenceRGBWeightsWithMIS[tmpIdx0.quot].y, tmpIdx0.rem) * nWeight1;
    get(_fluenceRGBWeightsWithMIS[tmpIdx1.quot].z, tmpIdx1.rem) = get(_fluenceRGBWeightsWithMIS[tmpIdx0.quot].z, tmpIdx0.rem) * nWeight1;

    get(_fluenceRGBWeightsWithMIS[tmpIdx0.quot].x, tmpIdx0.rem) *= nWeight0;
    get(_fluenceRGBWeightsWithMIS[tmpIdx0.quot].y, tmpIdx0.rem) *= nWeight0;
    get(_fluenceRGBWeightsWithMIS[tmpIdx0.quot].z, tmpIdx0.rem) *= nWeight0;

    get(_fluenceRGBWeights[tmpIdx1.quot].x, tmpIdx1.rem) = get(_fluenceRGBWeights[tmpIdx0.quot].x, tmpIdx0.rem) * nWeight1;
    get(_fluenceRGBWeights[tmpIdx1.quot].y, tmpIdx1.rem) = get(_fluenceRGBWeights[tmpIdx0.quot].y, tmpIdx0.rem) * nWeight1;
    get(_fluenceRGBWeights[tmpIdx1.quot].z, tmpIdx1.rem) = get(_fluenceRGBWeights[tmpIdx0.quot].z, tmpIdx0.rem) * nWeight1;

    get(_fluenceRGBWeights[tmpIdx0.quot].x, tmpIdx0.rem) *= nWeight0;
    get(_fluenceRGBWeights[tmpIdx0.quot].y, tmpIdx0.rem) *= nWeight0;
    get(_fluenceRGBWeights[tmpIdx0.quot].z, tmpIdx0.rem) *= nWeight0;
#endif

    if (idx1 == _numComponents)
    {
        _numComponents++;
    }
    _calculateNormalization();
}

template <class Kernel, int maxComponents, bool UseParallaxCompensation>
KERNEL_FUNCTION void ParallaxAwareVonMisesFisherMixture<Kernel, maxComponents, UseParallaxCompensation>::mergeComponents(const size_t &idx0, const size_t &idx1)
{
    const div_t tmpIdx0 = div_(idx0, VectorSize);
    const div_t tmpIdx1 = div_(idx1, VectorSize);

    // const div_t tmpIdx2 = div_( this->_numComponents -1, VectorSize);
    if (idx0 != idx1)
    {
        const float weight0 = get(_weights[tmpIdx0.quot], tmpIdx0.rem);
        const float weight1 = get(_weights[tmpIdx1.quot], tmpIdx1.rem);

        const float meanCosine0 = get(_meanCosines[tmpIdx0.quot], tmpIdx0.rem);
        const float meanCosine1 = get(_meanCosines[tmpIdx1.quot], tmpIdx1.rem);

        float kappa = 0.0f;
        float norm = ONE_OVER_FOUR_PI;
        float eMin2Kappa = 1.0f;

        float weight = weight0 + weight1;

        float meanDirectionX = weight0 * meanCosine0 * get(_meanDirections[tmpIdx0.quot].x, tmpIdx0.rem) + weight1 * meanCosine1 * get(_meanDirections[tmpIdx1.quot].x, tmpIdx1.rem);
        float meanDirectionY = weight0 * meanCosine0 * get(_meanDirections[tmpIdx0.quot].y, tmpIdx0.rem) + weight1 * meanCosine1 * get(_meanDirections[tmpIdx1.quot].y, tmpIdx1.rem);
        float meanDirectionZ = weight0 * meanCosine0 * get(_meanDirections[tmpIdx0.quot].z, tmpIdx0.rem) + weight1 * meanCosine1 * get(_meanDirections[tmpIdx1.quot].z, tmpIdx1.rem);

        // std::cout << "mergeComponents: cosTheta: " << get(_meanDirections[tmpIdx0.quot].x, tmpIdx0.rem) *get(_meanDirections[tmpIdx1.quot].x, tmpIdx1.rem) +
        //                                                 get(_meanDirections[tmpIdx0.quot].y, tmpIdx0.rem) *get(_meanDirections[tmpIdx1.quot].y, tmpIdx1.rem) +
        //                                                 get(_meanDirections[tmpIdx0.quot].z, tmpIdx0.rem) *get(_meanDirections[tmpIdx1.quot].z, tmpIdx1.rem) << std::endl;

        meanDirectionX /= weight;
        meanDirectionY /= weight;
        meanDirectionZ /= weight;

        float meanCosine = meanDirectionX * meanDirectionX + meanDirectionY * meanDirectionY + meanDirectionZ * meanDirectionZ;

        if (meanCosine > 0.0f)
        {
            meanCosine = std::sqrt(meanCosine);

            kappa = MeanCosineToKappa<float>(meanCosine);
            kappa = kappa < 1e-3f ? 0.f : kappa;
            // eMin2Kappa = math::fastexp( -2.0f * kappa );
            eMin2Kappa = embree::exp(-2.0f * kappa);
            norm = kappa / (2.0f * M_PI_F * (1.0f - eMin2Kappa));

            meanDirectionX /= meanCosine;
            meanDirectionY /= meanCosine;
            meanDirectionZ /= meanCosine;
        }
        else
        {
            meanDirectionX = get(_meanDirections[tmpIdx0.quot].x, tmpIdx0.rem);
            meanDirectionY = get(_meanDirections[tmpIdx0.quot].y, tmpIdx0.rem);
            meanDirectionZ = get(_meanDirections[tmpIdx0.quot].z, tmpIdx0.rem);
        }

        get(_weights[tmpIdx0.quot], tmpIdx0.rem) = weight;
        get(_kappas[tmpIdx0.quot], tmpIdx0.rem) = kappa;
        get(_meanCosines[tmpIdx0.quot], tmpIdx0.rem) = meanCosine;

        get(_normalizations[tmpIdx0.quot], tmpIdx0.rem) = norm;
        get(_eMinus2Kappa[tmpIdx0.quot], tmpIdx0.rem) = eMin2Kappa;

        get(_meanDirections[tmpIdx0.quot].x, tmpIdx0.rem) = meanDirectionX;
        get(_meanDirections[tmpIdx0.quot].y, tmpIdx0.rem) = meanDirectionY;
        get(_meanDirections[tmpIdx0.quot].z, tmpIdx0.rem) = meanDirectionZ;

        const float distance0 = get(_distances[tmpIdx0.quot], tmpIdx0.rem);
        const float distance1 = get(_distances[tmpIdx1.quot], tmpIdx1.rem);

        float newDistance = weight0 * distance0 + weight1 * distance1;
        newDistance /= (weight0 + weight1);

        get(_distances[tmpIdx0.quot], tmpIdx0.rem) = newDistance;
#ifdef OPENPGL_RADIANCE_CACHES
        const Vector3 fluenceRGBWeightsWithMIS0(get(_fluenceRGBWeightsWithMIS[tmpIdx0.quot].x, tmpIdx0.rem), get(_fluenceRGBWeightsWithMIS[tmpIdx0.quot].y, tmpIdx0.rem),
                                                get(_fluenceRGBWeightsWithMIS[tmpIdx0.quot].z, tmpIdx0.rem));
        const Vector3 fluenceRGBWeightsWithMIS1(get(_fluenceRGBWeightsWithMIS[tmpIdx1.quot].x, tmpIdx1.rem), get(_fluenceRGBWeightsWithMIS[tmpIdx1.quot].y, tmpIdx1.rem),
                                                get(_fluenceRGBWeightsWithMIS[tmpIdx1.quot].z, tmpIdx1.rem));
        get(_fluenceRGBWeightsWithMIS[tmpIdx0.quot].x, tmpIdx0.rem) = fluenceRGBWeightsWithMIS0.x + fluenceRGBWeightsWithMIS1.x;
        get(_fluenceRGBWeightsWithMIS[tmpIdx0.quot].y, tmpIdx0.rem) = fluenceRGBWeightsWithMIS0.y + fluenceRGBWeightsWithMIS1.y;
        get(_fluenceRGBWeightsWithMIS[tmpIdx0.quot].z, tmpIdx0.rem) = fluenceRGBWeightsWithMIS0.z + fluenceRGBWeightsWithMIS1.z;

        const Vector3 fluenceRGBWeights0(get(_fluenceRGBWeights[tmpIdx0.quot].x, tmpIdx0.rem), get(_fluenceRGBWeights[tmpIdx0.quot].y, tmpIdx0.rem),
                                         get(_fluenceRGBWeights[tmpIdx0.quot].z, tmpIdx0.rem));
        const Vector3 fluenceRGBWeights1(get(_fluenceRGBWeights[tmpIdx1.quot].x, tmpIdx1.rem), get(_fluenceRGBWeights[tmpIdx1.quot].y, tmpIdx1.rem),
                                         get(_fluenceRGBWeights[tmpIdx1.quot].z, tmpIdx1.rem));
        get(_fluenceRGBWeights[tmpIdx0.quot].x, tmpIdx0.rem) = fluenceRGBWeights0.x + fluenceRGBWeights1.x;
        get(_fluenceRGBWeights[tmpIdx0.quot].y, tmpIdx0.rem) = fluenceRGBWeights0.y + fluenceRGBWeights1.y;
        get(_fluenceRGBWeights[tmpIdx0.quot].z, tmpIdx0.rem) = fluenceRGBWeights0.z + fluenceRGBWeights1.z;
#endif
        // std::cout << "mergeComponents: weight: " << weight << "\tkappa: " << kappa << "\tmeanDirection: " << meanDirectionX << "\t" << meanDirectionY << "\t" << meanDirectionZ
        // << std::endl;
        swapComponents(idx1, _numComponents - 1);
        clearComponent(_numComponents - 1);
        _numComponents -= 1;
    }
}

template <class Kernel, int maxComponents, bool UseParallaxCompensation>
KERNEL_FUNCTION void ParallaxAwareVonMisesFisherMixture<Kernel, maxComponents, UseParallaxCompensation>::swapComponents(const size_t &idx0, const size_t &idx1)
{
    const div_t tmpIdx0 = div_(idx0, VectorSize);
    const div_t tmpIdx1 = div_(idx1, VectorSize);

    if (idx0 != idx1)
    {
        swap_(get(_weights[tmpIdx0.quot], tmpIdx0.rem), get(_weights[tmpIdx1.quot], tmpIdx1.rem));
        swap_(get(_kappas[tmpIdx0.quot], tmpIdx0.rem), get(_kappas[tmpIdx1.quot], tmpIdx1.rem));
        swap_(get(_eMinus2Kappa[tmpIdx0.quot], tmpIdx0.rem), get(_eMinus2Kappa[tmpIdx1.quot], tmpIdx1.rem));
        swap_(get(_meanCosines[tmpIdx0.quot], tmpIdx0.rem), get(_meanCosines[tmpIdx1.quot], tmpIdx1.rem));
        swap_(get(_normalizations[tmpIdx0.quot], tmpIdx0.rem), get(_normalizations[tmpIdx1.quot], tmpIdx1.rem));

        swap_(get(_meanDirections[tmpIdx0.quot].x, tmpIdx0.rem), get(_meanDirections[tmpIdx1.quot].x, tmpIdx1.rem));
        swap_(get(_meanDirections[tmpIdx0.quot].y, tmpIdx0.rem), get(_meanDirections[tmpIdx1.quot].y, tmpIdx1.rem));
        swap_(get(_meanDirections[tmpIdx0.quot].z, tmpIdx0.rem), get(_meanDirections[tmpIdx1.quot].z, tmpIdx1.rem));

        swap_(get(_distances[tmpIdx0.quot], tmpIdx0.rem), get(_distances[tmpIdx1.quot], tmpIdx1.rem));
#ifdef OPENPGL_RADIANCE_CACHES
        swap_(get(_fluenceRGBWeightsWithMIS[tmpIdx0.quot].x, tmpIdx0.rem), get(_fluenceRGBWeightsWithMIS[tmpIdx1.quot].x, tmpIdx1.rem));
        swap_(get(_fluenceRGBWeightsWithMIS[tmpIdx0.quot].y, tmpIdx0.rem), get(_fluenceRGBWeightsWithMIS[tmpIdx1.quot].y, tmpIdx1.rem));
        swap_(get(_fluenceRGBWeightsWithMIS[tmpIdx0.quot].z, tmpIdx0.rem), get(_fluenceRGBWeightsWithMIS[tmpIdx1.quot].z, tmpIdx1.rem));

        swap_(get(_fluenceRGBWeights[tmpIdx0.quot].x, tmpIdx0.rem), get(_fluenceRGBWeights[tmpIdx1.quot].x, tmpIdx1.rem));
        swap_(get(_fluenceRGBWeights[tmpIdx0.quot].y, tmpIdx0.rem), get(_fluenceRGBWeights[tmpIdx1.quot].y, tmpIdx1.rem));
        swap_(get(_fluenceRGBWeights[tmpIdx0.quot].z, tmpIdx0.rem), get(_fluenceRGBWeights[tmpIdx1.quot].z, tmpIdx1.rem));
#endif
    }
}

template <class Kernel, int maxComponents, bool UseParallaxCompensation>
KERNEL_FUNCTION void ParallaxAwareVonMisesFisherMixture<Kernel, maxComponents, UseParallaxCompensation>::clearComponent(const size_t &idx)
{
    const div_t tmpIdx = div_(idx, VectorSize);

    get(_weights[tmpIdx.quot], tmpIdx.rem) = 0.f;
    get(_kappas[tmpIdx.quot], tmpIdx.rem) = 0.f;
    get(_eMinus2Kappa[tmpIdx.quot], tmpIdx.rem) = 1.f;
    get(_meanCosines[tmpIdx.quot], tmpIdx.rem) = 0.f;
    get(_normalizations[tmpIdx.quot], tmpIdx.rem) = ONE_OVER_FOUR_PI;

    get(_meanDirections[tmpIdx.quot].x, tmpIdx.rem) = 0.f;
    get(_meanDirections[tmpIdx.quot].y, tmpIdx.rem) = 0.f;
    get(_meanDirections[tmpIdx.quot].z, tmpIdx.rem) = 1.f;

    get(_distances[tmpIdx.quot], tmpIdx.rem) = 0.0f;

#ifdef OPENPGL_RADIANCE_CACHES
    get(_fluenceRGBWeightsWithMIS[tmpIdx.quot].x, tmpIdx.rem) = 0.f;
    get(_fluenceRGBWeightsWithMIS[tmpIdx.quot].y, tmpIdx.rem) = 0.f;
    get(_fluenceRGBWeightsWithMIS[tmpIdx.quot].z, tmpIdx.rem) = 0.f;

    get(_fluenceRGBWeights[tmpIdx.quot].x, tmpIdx.rem) = 0.f;
    get(_fluenceRGBWeights[tmpIdx.quot].y, tmpIdx.rem) = 0.f;
    get(_fluenceRGBWeights[tmpIdx.quot].z, tmpIdx.rem) = 0.f;
#endif
}

template <class Kernel, int maxComponents, bool UseParallaxCompensation>
KERNEL_FUNCTION void ParallaxAwareVonMisesFisherMixture<Kernel, maxComponents, UseParallaxCompensation>::serialize(std::ostream &stream) const
{
    serializeFloatVectors<NumVectors>(stream, _weights);
    serializeFloatVectors<NumVectors>(stream, _kappas);
    serializeVec3Vectors<NumVectors>(stream, _meanDirections);
    serializeFloatVectors<NumVectors>(stream, _normalizations);
    serializeFloatVectors<NumVectors>(stream, _eMinus2Kappa);
    serializeFloatVectors<NumVectors>(stream, _meanCosines);
    serializeFloatVectors<NumVectors>(stream, _distances);
#ifdef OPENPGL_RADIANCE_CACHES
    serializeVec3Vectors<NumVectors>(stream, _fluenceRGBWeightsWithMIS);
    serializeVec3Vectors<NumVectors>(stream, _fluenceRGBWeights);
#endif
    stream.write(reinterpret_cast<const char *>(&_numComponents), sizeof(_numComponents));
    stream.write(reinterpret_cast<const char *>(&_pivotPosition), sizeof(Point3));

#ifdef OPENPGL_RADIANCE_CACHES
    // stream.write(reinterpret_cast<const char*>(&_fluence), sizeof(float));
    stream.write(reinterpret_cast<const char *>(&_fluenceRGB), sizeof(Vector3));
    stream.write(reinterpret_cast<const char *>(&_fluenceRGBWithMIS), sizeof(Vector3));
    stream.write(reinterpret_cast<const char *>(&_numFluenceSamples), sizeof(float));
#endif
}

template <class Kernel, int maxComponents, bool UseParallaxCompensation>
KERNEL_FUNCTION void ParallaxAwareVonMisesFisherMixture<Kernel, maxComponents, UseParallaxCompensation>::deserialize(std::istream &stream)
{
    deserializeFloatVectors<NumVectors>(stream, _weights);
    deserializeFloatVectors<NumVectors>(stream, _kappas);
    deserializeVec3Vectors<NumVectors>(stream, _meanDirections);
    deserializeFloatVectors<NumVectors>(stream, _normalizations);
    deserializeFloatVectors<NumVectors>(stream, _eMinus2Kappa);
    deserializeFloatVectors<NumVectors>(stream, _meanCosines);
    deserializeFloatVectors<NumVectors>(stream, _distances);
#ifdef OPENPGL_RADIANCE_CACHES
    deserializeVec3Vectors<NumVectors>(stream, _fluenceRGBWeightsWithMIS);
    deserializeVec3Vectors<NumVectors>(stream, _fluenceRGBWeights);
#endif
    stream.read(reinterpret_cast<char *>(&_numComponents), sizeof(_numComponents));
    stream.read(reinterpret_cast<char *>(&_pivotPosition), sizeof(Point3));

#ifdef OPENPGL_RADIANCE_CACHES
    // stream.read(reinterpret_cast<char*>(&_fluence), sizeof(float));
    stream.read(reinterpret_cast<char *>(&_fluenceRGB), sizeof(Vector3));
    stream.read(reinterpret_cast<char *>(&_fluenceRGBWithMIS), sizeof(Vector3));
    stream.read(reinterpret_cast<char *>(&_numFluenceSamples), sizeof(float));
#endif
}

template <class Kernel, int maxComponents, bool UseParallaxCompensation>
KERNEL_FUNCTION bool ParallaxAwareVonMisesFisherMixture<Kernel, maxComponents, UseParallaxCompensation>::isValid() const
{
    bool valid = true;
    float sumWeights = 0.0f;

    for (size_t k = 0; k < _numComponents; k++)
    {
        const div_t tmpK = div_(k, VectorSize);
        sumWeights += get(_weights[tmpK.quot], tmpK.rem);

        valid = valid && embree::isvalid(get(_weights[tmpK.quot], tmpK.rem));
        valid = valid && get(_weights[tmpK.quot], tmpK.rem) >= 0.0f;
        valid = valid && get(_weights[tmpK.quot], tmpK.rem) <= 1.0f + 1e-6f;
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(get(_kappas[tmpK.quot], tmpK.rem));
        valid = valid && get(_kappas[tmpK.quot], tmpK.rem) >= 0.0f;
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(get(_meanCosines[tmpK.quot], tmpK.rem));
        valid = valid && get(_meanCosines[tmpK.quot], tmpK.rem) >= 0.0f;
        valid = valid && get(_meanCosines[tmpK.quot], tmpK.rem) <= 1.0f;
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(get(_meanDirections[tmpK.quot].x, tmpK.rem));
        valid = valid && get(_meanDirections[tmpK.quot].x, tmpK.rem) >= -1.0f;
        valid = valid && get(_meanDirections[tmpK.quot].x, tmpK.rem) <= 1.0f;
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(get(_meanDirections[tmpK.quot].y, tmpK.rem));
        valid = valid && get(_meanDirections[tmpK.quot].y, tmpK.rem) >= -1.0f;
        valid = valid && get(_meanDirections[tmpK.quot].y, tmpK.rem) <= 1.0f;
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(get(_meanDirections[tmpK.quot].z, tmpK.rem));
        valid = valid && get(_meanDirections[tmpK.quot].z, tmpK.rem) >= -1.0f;
        valid = valid && get(_meanDirections[tmpK.quot].z, tmpK.rem) <= 1.0f;
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(get(_normalizations[tmpK.quot], tmpK.rem));
        valid = valid && get(_normalizations[tmpK.quot], tmpK.rem) >= 0.0f;
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(get(_eMinus2Kappa[tmpK.quot], tmpK.rem));
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(get(_distances[tmpK.quot], tmpK.rem));
        valid = valid && get(_distances[tmpK.quot], tmpK.rem) >= 0.0f;
        OPENPGL_ASSERT(valid);
    }

    // check unused componets
    for (int k = _numComponents; k < MaxComponents; k++)
    {
        const div_t tmpK = div_(k, VectorSize);
        valid = valid && embree::isvalid(get(_weights[tmpK.quot], tmpK.rem));
        valid = valid && get(_weights[tmpK.quot], tmpK.rem) == 0.0f;
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(get(_kappas[tmpK.quot], tmpK.rem));
        valid = valid && get(_kappas[tmpK.quot], tmpK.rem) == 0.0f;
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(get(_meanDirections[tmpK.quot].x, tmpK.rem));
        valid = valid && get(_meanDirections[tmpK.quot].x, tmpK.rem) == 0.0f;
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(get(_meanDirections[tmpK.quot].y, tmpK.rem));
        valid = valid && get(_meanDirections[tmpK.quot].y, tmpK.rem) == 0.0f;
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(get(_meanDirections[tmpK.quot].z, tmpK.rem));
        valid = valid && get(_meanDirections[tmpK.quot].z, tmpK.rem) == 1.0f;
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(get(_meanCosines[tmpK.quot], tmpK.rem));
        valid = valid && get(_meanCosines[tmpK.quot], tmpK.rem) == 0.0f;
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(get(_normalizations[tmpK.quot], tmpK.rem));
        valid = valid && std::fabs(get(_normalizations[tmpK.quot], tmpK.rem) - ONE_OVER_FOUR_PI) < 1e-6f;
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(get(_eMinus2Kappa[tmpK.quot], tmpK.rem));
        valid = valid && get(_eMinus2Kappa[tmpK.quot], tmpK.rem) == 1.0f;
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(get(_distances[tmpK.quot], tmpK.rem));
        valid = valid && get(_distances[tmpK.quot], tmpK.rem) == 0.0f;
        OPENPGL_ASSERT(valid);
    }

    return valid;
}

template <class Kernel, int maxComponents, bool UseParallaxCompensation>
KERNEL_FUNCTION void ParallaxAwareVonMisesFisherMixture<Kernel, maxComponents, UseParallaxCompensation>::setComponentWeight(const size_t idx, const float &weight)
{
    const div_t tmpIdx = div_(idx, VectorSize);

    get(_weights[tmpIdx.quot], tmpIdx.rem) = weight;
}

template <class Kernel, int maxComponents, bool UseParallaxCompensation>
KERNEL_FUNCTION void ParallaxAwareVonMisesFisherMixture<Kernel, maxComponents, UseParallaxCompensation>::setComponentKappa(const size_t idx, const float &kappa)
{
    const div_t tmpIdx = div_(idx, VectorSize);

    get(_kappas[tmpIdx.quot], tmpIdx.rem) = kappa;
    _calculateNormalization();
    _calculateMeanCosines();
}

template <class Kernel, int maxComponents, bool UseParallaxCompensation>
KERNEL_FUNCTION void ParallaxAwareVonMisesFisherMixture<Kernel, maxComponents, UseParallaxCompensation>::setComponentMeanDirection(const size_t idx, const Vector3 &meanDirection)
{
    const div_t tmpIdx = div_(idx, VectorSize);

    get(_meanDirections[tmpIdx.quot].x, tmpIdx.rem) = meanDirection.x;
    get(_meanDirections[tmpIdx.quot].y, tmpIdx.rem) = meanDirection.y;
    get(_meanDirections[tmpIdx.quot].z, tmpIdx.rem) = meanDirection.z;
}

template <class Kernel, int maxComponents, bool UseParallaxCompensation>
KERNEL_FUNCTION Vector3 ParallaxAwareVonMisesFisherMixture<Kernel, maxComponents, UseParallaxCompensation>::getComponentMeanDirection(const size_t &idx) const
{
    const div_t tmpIdx = div_(idx, VectorSize);
    return Vector3(get(_meanDirections[tmpIdx.quot].x, tmpIdx.rem), get(_meanDirections[tmpIdx.quot].y, tmpIdx.rem), get(_meanDirections[tmpIdx.quot].z, tmpIdx.rem));
}

template <class Kernel, int maxComponents, bool UseParallaxCompensation>
KERNEL_FUNCTION float ParallaxAwareVonMisesFisherMixture<Kernel, maxComponents, UseParallaxCompensation>::getComponentWeight(const size_t &idx) const
{
    const div_t tmpIdx = div_(idx, VectorSize);
    return get(_weights[tmpIdx.quot], tmpIdx.rem);
}

template <class Kernel, int maxComponents, bool UseParallaxCompensation>
KERNEL_FUNCTION float ParallaxAwareVonMisesFisherMixture<Kernel, maxComponents, UseParallaxCompensation>::getComponentKappa(const size_t &idx) const
{
    const div_t tmpIdx = div_(idx, VectorSize);
    return get(_kappas[tmpIdx.quot], tmpIdx.rem);
}

template <class Kernel, int maxComponents, bool UseParallaxCompensation>
KERNEL_FUNCTION void ParallaxAwareVonMisesFisherMixture<Kernel, maxComponents, UseParallaxCompensation>::setComponentDistance(const size_t &idx, const float &distance)
{
    const div_t tmpIdx = div_(idx, VectorSize);
    get(_distances[tmpIdx.quot], tmpIdx.rem) = distance;
}

template <class Kernel, int maxComponents, bool UseParallaxCompensation>
KERNEL_FUNCTION float ParallaxAwareVonMisesFisherMixture<Kernel, maxComponents, UseParallaxCompensation>::getComponentDistance(const size_t &idx) const
{
    const div_t tmpIdx = div_(idx, VectorSize);
    return get(_distances[tmpIdx.quot], tmpIdx.rem);
}

template <class Kernel, int maxComponents, bool UseParallaxCompensation>
KERNEL_FUNCTION void ParallaxAwareVonMisesFisherMixture<Kernel, maxComponents, UseParallaxCompensation>::convole(const float &_meanCosine)
{
    const int cnt = (_numComponents + VectorSize - 1) / VectorSize;
    const vfloat meanCosine = _meanCosine;

    for (int k = 0; k < cnt; k++)
    {
        _meanCosines[k] *= meanCosine;
        _kappas[k] = MeanCosineToKappa<vfloat>(meanCosine);
    }
    _calculateNormalization();
}

template <class Kernel, int maxComponents, bool UseParallaxCompensation>
KERNEL_FUNCTION float ParallaxAwareVonMisesFisherMixture<Kernel, maxComponents, UseParallaxCompensation>::product(const float &_weight, const Vector3 &_meanDirection, const float &_kappa)
{
    float _normalization = ONE_OVER_FOUR_PI;
    // float _eMinus2Kappa = embree::fastapprox::exp< float >(-2.0f * _kappa);
    //  TODO: use faster exp
    float _eMinus2Kappa = std::exp(-2.0f * _kappa);

    if (_kappa > 0.0f)
    {
        _normalization = _kappa / (2.0f * M_PI_F * (1.0f - _eMinus2Kappa));
    }
    return this->product(_weight, _meanDirection, _kappa, _normalization);
}

template <class Kernel, int maxComponents, bool UseParallaxCompensation>
KERNEL_FUNCTION float ParallaxAwareVonMisesFisherMixture<Kernel, maxComponents, UseParallaxCompensation>::product(const float &_weight, const Vector3 &_meanDirection, const float &_kappa,
                                                                                                   const float &_normalization)
{
    const vfloat twoPi(2.0f * M_PI_F);
    const vfloat ones(1.0f);
    const vfloat minusTwos(-2.0f);
    const vfloat zeros(0.0f);
    const vfloat zeroKappaNorm(ONE_OVER_FOUR_PI);

    const int cnt = (_numComponents + VectorSize - 1) / VectorSize;
    const int rem = _numComponents % VectorSize;

    const vfloat weight = _weight;
    const vfloat kappa = _kappa;
    const vfloat normalization = _normalization;

    const embree::Vec3<vfloat> meanDirection = _meanDirection;

    vfloat productIntegralVec(0.f);

    for (int k = 0; k < cnt; k++)
    {
        embree::Vec3<vfloat> newMeanDirection = _kappas[k] * _meanDirections[k] + kappa * meanDirection;
        vfloat newKappa = embree::sqrt(dot(newMeanDirection, newMeanDirection));
        auto checkNewKappa = (newKappa > 1e-3f);
        newKappa = select(checkNewKappa, newKappa, zeros);

        // TODO: update meanCosine
        newMeanDirection.x = select(checkNewKappa, newMeanDirection.x / newKappa, _meanDirections[k].x);
        newMeanDirection.y = select(checkNewKappa, newMeanDirection.y / newKappa, _meanDirections[k].y);
        newMeanDirection.z = select(checkNewKappa, newMeanDirection.z / newKappa, _meanDirections[k].z);

        vfloat newEMinus2Kappa = embree::fastapprox::exp(minusTwos * newKappa);
        vfloat newNormalization = newKappa / (twoPi * (ones - newEMinus2Kappa));
        newNormalization = select(checkNewKappa, newNormalization, zeroKappaNorm);

        vfloat scale = (_normalizations[k] * normalization) / newNormalization;

        vfloat cosTheta0 = dot(_meanDirections[k], newMeanDirection);
        vfloat cosTheta1 = dot(meanDirection, newMeanDirection);

        // std::cout << "cosTheta0: " << cosTheta0 <<"\tcosTheta1: " << cosTheta1 << std::endl;
        // std::cout << "_kappas[k]: " << _kappas[k] <<"\tkappa: " << kappa << std::endl;
        // std::cout << "tmp: " <<  _kappas[k] * (cosTheta0 - ones) + kappa * (cosTheta1 - ones) << std::endl;
        vfloat eval = embree::fastapprox::exp(_kappas[k] * (cosTheta0 - ones) + kappa * (cosTheta1 - ones));
        // std::cout << "scale: " << scale <<"\teval: " << eval << std::endl;
        scale *= eval;
        scale *= _weights[k] * weight;

        _weights[k] = scale;
        _kappas[k] = newKappa;
        _meanDirections[k] = newMeanDirection;
        _normalizations[k] = newNormalization;
        _eMinus2Kappa[k] = newEMinus2Kappa;

        productIntegralVec += _weights[k];
    }

    float productIntegral = embree::reduce_add(productIntegralVec);
    for (int k = 0; k < cnt; k++)
    {
        _weights[k] /= productIntegral;
    }

    if (rem > 0)
    {
        for (size_t i = rem; i < VectorSize; i++)
        {
            _meanDirections[cnt - 1].x[i] = 0.0f;
            _meanDirections[cnt - 1].y[i] = 0.0f;
            _meanDirections[cnt - 1].z[i] = 1.0f;

            _meanCosines[cnt - 1][i] = 0.0f;
            _kappas[cnt - 1][i] = 0.0f;
            _normalizations[cnt - 1][i] = ONE_OVER_FOUR_PI;
            _eMinus2Kappa[cnt - 1][i] = 1.0f;
            _distances[cnt - 1][i] = 0.0f;
        }
    }

    return productIntegral;
    //_normalizeWeights();
}

template <class Kernel, int maxComponents, bool UseParallaxCompensation>
KERNEL_FUNCTION float ParallaxAwareVonMisesFisherMixture<Kernel, maxComponents, UseParallaxCompensation>::pdf(Vector3 direction) const
{
    const int cnt = (_numComponents + VectorSize - 1) / VectorSize;

    vfloat pdf = {0.0f};
    embree::Vec3<vfloat> vec3Direction(direction[0], direction[1], direction[2]);

    const vfloat ones(1.0f);
    const vfloat zeros(0.0f);

    for (int k = 0; k < cnt; k++)
    {
        const vfloat cosTheta = dot(vec3Direction, _meanDirections[k]);
        const vfloat cosThetaMinusOne = embree::min(cosTheta - ones, zeros);
        const vfloat eval = _normalizations[k] * embree::fastapprox::exp<vfloat>(_kappas[k] * cosThetaMinusOne);
        pdf += _weights[k] * eval;
    }

    #if OPENPGL_VEC_SIZE == 1
        return pdf;
    #else
        return reduce_add(pdf);
    #endif
}

template <class Kernel, int maxComponents, bool UseParallaxCompensation>
KERNEL_FUNCTION bool ParallaxAwareVonMisesFisherMixture<Kernel, maxComponents, UseParallaxCompensation>::softAssignment(
    Vector3 direction, typename ParallaxAwareVonMisesFisherMixture<Kernel, maxComponents, UseParallaxCompensation>::SoftAssignment &softAssign) const
{
    const int cnt = (_numComponents + VectorSize - 1) / VectorSize;

    vfloat pdf = {0.0f};
    embree::Vec3<vfloat> vec3Direction(direction[0], direction[1], direction[2]);

    const vfloat ones(1.0f);
    const vfloat zeros(0.0f);

    for (int k = 0; k < cnt; k++)
    {
        const vfloat cosTheta = dot(vec3Direction, _meanDirections[k]);
        const vfloat cosThetaMinusOne = embree::min(cosTheta - ones, zeros);
        const vfloat eval = _normalizations[k] * embree::fastapprox::exp<vfloat>(_kappas[k] * cosThetaMinusOne);
        softAssign.assignments[k] = _weights[k] * eval;
        OPENPGL_ASSERT(embree::isvalid(softAssign.assignments[k]));
        pdf += softAssign.assignments[k];
    }
    OPENPGL_ASSERT(embree::isvalid(pdf));
    softAssign.pdf = embree::reduce_add(pdf);
    softAssign.size = _numComponents;

    if (softAssign.pdf <= 1e-16f)
    {
        return false;
    }

    vfloat inv_pdf = embree::rcp(softAssign.pdf);
    OPENPGL_ASSERT(embree::isvalid(inv_pdf));
    for (int k = 0; k < cnt; k++)
    {
        softAssign.assignments[k] *= inv_pdf;
    }

    return true;
}

#if defined(__CUDACC__)
template <class Kernel, int maxComponents, bool UseParallaxCompensation>
KERNEL_FUNCTION void ParallaxAwareVonMisesFisherMixture<Kernel, maxComponents, UseParallaxCompensation>::selectComponent(uint32_t &selectedVector, uint32_t &selectedComponent,
                                                                                                          Vector2 &_sample) const
{
    float searched = _sample[1];
    float sumWeights = 0.0f;
    float cdf = 0.0f;
    // int k0 = 0;
    // int k1 = 0;
    //  find comp

    const div_t tmp = div_(_numComponents - 1, VectorSize);

    while (true)
    {
        cdf = reduce_add(_weights[selectedVector]);
        if (sumWeights + cdf >= searched || selectedVector + 1 >= (tmp.quot + 1))
        {
            break;
        }
        else
        {
            sumWeights += cdf;
            selectedVector++;
        }
    }

    selectedComponent = 0;

    _sample[1] = std::min(1 - FLT_EPSILON, (searched - sumWeights) / cdf);
}
#else
template <class Kernel, int maxComponents, bool UseParallaxCompensation>
KERNEL_FUNCTION void ParallaxAwareVonMisesFisherMixture<Kernel, maxComponents, UseParallaxCompensation>::selectComponent(uint32_t &selectedVector, uint32_t &selectedComponent,
                                                                                                          Vector2 &_sample) const
{
    float searched = _sample[1];
    float sumWeights = 0.0f;
    float cdf = 0.0f;
    // int k0 = 0;
    // int k1 = 0;
    //  find comp

    const div_t tmp = div_(_numComponents - 1, VectorSize);

    while (true)
    {
        cdf = reduce_add(_weights[selectedVector]);
        if (sumWeights + cdf >= searched || selectedVector + 1 >= (tmp.quot + 1))
        {
            break;
        }
        else
        {
            sumWeights += cdf;
            selectedVector++;
        }
    }

    int maxSelectedComponent = selectedVector == tmp.quot ? tmp.rem + 1 : VectorSize;

    while (true)
    {
        cdf = _weights[selectedVector][selectedComponent];
        if (sumWeights + cdf >= searched || selectedComponent + 1 >= maxSelectedComponent)
        {
            break;
        }
        else
        {
            sumWeights += cdf;
            selectedComponent++;
        }
    }

    _sample[1] = std::min(1 - FLT_EPSILON, (searched - sumWeights) / cdf);
}

#ifdef USE_SIMD_CDF_SAMPLING
template <class Kernel, int maxComponents, bool UseParallaxCompensation>
KERNEL_FUNCTION inline void ParallaxAwareVonMisesFisherMixture<Kernel, maxComponents, UseParallaxCompensation>::selectComponentSIMD(uint32_t &selectedVector, uint32_t &selectedComponent,
                                                                                                                     Vector2 &_sample) const
{
    vfloat cdfs[NumVectors];

    const float searched = _sample[1];
#ifdef VALIDATE_SELECT_COMPONENT_SIMD
    const float searchedScalar = _sample[1];
#endif
    float sumWeights = 0.0f;
    float cdf = 0.0f;

    const div_t tmp = div_(_numComponents - 1, VectorSize);

    selectedVector = 0;
    selectedComponent = 0;
#if (1)
    for (selectedVector = 0; selectedVector < NumVectors; selectedVector++)
    {
        cdfs[selectedVector] = vinclusive_prefix_sum(_weights[selectedVector]);
        cdf = cdfs[selectedVector][VectorSize - 1];
        if (sumWeights + cdf >= searched || selectedVector >= tmp.quot)
        {
            break;
        }
        else
        {
            sumWeights += cdf;
        }
    }
#else
    while (true)
    {
        cdfs[selectedVector] = vinclusive_prefix_sum(_weights[selectedVector]);
        cdf = cdfs[selectedVector][VectorSize - 1];
        if (sumWeights + cdf >= searched || selectedVector + 1 >= (tmp.quot + 1))
        {
            break;
        }
        else
        {
            sumWeights += cdf;
            selectedVector++;
        }
    }
#endif
#ifdef VALIDATE_SELECT_COMPONENT_SIMD
    float sumWeightsCheckPoint = sumWeights;
#endif

    const uint32_t maxSelectedComponent = selectedVector == tmp.quot ? tmp.rem + 1 : VectorSize;

    cdfs[selectedVector] += sumWeights;
    selectedComponent = embree::select_min(cdfs[selectedVector] >= searched, cdfs[selectedVector]);

    selectedComponent = std::min(selectedComponent, maxSelectedComponent - 1);
    sumWeights = selectedComponent > 0 ? cdfs[selectedVector][selectedComponent - 1] : sumWeights;

    cdf = _weights[selectedVector][selectedComponent];

    cdf = std::max(FLT_EPSILON, cdf);
    _sample[1] = std::min(1 - FLT_EPSILON, (searched - sumWeights) / cdf);

#ifdef VALIDATE_SELECT_COMPONENT_SIMD
    Vector2 _sampleScalar = _sample;
    uint32_t selectedVectorScalar = 0;
    uint32_t selectedComponentScalar = 0;

    float sumWeightsScalar = 0.0f;
    float cdfScalar = 0.0f;
    const div_t tmpScalar = div_(_numComponents - 1, VectorSize);

    while (true)
    {
        cdfScalar = reduce_add(_weights[selectedVectorScalar]);
        if (sumWeightsScalar + cdfScalar >= searchedScalar || selectedVectorScalar + 1 >= (tmpScalar.quot + 1))
        {
            break;
        }
        else
        {
            sumWeightsScalar += cdfScalar;
            selectedVectorScalar++;
        }
    }

    OPENPGL_ASSERT(searched == searchedScalar);
    OPENPGL_ASSERT(sumWeightsCheckPoint == sumWeightsScalar);
    OPENPGL_ASSERT(selectedVector == selectedVectorScalar);

    int maxSelectedComponentScalar = selectedVectorScalar == tmpScalar.quot ? tmpScalar.rem + 1 : VectorSize;

    while (true)
    {
        cdfScalar = _weights[selectedVectorScalar][selectedComponentScalar];
        if (sumWeightsScalar + cdfScalar >= searchedScalar || selectedComponentScalar + 1 >= maxSelectedComponentScalar)
        {
            break;
        }
        else
        {
            sumWeightsScalar += cdfScalar;
            selectedComponentScalar++;
        }
    }

    // OPENPGL_ASSERT(selectedComponent == selectedComponentScalar);
    // OPENPGL_ASSERT(cdf == cdfScalar);
    // OPENPGL_ASSERT(sumWeights == sumWeightsScalar);
    _sampleScalar[1] = std::min(1 - FLT_EPSILON, (searchedScalar - sumWeightsScalar) / cdfScalar);
    // OPENPGL_ASSERT(_sample[1] == _sampleScalar[1]);
#endif
}
#endif
#endif

template <class Kernel, int maxComponents, bool UseParallaxCompensation>
KERNEL_FUNCTION Vector3 ParallaxAwareVonMisesFisherMixture<Kernel, maxComponents, UseParallaxCompensation>::sample(const Vector2 sample) const
{
    uint32_t selectedVector{0};
    uint32_t selectedComponent{0};
    // First, identify component we want to sample

    Vector2 _sample = sample;
#ifdef USE_SIMD_CDF_SAMPLING
    selectComponentSIMD(selectedVector, selectedComponent, _sample);
#else
    selectComponent(selectedVector, selectedComponent, _sample);
#endif

    embree::Vec3<float> sampledDirection(0.f, 0.f, 1.f);

    // Second, sample selected component
    const float sKappa = _kappas[selectedVector][selectedComponent];
    const float sEMinus2Kappa = _eMinus2Kappa[selectedVector][selectedComponent];
    const embree::Vec3<float> meanDirection(_meanDirections[selectedVector].x[selectedComponent], _meanDirections[selectedVector].y[selectedComponent],
                                            _meanDirections[selectedVector].z[selectedComponent]);

    if (sKappa == 0.0f)
    {
        sampledDirection = squareToUniformSphere(_sample);
    }
    else
    {
        float cosTheta = 1.f + (embree::fastapprox::log<float>(1.0f + ((sEMinus2Kappa - 1.f) * _sample[0]))) / sKappa;
        // float cosTheta = 1.f + (std::log1p((sEMinus2Kappa-1.f) * _sample[0])) / sKappa;

        // safeguard for numerical imprecisions (if sample[0] is 0.999999999)
        cosTheta = std::min(1.0f, std::max(cosTheta, -1.f));

        const float sinTheta = std::sqrt(1.f - cosTheta * cosTheta);

        const float phi = 2.f * M_PI_F * _sample[1];

        float sinPhi, cosPhi;
        sincosf(phi, &sinPhi, &cosPhi);
        sampledDirection = openpgl::sphericalDirection(cosTheta, sinTheta, cosPhi, sinPhi);
    }

    return embree::frame(meanDirection) * sampledDirection;
}

template <class Kernel, int maxComponents, bool UseParallaxCompensation>
KERNEL_FUNCTION void ParallaxAwareVonMisesFisherMixture<Kernel, maxComponents, UseParallaxCompensation>::performRelativeParallaxShift(const Vector3 &shiftDirection)
{
    const vfloat ones(1.0f);
    const vfloat zeros(0.0f);

    if (embree::length(shiftDirection) < FLT_EPSILON)
    {
        return;
    }

    const int cnt = (this->_numComponents + VectorSize - 1) / VectorSize;
    // const int rem = this->_numComponents % VectorSize;

    const embree::Vec3<vfloat> shiftDirectionVec(shiftDirection);
    embree::Vec3<vfloat> parallaxCorrectedMeanDirections;
    vfloat lengths;
    for (uint32_t k = 0; k < cnt; k++)
    {
        parallaxCorrectedMeanDirections = this->_meanDirections[k] * _distances[k] + shiftDirectionVec;
        lengths = embree::length(parallaxCorrectedMeanDirections);
        parallaxCorrectedMeanDirections /= lengths;
        this->_meanDirections[k].x =
            select((_distances[k] > 0.0f) & (lengths > FLT_EPSILON) & embree::isfinite<VectorSize>(_distances[k]), parallaxCorrectedMeanDirections.x, this->_meanDirections[k].x);
        this->_meanDirections[k].y =
            select((_distances[k] > 0.0f) & (lengths > FLT_EPSILON) & embree::isfinite<VectorSize>(_distances[k]), parallaxCorrectedMeanDirections.y, this->_meanDirections[k].y);
        this->_meanDirections[k].z =
            select((_distances[k] > 0.0f) & (lengths > FLT_EPSILON) & embree::isfinite<VectorSize>(_distances[k]), parallaxCorrectedMeanDirections.z, this->_meanDirections[k].z);
        _distances[k] = select(_distances[k] > 0.0f, lengths, zeros);
    }

    _pivotPosition -= shiftDirection;
}

template <class Kernel, int maxComponents, bool UseParallaxCompensation>
KERNEL_FUNCTION void ParallaxAwareVonMisesFisherMixture<Kernel, maxComponents, UseParallaxCompensation>::_normalizeWeights()
{
    const int cnt = (_numComponents + VectorSize - 1) / VectorSize;
    vfloat sumWeights = 0.0f;
    for (int k = 0; k < cnt; k++)
    {
        sumWeights += _weights[k];
    }

    vfloat inv_sumWeights = 1.0f / embree::reduce_add(sumWeights);
    for (int k = 0; k < cnt; k++)
    {
        _weights[k] *= inv_sumWeights;
    }
}

template <class Kernel, int maxComponents, bool UseParallaxCompensation>
KERNEL_FUNCTION void ParallaxAwareVonMisesFisherMixture<Kernel, maxComponents, UseParallaxCompensation>::_calculateNormalization()
{
    const vfloat zeroKappaNorm(ONE_OVER_FOUR_PI);

    const int cnt = (_numComponents + VectorSize - 1) / VectorSize;
    const vfloat minusTwo(-2.0f);
    for (int k = 0; k < cnt; k++)
    {
        _eMinus2Kappa[k] = embree::fastapprox::exp<vfloat>(minusTwo * _kappas[k]);
        const vfloat norm = _kappas[k] / (2.0f * M_PI_F * (1.0f - _eMinus2Kappa[k]));
        _normalizations[k] = select(_kappas[k] > 0.f, norm, zeroKappaNorm);
    }
}

template <class Kernel, int maxComponents, bool UseParallaxCompensation>
KERNEL_FUNCTION void ParallaxAwareVonMisesFisherMixture<Kernel, maxComponents, UseParallaxCompensation>::_calculateMeanCosines()
{
    const int cnt = (_numComponents + VectorSize - 1) / VectorSize;
    const vfloat zeros(0.0f);
    const vfloat ones(1.0f);
    for (int k = 0; k < cnt; k++)
    {
        vfloat tanh = ones - 2.0f / (embree::fastapprox::exp(2.0f * _kappas[k]) - ones);
        vfloat meanCosine = ones / tanh - ones / _kappas[k];
        // std::cout << "meanCosine: " << meanCosine << std::endl;
        _meanCosines[k] = select(_kappas[k] > 0.f, meanCosine, zeros);
    }
}

template <class Kernel, int maxComponents, bool UseParallaxCompensation>
KERNEL_FUNCTION bool ParallaxAwareVonMisesFisherMixture<Kernel, maxComponents, UseParallaxCompensation>::operator==(const ParallaxAwareVonMisesFisherMixture &b) const
{
    bool equal = true;
    if (_numComponents != b._numComponents || _pivotPosition != b._pivotPosition)
    {
        equal = false;
    }

    for (int k = 0; k < NumVectors; k++)
    {
        if (embree::any(_weights[k] != b._weights[k]) || embree::any(_kappas[k] != b._kappas[k]) || embree::any(_meanDirections[k].x != b._meanDirections[k].x) ||
            embree::any(_meanDirections[k].y != b._meanDirections[k].y) || embree::any(_meanDirections[k].z != b._meanDirections[k].z) ||
            embree::any(_normalizations[k] != b._normalizations[k]) || embree::any(_eMinus2Kappa[k] != b._eMinus2Kappa[k]) || embree::any(_meanCosines[k] != b._meanCosines[k]) ||
            embree::any(_distances[k] != b._distances[k]))
        {
            equal = false;
        }
    }

    return equal;
}

#ifdef OPENPGL_RADIANCE_CACHES
template <class Kernel, int maxComponents, bool UseParallaxCompensation>
KERNEL_FUNCTION Vector3 ParallaxAwareVonMisesFisherMixture<Kernel, maxComponents, UseParallaxCompensation>::incomingRadiance(const Vector3 &direction, const bool directLightMIS) const
{
    const int cnt = (_numComponents + VectorSize - 1) / VectorSize;

    embree::Vec3<vfloat> incomingRadiance = {0.0f, 0.0f, 0.0f};
    embree::Vec3<vfloat> vec3Direction(direction[0], direction[1], direction[2]);

    const vfloat ones(1.0f);
    const vfloat zeros(0.0f);

    for (int k = 0; k < cnt; k++)
    {
        const vfloat cosTheta = dot(vec3Direction, _meanDirections[k]);
        const vfloat cosThetaMinusOne = embree::min(cosTheta - ones, zeros);
        const vfloat eval = _normalizations[k] * embree::fastapprox::exp<vfloat>(_kappas[k] * cosThetaMinusOne);
        incomingRadiance += directLightMIS ? _fluenceRGBWeightsWithMIS[k] * eval : _fluenceRGBWeights[k] * eval;
    }

    return Vector3(reduce_add(incomingRadiance.x), reduce_add(incomingRadiance.y), reduce_add(incomingRadiance.z));
}

template <class Kernel, int maxComponents, bool UseParallaxCompensation>
KERNEL_FUNCTION Vector3 ParallaxAwareVonMisesFisherMixture<Kernel, maxComponents, UseParallaxCompensation>::irradiance(const Vector3 &normal, const bool directLightMIS) const
{
    const vfloat cosine_meanCosine(KappaToMeanCosine<float>(2.18853f));  // TODO

    const int cnt = (_numComponents + VectorSize - 1) / VectorSize;

    embree::Vec3<vfloat> irradiance = {0.0f, 0.0f, 0.0f};
    embree::Vec3<vfloat> vec3Normal(normal[0], normal[1], normal[2]);

    const vfloat ones(1.0f);
    const vfloat zeros(0.0f);

    for (int k = 0; k < cnt; k++)
    {
        const vfloat eval = _convolvePDF(k, vec3Normal, cosine_meanCosine);
        irradiance += directLightMIS ? _fluenceRGBWeightsWithMIS[k] * eval : _fluenceRGBWeights[k] * eval;
    }
    return Vector3(reduce_add(irradiance.x), reduce_add(irradiance.y), reduce_add(irradiance.z));
}

template <class Kernel, int maxComponents, bool UseParallaxCompensation>
KERNEL_FUNCTION Vector3 ParallaxAwareVonMisesFisherMixture<Kernel, maxComponents, UseParallaxCompensation>::inscatteredRadiance(const Vector3 &dir, const float meanCosine,
                                                                                                                 const bool directLightMIS) const
{
    const vfloat meanCosineVec(meanCosine);

    const int cnt = (_numComponents + VectorSize - 1) / VectorSize;

    embree::Vec3<vfloat> inscatteredRadiance = {0.0f, 0.0f, 0.0f};
    embree::Vec3<vfloat> vec3Dir(-dir[0], -dir[1], -dir[2]);

    const vfloat ones(1.0f);
    const vfloat zeros(0.0f);

    for (int k = 0; k < cnt; k++)
    {
        const vfloat eval = _convolvePDF(k, vec3Dir, meanCosineVec);
        inscatteredRadiance += directLightMIS ? _fluenceRGBWeightsWithMIS[k] * eval : _fluenceRGBWeights[k] * eval;
    }
    return Vector3(reduce_add(inscatteredRadiance.x), reduce_add(inscatteredRadiance.y), reduce_add(inscatteredRadiance.z));
}

template <class Kernel, int maxComponents, bool UseParallaxCompensation>
KERNEL_FUNCTION Vector3 ParallaxAwareVonMisesFisherMixture<Kernel, maxComponents, UseParallaxCompensation>::fluence(const bool directLightMIS) const
{
    return directLightMIS ? _fluenceRGBWithMIS : _fluenceRGB;
}
#endif

template <class Kernel, int maxComponents, bool UseParallaxCompensation>
KERNEL_FUNCTION vfloat ParallaxAwareVonMisesFisherMixture<Kernel, maxComponents, UseParallaxCompensation>::_convolvePDF(const size_t k,
                                                                                                                          const embree::Vec3<vfloat> &normal,
                                                                                                                          const vfloat &meanCosine1) const
{
    const vfloat ones(1.0f);
    const vfloat zeros(0.0f);
    const vfloat invFourPi(1.0f / (4.0f * M_PI_F));

    const vfloat cosTheta = dot(normal, _meanDirections[k]);

    const vfloat meanCosine0 = _meanCosines[k];

    const vfloat meanCosine = meanCosine0 * meanCosine1;
    OPENPGL_ASSERT(embree::is_finite(meanCosine));

    vfloat kappa = MeanCosineToKappa<vfloat>(meanCosine);
    OPENPGL_ASSERT(embree::is_finite(kappa));
    kappa = select(kappa < OPENPGL_MIN_KAPPA, zeros, kappa);

    const vfloat eMinus2Kappa = embree::fastapprox::exp(-2.0f * kappa);
    vfloat normalization = kappa / (2.0f * M_PI_F * (1.0f - eMinus2Kappa));
    normalization = select(kappa > 0.f, normalization, invFourPi);
    OPENPGL_ASSERT(embree::is_finite(normalization));

    const vfloat cosThetaMinusOne = embree::min(cosTheta - ones, zeros);
    const vfloat eval = embree::fastapprox::exp(kappa * cosThetaMinusOne);
    return normalization * eval;
}

template <typename Type>
SHARED_FUNCTION inline Type KappaToMeanCosine(const Type &kappa)
{
    const Type ones(1.0f);
    const Type zeros(0.0f);

    Type meanCosine = ones / embree::tanh(kappa) - ones / kappa;
    return embree::select(kappa > 0.f, meanCosine, zeros);
}

template <typename Type>
KERNEL_FUNCTION inline Type MeanCosineToKappa(const Type &meanCosine)
{
    const Type ones(1.0f);
    const Type dim(3.0f);
    const Type meanCosine2 = meanCosine * meanCosine;
    return (meanCosine * dim - meanCosine * meanCosine2) / (ones - meanCosine2);
}

}
}  // namespace openpgl
