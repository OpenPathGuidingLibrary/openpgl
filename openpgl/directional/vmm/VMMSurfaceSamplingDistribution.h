// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../ISurfaceSamplingDistribution.h"

#include <array>

namespace openpgl
{

template<class TVMMDistribution>
struct VMMSurfaceSamplingDistribution: public ISurfaceSamplingDistribution
{
    
    VMMSurfaceSamplingDistribution(const bool useParallaxCompensation):ISurfaceSamplingDistribution(useParallaxCompensation)
    {};
    ~VMMSurfaceSamplingDistribution() = default;
    
    typedef std::integral_constant<size_t, 2> MaxNumProductDistributions;

    /// the region's Li distribution with applied parallax-compensation
    TVMMDistribution m_liDistribution;

    /// product guiding distribution (may be invalid)
    std::array<TVMMDistribution, MaxNumProductDistributions::value> m_distributions;
    /// distribution sampling weights, sum up to 1.0f
    std::array<float, MaxNumProductDistributions::value> m_weights;
    /// when 0 use the non-product distribution instead
    uint32_t m_numDistributions;
    /// guiding cosine/BSDF product integral (= irradiance/flux, for cosine)
    float m_productIntegral;

    inline void init(const void* distribution, Point3 samplePosition) override
    {
        m_liDistribution = *(const TVMMDistribution*)distribution;
        // prespare sampling distribution
        if(m_useParallaxCompensation)
        {
            const Point3 pivotPosition = this->m_liDistribution._pivotPosition;
            this->m_liDistribution.performRelativeParallaxShift(pivotPosition - samplePosition);
        }
        this->m_distributions[0] = m_liDistribution;
        this->m_weights[0] = 1.0f;
        this->m_numDistributions = 1;
        this->m_productIntegral = 1.0f;
    }

    inline void applyCosineProduct(const Vector3& normal) override
    {
        if(this->m_numDistributions > 0)
        {
            this->m_productIntegral = this->m_distributions[0].product(1.0f, normal, 2.18853f);
        }
    }

    inline bool supportsApplyCosineProduct() const
    {
        return true;
    }

    inline Vector3 sample(const Point2 sample) const override
    {
        OPENPGL_ASSERT(m_numDistributions > 0);

        float weight {0.0f};
        uint32_t i=0;
        for (; i<m_numDistributions-1; ++i)
        {
            const float nextWeight = weight+m_weights[i];

            if (nextWeight > sample.x)
                break;

            weight = nextWeight;
        }

        Vector3 dir = m_distributions[i].sample(openpgl::Vector2{(sample.x-weight)/m_weights[i], sample.y});

        return Vector3(dir[0], dir[1], dir[2]);
    }

    inline float pdf(const Vector3 dir) const override
    {
        OPENPGL_ASSERT(m_numDistributions > 0);

        float pdf {0.0f};
        for (uint32_t i=0; i<m_numDistributions; ++i)
            pdf += m_weights[i]*m_distributions[i].pdf(dir);

        return pdf;
    }

    inline float samplePdf(const Point2 sample, Vector3 &dir) const override
    {
        dir = this->sample(sample);
        return pdf(dir);
    }

    inline bool validate() const override
    {
        return m_numDistributions > 0;
    }

    inline void clear() override
    {
        m_numDistributions = 0;
    }

    std::string toString() const override
    {
        std::ostringstream oss;
        oss << "GuidingData [\n";
        for (uint32_t i=0; i<m_numDistributions; ++i)
        {
            oss << '[' << i << "]: " << m_distributions[i].toString() << '\n'
                << "weight: " << m_weights[i] << '\n';
        }
        oss << "product: " << m_productIntegral << '\n'
            << ']';
        return oss.str();
    }

};

}