// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>

#include "../ISurfaceSamplingDistribution.h"

namespace openpgl
{

template <class TVMMDistribution, bool UseParallaxCompensation>
struct __aligned(TVMMDistribution::VectorSize * 4) VMMSurfaceSamplingDistribution : public ISurfaceSamplingDistribution
{
    OPENPGL_ALIGNED_STRUCT_(TVMMDistribution::VectorSize * 4)

    VMMSurfaceSamplingDistribution(){};
    ~VMMSurfaceSamplingDistribution() override = default;

    typedef std::integral_constant<size_t, 2> MaxNumProductDistributions;

    /// the region's Li distribution with applied parallax-compensation
    TVMMDistribution m_liDistribution;

    /// product guiding distribution (may be invalid)
    std::array<TVMMDistribution, MaxNumProductDistributions::value> m_distributions;
    /// distribution sampling weights, sum up to 1.0f
    std::array<float, MaxNumProductDistributions::value> m_weights;
    /// when 0 use the non-product distribution instead
    uint32_t m_numDistributions{0};
    /// guiding cosine/BSDF product integral (= irradiance/flux, for cosine)
    float m_productIntegral{0.f};

    const IRegion *m_region{nullptr};

    inline void init(const void *distribution, Point3 samplePosition) override
    {
        m_liDistribution = *(const TVMMDistribution *)distribution;
        // prespare sampling distribution
        if (UseParallaxCompensation)
        {
            const Point3 pivotPosition = this->m_liDistribution._pivotPosition;
            this->m_liDistribution.performRelativeParallaxShift(pivotPosition - samplePosition);
        }
        this->m_distributions[0] = m_liDistribution;
        this->m_weights[0] = 1.0f;
        this->m_numDistributions = 1;
        this->m_productIntegral = 1.0f;
        // std::cout << this->m_liDistribution.toString() << std::endl;
    }

    inline void applyCosineProduct(const Vector3 &normal) override
    {
        const float cosine_kappa = 2.18853f;
        const float cosine_normalization = 2.18853f / (2.0f * M_PI_F * (1.0f - std::exp(-2.0f * 2.18853f)));
        if (this->m_numDistributions > 0)
        {
            this->m_productIntegral = this->m_distributions[0].product(1.0f, normal, cosine_kappa, cosine_normalization);
        }
    }

    inline bool supportsApplyCosineProduct() const override
    {
        return true;
    }

    inline Vector3 sample(const Point2 sample) const override
    {
        OPENPGL_ASSERT(m_numDistributions > 0);

        float weight{0.0f};
        uint32_t i = 0;
        for (; i < m_numDistributions - 1; ++i)
        {
            const float nextWeight = weight + m_weights[i];

            if (nextWeight > sample.x)
                break;

            weight = nextWeight;
        }

        Vector3 dir = m_distributions[i].sample(openpgl::Vector2{(sample.x - weight) / m_weights[i], sample.y});

        return Vector3(dir[0], dir[1], dir[2]);
    }

    inline float pdf(const Vector3 dir) const override
    {
        OPENPGL_ASSERT(m_numDistributions > 0);

        float pdf{0.0f};
        for (uint32_t i = 0; i < m_numDistributions; ++i)
            pdf += m_weights[i] * m_distributions[i].pdf(dir);

        return pdf;
    }

#ifdef OPENPGL_RADIANCE_CACHES
    inline Vector3 incomingRadiance(const Vector3 dir, const bool directLightMIS) const override
    {
        return m_liDistribution.incomingRadiance(dir, directLightMIS);
    }

    inline Vector3 outgoingRadiance(const Vector3 dir) const override
    {
        return m_region->getOutgoingRadiance(dir);
    }

    inline Vector3 irradiance(const Vector3 normal, const bool directLightMIS) const override
    {
        return m_liDistribution.irradiance(normal, directLightMIS);
    }
#endif

    inline float samplePdf(const Point2 sample, Vector3 &dir) const override
    {
        dir = this->sample(sample);
        return pdf(dir);
    }

    inline float pdfLi(const Vector3 dir) const override
    {
        return m_liDistribution.pdf(dir);
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
        for (uint32_t i = 0; i < m_numDistributions; ++i)
        {
            oss << '[' << i << "]: " << m_distributions[i].toString() << '\n' << "weight: " << m_weights[i] << '\n';
        }
        oss << "product: " << m_productIntegral << '\n' << ']';
        return oss.str();
    }

    inline const IRegion *getRegion() const override
    {
        return m_region;
    }

    inline void setRegion(const IRegion *region) override
    {
        m_region = region;
    }
};

}  // namespace openpgl