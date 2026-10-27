// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../IVolumeSamplingDistribution.h"
#include "VMMPhaseFunctions.h"

namespace openpgl
{

template <class TVMMDistribution, bool UseParallaxCompensation>
struct __aligned(TVMMDistribution::VectorSize * 4) VMMVolumeSamplingDistribution : public IVolumeSamplingDistribution
{
    OPENPGL_ALIGNED_STRUCT_(TVMMDistribution::VectorSize * 4)
    VMMVolumeSamplingDistribution() : IVolumeSamplingDistribution() {};
    ~VMMVolumeSamplingDistribution() override = default;

    typedef std::integral_constant<size_t, OPENPGL_VMM_NUM_PHASE_COMP> MaxNumProductDistributions;

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
        m_liDistribution = *(TVMMDistribution *)distribution;

        // m_liDistribution.uniformInit(0.0);

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
    }

    inline void applySingleLobeHenyeyGreensteinProduct(const Vector3 &dir, const float meanCosine) override
    {
        float sumWeights = 0.f;
        const VMMPhaseFunctionRepresentation pfRep = VMMSingleLobeHenyeyGreensteinOracle::getPhaseFunctionRepresentation(meanCosine);

        for (int i = 0; i < pfRep.K; i++)
        {
            this->m_distributions[i] = m_liDistribution;
            const Vector3 outDir = meanCosine * pfRep.meanCosines[i] > 0.f ? dir : -dir;
            /*
            const float absMeanCosine = std::fabs(pfRep.meanCosines[i]);
            float kappa = MeanCosineToKappa< float >(absMeanCosine);
            kappa = kappa<= 1e-4f ? 0.0f: kappa;
            this->m_weights[i] = this->m_distributions[i].product(pfRep.weights[i], outDir, kappa);
            */
            this->m_weights[i] = this->m_distributions[i].product(pfRep.weights[i], outDir, pfRep.kappas[i], pfRep.normalizations[i]);
            sumWeights += this->m_weights[i];
        }

        for (int i = 0; i < pfRep.K; i++)
        {
            this->m_weights[i] /= sumWeights;
        }

        this->m_numDistributions = pfRep.K;
        this->m_productIntegral = sumWeights;
    }

    inline bool supportsApplySingleLobeHenyeyGreensteinProduct() const override
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

    inline float samplePdf(const Point2 sample, Vector3 &dir) const override
    {
        dir = this->sample(sample);
        return pdf(dir);
    }

    inline float pdfLi(const Vector3 dir) const override
    {
        return m_liDistribution.pdf(dir);
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

    inline Vector3 inScatteredRadiance(const Vector3 dir, const float meanCosine, const bool directLightMIS) const override
    {
        const VMMPhaseFunctionRepresentation pfRep = VMMSingleLobeHenyeyGreensteinOracle::getPhaseFunctionRepresentation(meanCosine);

        Vector3 inscatteredRad(0.f);
        for (int i = 0; i < pfRep.K; i++)
        {
            const Vector3 outDir = meanCosine * pfRep.meanCosines[i] > 0.f ? dir : -dir;
            inscatteredRad += pfRep.weights[i] * m_liDistribution.inscatteredRadiance(outDir, pfRep.meanCosines[i], directLightMIS);
        }
        return inscatteredRad;
    }

    inline Vector3 fluence(const bool directLightMIS) const override
    {
        return m_liDistribution.fluence(directLightMIS);
    }
#endif

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

#ifdef OPENPGL_VSP_GUIDING
    float volumeScatterProbability(Vector3 dir) const override
    {
        return m_liDistribution.volumeScatterProbability(dir);
    }
#endif
};

}  // namespace openpgl