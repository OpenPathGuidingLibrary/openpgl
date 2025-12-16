#pragma once

#include "../ISurfaceSamplingDistribution.h"
#include "DQT.h"

namespace openpgl
{

template <class TDirectionalQuadtree>
struct DQTSurfaceSamplingDistribution : public ISurfaceSamplingDistribution
{
    DQTSurfaceSamplingDistribution(){};

    virtual ~DQTSurfaceSamplingDistribution() override{};

    virtual void init(const void *distribution, Point3) override
    {
        this->distribution = *(TDirectionalQuadtree *)distribution;
    };

    inline void applyCosineProduct(const Vector3 &) override
    {
        // not supported by quadtree
        return;
    };

    inline bool supportsApplyCosineProduct() const override
    {
        return false;
    }

    inline Vector3 sample(const Point2 sample) const override
    {
        return distribution.sample(sample);
    };

    inline float pdf(const Vector3 dir) const override
    {
        return distribution.pdf(dir);
    };

    inline float samplePdf(const Point2 sample, Vector3 &dir) const override
    {
        return distribution.samplePdf(sample, dir);
    }

    inline float pdfLi(const Vector3 dir) const override
    {
        return distribution.pdf(dir);
    };
#ifdef OPENPGL_RADIANCE_CACHES
    inline Vector3 incomingRadiance(const Vector3 dir, const bool directLightMIS) const override
    {
        return Vector3(0.f, 0.f, 0.f);
    }

    inline Vector3 irradiance(const Vector3 normal, const bool directLightMIS) const override
    {
        return Vector3(0.f, 0.f, 0.f);
    }

    inline Vector3 outgoingRadiance(const Vector3 dir) const override
    {
        return m_region->getOutgoingRadiance(dir);
    }
#endif

    inline bool validate() const override
    {
        return distribution.isValid();
    };

    inline void clear() override {};

    std::string toString() const override
    {
        return "";
    };

    const IRegion *getRegion() const override
    {
        return m_region;
    }

    void setRegion(const IRegion *region) override
    {
        m_region = region;
    }

#ifdef OPENPGL_VSP_GUIDING
    float volumeScatterProbability(Vector3 dir) const override
    {
        return 0.f;
    }
#endif

   private:
    TDirectionalQuadtree distribution;
    const IRegion *m_region{nullptr};
};

}  // namespace openpgl