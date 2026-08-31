// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../openpgl.h"
#include "Field.h"
#include "Region.h"

namespace openpgl
{
namespace cpp
{

/**
 * @brief The Sampling distribution used for guidiging directional sampling decisions on surfaces.
 *
 * The guided sampling distribution can be proportional to the incoming radiance or its product
 * with components of a BSDF model (e.g., cosine term). The class supports function for sampling and
 * PDF evaluations.
 *
 */
struct SurfaceSamplingDistribution
{
    /**
     * @brief Constructs new instance of a SurfaceSamplingDistribution.
     *
     * Reserves the memory need to store the guiding distribution.
     * Since the type/representation of distribution depends on the guiding field
     * a pointer to the @ref Field has to be provided. After construction
     * the SurfaceSamplingDistribution still need to be initialized using the @ref Init function.
     *
     * @param field Pointer to the -guiding- Field.
     */
    SurfaceSamplingDistribution(const Field *field);

    ~SurfaceSamplingDistribution();

    SurfaceSamplingDistribution(const SurfaceSamplingDistribution &) = delete;

    /**
     * @brief Initializes the guiding distribution for a given position in the scene.
     *
     * This function queries the guiding field for a surface guiding distribution for
     * given position in the scene and initializes the SurfaceSamplingDistribution
     * to this distribution. The resulting distribution is usually proportional to the local
     * incident radiance distribution at the query position. The SurfaceSamplingDistribution
     * can further being improved by applying products with BSDF components (e.g., cosine).
     *
     * @param field The guiding field of the scene.
     * @param pos The position the guiding distribution is queried for.
     * @param sample1D A random number used of a stochastic look-up is used.
     * @return true
     * @return false
     */
    bool Init(const Field *field, const pgl_point3f &pos, float &sample1D);

    /**
     * @brief Clears/resets the internal representation of the guiding distribution.
     *
     */
    void Clear();

    /**
     * @brief Importance samples a new direction based on the guiding distribution.
     *
     * @param sample2D A 2D random variable
     * @return pgl_vec3f The sampled direction
     */
    pgl_vec3f Sample(const pgl_point2f &sample2D) const;

    /**
     * @brief Returns the sampling PDF for a given direction when is sampled
     * according to the guiding distribution.
     *
     * @param direction
     * @return float The PDF for sampling @ref direction
     */
    float PDF(const pgl_vec3f &direction) const;

    /**
     * @brief Combined importance sampling and PDF calculation.
     * Can be more efficient to use for some distributions (e.g. DirectionQuadtree)
     *
     * @param sample2D A 2D random variable
     * @param direction Importance sampled direction
     * @return float The PDF for sampling @ref direction
     */
    float SamplePDF(const pgl_point2f &sample2D, pgl_vec3f &direction) const;

    /**
     * @brief Returns the PDF for the incoming radiance distibution.
     * This PDF does not need to relate to the sampling PDF. It is used
     * to estimate the normalized incoming radiance distriubtion (e.g., for RIS).
     *
     * @param direction
     * @return float The PDF for incoming radiance distribution for @ref direction
     */
    float IncomingRadiancePDF(const pgl_vec3f &direction) const;

    /**
     * @brief Returns if the used representation supports for including the cosine
     * product (e.g, for diffuse surfaces) into the guiding distribution.
     *
     * @return true
     * @return false
     */
    bool SupportsApplyCosineProduct() const;

    /**
     * @brief Applies the product with the cosine to the guiding distribution.
     *
     * @param normal The surface normal at the curren sampling position.
     */
    void ApplyCosineProduct(const pgl_vec3f &normal);

    /**
     * @brief Get the id of the guiding cache used to initialize the SurfaceSamplingDistribution.
     *
     * @return uint32_t The id of the cache.
     */
    uint32_t GetId() const;

    pgl_vec3f OutgoingRadiance() const;

#ifdef OPENPGL_RADIANCE_CACHES
    /**
     * @brief Returns the incoming radiance estimates.
     *
     * @param direction The direction of the incoming radiance
     * @return An estimate of the incoming direction
     */
    pgl_vec3f IncomingRadiance(pgl_vec3f &direction, const bool directLightMIS) const;

    /**
     * @brief Returns the outgoing (scattered) radiance estimates.
     *
     * @param direction The direction of the outgoing (scattered) radiance
     * @return An estimate of the outgoing direction
     */
    pgl_vec3f OutgoingRadiance(pgl_vec3f &direction) const;

    /**
     * @brief Returns the irradiance estimate for a given normal direction.
     * The irradiance is the integral of the product of the cosine and incident radiance.
     *
     * @param direction The direction of the surface normal.
     * @return An estimate of the irradiance.
     */
    pgl_vec3f Irradiance(pgl_vec3f &normal, const bool directLightMIS) const;
#endif
    ///////////////////////////////////////
    /// Future plans
    ///////////////////////////////////////

    /**
     * @brief
     *
     * @param normal
     * @param opaque
     * @param transmission
     */
    // void ApplyCosineProduct(const pgl_vec3f& normal, const bool opaque, const pgl_vec3f transmission);

    /**
     * @brief Validates the current guiding distribution.
     * The guiding distribution can be invalid if it was not
     * initialized before or due to (numerical) problems during the fitting process.
     *
     * Note: Due to the overhead of this function, it should only be called during debugging.
     *
     * @return true
     * @return false
     */
    bool Validate() const;

    /**
     * @brief @deprecated
     *
     * @return Region
     */
    Region GetRegion() const;

   private:
    PGLSurfaceSamplingDistribution m_surfaceSamplingDistributionHandle{nullptr};
};

////////////////////////////////////////////////////////////
/// Implementation
////////////////////////////////////////////////////////////

OPENPGL_INLINE SurfaceSamplingDistribution::SurfaceSamplingDistribution(const Field *field)
{
    m_surfaceSamplingDistributionHandle = pglFieldNewSurfaceSamplingDistribution(field->m_fieldHandle);
}

OPENPGL_INLINE SurfaceSamplingDistribution::~SurfaceSamplingDistribution()
{
    OPENPGL_ASSERT(m_surfaceSamplingDistributionHandle);
    if (m_surfaceSamplingDistributionHandle)
        pglReleaseSurfaceSamplingDistribution(m_surfaceSamplingDistributionHandle);
    m_surfaceSamplingDistributionHandle = nullptr;
}

OPENPGL_INLINE pgl_vec3f SurfaceSamplingDistribution::Sample(const pgl_point2f &sample2D) const
{
    OPENPGL_ASSERT(m_surfaceSamplingDistributionHandle);
    return pglSurfaceSamplingDistributionSample(m_surfaceSamplingDistributionHandle, sample2D);
}

OPENPGL_INLINE float SurfaceSamplingDistribution::PDF(const pgl_vec3f &direction) const
{
    OPENPGL_ASSERT(m_surfaceSamplingDistributionHandle);
    return pglSurfaceSamplingDistributionPDF(m_surfaceSamplingDistributionHandle, direction);
}

OPENPGL_INLINE float SurfaceSamplingDistribution::SamplePDF(const pgl_point2f &sample2D, pgl_vec3f &direction) const
{
    OPENPGL_ASSERT(m_surfaceSamplingDistributionHandle);
    return pglSurfaceSamplingDistributionSamplePDF(m_surfaceSamplingDistributionHandle, sample2D, direction);
}

OPENPGL_INLINE float SurfaceSamplingDistribution::IncomingRadiancePDF(const pgl_vec3f &direction) const
{
    OPENPGL_ASSERT(m_surfaceSamplingDistributionHandle);
    return pglSurfaceSamplingDistributionIncomingRadiancePDF(m_surfaceSamplingDistributionHandle, direction);
}

OPENPGL_INLINE bool SurfaceSamplingDistribution::Validate() const
{
    OPENPGL_ASSERT(m_surfaceSamplingDistributionHandle);
    return pglSurfaceSamplingDistributionValidate(m_surfaceSamplingDistributionHandle);
}

OPENPGL_INLINE void SurfaceSamplingDistribution::Clear()
{
    OPENPGL_ASSERT(m_surfaceSamplingDistributionHandle);
    return pglSurfaceSamplingDistributionClear(m_surfaceSamplingDistributionHandle);
}

OPENPGL_INLINE bool SurfaceSamplingDistribution::Init(const Field *field, const pgl_point3f &pos, float &sample1D)
{
    OPENPGL_ASSERT(m_surfaceSamplingDistributionHandle);
    OPENPGL_ASSERT(field->m_fieldHandle);
    return pglFieldInitSurfaceSamplingDistribution(field->m_fieldHandle, m_surfaceSamplingDistributionHandle, pos, &sample1D);
}

OPENPGL_INLINE void SurfaceSamplingDistribution::ApplyCosineProduct(const pgl_vec3f &normal)
{
    OPENPGL_ASSERT(m_surfaceSamplingDistributionHandle);
    pglSurfaceSamplingDistributionApplyCosineProduct(m_surfaceSamplingDistributionHandle, normal);
}

OPENPGL_INLINE bool SurfaceSamplingDistribution::SupportsApplyCosineProduct() const
{
    OPENPGL_ASSERT(m_surfaceSamplingDistributionHandle);
    return pglSurfaceSamplingDistributionSupportsApplyCosineProduct(m_surfaceSamplingDistributionHandle);
}

OPENPGL_INLINE Region SurfaceSamplingDistribution::GetRegion() const
{
    OPENPGL_ASSERT(m_surfaceSamplingDistributionHandle);
    PGLRegion regionHandle = pglSurfaceSamplingGetRegion(m_surfaceSamplingDistributionHandle);
    return Region(regionHandle);
}

OPENPGL_INLINE uint32_t SurfaceSamplingDistribution::GetId() const
{
    OPENPGL_ASSERT(m_surfaceSamplingDistributionHandle);
    return pglSurfaceSamplingDistributionGetId(m_surfaceSamplingDistributionHandle);
}

OPENPGL_INLINE pgl_vec3f SurfaceSamplingDistribution::OutgoingRadiance() const {
    OPENPGL_ASSERT(m_surfaceSamplingDistributionHandle);
    return pglSurfaceSamplingDistributionOutgoingRadiance(m_surfaceSamplingDistributionHandle);
}

#ifdef OPENPGL_RADIANCE_CACHES
OPENPGL_INLINE pgl_vec3f SurfaceSamplingDistribution::IncomingRadiance(pgl_vec3f &direction, const bool directLightMIS) const
{
    OPENPGL_ASSERT(m_surfaceSamplingDistributionHandle);
    return pglSurfaceSamplingDistributionIncomingRadiance(m_surfaceSamplingDistributionHandle, direction, directLightMIS);
}

OPENPGL_INLINE pgl_vec3f SurfaceSamplingDistribution::OutgoingRadiance(pgl_vec3f &direction) const
{
    OPENPGL_ASSERT(m_surfaceSamplingDistributionHandle);
    return pglSurfaceSamplingDistributionOutgoingRadiance(m_surfaceSamplingDistributionHandle, direction);
}

OPENPGL_INLINE pgl_vec3f SurfaceSamplingDistribution::Irradiance(pgl_vec3f &normal, const bool directLightMIS) const
{
    OPENPGL_ASSERT(m_surfaceSamplingDistributionHandle);
    return pglSurfaceSamplingDistributionIrradiance(m_surfaceSamplingDistributionHandle, normal, directLightMIS);
}
#endif

}  // namespace cpp
}  // namespace openpgl