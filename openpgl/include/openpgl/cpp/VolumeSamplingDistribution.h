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
 * @brief The Sampling distribution used for guidiging directional sampling decisions inside volumes.
 *
 * The guided sampling distribution can be proportional to the incoming radiance or to its product
 * with the phase function (e.g., single lobe HG). The class supports function for sampling and
 * PDF evaluations.
 *
 */

struct VolumeSamplingDistribution
{
    /**
     * @brief Constructs new instance of a VolumeSamplingDistribution.
     *
     * Reserves the memory need to store the guiding distribution.
     * Since the type/representation of distribution depends on the guiding field
     * a pointer to the @ref Field has to be provided. After construction
     * the VolumeSamplingDistribution still need to be initialized using the @ref Init function.
     *
     * @param field Pointer to the -guiding- Field.
     */
    VolumeSamplingDistribution(const Field *field);

    ~VolumeSamplingDistribution();

    VolumeSamplingDistribution(const VolumeSamplingDistribution &) = delete;

    /**
     * @brief Intitializes the guiding distibution for a given position in the scene.
     *
     * This function queries the guiding field for a surface guiding distribution for
     * given position in the scene and initializes the VolumeSamplingDistribution
     * to this Distribution. The resulting distribution is usually proportional to the local
     * incident radiance distribution at the query position. The VolumeSamplingDistribution
     * can further being imporoved by applying products with phase function (e.g., single lobe HG).
     *
     * Note: in anisotropic volumes it is highly recommended to add the phase function product to
     * avoid variance increase due to only guiding proportional to the incident radiance distribution.
     *
     * @param field The guiding field of the scene.
     * @param pos The position the guiding distribution is queried for.
     * @param sample1D A random number used of a stoachastic look-up is used.
     * @return true
     * @return false
     */
    bool Init(const Field *field, const pgl_point3f &pos, float &sample1D);

    /**
     * @brief Clears/resets the internal repesentation of the guiding distribution.
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
     * @brief Returns if the used representation supports for including the
     * product with a single lobe HenyeyGreenstein phase function into the guiding distribution.
     *
     * @return true
     * @return false
     */
    bool SupportsApplySingleLobeHenyeyGreensteinProduct() const;

    /**
     * @brief Applies the product with a single lobe HenyeyGreenstein phase function
     * to the sampling distribution.
     *
     * @param dir The direction the walk/path arrives a the sample position.
     * @param meanCosine The mean cosine of the HG phase function.
     */
    void ApplySingleLobeHenyeyGreensteinProduct(const pgl_vec3f &dir, const float meanCosine);

    /**
     * @brief Get the id of the guiding cache used to initialize the VolumeSamplingDistribution.
     *
     * @return uint32_t The id of the cache.
     */
    uint32_t GetId() const;

    pgl_vec3f OutgoingRadiance() const;

#ifdef OPENPGL_RADIANCE_CACHES
    /**
     * @brief Returns the incoming radiance estimates.
     *
     * @param direction The direction of the incoming radiance.
     * @return An estimate of the incoming direction.
     */
    pgl_vec3f IncomingRadiance(pgl_vec3f &direction, const bool directLightMIS) const;

    /**
     * @brief Returns the outgoing (scattered) radiance estimates.
     *
     * @param direction The direction of the outgoing (scattered) radiance.
     * @return An estimate of the outgoing direction.
     */
    pgl_vec3f OutgoingRadiance(pgl_vec3f &direction) const;

    /**
     * @brief Returns the in-scattered radiance estimates.
     * The outgoing and in-scattered radiance estimates are estimates of the same qunatity.
     * Depending of the used representation (e.g., VMM, QuadTree), more precise estimates for the inscattered
     * might be used (e.g., VMM -> convolution of the incomming radaince).
     *
     * @param direction The direction the volume radiance is scattered into.
     * @return An estimate of the in-scattered direction.
     */
    pgl_vec3f InscatteredRadiance(pgl_vec3f &direction, const float g, const bool directLightMIS) const;

    /**
     * @brief Returns the estimate of the volume fluence.
     * The volume fluence is the radiance passing throug a point in a volume before scattering.
     *
     * @return An estimate of the volume fluence direction.
     */
    pgl_vec3f Fluence(const bool directLightMIS) const;
#endif

    ///////////////////////////////////////
    /// Future plans
    ///////////////////////////////////////

    /*
        void ApplyDualLobeHGProduct(const float meanCosine0, const float meanCosine1, const float mixWeight);
    */

    /**
     * @brief Validates the current guiding distribution.
     * The guiding distribution can be invalid if it was not
     * initialized before or due to (numerical) porblems during the fitting process.
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

    friend struct openpgl::cpp::Field;

   private:
    PGLVolumeSamplingDistribution m_volumeSamplingDistributionHandle{nullptr};
};

////////////////////////////////////////////////////////////
/// Implementation
////////////////////////////////////////////////////////////

OPENPGL_INLINE VolumeSamplingDistribution::VolumeSamplingDistribution(const Field *field)
{
    m_volumeSamplingDistributionHandle = pglFieldNewVolumeSamplingDistribution(field->m_fieldHandle);
}

OPENPGL_INLINE VolumeSamplingDistribution::~VolumeSamplingDistribution()
{
    OPENPGL_ASSERT(m_volumeSamplingDistributionHandle);
    if (m_volumeSamplingDistributionHandle)
        pglReleaseVolumeSamplingDistribution(m_volumeSamplingDistributionHandle);
    m_volumeSamplingDistributionHandle = nullptr;
}

OPENPGL_INLINE pgl_vec3f VolumeSamplingDistribution::Sample(const pgl_point2f &sample2D) const
{
    OPENPGL_ASSERT(m_volumeSamplingDistributionHandle);
    return pglVolumeSamplingDistributionSample(m_volumeSamplingDistributionHandle, sample2D);
}

OPENPGL_INLINE float VolumeSamplingDistribution::PDF(const pgl_vec3f &direction) const
{
    OPENPGL_ASSERT(m_volumeSamplingDistributionHandle);
    return pglVolumeSamplingDistributionPDF(m_volumeSamplingDistributionHandle, direction);
}

OPENPGL_INLINE float VolumeSamplingDistribution::SamplePDF(const pgl_point2f &sample2D, pgl_vec3f &direction) const
{
    OPENPGL_ASSERT(m_volumeSamplingDistributionHandle);
    return pglVolumeSamplingDistributionSamplePDF(m_volumeSamplingDistributionHandle, sample2D, direction);
}

OPENPGL_INLINE float VolumeSamplingDistribution::IncomingRadiancePDF(const pgl_vec3f &direction) const
{
    OPENPGL_ASSERT(m_volumeSamplingDistributionHandle);
    return pglVolumeSamplingDistributionIncomingRadiancePDF(m_volumeSamplingDistributionHandle, direction);
}

OPENPGL_INLINE bool VolumeSamplingDistribution::Validate() const
{
    OPENPGL_ASSERT(m_volumeSamplingDistributionHandle);
    return pglVolumeSamplingDistributionValidate(m_volumeSamplingDistributionHandle);
}

OPENPGL_INLINE void VolumeSamplingDistribution::Clear()
{
    OPENPGL_ASSERT(m_volumeSamplingDistributionHandle);
    return pglVolumeSamplingDistributionClear(m_volumeSamplingDistributionHandle);
}

OPENPGL_INLINE void VolumeSamplingDistribution::ApplySingleLobeHenyeyGreensteinProduct(const pgl_vec3f &dir, const float meanCosine)
{
    OPENPGL_ASSERT(m_volumeSamplingDistributionHandle);
    return pglVolumeSamplingDistributionApplySingleLobeHenyeyGreensteinProduct(m_volumeSamplingDistributionHandle, dir, meanCosine);
}

OPENPGL_INLINE bool VolumeSamplingDistribution::SupportsApplySingleLobeHenyeyGreensteinProduct() const
{
    OPENPGL_ASSERT(m_volumeSamplingDistributionHandle);
    return pglVolumeSamplingDistributionSupportsApplySingleLobeHenyeyGreensteinProduct(m_volumeSamplingDistributionHandle);
}

OPENPGL_INLINE bool VolumeSamplingDistribution::Init(const Field *field, const pgl_point3f &pos, float &sample1D)
{
    OPENPGL_ASSERT(m_volumeSamplingDistributionHandle);
    OPENPGL_ASSERT(field->m_fieldHandle);
    return pglFieldInitVolumeSamplingDistribution(field->m_fieldHandle, m_volumeSamplingDistributionHandle, pos, &sample1D);
}

OPENPGL_INLINE Region VolumeSamplingDistribution::GetRegion() const
{
    OPENPGL_ASSERT(m_volumeSamplingDistributionHandle);
    PGLRegion regionHandle = pglVolumeSamplingGetRegion(m_volumeSamplingDistributionHandle);
    return Region(regionHandle);
}

OPENPGL_INLINE uint32_t VolumeSamplingDistribution::GetId() const
{
    OPENPGL_ASSERT(m_volumeSamplingDistributionHandle);
    return pglVolumeSamplingDistributionGetId(m_volumeSamplingDistributionHandle);
}

OPENPGL_INLINE pgl_vec3f VolumeSamplingDistribution::OutgoingRadiance() const {
    OPENPGL_ASSERT(m_volumeSamplingDistributionHandle);
    return pglVolumeSamplingDistributionOutgoingRadiance(m_volumeSamplingDistributionHandle);
}

#ifdef OPENPGL_RADIANCE_CACHES
OPENPGL_INLINE pgl_vec3f VolumeSamplingDistribution::IncomingRadiance(pgl_vec3f &direction, const bool directLightMIS) const
{
    OPENPGL_ASSERT(m_volumeSamplingDistributionHandle);
    return pglVolumeSamplingDistributionIncomingRadiance(m_volumeSamplingDistributionHandle, direction, directLightMIS);
}

OPENPGL_INLINE pgl_vec3f VolumeSamplingDistribution::OutgoingRadiance(pgl_vec3f &direction) const
{
    OPENPGL_ASSERT(m_volumeSamplingDistributionHandle);
    return pglVolumeSamplingDistributionOutgoingRadiance(m_volumeSamplingDistributionHandle, direction);
}

OPENPGL_INLINE pgl_vec3f VolumeSamplingDistribution::InscatteredRadiance(pgl_vec3f &direction, const float g, const bool directLightMIS) const
{
    OPENPGL_ASSERT(m_volumeSamplingDistributionHandle);
    return pglVolumeSamplingDistributionInscatteredRadiance(m_volumeSamplingDistributionHandle, direction, g, directLightMIS);
}

OPENPGL_INLINE pgl_vec3f VolumeSamplingDistribution::Fluence(const bool directLightMIS) const
{
    OPENPGL_ASSERT(m_volumeSamplingDistributionHandle);
    return pglVolumeSamplingDistributionFluence(m_volumeSamplingDistributionHandle, directLightMIS);
}
#endif

}  // namespace cpp
}  // namespace openpgl