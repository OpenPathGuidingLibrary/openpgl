
// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef __cplusplus
#include <cstdint>
#include <cstdlib>
#else
#include <stdint.h>
#include <stdlib.h>
#endif

#include "common.h"

struct PGLRange
{
    size_t start{0};
    size_t end{0};
};

/**
 * @brief
 *
 */
struct PGLSampleData
{
    enum Flags
    {
        /// point does not represent any real scene intersection point
        EInsideVolume = 1 << 0,
        /// if the samples represents direct light from a light source
        EDirectLight = 1 << 1
    };

    /// the position of the sample (i.e., at which energy arrives)
    pgl_point3f position;

    /// a vector pointing into the direction the energy (e.g., radiance or importance)
    /// comes from
    pgl_direction direction;

    /// a scalar representation of the incident radiance divide by @ref pdf
    float weight;

#ifdef OPENPGL_RADIANCE_CACHES
    /// a scalar representation of the incident radiance in RGB NOT divide by @ref pdf
    /// but probably weighted with @ref radianceInMISWeight if the racince comes from a
    /// light source.
    pgl_spectrum radianceIn;
    /// The nex
    float radianceInMISWeight;
#endif

    /// the PDF used for sampling @ref direction at @ref position
    float pdf;

    /// the distance to the source of the incident radiance
    float distance;

    ///
    uint32_t flags;

#ifdef OPENPGL_RADIANCE_CACHES
    /// a vector pointing into the outgoing direction the energy is reflected to(e.g., towards the camera)
    pgl_direction directionOut;
    /// the outgoing, reflected/scattered radiance (i.e., incident radiance time BSDF/phase divided by pdf)
    pgl_spectrum radianceOut;
#endif
};

struct PGLZeroValueSampleData
{
    /// the position of the invalid sample
    pgl_point3f position;
    /// a vector pointing into the direction the path generating the zero value sample went
    pgl_direction direction;
#ifdef OPENPGL_RADIANCE_CACHES
    /// a vector pointing into the outgoing direction the energy is reflected to(e.g., towards the camera)
    pgl_direction directionOut;
#endif
    /// if the position is inside a volume
    bool volume;
};

/**
 * @brief
 *
 */
struct PGLPathSegmentData
{
    /// The starting position of the path segment (on a surface or inside a volume)
    pgl_point3f position{0.f, 0.f, 0.f};
    /// The direction to the starting position of the next path segment
    pgl_vec3f directionIn{0.f, 1.f, 0.f};
    /// The direction to the starting position of the previous path segment
    pgl_vec3f directionOut{1.f, 0.f, 0.f};
    /// The surface normal at the current @ref position. Inside volumes @ref normal can be an arbritray valid direction.
    pgl_vec3f normal{0.f, 0.f, 1.f};

    // float distance {0.0f};

    /// If the corrent segment starts inside a volume
    bool volumeScatter{false};

    /// The PDF of sampling @ref directionIn at @ref position towards the next segment
    float pdfDirectionIn{1.0f};

    /// If the scattering interaction at @ref position was a delta Dirac (e.g., perfect mirror or glass).
    bool isDelta{false};

    /// The BSDF*cos or phase function evaluation at @ref position for the out-going direction @ref directionOut and the incoming direction @ref directionIn divided by the sampling
    /// PDF @ref pdfDirectionIn.
    pgl_vec3f scatteringWeight{1.0f, 1.0f, 1.0f};

    /// The transmittance from the current @ref position to the @ref position of the next segment divided by the corresponding distance sampling PDF.
    pgl_vec3f transmittanceWeight{1.0f, 1.0f, 1.0f};

    /// the direct contribution (i.e., emission) at @ref position into @ref directionOut
    pgl_vec3f directContribution{0.0f, 0.0f, 0.0f};

    /// The MIS weight which would be applied to @ref directContribution (e.g., miWeight = bsdfPDF^2/(bsdfPDF^2+neePDF^2)).
    float miWeight{1.0f};

    /// The contribution which is scattered at @ref position into @ref directionOut. There can be multiple source us such contribution (sub-surface-scattering (dipole) or scattered
    /// NEE).
    pgl_vec3f scatteredContribution{0.0f, 0.0f, 0.0f};

    /// The probability to survive Russian roulette
    float russianRouletteSurvivalProbability{1.0f};

    // BSDF information
    /// The refractive index (eta) of the material at @ref position.
    float eta{1.0f};

    /// The roughness of the material (e.g., GGX roughness on surfaces and mean cosine in volumes).
    float roughness{1.0f};

    /// The pointer to the Region (DEPRECATED)
    const void *regionPtr{nullptr};
};

#if defined(OPENPGL_IMAGE_SPACE_GUIDING_BUFFER)

/**
 * @brief The class representing the contribution and some auxiliary information (e.g., albedo, normal) about the MC (random walk) sample for estimating a pixel value.
 */
struct PGLImageSpaceSample
{
    enum Flags
    {
        /// if the first event of the random walk is inside a volume
        EVolumeEvent = 1 << 0,
    };
    /// @brief The contribution of the MC sample.
    pgl_vec3f contribution{0.f, 0.f, 0.f};
    /// @brief The albedo of the surface or volume at the first hit position (on a surface or in a volume).
    pgl_vec3f albedo{0.f, 0.f, 0.f};
    /// @brief The surface normal if the first hit position is on a surface or the primary ray
    /// direction if the first hit is inside a volume.
    pgl_vec3f normal{0.f, 0.f, -1.f};
    /// @brief Stores a set of binary sample attributes.
    uint32_t flags{0};

#ifdef __cplusplus
    /// @brief Sets the bit-flag if the sample originates from a surface envent or not.
    /// @param isSurfaceEvent If the samples originates from a surface event.
    void SetSurfaceEvent(bool isSurfaceEvent)
    {
        if (isSurfaceEvent)
            flags ^= EVolumeEvent;
        else
            flags |= EVolumeEvent;
    }

    /// @brief If the samples originates from a surface event.
    bool IsSurfaceEvent() const
    {
        return flags ^ EVolumeEvent;
    }
#endif
};

#endif

struct PGLString
{
    char *m_str;
    size_t m_size;
};

#ifdef __cplusplus
extern "C"
{
#endif

    OPENPGL_CORE_INTERFACE void pglReleaseString(PGLString str);

#ifdef __cplusplus
}  // extern "C"
#endif