// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common.h"
#include "region.h"

#ifdef __cplusplus
extern "C"
{
#endif

#ifdef __cplusplus
    struct SurfaceSamplingDistribution;
#else
typedef ManagedObject SurfaceSamplingDistribution;
#endif

    typedef SurfaceSamplingDistribution *PGLSurfaceSamplingDistribution;

    OPENPGL_CORE_INTERFACE void pglReleaseSurfaceSamplingDistribution(PGLSurfaceSamplingDistribution surfaceSamplingDistribution);

    OPENPGL_CORE_INTERFACE void pglSurfaceSamplingDistributionApplyCosineProduct(PGLSurfaceSamplingDistribution surfaceSamplingDistribution, pgl_vec3f normal);

    OPENPGL_CORE_INTERFACE bool pglSurfaceSamplingDistributionSupportsApplyCosineProduct(PGLSurfaceSamplingDistribution surfaceSamplingDistribution);

    OPENPGL_CORE_INTERFACE pgl_vec3f pglSurfaceSamplingDistributionSample(PGLSurfaceSamplingDistribution surfaceSamplingDistribution, pgl_point2f sample);

    OPENPGL_CORE_INTERFACE float pglSurfaceSamplingDistributionPDF(PGLSurfaceSamplingDistribution surfaceSamplingDistribution, pgl_vec3f direction);

    OPENPGL_CORE_INTERFACE float pglSurfaceSamplingDistributionSamplePDF(PGLSurfaceSamplingDistribution surfaceSamplingDistribution, pgl_point2f sample, pgl_vec3f &direction);

    OPENPGL_CORE_INTERFACE float pglSurfaceSamplingDistributionIncomingRadiancePDF(PGLSurfaceSamplingDistribution surfaceSamplingDistribution, pgl_vec3f direction);

    OPENPGL_CORE_INTERFACE uint32_t pglSurfaceSamplingDistributionGetId(PGLSurfaceSamplingDistribution surfaceSamplingDistribution);

    OPENPGL_CORE_INTERFACE pgl_vec3f pglSurfaceSamplingDistributionOutgoingRadiance(PGLSurfaceSamplingDistribution surfaceSamplingDistribution);

    OPENPGL_CORE_INTERFACE bool pglSurfaceSamplingDistributionValidate(PGLSurfaceSamplingDistribution surfaceSamplingDistribution);

    OPENPGL_CORE_INTERFACE void pglSurfaceSamplingDistributionClear(PGLSurfaceSamplingDistribution surfaceSamplingDistribution);

    OPENPGL_CORE_INTERFACE PGLRegion pglSurfaceSamplingGetRegion(PGLSurfaceSamplingDistribution surfaceSamplingDistribution);

#ifdef OPENPGL_RADIANCE_CACHES
    OPENPGL_CORE_INTERFACE pgl_vec3f pglSurfaceSamplingDistributionIncomingRadiance(PGLSurfaceSamplingDistribution surfaceSamplingDistribution, pgl_vec3f direction,
                                                                                    const bool directLightMIS);

    OPENPGL_CORE_INTERFACE pgl_vec3f pglSurfaceSamplingDistributionIrradiance(PGLSurfaceSamplingDistribution surfaceSamplingDistribution, pgl_vec3f normal,
                                                                              const bool directLightMIS);

    OPENPGL_CORE_INTERFACE pgl_vec3f pglSurfaceSamplingDistributionOutgoingRadiance(PGLSurfaceSamplingDistribution surfaceSamplingDistribution, pgl_vec3f direction);
#endif

#ifdef __cplusplus
}  // extern "C"
#endif