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
    struct VolumeSamplingDistribution;
#else
typedef ManagedObject VolumeSamplingDistribution;
#endif

    typedef VolumeSamplingDistribution *PGLVolumeSamplingDistribution;

    OPENPGL_CORE_INTERFACE void pglReleaseVolumeSamplingDistribution(PGLVolumeSamplingDistribution VolumeSamplingDistribution);

    OPENPGL_CORE_INTERFACE pgl_vec3f pglVolumeSamplingDistributionSample(PGLVolumeSamplingDistribution VolumeSamplingDistribution, pgl_point2f sample);

    OPENPGL_CORE_INTERFACE float pglVolumeSamplingDistributionPDF(PGLVolumeSamplingDistribution VolumeSamplingDistribution, pgl_vec3f direction);

    OPENPGL_CORE_INTERFACE float pglVolumeSamplingDistributionSamplePDF(PGLVolumeSamplingDistribution VolumeSamplingDistribution, pgl_point2f sample, pgl_vec3f &direction);

    OPENPGL_CORE_INTERFACE float pglVolumeSamplingDistributionIncomingRadiancePDF(PGLVolumeSamplingDistribution VolumeSamplingDistribution, pgl_vec3f direction);

    OPENPGL_CORE_INTERFACE uint32_t pglVolumeSamplingDistributionGetId(PGLVolumeSamplingDistribution VolumeSamplingDistribution);

    OPENPGL_CORE_INTERFACE pgl_vec3f pglVolumeSamplingDistributionOutgoingRadiance(PGLVolumeSamplingDistribution VolumeSamplingDistribution);

    OPENPGL_CORE_INTERFACE bool pglVolumeSamplingDistributionValidate(PGLVolumeSamplingDistribution VolumeSamplingDistribution);

    OPENPGL_CORE_INTERFACE void pglVolumeSamplingDistributionClear(PGLVolumeSamplingDistribution VolumeSamplingDistribution);

    OPENPGL_CORE_INTERFACE void pglVolumeSamplingDistributionApplySingleLobeHenyeyGreensteinProduct(PGLVolumeSamplingDistribution volumeSamplingDistribution, pgl_vec3f dir,
                                                                                                    const float meanCosine);

    OPENPGL_CORE_INTERFACE bool pglVolumeSamplingDistributionSupportsApplySingleLobeHenyeyGreensteinProduct(PGLVolumeSamplingDistribution volumeSamplingDistribution);

    OPENPGL_CORE_INTERFACE PGLRegion pglVolumeSamplingGetRegion(PGLVolumeSamplingDistribution VolumeSamplingDistribution);

#ifdef OPENPGL_RADIANCE_CACHES
    OPENPGL_CORE_INTERFACE pgl_vec3f pglVolumeSamplingDistributionIncomingRadiance(PGLVolumeSamplingDistribution volumeSamplingDistribution, pgl_vec3f direction,
                                                                                   const bool directLightMIS);

    OPENPGL_CORE_INTERFACE pgl_vec3f pglVolumeSamplingDistributionOutgoingRadiance(PGLVolumeSamplingDistribution volumeSamplingDistribution, pgl_vec3f direction);

    OPENPGL_CORE_INTERFACE pgl_vec3f pglVolumeSamplingDistributionInscatteredRadiance(PGLVolumeSamplingDistribution volumeSamplingDistribution, pgl_vec3f direction, float g,
                                                                                      const bool directLightMIS);

    OPENPGL_CORE_INTERFACE pgl_vec3f pglVolumeSamplingDistributionFluence(PGLVolumeSamplingDistribution volumeSamplingDistribution, const bool directLightMIS);
#endif

#ifdef __cplusplus
}  // extern "C"
#endif