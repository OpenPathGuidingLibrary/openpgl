// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <math.h>

#include <array>

#include "../../openpgl_common.h"

#define OPENPGL_VMM_NUM_PHASE_COMP 4

#define OPENPGL_VMM_NUM_PHASE_REP 128
#define OPENPGL_VMM_PHASE_MIN_MEAN_COSINE 0.f
#define OPENPGL_VMM_PHASE_MAX_MEAN_COSINE 0.99f

namespace openpgl
{
struct VMMPhaseFunctionRepresentation
{
    int K{OPENPGL_VMM_NUM_PHASE_COMP};
    float g{0.f};
    float weights[OPENPGL_VMM_NUM_PHASE_COMP];
    float meanCosines[OPENPGL_VMM_NUM_PHASE_COMP];
    float kappas[OPENPGL_VMM_NUM_PHASE_COMP];
    float normalizations[OPENPGL_VMM_NUM_PHASE_COMP];

    // const std::string toString() const{

    //}
};

class VMMSingleLobeHenyeyGreensteinOracle
{
    static float minMeanCosine;     //{0.f};
    static float maxMeanCosine;     //{0.99f};
    static int numRepresentations;  //{128};
   public:
    static std::array<VMMPhaseFunctionRepresentation, OPENPGL_VMM_NUM_PHASE_REP> representations;

    static void init();

    static const VMMPhaseFunctionRepresentation &getPhaseFunctionRepresentation(const float meanCosine)
    {
        // OPENPGL_ASSERT(std::fabs(meanCosine) >= minMeanCosine);
        // OPENPGL_ASSERT(std::fabs(meanCosine) <= maxMeanCosine);

        const float absMeanCosine = std::fabs(meanCosine);
        const float stepSize = (maxMeanCosine - minMeanCosine) / float(numRepresentations);
        int idx = std::floor((absMeanCosine - minMeanCosine) / stepSize);
        idx = std::min(idx, numRepresentations - 1);

        OPENPGL_ASSERT(idx >= 0);
        OPENPGL_ASSERT(idx < numRepresentations);
        return representations[idx];
    }
};

}  // namespace openpgl
