// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#define OPENPGL_VMM_NUM_PHASE_COMP 4

#define OPENPGL_VMM_NUM_PHASE_REP 128
#define OPENPGL_VMM_PHASE_MIN_MEAN_COSINE 0.f
#define OPENPGL_VMM_PHASE_MAX_MEAN_COSINE 0.99f

struct VMMPhaseFunctionRepresentation
{
    int K;
    float g;
    float weights[OPENPGL_VMM_NUM_PHASE_COMP];
    float meanCosines[OPENPGL_VMM_NUM_PHASE_COMP];
    float kappas[OPENPGL_VMM_NUM_PHASE_COMP];
    float normalizations[OPENPGL_VMM_NUM_PHASE_COMP];
};

struct VMMSingleLobeHenyeyGreensteinOracle
{
    float minMeanCosine;     //{0.f};
    float maxMeanCosine;     //{0.99f};
    int numRepresentations;  //{128};

    VMMPhaseFunctionRepresentation representations[OPENPGL_VMM_NUM_PHASE_REP];

    inline void init();
    OPENPGL_GPU_CALLABLE const VMMPhaseFunctionRepresentation &getPhaseFunctionRepresentation(const float meanCosine)
    {
        // OPENPGL_ASSERT(std::fabs(meanCosine) >= minMeanCosine);
        // OPENPGL_ASSERT(std::fabs(meanCosine) <= maxMeanCosine);
    
        const float absMeanCosine = std::fabs(meanCosine);
        const float stepSize = (maxMeanCosine - minMeanCosine) / float(numRepresentations);
        int idx = std::floor((absMeanCosine - minMeanCosine) / stepSize);
        idx = std::min(idx, numRepresentations - 1);
    
        //OPENPGL_ASSERT(idx >= 0);
        //OPENPGL_ASSERT(idx < numRepresentations);
        return representations[idx];
    }
};

__device__ VMMSingleLobeHenyeyGreensteinOracle vMMSingleLobeHenyeyGreensteinOracle;

#include "VMMPhaseFunctionDefs.h"
