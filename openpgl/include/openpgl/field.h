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
#include "config.h"
#include "fieldstatistics.h"
#include "region.h"
#include "samplestorage.h"
#include "surfacesamplingdistribution.h"
#include "volumesamplingdistribution.h"
#include "sdump.h"

#ifdef __cplusplus
extern "C"
{
#endif

#ifdef __cplusplus
    struct Field;
#else
typedef ManagedObject Field;
#endif

    typedef Field *PGLField;

    OPENPGL_CORE_INTERFACE void pglReleaseField(PGLField field);

    OPENPGL_CORE_INTERFACE bool pglFieldStoreToFile(PGLField field, const char *fieldFileName);

    OPENPGL_CORE_INTERFACE size_t pglFieldGetIteration(PGLField field);

    OPENPGL_CORE_INTERFACE void pglFieldSetSceneBounds(PGLField field, pgl_box3f bounds);

    OPENPGL_CORE_INTERFACE pgl_box3f pglFieldGetSceneBounds(PGLField field);

    OPENPGL_CORE_INTERFACE bool pglFieldDump(PGLField field, const char *dumpFileName);

    OPENPGL_CORE_INTERFACE void pglFieldUpdate(PGLField field, PGLSampleStorage sampleStorage);

    OPENPGL_CORE_INTERFACE void pglFieldUpdateSurface(PGLField field, PGLSampleStorage sampleStorage);

    OPENPGL_CORE_INTERFACE void pglFieldUpdateVolume(PGLField field, PGLSampleStorage sampleStorage);

    OPENPGL_CORE_INTERFACE void pglFieldReset(PGLField field);

    OPENPGL_CORE_INTERFACE PGLSurfaceSamplingDistribution pglFieldNewSurfaceSamplingDistribution(PGLField field);

    OPENPGL_CORE_INTERFACE bool pglFieldInitSurfaceSamplingDistribution(PGLField field, PGLSurfaceSamplingDistribution surfaceSamplingDistribution, pgl_point3f position,
                                                                        float *sample1D);

    OPENPGL_CORE_INTERFACE PGLVolumeSamplingDistribution pglFieldNewVolumeSamplingDistribution(PGLField field);

    OPENPGL_CORE_INTERFACE bool pglFieldInitVolumeSamplingDistribution(PGLField field, PGLVolumeSamplingDistribution volumeSamplingDistribution, pgl_point3f position,
                                                                       float *sample1D);

    OPENPGL_CORE_INTERFACE bool pglFieldValidate(PGLField field);

    OPENPGL_CORE_INTERFACE bool pglFieldCompare(PGLField fieldA, PGLField fieldB);

    OPENPGL_CORE_INTERFACE PGLFieldStatistics pglFieldGetSurfaceStatistics(PGLField field);

    OPENPGL_CORE_INTERFACE PGLFieldStatistics pglFieldGetVolumeStatistics(PGLField field);

    // Stefan SYCL
    OPENPGL_CORE_INTERFACE void pglFieldFillFieldGPU(PGLField field, void *fieldData, void *deviceGPU);
    OPENPGL_CORE_INTERFACE void pglFieldReleaseFieldGPU(PGLField field, void *fieldData, void *deviceGPU);
/*
OPENPGL_CORE_INTERFACE int pglFieldGetNumSurfaceNodes(PGLField field);
OPENPGL_CORE_INTERFACE void *pglFieldGetSurfaceNodes(PGLField field);
OPENPGL_CORE_INTERFACE int pglFieldGetNumSurfaceDistributions(PGLField field);
OPENPGL_CORE_INTERFACE void pglFieldCopySurfaceDistributions(PGLField field, void *out);

OPENPGL_CORE_INTERFACE int pglFieldGetNumVolumeNodes(PGLField field);
OPENPGL_CORE_INTERFACE void *pglFieldGetVolumeNodes(PGLField field);
OPENPGL_CORE_INTERFACE int pglFieldGetNumVolumeDistributions(PGLField field);
OPENPGL_CORE_INTERFACE void pglFieldCopyVolumeDistributions(PGLField field, void *out);
*/
    OPENPGL_CORE_INTERFACE PGLRange pglFieldGetSurfaceSampleRange(PGLField field, size_t id);

    OPENPGL_CORE_INTERFACE PGLRange pglFieldGetVolumeSampleRange(PGLField field, size_t id);

    OPENPGL_CORE_INTERFACE void pglFieldRunUpdateDump(PGLField field, const char *updateDumpFileName, bool surface);

    OPENPGL_CORE_INTERFACE void pglFieldSDump(PGLField field, SDump* sdump);

#ifdef __cplusplus
}  // extern "C"
#endif
