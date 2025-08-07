// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstring>

#include "../openpgl.h"
namespace openpgl
{
namespace cpp
{

struct Field;

/**
 * @brief The configuration for used to initialize the guiding Field.
 *
 * This class contains and defines all parameters needed to initialize a guiding Field object.
 * These parameters contain the spatial structure or directional distribution types and the parameters
 * to set up their building (i.e, sub-division behavior) and training.
 */
struct FieldConfig
{
    FieldConfig() = default;

    ~FieldConfig() = default;

    FieldConfig(const FieldConfig &) = delete;

    /**
     * @brief Initializes the field configurations by setting all default parameters for a given combination
     * of spatial structure and directional distribution type and some additional parameters.
     *
     * @param spatialType The spatial structure type.
     * @param directionalType The directional distribution type.
     * @param deterministic If the training/updating of the field should be deterministic (default = true).
     * @param maxSamplesPerLeaf The maximum number of samples per tree node (default = 32K).
     */
    void Init(const PGL_SPATIAL_STRUCTURE_TYPE spatialType, const PGL_DIRECTIONAL_DISTRIBUTION_TYPE directionalType, const bool deterministic = true,
              const size_t maxSamplesPerLeaf = 32000);

    /**
     * @brief Sets the maximum depth of the tree structure (e.g., 16).
     *
     * @param maxDepth The maximum depth of the tree structure.
     */
    void SetSpatialStructureArgMaxDepth(const size_t maxDepth);

    /**
     * @brief Enables or disables K-nearest neighbor lookup when querying a guiding cache.
     *
     * @param useKnnLookup if KNN lookup should be used
     */
    void SetUseKnnLookup(const bool useKnnLookup);

    /**
     * @brief Enables or disables if selected neighbor from the KNN-lookup is imporatnce sampled based on the distance (i.e., using a Gaussian kernel).
     *
     * @param useKnnIsLookup if distance based importance sampling of the neighbors should be used
     */
    void SetUseKnnIsLookup(const bool useKnnIsLookup);

    /**
     * @brief For debugging and benchmarking the update of the spatial structure this function can disable
     * the training of the directional distribution during the update iterations.
     *
     * @param fitRegions If the directional distributions should be trained during an update iteration.
     */
    void SetDebugArgFitRegions(const bool fitRegions);

    void SetDebugDumpCacheCellData(const bool dumpCacheCellData);

    void SetDebugDumpCacheCellPosition(const pgl_point3f &dumpCacheCellPosition);

    void SetDebugDumpCacheCellLocation(const std::string &dumpCacheCellLocation);

    friend struct openpgl::cpp::Field;

   private:
    PGLFieldArguments m_args;
};

////////////////////////////////////////////////////////////
/// Implementation
////////////////////////////////////////////////////////////

OPENPGL_INLINE void FieldConfig::Init(const PGL_SPATIAL_STRUCTURE_TYPE spatialType, const PGL_DIRECTIONAL_DISTRIBUTION_TYPE directionalType, const bool deterministic,
                                      const size_t maxSamplesPerLeaf)
{
    pglFieldArgumentsSetDefaults(m_args, spatialType, directionalType, deterministic, maxSamplesPerLeaf);
}

OPENPGL_INLINE void FieldConfig::SetDebugArgFitRegions(const bool fitRegions)
{
    m_args.debugArguments.fitRegions = fitRegions;
}

OPENPGL_INLINE void FieldConfig::SetSpatialStructureArgMaxDepth(const size_t maxDepth)
{
    reinterpret_cast<PGLKDTreeArguments *>(m_args.spatialSturctureArguments)->maxDepth = maxDepth;
}

OPENPGL_INLINE void FieldConfig::SetUseKnnLookup(const bool useKnnLookup)
{
    reinterpret_cast<PGLKDTreeArguments *>(m_args.spatialSturctureArguments)->knnLookup = useKnnLookup;
}

OPENPGL_INLINE void FieldConfig::SetUseKnnIsLookup(const bool useKnnIsLookup)
{
    reinterpret_cast<PGLKDTreeArguments *>(m_args.spatialSturctureArguments)->isKnnLookup = useKnnIsLookup;
}

OPENPGL_INLINE void FieldConfig::SetDebugDumpCacheCellData(const bool dumpCacheCellData)
{
    m_args.debugArguments.dumpCacheCellData = dumpCacheCellData;
}

OPENPGL_INLINE void FieldConfig::SetDebugDumpCacheCellPosition(const pgl_point3f &dumpCacheCellPosition)
{
    m_args.debugArguments.dumpCacheCellPosition = dumpCacheCellPosition;
}

OPENPGL_INLINE void FieldConfig::SetDebugDumpCacheCellLocation(const std::string &dumpCacheCellLocation)
{
    strcpy(m_args.debugArguments.dumpCacheCellLocation, dumpCacheCellLocation.c_str());
}

}  // namespace cpp
}  // namespace openpgl