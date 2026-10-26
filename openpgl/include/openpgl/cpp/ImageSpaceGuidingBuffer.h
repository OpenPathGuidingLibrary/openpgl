// Copyright 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

#include "../openpgl.h"
#include "Common.h"

namespace openpgl
{
namespace cpp
{
namespace util
{

/**
 * @brief The ImageSpaceGuidingBuffer class calculates image-space guiding information from pixel samples.
 *
 * The class collects and stores the Monte-Carlo random work pixels samples generated during rendering.
 * The information gathered by these samples is then used, during the @ref Update step to calculate/estimate
 * image-space guiding information (e.g., pixel contribution estimates for guided/adjoint-driven RR).
 *
 */
struct ImageSpaceGuidingBuffer
{
    struct Config
    {
        Config(Point2i resolution)
        {
            data.resolution = {resolution.x, resolution.y};
        }

        void EnableContributionEstimate(const bool contributionEstimate)
        {
            data.contributionEstimate = contributionEstimate;
        }

        bool ContributionEstimate() const
        {
            return data.contributionEstimate;
        }

        Point2i GetResolution() const
        {
            return {data.resolution.x, data.resolution.y};
        }

        void SetContributionType(PGLContributionTypes type)
        {
            data.contributionType = type;
        }

        PGLContributionTypes GetContributionType()
        {
            return data.contributionType;
        }

#if defined(OPENPGL_VSP_GUIDING)
        void EnableVolumeScatterProbabilityEstimate(const bool vspEstimate)
        {
            data.vspEstimate = vspEstimate;
        }

        bool VolumeScatterProbabilityEstimate() const
        {
            return data.vspEstimate;
        }

        void SetVolumeScatterProbabilityType(PGLVSPTypes type)
        {
            data.vspType = type;
        }

        PGLVSPTypes GetVolumeScatterProbabilityType()
        {
            return data.vspType;
        }
#endif
        friend struct ImageSpaceGuidingBuffer;

       private:
        PGLImageSpaceGuidingBufferConfig data;
    };

    typedef PGLImageSpaceSample Sample;

    /**
     * Creates an ImageSpaceGuidingBuffer for a given image resolution.
     *
     * @param resolution The size/reslution of the image buffer
     */
    ImageSpaceGuidingBuffer(const Config cfg);

    /**
     * Creates/Loads an ImageSpaceGuidingBuffer from  multi-channel EXR file.
     *
     * @param fileName The location of the multi-channel EXR file.
     */
    ImageSpaceGuidingBuffer(const std::string &fileName);

    ~ImageSpaceGuidingBuffer();

    /**
     * @brief Updates the image-space estimates using the previously collected/aggregated pixel samples.
     *
     * Internaly this step denoises the agregated sample data and calulated all necessary image-space guiding buffer.
     */
    void Update();

    /**
     * @brief Adds a pixel sample to the buffer.
     *
     * @param pixel The 2D pixel coordinate of the sample
     *
     * @param sample The sample added to the buffer at the given pixel coordinate @ref pixel
     */
    void AddSample(const Point2i pixel, const Sample &sample);

    /**
     * @brief Stores the ImageSpaceGuidingBuffer into a multi-channel EXR file.
     *
     * @param fileName
     */
    void Store(const std::string &fileName) const;

    /**
     * @brief Returns the estimate of the contirbution (i.e., expected value) of a pixel.
     *
     * This quantity is usefull guided/adjoint-driven Russina roulette decisions.
     */
    Vector3f GetContributionEstimate(const Point2i pixel) const;

#if defined(OPENPGL_VSP_GUIDING)
    float GetVolumeScatterProbabilityEstimate(const Point2i pixel) const;
#endif

    /**
     * @brief If the image-space guiding buffer is ready and can be used.
     * (e.g., when loaded from file or after the first Update step).
     */
    bool IsReady() const;

    /**
     * @brief Resets the ImageSpaceGuidingBuffer (e.g., if the camera is moved or the scene changed).
     */
    void Reset();

   private:
    PGLImageSpaceGuidingBuffer m_imageSpaceGuidingBufferHandle{nullptr};
};

////////////////////////////////////////////////////////////
/// Implementation
////////////////////////////////////////////////////////////

OPENPGL_INLINE ImageSpaceGuidingBuffer::ImageSpaceGuidingBuffer(const Config cfg)
{
    m_imageSpaceGuidingBufferHandle = pglFieldNewImageSpaceGuidingBuffer(cfg.data);
}

OPENPGL_INLINE ImageSpaceGuidingBuffer::ImageSpaceGuidingBuffer(const std::string &fileName)
{
    m_imageSpaceGuidingBufferHandle = pglFieldNewImageSpaceGuidingBufferFromFile(fileName.c_str());
}

OPENPGL_INLINE ImageSpaceGuidingBuffer::~ImageSpaceGuidingBuffer()
{
    OPENPGL_ASSERT(m_imageSpaceGuidingBufferHandle);
    if (m_imageSpaceGuidingBufferHandle)
        pglReleaseImageSpaceGuidingBuffer(m_imageSpaceGuidingBufferHandle);
    m_imageSpaceGuidingBufferHandle = nullptr;
}

OPENPGL_INLINE void ImageSpaceGuidingBuffer::Update()
{
    OPENPGL_ASSERT(m_imageSpaceGuidingBufferHandle);
    pglImageSpaceGuidingBufferUpdate(m_imageSpaceGuidingBufferHandle);
}

OPENPGL_INLINE void ImageSpaceGuidingBuffer::AddSample(const Point2i pixel, const Sample &sample)
{
    OPENPGL_ASSERT(m_imageSpaceGuidingBufferHandle);
    pglImageSpaceGuidingBufferAddSample(m_imageSpaceGuidingBufferHandle, pixel, sample);
}

OPENPGL_INLINE void ImageSpaceGuidingBuffer::Store(const std::string &fileName) const
{
    OPENPGL_ASSERT(m_imageSpaceGuidingBufferHandle);
    pglImageSpaceGuidingBufferStore(m_imageSpaceGuidingBufferHandle, fileName.c_str());
}

OPENPGL_INLINE Vector3f ImageSpaceGuidingBuffer::GetContributionEstimate(const Point2i pixel) const
{
    OPENPGL_ASSERT(m_imageSpaceGuidingBufferHandle);
    return pglImageSpaceGuidingBufferGetContributionEstimate(m_imageSpaceGuidingBufferHandle, pixel);
}

#if defined(OPENPGL_VSP_GUIDING)
OPENPGL_INLINE float ImageSpaceGuidingBuffer::GetVolumeScatterProbabilityEstimate(const Point2i pixel) const
{
    OPENPGL_ASSERT(m_imageSpaceGuidingBufferHandle);
    return pglImageSpaceGuidingBufferGetVolumeScatterProbabilityEstimate(m_imageSpaceGuidingBufferHandle, pixel);
}
#endif

OPENPGL_INLINE bool ImageSpaceGuidingBuffer::IsReady() const
{
    OPENPGL_ASSERT(m_imageSpaceGuidingBufferHandle);
    return pglImageSpaceGuidingBufferIsReady(m_imageSpaceGuidingBufferHandle);
}

OPENPGL_INLINE void ImageSpaceGuidingBuffer::Reset()
{
    OPENPGL_ASSERT(m_imageSpaceGuidingBufferHandle);
    pglImageSpaceGuidingBufferReset(m_imageSpaceGuidingBufferHandle);
}

}  // namespace util
}  // namespace cpp
}  // namespace openpgl