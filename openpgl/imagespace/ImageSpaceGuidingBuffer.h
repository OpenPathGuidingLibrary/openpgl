// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../openpgl_common.h"
#include "Denoiser.h"

#ifdef USE_EMBREE_PARALLEL
#define TASKING_TBB
#include <embreeSrc/common/algorithms/parallel_for.h>
#else
#include <tbb/parallel_for.h>
#endif

// Please include your own zlib-compatible API header before
// including `tinyexr.h` when you disable `TINYEXR_USE_MINIZ`
#define TINYEXR_USE_MINIZ 0
#define TINYEXR_USE_NANOZLIB 1
// #include "zlib.h"
// Or, if your project uses `stb_image[_write].h`, use their
// zlib implementation:
// #define TINYEXR_USE_STB_ZLIB 1
// #define TINYEXR_USE_THREAD 1
#define TINYEXR_IMPLEMENTATION
#include <tinyexr/tinyexr.h>

#define VSP_USE_PVOL_EST

namespace openpgl
{

struct ImageSpaceGuidingBuffer
{
    struct Buffers
    {
        Buffers(pgl_point2i resolution) : numPixels(resolution.x * resolution.y)
        {
            contribution = new pgl_vec3f[numPixels];

            albedo = new pgl_vec3f[numPixels];
            normal = new pgl_vec3f[numPixels];
            spp = new float[numPixels];

            filteredContribution = new pgl_vec3f[numPixels];
        }

        Buffers(const Buffers &buffer) = delete;

        Buffers &operator=(const Buffers &) = delete;

        ~Buffers()
        {
            delete[] contribution;

            delete[] albedo;
            delete[] normal;
            delete[] spp;

            delete[] filteredContribution;
        }

        void reset()
        {
#ifdef USE_EMBREE_PARALLEL
            embree::parallel_for(0, (int)numPixels, 1, [&](const embree::range<unsigned> &r) {
                for (size_t pIdx = r.begin(); pIdx < r.end(); pIdx++)
#else
            tbb::parallel_for(tbb::blocked_range<int>(0, numPixels), [&](tbb::blocked_range<int> r) {
                for (int pIdx = r.begin(); pIdx < r.end(); ++pIdx)
#endif
                {
                    contribution[pIdx] = {0.f, 0.f, 0.f};

                    albedo[pIdx] = {0.f, 0.f, 0.f};
                    normal[pIdx] = {0.f, 0.f, 0.f};
                    spp[pIdx] = 0.f;

                    filteredContribution[pIdx] = {0.f, 0.f, 0.f};
                }
            });
        }

        int numPixels{0};

        pgl_vec3f *contribution{nullptr};

        pgl_vec3f *albedo{nullptr};
        pgl_vec3f *normal{nullptr};
        float *spp{nullptr};

        pgl_vec3f *filteredContribution{nullptr};
    };

    ImageSpaceGuidingBuffer(PGLImageSpaceGuidingBufferConfig cfg) : m_cfg(cfg)
    {
        m_resolution = cfg.resolution;
        m_denoiser = new Denoiser(cfg.resolution, false);
        if (m_cfg.contributionEstimate)
        {
            m_contributionEstimateBuffers = new Buffers(cfg.resolution);
        }
#if defined(OPENPGL_VSP_GUIDING)
        if (m_cfg.vspEstimate)
        {
            m_surfaceContributionEstimateBuffers = new Buffers(cfg.resolution);
            m_volumeContributionEstimateBuffers = new Buffers(cfg.resolution);
            m_pVolBuffer = new float[m_resolution.x * m_resolution.y];
            m_filteredPVolBuffer = new float[m_resolution.x * m_resolution.y];
            m_vspContributionBuffer = new float[m_resolution.x * m_resolution.y];
        }
#endif
        this->reset();
        m_ready = false;
    }

    ImageSpaceGuidingBuffer(const ImageSpaceGuidingBuffer &isgbuffer) = delete;

    ImageSpaceGuidingBuffer &operator=(const ImageSpaceGuidingBuffer &) = delete;

    ImageSpaceGuidingBuffer(const std::string &fileName)
    {
        EXRVersion exrVersion;
        EXRHeader exrHeader;
        EXRImage exrImage;

        InitEXRHeader(&exrHeader);
        InitEXRImage(&exrImage);

        if (IsEXR(fileName.c_str()) == TINYEXR_SUCCESS)
        {
            m_ready = true;

            ParseEXRVersionFromFile(&exrVersion, fileName.c_str());
            ParseEXRHeaderFromFile(&exrHeader, &exrVersion, fileName.c_str(), nullptr);
            LoadEXRImageFromFile(&exrImage, &exrHeader, fileName.c_str(), nullptr);

            m_resolution.x = exrImage.width;
            m_resolution.y = exrImage.height;

            const size_t numPixels = m_resolution.x * m_resolution.y;

            m_denoiser = new Denoiser(m_resolution, false);

            std::vector<std::string> layer_names;
            tinyexr::GetLayers(exrHeader, layer_names);

            // identify config based on the stored layers
            m_cfg.contributionEstimate = false;
            m_cfg.contributionType = PGLContributionTypes::EContribContribution;
#if defined(OPENPGL_VSP_GUIDING)
            m_cfg.vspEstimate = false;
            m_cfg.vspType = PGLVSPTypes::EVSPContribution;
#endif
            for (int i = 0; i < layer_names.size(); i++)
            {
                std::string layerName = layer_names[i];
                if (layerName == "Contrib" || layerName == "Contrib2nd")
                {
                    m_cfg.contributionEstimate = true;
                    if (layerName == "Contrib2nd")
                    {
                        m_cfg.contributionType = PGLContributionTypes::EContribVariance;
                    }
                }
#if defined(OPENPGL_VSP_GUIDING)
                if (layerName == "surfContrib" || layerName == "surfContrib2nd")
                {
                    m_cfg.vspEstimate = true;
                    if (layerName == "surfContrib2nd")
                    {
                        m_cfg.vspType = PGLVSPTypes::EVSPVariance;
                    }
                }
#endif
            }

            if (m_cfg.contributionEstimate)
            {
                m_contributionEstimateBuffers = new Buffers(m_resolution);
            }

            m_cfg.resolution = m_resolution;
#if defined(OPENPGL_VSP_GUIDING)
            if (m_cfg.vspEstimate)
            {
                m_surfaceContributionEstimateBuffers = new Buffers(m_resolution);
                m_volumeContributionEstimateBuffers = new Buffers(m_resolution);
                m_pVolBuffer = new float[m_resolution.x * m_resolution.y];
                m_filteredPVolBuffer = new float[m_resolution.x * m_resolution.y];
                m_vspContributionBuffer = new float[m_resolution.x * m_resolution.y];
            }
#endif

            for (int i = 0; i < layer_names.size(); i++)
            {
                std::string layerName = layer_names[i];
                std::vector<tinyexr::LayerChannel> channels;
                tinyexr::ChannelsInLayer(exrHeader, layerName, channels);

                int nChannels = channels.size();
                int idxR = -1;
                int idxG = -1;
                int idxB = -1;
                // int idxA = -1;

                int idxY = -1;

                for (int j = 0; j < nChannels; j++)
                {
                    const tinyexr::LayerChannel &ch = channels[j];
                    if (ch.name == "R")
                    {
                        idxR = int(ch.index);
                    }
                    else if (ch.name == "G")
                    {
                        idxG = int(ch.index);
                    }
                    else if (ch.name == "B")
                    {
                        idxB = int(ch.index);
                        //} else if (ch.name == "A") {
                        //    idxA = int(ch.index);
                    }
                    else if (ch.name == "Y")
                    {
                        idxY = int(ch.index);
                    }
                }

                float *bufferPtr = nullptr;
                if (layerName == "Contrib" || layerName == "Contrib2nd")
                {
                    bufferPtr = (float *)m_contributionEstimateBuffers->filteredContribution;
                }
                else if (layerName == "Spp")
                {
                    bufferPtr = (float *)m_contributionEstimateBuffers->spp;
                }
                else if (layerName == "ContribRaw" || layerName == "Contrib2ndRaw")
                {
                    bufferPtr = (float *)m_contributionEstimateBuffers->contribution;
                }
                else if (layerName == "Albedo")
                {
                    bufferPtr = (float *)m_contributionEstimateBuffers->albedo;
                }
                else if (layerName == "N")
                {
                    bufferPtr = (float *)m_contributionEstimateBuffers->normal;
                }
#if defined(OPENPGL_VSP_GUIDING)
                if (layerName == "surfContrib" || layerName == "surfContrib2nd")
                {
                    bufferPtr = (float *)m_surfaceContributionEstimateBuffers->filteredContribution;
                }
                else if (layerName == "surfSpp")
                {
                    bufferPtr = (float *)m_surfaceContributionEstimateBuffers->spp;
                }
                else if (layerName == "surfContribRaw" || layerName == "surfContrib2ndRaw")
                {
                    bufferPtr = (float *)m_surfaceContributionEstimateBuffers->contribution;
                }
                else if (layerName == "surfAlbedo")
                {
                    bufferPtr = (float *)m_surfaceContributionEstimateBuffers->albedo;
                }
                else if (layerName == "surfN")
                {
                    bufferPtr = (float *)m_surfaceContributionEstimateBuffers->normal;
                }
                else if (layerName == "volContrib" || layerName == "volContrib2nd")
                {
                    bufferPtr = (float *)m_volumeContributionEstimateBuffers->filteredContribution;
                }
                else if (layerName == "volSpp")
                {
                    bufferPtr = (float *)m_volumeContributionEstimateBuffers->spp;
                }
                else if (layerName == "volContribRaw" || layerName == "volContrib2ndRaw")
                {
                    bufferPtr = (float *)m_volumeContributionEstimateBuffers->contribution;
                }
                else if (layerName == "volAlbedo")
                {
                    bufferPtr = (float *)m_volumeContributionEstimateBuffers->albedo;
                }
                else if (layerName == "volN")
                {
                    bufferPtr = (float *)m_volumeContributionEstimateBuffers->normal;
                }
                else if (layerName == "VSP")
                {
                    bufferPtr = (float *)m_vspContributionBuffer;
                }
                else if (layerName == "PVolRAW")
                {
                    bufferPtr = (float *)m_pVolBuffer;
                }
                else if (layerName == "PVol")
                {
                    bufferPtr = (float *)m_filteredPVolBuffer;
                }
#endif
                else
                {
                }

#ifdef USE_EMBREE_PARALLEL
                embree::parallel_for(0, (int)numPixels, 1, [&](const embree::range<unsigned> &r) {
                    for (size_t pIdx = r.begin(); pIdx < r.end(); pIdx++)
#else
                tbb::parallel_for(tbb::blocked_range<int>(0, numPixels), [&](tbb::blocked_range<int> r) {
                    for (int pIdx = r.begin(); pIdx < r.end(); ++pIdx)
#endif
                    {
                        switch (nChannels)
                        {
                            case 1:
                            {
                                const float val = reinterpret_cast<float **>(exrImage.images)[idxY][pIdx];

                                bufferPtr[pIdx * nChannels + 0] = val;
                                break;
                            }
                            case 3:
                            {
                                const float valR = reinterpret_cast<float **>(exrImage.images)[idxR][pIdx];
                                const float valG = reinterpret_cast<float **>(exrImage.images)[idxG][pIdx];
                                const float valB = reinterpret_cast<float **>(exrImage.images)[idxB][pIdx];

                                bufferPtr[pIdx * nChannels + 0] = valR;
                                bufferPtr[pIdx * nChannels + 1] = valG;
                                bufferPtr[pIdx * nChannels + 2] = valB;
                                break;
                            }
                        }
                    }
                });
            }
            m_ready = true;
        }
        else
        {
            m_ready = false;
        }
    }

    ~ImageSpaceGuidingBuffer()
    {
        delete m_denoiser;
        delete m_contributionEstimateBuffers;
#if defined(OPENPGL_VSP_GUIDING)
        delete m_surfaceContributionEstimateBuffers;
        delete m_volumeContributionEstimateBuffers;
        delete m_pVolBuffer;
        delete m_filteredPVolBuffer;
        delete m_vspContributionBuffer;
#endif
    }

    void store(const std::string &fileName) const
    {
        EXRHeader exrHeader;
        EXRImage exrImage;

        InitEXRHeader(&exrHeader);
        InitEXRImage(&exrImage);

        int width = m_resolution.x;
        int height = m_resolution.y;

        std::vector<int> numChannels;
        std::vector<tinyexr::LayerChannel> layerChannels;
        std::vector<const float *> channelValues;
        int cIdx = 0;
        if (m_cfg.contributionEstimate)
        {
            cIdx = layerChannels.size();
            if (m_cfg.contributionType == PGLContributionTypes::EContribContribution)
            {
                layerChannels.emplace_back(cIdx, "Contrib");
            }
            else
            {
                layerChannels.emplace_back(cIdx, "Contrib2nd");
            }
            numChannels.emplace_back(3);
            channelValues.emplace_back((const float *)m_contributionEstimateBuffers->filteredContribution);

            cIdx = layerChannels.size();
            layerChannels.emplace_back(cIdx, "Spp");
            numChannels.emplace_back(1);
            channelValues.emplace_back((const float *)m_contributionEstimateBuffers->spp);

            cIdx = layerChannels.size();
            if (m_cfg.contributionType == PGLContributionTypes::EContribContribution)
            {
                layerChannels.emplace_back(cIdx, "ContribRaw");
            }
            else
            {
                layerChannels.emplace_back(cIdx, "Contrib2ndRaw");
            }
            numChannels.emplace_back(3);
            channelValues.emplace_back((const float *)m_contributionEstimateBuffers->contribution);

            cIdx = layerChannels.size();
            layerChannels.emplace_back(cIdx, "Albedo");
            numChannels.emplace_back(3);
            channelValues.emplace_back((const float *)m_contributionEstimateBuffers->albedo);

            cIdx = layerChannels.size();
            layerChannels.emplace_back(cIdx, "N");
            numChannels.emplace_back(3);
            channelValues.emplace_back((const float *)m_contributionEstimateBuffers->normal);
        }
#if defined(OPENPGL_VSP_GUIDING)
        if (m_cfg.vspEstimate)
        {
            cIdx = layerChannels.size();
            if (m_cfg.vspType == PGLVSPTypes::EVSPContribution)
            {
                layerChannels.emplace_back(cIdx, "surfContrib");
            }
            else
            {
                layerChannels.emplace_back(cIdx, "surfContrib2nd");
            }
            numChannels.emplace_back(3);
            channelValues.emplace_back((const float *)m_surfaceContributionEstimateBuffers->filteredContribution);

            cIdx = layerChannels.size();
            layerChannels.emplace_back(cIdx, "surfSpp");
            numChannels.emplace_back(1);
            channelValues.emplace_back((const float *)m_surfaceContributionEstimateBuffers->spp);

            cIdx = layerChannels.size();
            if (m_cfg.vspType == PGLVSPTypes::EVSPContribution)
            {
                layerChannels.emplace_back(cIdx, "surfContribRaw");
            }
            else
            {
                layerChannels.emplace_back(cIdx, "surfContrib2ndRaw");
            }
            numChannels.emplace_back(3);
            channelValues.emplace_back((const float *)m_surfaceContributionEstimateBuffers->contribution);

            cIdx = layerChannels.size();
            layerChannels.emplace_back(cIdx, "surfAlbedo");
            numChannels.emplace_back(3);
            channelValues.emplace_back((const float *)m_surfaceContributionEstimateBuffers->albedo);

            cIdx = layerChannels.size();
            layerChannels.emplace_back(cIdx, "surfN");
            numChannels.emplace_back(3);
            channelValues.emplace_back((const float *)m_surfaceContributionEstimateBuffers->normal);

            cIdx = layerChannels.size();
            if (m_cfg.vspType == PGLVSPTypes::EVSPContribution)
            {
                layerChannels.emplace_back(cIdx, "volContrib");
            }
            else
            {
                layerChannels.emplace_back(cIdx, "volContrib2nd");
            }
            numChannels.emplace_back(3);
            channelValues.emplace_back((const float *)m_volumeContributionEstimateBuffers->filteredContribution);

            cIdx = layerChannels.size();
            layerChannels.emplace_back(cIdx, "volSpp");
            numChannels.emplace_back(1);
            channelValues.emplace_back((const float *)m_volumeContributionEstimateBuffers->spp);

            cIdx = layerChannels.size();
            if (m_cfg.vspType == PGLVSPTypes::EVSPContribution)
            {
                layerChannels.emplace_back(cIdx, "volContribRaw");
            }
            else
            {
                layerChannels.emplace_back(cIdx, "volContrib2ndRaw");
            }
            numChannels.emplace_back(3);
            channelValues.emplace_back((const float *)m_volumeContributionEstimateBuffers->contribution);

            cIdx = layerChannels.size();
            layerChannels.emplace_back(cIdx, "volAlbedo");
            numChannels.emplace_back(3);
            channelValues.emplace_back((const float *)m_volumeContributionEstimateBuffers->albedo);

            cIdx = layerChannels.size();
            layerChannels.emplace_back(cIdx, "volN");
            numChannels.emplace_back(3);
            channelValues.emplace_back((const float *)m_volumeContributionEstimateBuffers->normal);

            cIdx = layerChannels.size();
            layerChannels.emplace_back(cIdx, "VSP");
            numChannels.emplace_back(1);
            channelValues.emplace_back((const float *)m_vspContributionBuffer);

            cIdx = layerChannels.size();
            layerChannels.emplace_back(cIdx, "PVolRAW");
            numChannels.emplace_back(1);
            channelValues.emplace_back((const float *)m_pVolBuffer);

            cIdx = layerChannels.size();
            layerChannels.emplace_back(cIdx, "PVol");
            numChannels.emplace_back(1);
            channelValues.emplace_back((const float *)m_filteredPVolBuffer);
        }
#endif

        int totalNumLayers = layerChannels.size();
        int totalNumChannels = 0;
        for (int i = 0; i < numChannels.size(); i++)
        {
            totalNumChannels += numChannels[i];
        }

        exrHeader.num_channels = totalNumChannels;
        exrImage.num_channels = totalNumChannels;
        exrImage.width = width;
        exrImage.height = height;

        const size_t numPixels = width * height;

        // Convert image data from AoS to SoA (i.e. one image per channel in each layer)
        std::vector<std::vector<float>> channelImages;
        for (int i = 0; i < totalNumLayers; i++)
        {
            for (int j = 0; j < numChannels[i]; j++)
            {
                channelImages.emplace_back(width * height);
            }
            int offset = channelImages.size() - numChannels[i];

#ifdef USE_EMBREE_PARALLEL
            embree::parallel_for(0, (int)numPixels, 1, [&](const embree::range<unsigned> &r) {
                for (size_t pIdx = r.begin(); pIdx < r.end(); pIdx++)
#else
            tbb::parallel_for(tbb::blocked_range<int>(0, numPixels), [&](tbb::blocked_range<int> r) {
                for (int pIdx = r.begin(); pIdx < r.end(); ++pIdx)
#endif
                {
                    for (int c = 0; c < numChannels[i]; c++)
                    {
                        channelImages[offset + c][pIdx] = channelValues[i][(pIdx)*numChannels[i] + c];
                    }
                }
            });
        }

        // Set the channel names
        std::vector<EXRChannelInfo> channels(totalNumChannels);
        int offset = 0;
        for (int lay = 0; lay < totalNumLayers; ++lay)
        {
            char namePrefix[256];
            std::size_t prefixLen = 0;
            const char *layerName = layerChannels[lay].name.c_str();
            if (!layerName || layerName[0] == 0)
            {
                prefixLen = 0;
                namePrefix[0] = 0;
            }
            else
            {
                prefixLen = strlen(layerName) + 1;
                strncpy(namePrefix, layerName, 255);
                namePrefix[prefixLen - 1] = '.';
            }

            if (numChannels[lay] == 1)
            {
                strncpy(channels[offset + 0].name, namePrefix, 255);
                strncpy(channels[offset + 0].name + prefixLen, "Y", 255 - prefixLen);
            }
            else if (numChannels[lay] == 3)
            {
                strncpy(channels[offset + 0].name, namePrefix, 255);
                strncpy(channels[offset + 0].name + prefixLen, "R", 255 - prefixLen);

                strncpy(channels[offset + 1].name, namePrefix, 255);
                strncpy(channels[offset + 1].name + prefixLen, "G", 255 - prefixLen);

                strncpy(channels[offset + 2].name, namePrefix, 255);
                strncpy(channels[offset + 2].name + prefixLen, "B", 255 - prefixLen);
                /*
                } else if (numChannels[lay] == 4) {
                    strncpy(channels[offset + 0].name, namePrefix, 255);
                    strncpy(channels[offset + 0].name + prefixLen, "R", 255 - prefixLen);

                    strncpy(channels[offset + 1].name, namePrefix, 255);
                    strncpy(channels[offset + 1].name + prefixLen, "G", 255 - prefixLen);

                    strncpy(channels[offset + 2].name, namePrefix, 255);
                    strncpy(channels[offset + 2].name + prefixLen, "B", 255 - prefixLen);

                    strncpy(channels[offset + 3].name, namePrefix, 255);
                    strncpy(channels[offset + 3].name + prefixLen, "A", 255 - prefixLen);
                */
            }
            else
            {
                std::cerr << "ERROR while writing " << fileName << ": images with " << totalNumChannels << " channels are currently not supported. " << "no file has been written."
                          << std::endl;
                return;
            }

            offset += numChannels[lay];
        }

        // Sort channels by the ASCII byte code of their names because thats what OpenEXR expects
        std::vector<int> channelIndices;
        for (int i = 0; i < totalNumChannels; ++i)
            channelIndices.emplace_back(i);
        std::sort(channelIndices.begin(), channelIndices.end(), [&channels](int a, int b) {
            return strcmp(channels[a].name, channels[b].name) < 0;
        });

        std::vector<EXRChannelInfo> sortedChannels(totalNumChannels);
        float **imagePtr = (float **)alloca(sizeof(float *) * exrImage.num_channels);
        for (int i = 0; i < totalNumChannels; ++i)
        {
            sortedChannels[i] = channels[channelIndices[i]];
            imagePtr[i] = channelImages[channelIndices[i]].data();
        }
        exrHeader.channels = sortedChannels.data();
        exrImage.images = (unsigned char **)imagePtr;

        bool writeHalf = false;
        // Define pixel type of the buffer and requested output pixel type in the file
        exrHeader.pixel_types = (int *)alloca(sizeof(int) * exrHeader.num_channels);
        exrHeader.requested_pixel_types = (int *)alloca(sizeof(int) * exrHeader.num_channels);
        for (int i = 0; i < exrHeader.num_channels; i++)
        {
            exrHeader.pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;
            exrHeader.requested_pixel_types[i] = writeHalf ? TINYEXR_PIXELTYPE_HALF : TINYEXR_PIXELTYPE_FLOAT;
        }

        const char *errorMsg = nullptr;
        int retCode = 0;
        retCode = SaveEXRImageToFile(&exrImage, &exrHeader, fileName.c_str(), &errorMsg);
        if (retCode != TINYEXR_SUCCESS)
        {
            std::cerr << "TinyEXR error (" << retCode << "): " << errorMsg << std::endl;
            FreeEXRErrorMessage(errorMsg);
        }
    }

    void update()
    {
        if (m_cfg.contributionEstimate)
        {
            m_denoiser->denoise(m_contributionEstimateBuffers->contribution, m_contributionEstimateBuffers->normal, m_contributionEstimateBuffers->albedo,
                                m_contributionEstimateBuffers->filteredContribution);
        }
#if defined(OPENPGL_VSP_GUIDING)
        if (m_cfg.vspEstimate)
        {
            const int numPixels = m_resolution.x * m_resolution.y;

#ifdef USE_EMBREE_PARALLEL
            embree::parallel_for(0, (int)numPixels, 1, [&](const embree::range<unsigned> &r) {
                for (size_t pIdx = r.begin(); pIdx < r.end(); pIdx++)
#else
            tbb::parallel_for(tbb::blocked_range<int>(0, numPixels), [&](tbb::blocked_range<int> r) {
                for (int pIdx = r.begin(); pIdx < r.end(); ++pIdx)
#endif
                {
                    const float surfaceSampleCount = m_surfaceContributionEstimateBuffers->spp[pIdx];
                    const float volumeSampleCount = m_volumeContributionEstimateBuffers->spp[pIdx];
                    const float pVolEst = volumeSampleCount / (surfaceSampleCount + volumeSampleCount);
                    m_pVolBuffer[pIdx] = pVolEst;
                }
            });

            m_denoiser->denoise(m_pVolBuffer, m_filteredPVolBuffer);
            m_denoiser->denoise(m_surfaceContributionEstimateBuffers->contribution, m_surfaceContributionEstimateBuffers->normal, m_surfaceContributionEstimateBuffers->albedo,
                                m_surfaceContributionEstimateBuffers->filteredContribution);
            m_denoiser->denoise(m_volumeContributionEstimateBuffers->contribution, m_volumeContributionEstimateBuffers->normal, m_volumeContributionEstimateBuffers->albedo,
                                m_volumeContributionEstimateBuffers->filteredContribution);

#ifdef USE_EMBREE_PARALLEL
            embree::parallel_for(0, (int)numPixels, 1, [&](const embree::range<unsigned> &r) {
                for (size_t pIdx = r.begin(); pIdx < r.end(); pIdx++)
                {
#else
            tbb::parallel_for(tbb::blocked_range<int>(0, numPixels), [&](tbb::blocked_range<int> r) {
                for (int pIdx = r.begin(); pIdx < r.end(); ++pIdx)
                {
#endif
                    float pVolEst = m_filteredPVolBuffer[pIdx];
                    pVolEst = std::max(0.f, std::min(1.f, pVolEst));
#ifndef VSP_USE_PVOL_EST
                    // If we add zero samples to the other buffers (e.g., to the volume buffer when we have a surface sample)
                    // Then we do not need to multiply with (1.f -pVolEst) or pVolEst.
                    // Note for the second moment look for the USE_PVOL_CORRECTION earlier whne calcualting the sqrt of the second moment.
                    pgl_vec3f surfaceContribution = m_surfaceContributionEstimateBuffers->filteredContribution[pIdx];
                    pgl_vec3f volumeContribution = m_volumeContributionEstimateBuffers->filteredContribution[pIdx];
#else
                    // If the surface/volume buffers only include volume or surface samples we have
                    // to correct with (1.f -pVolEst) and pVolEst.
                    // Not since we already use the sqrt of the second moment we only need to multiply
                    // with pVolEst and not pVolEst°2

                    pgl_vec3f surfaceContribution, volumeContribution;
                    if (m_cfg.vspType == PGLVSPTypes::EVSPContribution)
                    {
                        surfaceContribution = (1.f - pVolEst) * m_surfaceContributionEstimateBuffers->filteredContribution[pIdx];
                        volumeContribution = pVolEst * m_volumeContributionEstimateBuffers->filteredContribution[pIdx];
                    }
                    else
                    {
                        surfaceContribution.x = m_surfaceContributionEstimateBuffers->filteredContribution[pIdx].x > 0.f
                                                    ? (1.f - pVolEst) * std::sqrt(m_surfaceContributionEstimateBuffers->filteredContribution[pIdx].x)
                                                    : 0.f;
                        surfaceContribution.y = m_surfaceContributionEstimateBuffers->filteredContribution[pIdx].y > 0.f
                                                    ? (1.f - pVolEst) * std::sqrt(m_surfaceContributionEstimateBuffers->filteredContribution[pIdx].y)
                                                    : 0.f;
                        surfaceContribution.z = m_surfaceContributionEstimateBuffers->filteredContribution[pIdx].z > 0.f
                                                    ? (1.f - pVolEst) * std::sqrt(m_surfaceContributionEstimateBuffers->filteredContribution[pIdx].z)
                                                    : 0.f;

                        volumeContribution.x = m_volumeContributionEstimateBuffers->filteredContribution[pIdx].x > 0.f
                                                   ? pVolEst * std::sqrt(m_volumeContributionEstimateBuffers->filteredContribution[pIdx].x)
                                                   : 0.f;
                        volumeContribution.y = m_volumeContributionEstimateBuffers->filteredContribution[pIdx].y > 0.f
                                                   ? pVolEst * std::sqrt(m_volumeContributionEstimateBuffers->filteredContribution[pIdx].y)
                                                   : 0.f;
                        volumeContribution.z = m_volumeContributionEstimateBuffers->filteredContribution[pIdx].z > 0.f
                                                   ? pVolEst * std::sqrt(m_volumeContributionEstimateBuffers->filteredContribution[pIdx].z)
                                                   : 0.f;
                    }

#endif
                    pgl_vec3f contribution = surfaceContribution + volumeContribution;
                    float contributionScalar = pglVec3fMax(contribution);
                    float volumeContributionScalar = pglVec3fMax(volumeContribution);

                    m_vspContributionBuffer[pIdx] = contributionScalar > 0.f ? volumeContributionScalar / (contributionScalar) : -1.f;
                }
            });
        }
#endif
        m_ready = true;
    }

    void addSample(const pgl_point2i pixel, const PGLImageSpaceSample &sample)
    {
        std::size_t pixelIdx = pixel.y * m_resolution.x + pixel.x;
        if (m_cfg.contributionEstimate)
        {
            m_contributionEstimateBuffers->spp[pixelIdx] += 1;
            float alpha = 1.f / m_contributionEstimateBuffers->spp[pixelIdx];

            const pgl_vec3f quantity = m_cfg.contributionType == PGLContributionTypes::EContribContribution ? sample.contribution : sample.contribution * sample.contribution;
            m_contributionEstimateBuffers->contribution[pixelIdx] = (1.f - alpha) * m_contributionEstimateBuffers->contribution[pixelIdx] + alpha * sample.contribution;
            m_contributionEstimateBuffers->albedo[pixelIdx] = (1.f - alpha) * m_contributionEstimateBuffers->albedo[pixelIdx] + alpha * sample.albedo;
            m_contributionEstimateBuffers->normal[pixelIdx] = (1.f - alpha) * m_contributionEstimateBuffers->normal[pixelIdx] + alpha * sample.normal;
        }
#if defined(OPENPGL_VSP_GUIDING)
        if (m_cfg.vspEstimate)
        {
            if (sample.IsSurfaceEvent())
            {
                m_surfaceContributionEstimateBuffers->spp[pixelIdx] += 1;
#ifdef VSP_USE_PVOL_EST
                // calculating the alpha using only the number of surface samples
                float alpha = 1.f / m_surfaceContributionEstimateBuffers->spp[pixelIdx];
#else
                // calculating the alpha simulating we added zero samples for each volume sample as well
                float alpha = 1.f / (m_surfaceContributionEstimateBuffers->spp[pixelIdx] + m_volumeContributionEstimateBuffers->spp[pixelIdx]);
#endif
                pgl_vec3f quantity = m_cfg.vspType == EVSPVariance ? sample.contribution * sample.contribution : sample.contribution;
                m_surfaceContributionEstimateBuffers->contribution[pixelIdx] = (1.f - alpha) * m_surfaceContributionEstimateBuffers->contribution[pixelIdx] + alpha * quantity;
                m_surfaceContributionEstimateBuffers->albedo[pixelIdx] = (1.f - alpha) * m_surfaceContributionEstimateBuffers->albedo[pixelIdx] + alpha * sample.albedo;
                m_surfaceContributionEstimateBuffers->normal[pixelIdx] = (1.f - alpha) * m_surfaceContributionEstimateBuffers->normal[pixelIdx] + alpha * sample.normal;
#ifndef VSP_USE_PVOL_EST
                // adding zero value samples to the volume buffer
                m_volumeContributionEstimateBuffers->contribution[pixelIdx] = (1.f - alpha) * m_volumeContributionEstimateBuffers->contribution[pixelIdx];
#endif
            }
            else
            {
                m_volumeContributionEstimateBuffers->spp[pixelIdx] += 1;
#ifdef VSP_USE_PVOL_EST
                // calculating the alpha using only the number of volume samples
                float alpha = 1.f / m_volumeContributionEstimateBuffers->spp[pixelIdx];
#else
                // calculating the alpha simulating we added zero samples for each surface sample as well
                float alpha = 1.f / (m_surfaceContributionEstimateBuffers->spp[pixelIdx] + m_volumeContributionEstimateBuffers->spp[pixelIdx]);
#endif
                pgl_vec3f quantity = m_cfg.vspType == EVSPVariance ? sample.contribution * sample.contribution : sample.contribution;
                m_volumeContributionEstimateBuffers->contribution[pixelIdx] = (1.f - alpha) * m_volumeContributionEstimateBuffers->contribution[pixelIdx] + alpha * quantity;
                m_volumeContributionEstimateBuffers->albedo[pixelIdx] = (1.f - alpha) * m_volumeContributionEstimateBuffers->albedo[pixelIdx] + alpha * sample.albedo;
                m_volumeContributionEstimateBuffers->normal[pixelIdx] = (1.f - alpha) * m_volumeContributionEstimateBuffers->normal[pixelIdx] + alpha * sample.normal;
#ifndef VSP_USE_PVOL_EST
                // adding zero value samples to the surface buffer
                m_surfaceContributionEstimateBuffers->contribution[pixelIdx] = (1.f - alpha) * m_surfaceContributionEstimateBuffers->contribution[pixelIdx];
#endif
            }
        }
#endif
    }

    pgl_vec3f getContributionEstimate(const pgl_point2i pixel) const
    {
        if (!m_ready || !m_cfg.contributionEstimate)
        {
            return {0.f, 0.f, 0.f};
        }

        std::size_t pixelIdx = pixel.y * m_resolution.x + pixel.x;
        const pgl_vec3f c = m_contributionEstimateBuffers->filteredContribution[pixelIdx];
        return c;
    }

#if defined(OPENPGL_VSP_GUIDING)
    float getVolumeScatterProbabilityEstimate(const pgl_point2i pixel) const
    {
        if (!m_ready || !m_cfg.vspEstimate)
        {
            return 0.5f;
        }
        std::size_t pixelIdx = pixel.y * m_resolution.x + pixel.x;
        return m_vspContributionBuffer[pixelIdx];
    }
#endif
    bool isReady() const
    {
        return m_ready;
    }

    void reset()
    {
        if (m_cfg.contributionEstimate && m_contributionEstimateBuffers)
        {
            m_contributionEstimateBuffers->reset();
        }
#if defined(OPENPGL_VSP_GUIDING)
        if (m_cfg.vspEstimate)
        {
            if (m_surfaceContributionEstimateBuffers)
            {
                m_surfaceContributionEstimateBuffers->reset();
            }
            if (m_volumeContributionEstimateBuffers)
            {
                m_volumeContributionEstimateBuffers->reset();
            }
            const int numPixels = m_resolution.x * m_resolution.y;

#ifdef USE_EMBREE_PARALLEL
            embree::parallel_for(0, (int)numPixels, 1, [&](const embree::range<unsigned> &r) {
                for (size_t pIdx = r.begin(); pIdx < r.end(); pIdx++)
#else
            tbb::parallel_for(tbb::blocked_range<int>(0, numPixels), [&](tbb::blocked_range<int> r) {
                for (int pIdx = r.begin(); pIdx < r.end(); ++pIdx)
#endif
                {
                    m_pVolBuffer[pIdx] = 0.f;
                    m_filteredPVolBuffer[pIdx] = 0.f;
                    m_vspContributionBuffer[pIdx] = 0.f;
                }
            });
        }
#endif
        m_ready = false;
    }

   private:
    bool m_ready{false};
    PGLImageSpaceGuidingBufferConfig m_cfg;
    pgl_point2i m_resolution{0, 0};
    Denoiser *m_denoiser{nullptr};

    Buffers *m_contributionEstimateBuffers{nullptr};

#if defined(OPENPGL_VSP_GUIDING)
    Buffers *m_surfaceContributionEstimateBuffers{nullptr};
    Buffers *m_volumeContributionEstimateBuffers{nullptr};

    float *m_pVolBuffer{nullptr};
    float *m_filteredPVolBuffer{nullptr};
    float *m_vspContributionBuffer{nullptr};
#endif
};

}  // namespace openpgl