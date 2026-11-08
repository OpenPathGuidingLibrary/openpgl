// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../include/openpgl/compression.h"
#include "../openpgl_common.h"
#include "../spatial/Region.h"
#include "PathSegmentData.h"
#include "SampleData.h"
#include "SampleDataStorage.h"

#define OPENPGL_PATHSEGMENT_STORAGE_USE_ARRAY

namespace openpgl
{
struct PathSegmentDataStorage
{
    PathSegmentDataStorage() {}

    PathSegmentDataStorage(const PathSegmentDataStorage &) = delete;

    PathSegmentDataStorage &operator=(const PathSegmentDataStorage &) = delete;

    ~PathSegmentDataStorage()
    {
#if defined(OPENPGL_PATHSEGMENT_STORAGE_USE_ARRAY)
        if (m_segmentStorage)
            delete[] m_segmentStorage;

        if (m_sampleStorage)
            delete[] m_sampleStorage;

        if (m_zeroValueSampleStorage)
            delete[] m_zeroValueSampleStorage;
#endif
    };

   private:
    float m_max_distance = {1e6f};
#if defined(OPENPGL_PATHSEGMENT_STORAGE_USE_ARRAY)
    PathSegmentData *m_segmentStorage{nullptr};
    int m_seg_idx = {-1};
    int m_max_seg_size = {0};

    SampleData *m_sampleStorage{nullptr};
    int m_sample_idx = {-1};
    int m_max_sample_size = {0};

#ifdef OPENPGL_RADIANCE_CACHES
    bool m_track_zero_value_samples{true};
#else
    bool m_track_zero_value_samples{false};
#endif

    ZeroValueSampleData *m_zeroValueSampleStorage{nullptr};
    int m_zero_value_sample_idx = {-1};
    int m_max_zero_value_sample_size = {0};

#else
    std::vector<PathSegmentData> m_segmentStorage;
    std::vector<SampleData> m_sampleStorage;
    std::vector<ZeroValueSampleData> m_zeroValueSampleStorage;
#endif
   public:
    void reserve(const size_t &size)
    {
#if defined(OPENPGL_PATHSEGMENT_STORAGE_USE_ARRAY)
        if (m_max_sample_size == size)
            return;
        if (m_segmentStorage)
            delete[] m_segmentStorage;

        m_segmentStorage = new PathSegmentData[size];
        m_seg_idx = -1;
        m_max_seg_size = size;

        if (m_sampleStorage)
            delete[] m_sampleStorage;

        m_sampleStorage = new SampleData[size];
        m_sample_idx = -1;
        m_max_sample_size = size;

        if (m_zeroValueSampleStorage)
            delete[] m_zeroValueSampleStorage;

        m_zeroValueSampleStorage = new ZeroValueSampleData[size];
        m_zero_value_sample_idx = -1;
        m_max_zero_value_sample_size = size;

#else
        if (m_segmentStorage.size() == size)
            return;
        m_segmentStorage.reserve(size);
        m_sampleStorage.reserve(size);
        m_zeroValueSampleStorage.reserve(size);
#endif
    }

    size_t size()
    {
#if defined(OPENPGL_PATHSEGMENT_STORAGE_USE_ARRAY)
        return m_seg_idx + 1;
#else
        return m_segmentStorage.size();
#endif
    }

    void clear()
    {
#if defined(OPENPGL_PATHSEGMENT_STORAGE_USE_ARRAY)
        // std::cout << "PathSegmentDataStorage::clear: " << std::endl;
        m_seg_idx = -1;
        m_sample_idx = -1;
        m_zero_value_sample_idx = -1;
#else
        // m_segmentStorage.clear();
        // m_sampleStorage.clear();
        m_segmentStorage.resize(0);
        m_sampleStorage.resize(0);
        m_zeroValueSampleStorage.resize(0);
#endif
    }

    PathSegmentData *next()
    {
#if defined(OPENPGL_PATHSEGMENT_STORAGE_USE_ARRAY)
        if (m_seg_idx + 1 < m_max_seg_size)
        {
            m_seg_idx++;
            m_segmentStorage[m_seg_idx] = PathSegmentData();
            return &m_segmentStorage[m_seg_idx];
        }
        else
        {
            // std::cout << "PathSegmentDataStorage::next: idx = " << m_seg_idx << "max_size = " << m_max_seg_size << std::endl;
            return nullptr;
        }
#else
        m_segmentStorage.emplace_back();
        return &m_segmentStorage.back();
#endif
    }

    void addSegment(const PGLPathSegmentData &segment)
    {
#if defined(OPENPGL_PATHSEGMENT_STORAGE_USE_ARRAY)
        if (m_seg_idx + 1 < m_max_seg_size)
        {
            m_seg_idx++;
            m_segmentStorage[m_seg_idx] = segment;
        }
#else
        m_segmentStorage.push_back(segment);
#endif
    }

    void push_back(const PathSegmentData &psData)
    {
#if defined(OPENPGL_PATHSEGMENT_STORAGE_USE_ARRAY)
        if (m_seg_idx + 1 < m_max_seg_size)
        {
            m_seg_idx++;
            m_segmentStorage[m_seg_idx] = psData;
        }
#else
        m_segmentStorage.push_back(psData);
#endif
    }

    float getMaxDistance() const
    {
        return m_max_distance;
    }

    int getNumSegments() const
    {
#if defined(OPENPGL_PATHSEGMENT_STORAGE_USE_ARRAY)
        return m_seg_idx + 1;
#else
        return m_segmentStorage.size();
#endif
    }

    void setMaxDistance(const float maxDistance)
    {
        m_max_distance = maxDistance;
    }

    size_t prepareSamples(const bool useNEEMiWeights = false, const bool guideDirectLight = false, const bool rrAffectsDirectContribution = true)
    {
        // clamping thresholds to avoid generating extremly strong fireflies that
        // could mess-up the fitting of the distributions
        const float minPDF{0.01f};
        const openpgl::Vector3 maxThroughput{10.0f};

#if defined(OPENPGL_PATHSEGMENT_STORAGE_USE_ARRAY)
        size_t numSegments = m_seg_idx + 1;
#else
        size_t numSegments = m_segmentStorage.size();
#endif
        float lastDistance = 0.0f;
        // go revese from the start vertex of the last path segment (the end vertex of 2nd last segment)
        // towards the vertex of the first path segment
        for (int i = numSegments - 2; i >= 0; --i)
        {
            const openpgl::PathSegmentData &currentPathSegment = m_segmentStorage[i];
            float currentDistance = embree::length(openpgl::Point3(m_segmentStorage[i + 1].position.x, m_segmentStorage[i + 1].position.y, m_segmentStorage[i + 1].position.z) -
                                                   openpgl::Point3(currentPathSegment.position.x, currentPathSegment.position.y, currentPathSegment.position.z));
            // calcualte the distance to the source
            float distance = std::fmin(currentDistance + lastDistance, 2.0f * m_max_distance);

            ////// Calculating/updating the distance to the source of the incident radiance for the next vertex
            // if the current vertex is not a delta intercation or rough we reset the previous distacne
            // TODO: double check the roughness threshold at some point
            if (!currentPathSegment.isDelta && currentPathSegment.roughness >= 0.3f)
            {
                lastDistance = 0.0f;
            }
            // else if the curent vetex is a dirac/delta interaction update the previous distance
            else
            {
                lastDistance = distance;
                if (currentPathSegment.eta != 1.0f)
                {
                    float cosThetaI = embree::dot(openpgl::Vector3(currentPathSegment.normal.x, currentPathSegment.normal.y, currentPathSegment.normal.z),
                                                  openpgl::Vector3(currentPathSegment.directionOut.x, currentPathSegment.directionOut.y, currentPathSegment.directionOut.z));
                    float cosThetaO = embree::dot(openpgl::Vector3(currentPathSegment.normal.x, currentPathSegment.normal.y, currentPathSegment.normal.z),
                                                  openpgl::Vector3(currentPathSegment.directionIn.x, currentPathSegment.directionIn.y, currentPathSegment.directionIn.z));
                    lastDistance *= std::fabs(cosThetaI / (cosThetaO * currentPathSegment.eta));
                }
            }

            // we only collect samples on non delta surfaces and which are generated
            // using a non-direct (delta) sampling method
            if (!currentPathSegment.isDelta && currentPathSegment.roughness > 0.01f)
            {
                // prepare the current pos, direction, distance, pdf at the current
                // path vertex
                openpgl::Point3 pos = openpgl::Point3(currentPathSegment.position.x, currentPathSegment.position.y, currentPathSegment.position.z);
                // using the direction directly is numerically more stable than recalcuating
                // it using position of the next segment when the distance is small.
                openpgl::Vector3 dir = openpgl::Vector3(currentPathSegment.directionIn.x, currentPathSegment.directionIn.y, currentPathSegment.directionIn.z);
#ifdef OPENPGL_RADIANCE_CACHES
                openpgl::Vector3 dirOut = openpgl::Vector3(currentPathSegment.directionOut.x, currentPathSegment.directionOut.y, currentPathSegment.directionOut.z);
                openpgl::Vector3 scatteringWeight =
                    openpgl::Vector3(currentPathSegment.scatteringWeight.x, currentPathSegment.scatteringWeight.y, currentPathSegment.scatteringWeight.z);
                openpgl::Vector3 scatteredContribution =
                    openpgl::Vector3(currentPathSegment.scatteredContribution.x, currentPathSegment.scatteredContribution.y, currentPathSegment.scatteredContribution.z);
                openpgl::Vector3 directContribution =
                    openpgl::Vector3(currentPathSegment.directContribution.x, currentPathSegment.directContribution.y, currentPathSegment.directContribution.z);
#endif
                float pdf = std::max(minPDF, currentPathSegment.pdfDirectionIn);
                uint32_t flags{0};

                bool insideVolume = currentPathSegment.volumeScatter;
                if (insideVolume)
                {
                    flags |= SampleData::EInsideVolume;
                }

                if (m_segmentStorage[i + 1].volumeScatter)
                {
                    flags |= SampleData::ENextEventVolume;
                }

                bool directLightSample = false;
#ifdef OPENPGL_RADIANCE_CACHES
                float misWeight = 1.f;
#endif
                // evalaute the incident radiance the incident
                openpgl::Vector3 throughput{1.0f};
                openpgl::Vector3 contribution{0.0f};
                float previousRRSurvivalProb = 1.0f;
                for (size_t j = i + 1; j < numSegments; ++j)
                {
                    const openpgl::PathSegmentData &nextPathSegment = m_segmentStorage[j];

                    throughput = throughput * openpgl::Vector3(m_segmentStorage[j - 1].transmittanceWeight.x, m_segmentStorage[j - 1].transmittanceWeight.y,
                                                               m_segmentStorage[j - 1].transmittanceWeight.z);
                    OPENPGL_ASSERT(embree::isvalid(throughput));
                    OPENPGL_ASSERT(throughput[0] >= 0.f && throughput[1] >= 0.f && throughput[2] >= 0.f)
                    openpgl::Vector3 clampedThroughput = embree::min(throughput, maxThroughput);
                    contribution += clampedThroughput *
                                    openpgl::Vector3(nextPathSegment.scatteredContribution.x, nextPathSegment.scatteredContribution.y, nextPathSegment.scatteredContribution.z);
                    OPENPGL_ASSERT(embree::isvalid(contribution));
                    OPENPGL_ASSERT(contribution[0] >= 0.f && contribution[1] >= 0.f && contribution[2] >= 0.f);

                    openpgl::Vector3 directContribution =
                        openpgl::Vector3(nextPathSegment.directContribution.x, nextPathSegment.directContribution.y, nextPathSegment.directContribution.z);
                    // Todo: need explanation
                    if (!rrAffectsDirectContribution)
                    {
                        directContribution *= previousRRSurvivalProb;
                    }
                    if (directContribution[0] > 0.f || directContribution[1] > 0.f || directContribution[2] > 0.f)
                    {
                        if (j == i + 1)
                        {
                            if (contribution[0] > 0.f || contribution[1] > 0.f || contribution[2] > 0.f)
                            {
#ifdef OPENPGL_SHOW_PRINT_OUTS
                                std::cout << "scateredContribution" << std::endl;
#endif
                            }
                            else
                            {
                                directLightSample = true;
                            }
                            if (guideDirectLight)
                            {
                                if (!useNEEMiWeights)
                                {
                                    contribution += clampedThroughput * directContribution;
                                    OPENPGL_ASSERT(embree::isvalid(contribution));
                                    OPENPGL_ASSERT(contribution[0] >= 0.f && contribution[1] >= 0.f && contribution[2] >= 0.f);
                                }
                                else
                                {
                                    contribution += clampedThroughput * nextPathSegment.miWeight * directContribution;
#ifdef OPENPGL_RADIANCE_CACHES
                                    misWeight = nextPathSegment.miWeight;
#endif
                                    OPENPGL_ASSERT(embree::isvalid(contribution));
                                    OPENPGL_ASSERT(contribution[0] >= 0.f && contribution[1] >= 0.f && contribution[2] >= 0.f);
                                }
                            }
                        }
                        else
                        {
                            contribution += clampedThroughput * nextPathSegment.miWeight * directContribution;
                            OPENPGL_ASSERT(embree::isvalid(contribution));
                            OPENPGL_ASSERT(contribution[0] >= 0.f && contribution[1] >= 0.f && contribution[2] >= 0.f);
                        }
                    }
                    throughput = throughput * openpgl::Vector3(nextPathSegment.scatteringWeight.x, nextPathSegment.scatteringWeight.y, nextPathSegment.scatteringWeight.z);
                    if (nextPathSegment.russianRouletteSurvivalProbability > 0.f)
                    {
                        throughput /= nextPathSegment.russianRouletteSurvivalProbability;
                    }
                    else
                    {
                        throughput = openpgl::Vector3(0.f);
                    }
                    previousRRSurvivalProb = nextPathSegment.russianRouletteSurvivalProbability;

                    OPENPGL_ASSERT(embree::isvalid(throughput));
                    OPENPGL_ASSERT(throughput[0] >= 0.f && throughput[1] >= 0.f && throughput[2] >= 0.f)
                }
                if (directLightSample)
                {
                    flags |= SampleData::EDirectLight;
                }

                OPENPGL_ASSERT(embree::isvalid(contribution));
                OPENPGL_ASSERT(contribution[0] >= 0.f && contribution[1] >= 0.f && contribution[2] >= 0.f);
                if (contribution[0] > 0.0f || contribution[1] > 0.0f || contribution[2] > 0.0f)
                {
                    OPENPGL_ASSERT(embree::isvalid(distance));
                    if (distance > 0)
                    {
                        const float weight = OPENPGL_SPECTRUM_TO_FLOAT(contribution) / pdf;
                        OPENPGL_ASSERT(embree::isvalid(weight));
                        OPENPGL_ASSERT(weight >= 0.f);
                        SampleData dsd;
                        dsd.position = {pos[0], pos[1], pos[2]};
                        pgl_vec3f pglDirection = {dir[0], dir[1], dir[2]};
                        dsd.direction = pglDirection;
                        dsd.weight = weight;
#ifdef OPENPGL_RADIANCE_CACHES
                        pgl_vec3f pgl_directionOut = {dirOut[0], dirOut[1], dirOut[2]};
                        dsd.directionOut = pgl_directionOut;

                        pgl_vec3f colorOut;
                        colorOut.x = /*directContribution.x*/ +scatteredContribution.x + scatteringWeight.x * contribution[0];
                        colorOut.y = /*directContribution.y*/ +scatteredContribution.y + scatteringWeight.y * contribution[1];
                        colorOut.z = /*directContribution.z*/ +scatteredContribution.z + scatteringWeight.z * contribution[2];
                        dsd.radianceOut = colorOut;
                        pgl_vec3f colorIn = {contribution[0], contribution[1], contribution[2]};
                        dsd.radianceIn = colorIn;
                        dsd.radianceInMISWeight = misWeight;
#endif
                        dsd.pdf = pdf;
                        dsd.distance = distance;
                        dsd.flags = flags;
#if defined(OPENPGL_PATHSEGMENT_STORAGE_USE_ARRAY)
                        if (m_sample_idx + 1 <= m_max_sample_size)
                        {
                            m_sample_idx++;
                            m_sampleStorage[m_sample_idx] = dsd;
                        }
#else
                        m_sampleStorage.emplace_back(dsd);
#endif
                    }
                }
                else if (m_track_zero_value_samples)
                {
                    ZeroValueSampleData isd;
                    isd.position = {pos[0], pos[1], pos[2]};
                    pgl_vec3f pglDirection = {dir[0], dir[1], dir[2]};
                    isd.direction = pglDirection;
#ifdef OPENPGL_RADIANCE_CACHES
                    pglDirection = {dirOut[0], dirOut[1], dirOut[2]};
                    isd.directionOut = pglDirection;
#endif
                    isd.flags = flags;
#if defined(OPENPGL_PATHSEGMENT_STORAGE_USE_ARRAY)
                    if (m_zero_value_sample_idx + 1 <= m_max_zero_value_sample_size)
                    {
                        m_zero_value_sample_idx++;
                        m_zeroValueSampleStorage[m_zero_value_sample_idx] = isd;
                    }
#else
                    m_zeroValueSampleStorage.emplace_back(isd);
#endif
                }
            }
        }

#if defined(OPENPGL_PATHSEGMENT_STORAGE_USE_ARRAY)
        return m_sample_idx + 1;
#else
        return m_sampleStorage.size();
#endif
    }

    pgl_vec3f calculatePixelEstimate(const bool rrAffectsDirectContribution = true)
    {
#if defined(OPENPGL_PATHSEGMENT_STORAGE_USE_ARRAY)
        size_t numSegments = m_seg_idx + 1;
#else
        size_t numSegments = m_segmentStorage.size();
#endif

        pgl_vec3f finalColor;
        finalColor.x = 0.f;
        finalColor.y = 0.f;
        finalColor.z = 0.f;

        if (numSegments == 0)
            return finalColor;

        // evalaute the incident radiance the incident
        openpgl::Vector3 throughput{1.0f};
        openpgl::Vector3 contribution{0.0f};
        float previousRRSurvivalProb = m_segmentStorage[0].russianRouletteSurvivalProbability;
        for (size_t j = 0 + 1; j < numSegments; ++j)
        {
            const openpgl::PathSegmentData &nextPathSegment = m_segmentStorage[j];

            throughput = throughput * openpgl::Vector3(m_segmentStorage[j - 1].transmittanceWeight.x, m_segmentStorage[j - 1].transmittanceWeight.y,
                                                       m_segmentStorage[j - 1].transmittanceWeight.z);
            OPENPGL_ASSERT(embree::isvalid(throughput));
            OPENPGL_ASSERT(throughput[0] >= 0.f && throughput[1] >= 0.f && throughput[2] >= 0.f)

            contribution +=
                throughput * openpgl::Vector3(nextPathSegment.scatteredContribution.x, nextPathSegment.scatteredContribution.y, nextPathSegment.scatteredContribution.z);
            OPENPGL_ASSERT(embree::isvalid(contribution));
            OPENPGL_ASSERT(contribution[0] >= 0.f && contribution[1] >= 0.f && contribution[2] >= 0.f);

            openpgl::Vector3 directContribution =
                openpgl::Vector3(nextPathSegment.directContribution.x, nextPathSegment.directContribution.y, nextPathSegment.directContribution.z);
            if (!rrAffectsDirectContribution)
            {
                directContribution *= previousRRSurvivalProb;
            }
            contribution += throughput * nextPathSegment.miWeight * directContribution;
            OPENPGL_ASSERT(embree::isvalid(contribution));
            OPENPGL_ASSERT(contribution[0] >= 0.f && contribution[1] >= 0.f && contribution[2] >= 0.f);

            throughput = throughput * openpgl::Vector3(nextPathSegment.scatteringWeight.x, nextPathSegment.scatteringWeight.y, nextPathSegment.scatteringWeight.z);
            if (nextPathSegment.russianRouletteSurvivalProbability > 0.f)
            {
                throughput /= nextPathSegment.russianRouletteSurvivalProbability;
            }
            else
            {
                throughput = openpgl::Vector3(0.f);
            }
            previousRRSurvivalProb = nextPathSegment.russianRouletteSurvivalProbability;
            OPENPGL_ASSERT(embree::isvalid(throughput));
            OPENPGL_ASSERT(throughput[0] >= 0.f && throughput[1] >= 0.f && throughput[2] >= 0.f)
        }

        OPENPGL_ASSERT(embree::isvalid(contribution));
        OPENPGL_ASSERT(contribution[0] >= 0.f && contribution[1] >= 0.f && contribution[2] >= 0.f);

        if (m_segmentStorage[0].russianRouletteSurvivalProbability > 0.f)
        {
            finalColor.x =
                m_segmentStorage[0].directContribution.x + m_segmentStorage[0].scatteredContribution.x +
                m_segmentStorage[0].scatteringWeight.x /** m_segmentStorage[0].transmittanceWeight.x*/ * contribution[0] / m_segmentStorage[0].russianRouletteSurvivalProbability;
            finalColor.y =
                m_segmentStorage[0].directContribution.y + m_segmentStorage[0].scatteredContribution.y +
                m_segmentStorage[0].scatteringWeight.y /** m_segmentStorage[0].transmittanceWeight.y*/ * contribution[1] / m_segmentStorage[0].russianRouletteSurvivalProbability;
            finalColor.z =
                m_segmentStorage[0].directContribution.z + m_segmentStorage[0].scatteredContribution.z +
                m_segmentStorage[0].scatteringWeight.z /** m_segmentStorage[0].transmittanceWeight.z*/ * contribution[2] / m_segmentStorage[0].russianRouletteSurvivalProbability;
        }
        else
        {
            finalColor.x = m_segmentStorage[0].directContribution.x + m_segmentStorage[0].scatteredContribution.x +
                           m_segmentStorage[0].scatteringWeight.x /** m_segmentStorage[0].transmittanceWeight.x*/ * contribution[0];
            finalColor.y = m_segmentStorage[0].directContribution.y + m_segmentStorage[0].scatteredContribution.y +
                           m_segmentStorage[0].scatteringWeight.y /** m_segmentStorage[0].transmittanceWeight.y*/ * contribution[1];
            finalColor.z = m_segmentStorage[0].directContribution.z + m_segmentStorage[0].scatteredContribution.z +
                           m_segmentStorage[0].scatteringWeight.z /** m_segmentStorage[0].transmittanceWeight.z*/ * contribution[2];
        }

        if (numSegments == 1)
        {
            finalColor.x = m_segmentStorage[0].directContribution.x + m_segmentStorage[0].scatteredContribution.x;  // + currentPathSegment.scatteringWeight.x * contribution[0];
            finalColor.y = m_segmentStorage[0].directContribution.y + m_segmentStorage[0].scatteredContribution.y;  // + currentPathSegment.scatteringWeight.y * contribution[1];
            finalColor.z = m_segmentStorage[0].directContribution.z + m_segmentStorage[0].scatteredContribution.z;  // + currentPathSegment.scatteringWeight.z * contribution[2];
        }

        return finalColor;
    }

    const SampleData *getSamples() const
    {
#if defined(OPENPGL_PATHSEGMENT_STORAGE_USE_ARRAY)
        return m_sampleStorage;
#else
        return m_sampleStorage.data();
#endif
    }

    int getNumSamples() const
    {
#if defined(OPENPGL_PATHSEGMENT_STORAGE_USE_ARRAY)
        return m_sample_idx + 1;
#else
        return m_sampleStorage.size();
#endif
    }

    void addSample(const SampleData &sampleData)
    {
        OPENPGL_ASSERT(isValid(sampleData));
        OPENPGL_ASSERT(sampleData.distance > 0);
        OPENPGL_ASSERT(embree::isvalid(sampleData.distance));
#if defined(OPENPGL_PATHSEGMENT_STORAGE_USE_ARRAY)
        if (m_sample_idx + 1 <= m_max_sample_size)
        {
            m_sample_idx++;
            m_sampleStorage[m_sample_idx] = sampleData;
        }
#else
        m_sampleStorage.push_back(sampleData);
#endif
    }

    bool validateSamples() const
    {
        bool valid = true;
#if defined(OPENPGL_PATHSEGMENT_STORAGE_USE_ARRAY)
        int nSamples = m_sample_idx + 1;
#else
        int nSamples = m_sampleStorage.size();
#endif
        for (int s = 0; s < nSamples; s++)
        {
            SampleData sample = m_sampleStorage[s];
            valid = valid && isValid(sample);
            OPENPGL_ASSERT(valid);
        }
        return valid;
    }

    bool validateSegments() const
    {
#if defined(OPENPGL_PATHSEGMENT_STORAGE_USE_ARRAY)
        size_t numSegments = m_seg_idx + 1;
#else
        size_t numSegments = m_segmentStorage.size();
#endif

        bool valid = true;
        for (int s = 0; s < numSegments; s++)
        {
            PathSegmentData psd = m_segmentStorage[s];
            valid = valid && isValid(psd);
            OPENPGL_ASSERT(valid);
        }
        /*
        if(!valid)
        {
            std::cout << "Segments not Valid:" << std::endl;
            std::cout << toString() << std::endl;
        }
        */
        return valid;
    }

    std::string toString() const
    {
#if defined(OPENPGL_PATHSEGMENT_STORAGE_USE_ARRAY)
        size_t numSegments = m_seg_idx + 1;
#else
        size_t numSegments = m_segmentStorage.size();
#endif
        std::stringstream ss;
        ss << "PathSegmentDataStorage:" << std::endl;
        ss << "segment storage: size = " << numSegments << std::endl;
        for (int s = 0; s < numSegments; s++)
        {
            PathSegmentData psd = m_segmentStorage[s];
            ss << "seg[" << s << "]: " << openpgl::toString(psd);
            ss << "\t valid = " << isValid(psd);
            ss << std::endl;
        }
#if defined(OPENPGL_PATHSEGMENT_STORAGE_USE_ARRAY)
        int nSamples = m_sample_idx + 1;
#else
        int nSamples = m_sampleStorage.size();
#endif

        for (int s = 0; s < nSamples; s++)
        {
            SampleData sample = m_sampleStorage[s];
            ss << "sample[" << s << "]: " << openpgl::toString(sample);
            ss << "\t valid = " << isValid(sample);
            ss << std::endl;
        }
        return ss.str();
    }

    void propagateSamples(SampleDataStorage *sampleDataStorage, const bool guideDirectLight = false, const bool useNEEMiWeights = false,
                          const bool rrAffectsDirectContribution = true)
    {
        prepareSamples(useNEEMiWeights, guideDirectLight, rrAffectsDirectContribution);
#if defined(OPENPGL_PATHSEGMENT_STORAGE_USE_ARRAY)
        sampleDataStorage->addSamples(m_sampleStorage, m_sample_idx + 1);
#else
        sampleDataStorage->addSamples(m_sampleStorage.data(), m_sampleData.size());
#endif
        if (m_track_zero_value_samples)
        {
#if defined(OPENPGL_PATHSEGMENT_STORAGE_USE_ARRAY)
            sampleDataStorage->addZeroValueSamples(m_zeroValueSampleStorage, m_zero_value_sample_idx + 1);
#else
            sampleDataStorage->addZeroValueSamples(m_zeroValueSampleStorage.data(), m_zeroValueSampleStorage.size());
#endif
        }
        clear();
    }

    const ZeroValueSampleData *getZeroValueSamples() const
    {
#if defined(OPENPGL_PATHSEGMENT_STORAGE_USE_ARRAY)
        return m_zeroValueSampleStorage;
#else
        return m_zeroValueSampleStorage.data();
#endif
    }

    int getNumZeroValueSamples() const
    {
#if defined(OPENPGL_PATHSEGMENT_STORAGE_USE_ARRAY)
        return m_zero_value_sample_idx + 1;
#else
        return m_zeroValueSampleStorage.size();
#endif
    }
};
}  // namespace openpgl