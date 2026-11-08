// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../openpgl_common.h"

#ifdef PGL_USE_DIRECTION_COMPRESSION
#include "../include/openpgl/compression.h"
#endif

#include <tbb/concurrent_vector.h>
#include <tbb/parallel_sort.h>

#include "SampleData.h"

#define SAMPLE_DATA_STORAGE_FILE_HEADER_STRING "OPENPGL_" OPENPGL_VERSION_STRING "_SAMPLE_STORAGE"

namespace openpgl
{

struct SampleDataStorage
{
    typedef tbb::concurrent_vector<SampleData> SampleDataContainer;
    typedef tbb::concurrent_vector<ZeroValueSampleData> ZeroValueSampleDataContainer;
    struct SampleContainer
    {
        SampleDataContainer samples;
        ZeroValueSampleDataContainer zeroValueSamples;
    };

    SampleContainer m_surfaceContainer;
    SampleContainer m_volumeContainer;

    static SampleDataStorage *newSampleDataStorage()
    {
        return new openpgl::SampleDataStorage();
    }

    static SampleDataStorage *newSampleDataStorageFromFile(const std::string sampleDataStorageFileName)
    {
        std::filebuf fb;
        fb.open(sampleDataStorageFileName, std::ios::in | std::ios::binary);
        if (!fb.is_open())
            throw std::runtime_error("error: couldn't open file");
        std::istream is(&fb);

        auto size = strlen(SAMPLE_DATA_STORAGE_FILE_HEADER_STRING) + 1;
        OPENPGL_ASSERT(size <= 256);
        char buf[256];
        is.read(&buf[0], size);
        if (!is)
            throw std::runtime_error("error: invalid file header");
#ifdef OPENPGL_STRICT_IO_VERSION_CHECKING
        for (auto i = 0; i < size; i++)
        {
            if (buf[i] != SAMPLE_DATA_STORAGE_FILE_HEADER_STRING[i])
                throw std::runtime_error("error: invalid file header");
        }
#endif
        openpgl::SampleDataStorage *gSampleStorage = new openpgl::SampleDataStorage();
        gSampleStorage->deserialize(is);

        fb.close();

        return gSampleStorage;
    }

    static void storeSampleDataStorageToFile(SampleDataStorage *gSampleDataStorage, const std::string sampleDataStorageFileName)
    {
        std::filebuf fb;
        fb.open(sampleDataStorageFileName, std::ios::out | std::ios::binary);
        if (!fb.is_open())
            throw std::runtime_error("error: couldn't open file!");
        std::ostream os(&fb);

        os.write(SAMPLE_DATA_STORAGE_FILE_HEADER_STRING, strlen(SAMPLE_DATA_STORAGE_FILE_HEADER_STRING) + 1);

        gSampleDataStorage->serialize(os);

        os.flush();
        fb.close();
    }

    inline void addSample(const SampleData &sample)
    {
        if (isInsideVolume(sample))
        {
            m_volumeContainer.samples.push_back(sample);
        }
        else
        {
            m_surfaceContainer.samples.push_back(sample);
        }
    }

    inline void addSamples(const SampleData *samples, int nSamples)
    {
        for (int i = 0; i < nSamples; i++)
        {
            addSample(samples[i]);
        }
    }

    inline void addZeroValueSample(const ZeroValueSampleData &sample)
    {
        if (isInsideVolume(sample))
        {
            m_volumeContainer.zeroValueSamples.push_back(sample);
        }
        else
        {
            m_surfaceContainer.zeroValueSamples.push_back(sample);
        }
    }

    inline void addZeroValueSamples(const ZeroValueSampleData *samples, int nSamples)
    {
        for (int i = 0; i < nSamples; i++)
        {
            addZeroValueSample(samples[i]);
        }
    }

    inline void reserveSurface(const size_t &size)
    {
        m_surfaceContainer.samples.reserve(size);
    }

    inline size_t sizeSurface() const
    {
        return m_surfaceContainer.samples.size();
    }

    inline SampleData getSampleSurface(const uint32_t idx) const
    {
        OPENPGL_ASSERT(idx >= 0);
        OPENPGL_ASSERT(idx < m_surfaceContainer.samples.size());

        SampleData sd;
        if (idx < m_surfaceContainer.samples.size())
        {
            sd = m_surfaceContainer.samples[idx];
        }
        return sd;
    }

    inline void clearSurface()
    {
        m_surfaceContainer.samples.clear();
    }

    void sortSurface()
    {
        std::sort(m_surfaceContainer.samples.begin(), m_surfaceContainer.samples.end(), SampleDataLess);
    }

    inline void reserveVolume(const size_t &size)
    {
        m_volumeContainer.samples.reserve(size);
    }

    inline size_t sizeVolume() const
    {
        return m_volumeContainer.samples.size();
    }

    inline SampleData getSampleVolume(const uint32_t idx) const
    {
        OPENPGL_ASSERT(idx >= 0);
        OPENPGL_ASSERT(idx < m_volumeContainer.samples.size());

        SampleData sd;
        if (idx < m_volumeContainer.samples.size())
        {
            sd = m_volumeContainer.samples[idx];
        }
        return sd;
    }

    inline void clearVolume()
    {
        m_volumeContainer.samples.clear();
    }

    void sortVolume()
    {
        std::sort(m_volumeContainer.samples.begin(), m_volumeContainer.samples.end(), SampleDataLess);
    }

    inline void reserveInvalidSurface(const size_t &size)
    {
        m_surfaceContainer.zeroValueSamples.reserve(size);
    }

    inline size_t sizeInvalidSurface() const
    {
        return m_surfaceContainer.zeroValueSamples.size();
    }

    inline ZeroValueSampleData getZeroValueSampleSurface(const uint32_t idx) const
    {
        OPENPGL_ASSERT(idx >= 0);
        OPENPGL_ASSERT(idx < m_surfaceContainer.zeroValueSamples.size());

        ZeroValueSampleData isd;
        if (idx < m_surfaceContainer.zeroValueSamples.size())
        {
            isd = m_surfaceContainer.zeroValueSamples[idx];
        }
        return isd;
    }

    inline void clearInvalidSurface()
    {
        m_surfaceContainer.zeroValueSamples.clear();
    }

    void sortInvalidSurface()
    {
        std::sort(m_surfaceContainer.zeroValueSamples.begin(), m_surfaceContainer.zeroValueSamples.end(), ZeroValueSampleDataLess);
    }

    inline void reserveInvalidVolume(const size_t &size)
    {
        m_volumeContainer.zeroValueSamples.reserve(size);
    }

    inline size_t sizeInvalidVolume() const
    {
        return m_volumeContainer.zeroValueSamples.size();
    }

    inline ZeroValueSampleData getZeroValueSampleVolume(const uint32_t idx) const
    {
        OPENPGL_ASSERT(idx >= 0);
        OPENPGL_ASSERT(idx < m_volumeContainer.zeroValueSamples.size());

        ZeroValueSampleData isd;
        if (idx < m_volumeContainer.zeroValueSamples.size())
        {
            isd = m_volumeContainer.zeroValueSamples[idx];
        }
        return isd;
    }

    inline void clearInvalidVolume()
    {
        m_volumeContainer.zeroValueSamples.clear();
    }

    void sortInvalidVolume()
    {
        std::sort(m_volumeContainer.zeroValueSamples.begin(), m_volumeContainer.zeroValueSamples.end(), ZeroValueSampleDataLess);
    }

    void exportSurfaceSamplesToObj(std::string objFileName, bool pointsOnly = true)
    {
        std::ofstream objFile;
        objFile.open(objFileName.c_str());
        exportSamplesToObj(objFile, m_surfaceContainer.samples, pointsOnly);
        objFile.close();
    }

    void exportVolumeSamplesToObj(std::string objFileName, bool pointsOnly = true)
    {
        std::ofstream objFile;
        objFile.open(objFileName.c_str());
        exportSamplesToObj(objFile, m_volumeContainer.samples, pointsOnly);
        objFile.close();
    }

    void serialize(std::ostream &stream) const
    {
        size_t num_surface_samples = m_surfaceContainer.samples.size();
        stream.write(reinterpret_cast<const char *>(&num_surface_samples), sizeof(size_t));
        for (size_t n = 0; n < num_surface_samples; n++)
        {
            SampleData dsd = m_surfaceContainer.samples[n];
            stream.write(reinterpret_cast<const char *>(&dsd), sizeof(SampleData));
        }

        size_t num_volume_samples = m_volumeContainer.samples.size();
        stream.write(reinterpret_cast<const char *>(&num_volume_samples), sizeof(size_t));
        for (size_t n = 0; n < num_volume_samples; n++)
        {
            SampleData dsd = m_volumeContainer.samples[n];
            stream.write(reinterpret_cast<const char *>(&dsd), sizeof(SampleData));
        }

        size_t num_zero_value_surface_samples = m_surfaceContainer.zeroValueSamples.size();
        stream.write(reinterpret_cast<const char *>(&num_zero_value_surface_samples), sizeof(size_t));
        for (size_t n = 0; n < num_zero_value_surface_samples; n++)
        {
            ZeroValueSampleData isd = m_surfaceContainer.zeroValueSamples[n];
            stream.write(reinterpret_cast<const char *>(&isd), sizeof(ZeroValueSampleData));
        }

        size_t num_zero_value_volume_samples = m_volumeContainer.zeroValueSamples.size();
        stream.write(reinterpret_cast<const char *>(&num_zero_value_volume_samples), sizeof(size_t));
        for (size_t n = 0; n < num_zero_value_volume_samples; n++)
        {
            ZeroValueSampleData isd = m_volumeContainer.zeroValueSamples[n];
            stream.write(reinterpret_cast<const char *>(&isd), sizeof(ZeroValueSampleData));
        }
    }

    void deserialize(std::istream &stream)
    {
        size_t num_surface_samples;
        stream.read(reinterpret_cast<char *>(&num_surface_samples), sizeof(size_t));
        m_surfaceContainer.samples.reserve(num_surface_samples);
        for (size_t n = 0; n < num_surface_samples; n++)
        {
            SampleData dsd;
            stream.read(reinterpret_cast<char *>(&dsd), sizeof(SampleData));
            m_surfaceContainer.samples.push_back(dsd);
        }

        size_t num_volume_samples;
        stream.read(reinterpret_cast<char *>(&num_volume_samples), sizeof(size_t));
        m_volumeContainer.samples.reserve(num_volume_samples);
        for (size_t n = 0; n < num_volume_samples; n++)
        {
            SampleData dsd;
            stream.read(reinterpret_cast<char *>(&dsd), sizeof(SampleData));
            m_volumeContainer.samples.push_back(dsd);
        }

        size_t num_zero_value_surface_samples;
        stream.read(reinterpret_cast<char *>(&num_zero_value_surface_samples), sizeof(size_t));
        m_surfaceContainer.zeroValueSamples.reserve(num_zero_value_surface_samples);
        for (size_t n = 0; n < num_zero_value_surface_samples; n++)
        {
            ZeroValueSampleData isd;
            stream.read(reinterpret_cast<char *>(&isd), sizeof(ZeroValueSampleData));
            m_surfaceContainer.zeroValueSamples.push_back(isd);
        }

        size_t num_zero_value_volume_samples;
        stream.read(reinterpret_cast<char *>(&num_zero_value_volume_samples), sizeof(size_t));
        m_volumeContainer.zeroValueSamples.reserve(num_zero_value_volume_samples);
        for (size_t n = 0; n < num_zero_value_volume_samples; n++)
        {
            ZeroValueSampleData isd;
            stream.read(reinterpret_cast<char *>(&isd), sizeof(ZeroValueSampleData));
            m_volumeContainer.zeroValueSamples.push_back(isd);
        }
    }

    bool validate() const
    {
        bool valid = true;
        for (uint32_t i = 0; i < m_surfaceContainer.samples.size(); i++)
        {
            valid = valid && isValid(m_surfaceContainer.samples[i]);
        }

        for (uint32_t i = 0; i < m_volumeContainer.samples.size(); i++)
        {
            valid = valid && isValid(m_volumeContainer.samples[i]);
        }
        return valid;
    }

    void merge(const SampleDataStorage &b)
    {
        for (uint32_t i = 0; i < b.m_surfaceContainer.samples.size(); i++)
        {
            m_surfaceContainer.samples.push_back(b.m_surfaceContainer.samples[i]);
        }

        for (uint32_t i = 0; i < b.m_volumeContainer.samples.size(); i++)
        {
            m_volumeContainer.samples.push_back(b.m_volumeContainer.samples[i]);
        }

        for (uint32_t i = 0; i < b.m_surfaceContainer.zeroValueSamples.size(); i++)
        {
            m_surfaceContainer.zeroValueSamples.push_back(b.m_surfaceContainer.zeroValueSamples[i]);
        }

        for (uint32_t i = 0; i < b.m_volumeContainer.zeroValueSamples.size(); i++)
        {
            m_volumeContainer.zeroValueSamples.push_back(b.m_volumeContainer.zeroValueSamples[i]);
        }
    }

    bool operator==(const SampleDataStorage &b) const
    {
        std::vector<SampleData> surfaceSampleDataA;
        std::vector<SampleData> volumeSampleDataA;

        std::vector<ZeroValueSampleData> surfaceZeroValueSampleDataA;
        std::vector<ZeroValueSampleData> volumeZeroValueSampleDataA;

        std::vector<SampleData> surfaceSampleDataB;
        std::vector<SampleData> volumeSampleDataB;

        std::vector<ZeroValueSampleData> surfaceZeroValueSampleDataB;
        std::vector<ZeroValueSampleData> volumeZeroValueSampleDataB;

        surfaceSampleDataA.resize(m_surfaceContainer.samples.size());
        volumeSampleDataA.resize(m_volumeContainer.samples.size());

        for (uint32_t i = 0; i < m_surfaceContainer.samples.size(); i++)
        {
            surfaceSampleDataA[i] = m_surfaceContainer.samples[i];
        }
        tbb::parallel_sort(surfaceSampleDataA.begin(), surfaceSampleDataA.end(), SampleDataLess);

        for (uint32_t i = 0; i < m_volumeContainer.samples.size(); i++)
        {
            volumeSampleDataA[i] = m_volumeContainer.samples[i];
        }
        tbb::parallel_sort(volumeSampleDataA.begin(), volumeSampleDataA.end(), SampleDataLess);

        surfaceZeroValueSampleDataA.resize(m_surfaceContainer.zeroValueSamples.size());
        volumeZeroValueSampleDataA.resize(m_volumeContainer.zeroValueSamples.size());

        for (uint32_t i = 0; i < m_surfaceContainer.zeroValueSamples.size(); i++)
        {
            surfaceZeroValueSampleDataA[i] = m_surfaceContainer.zeroValueSamples[i];
        }
        tbb::parallel_sort(surfaceZeroValueSampleDataA.begin(), surfaceZeroValueSampleDataA.end(), ZeroValueSampleDataLess);

        for (uint32_t i = 0; i < m_volumeContainer.zeroValueSamples.size(); i++)
        {
            volumeZeroValueSampleDataA[i] = m_volumeContainer.zeroValueSamples[i];
        }
        tbb::parallel_sort(volumeZeroValueSampleDataA.begin(), volumeZeroValueSampleDataA.end(), ZeroValueSampleDataLess);

        surfaceSampleDataB.resize(b.m_surfaceContainer.samples.size());
        volumeSampleDataB.resize(b.m_volumeContainer.samples.size());

        for (uint32_t i = 0; i < b.m_surfaceContainer.samples.size(); i++)
        {
            surfaceSampleDataB[i] = b.m_surfaceContainer.samples[i];
        }
        tbb::parallel_sort(surfaceSampleDataB.begin(), surfaceSampleDataB.end(), SampleDataLess);

        for (uint32_t i = 0; i < b.m_volumeContainer.samples.size(); i++)
        {
            volumeSampleDataB[i] = b.m_volumeContainer.samples[i];
        }
        tbb::parallel_sort(volumeSampleDataB.begin(), volumeSampleDataB.end(), SampleDataLess);

        surfaceZeroValueSampleDataB.resize(b.m_surfaceContainer.zeroValueSamples.size());
        volumeZeroValueSampleDataB.resize(b.m_volumeContainer.zeroValueSamples.size());

        for (uint32_t i = 0; i < b.m_surfaceContainer.zeroValueSamples.size(); i++)
        {
            surfaceZeroValueSampleDataB[i] = b.m_surfaceContainer.zeroValueSamples[i];
        }
        tbb::parallel_sort(surfaceZeroValueSampleDataB.begin(), surfaceZeroValueSampleDataB.end(), ZeroValueSampleDataLess);

        for (uint32_t i = 0; i < b.m_volumeContainer.zeroValueSamples.size(); i++)
        {
            volumeZeroValueSampleDataB[i] = b.m_volumeContainer.zeroValueSamples[i];
        }
        tbb::parallel_sort(volumeZeroValueSampleDataB.begin(), volumeZeroValueSampleDataB.end(), ZeroValueSampleDataLess);

        bool equal = true;
        uint32_t sizeB = surfaceSampleDataB.size();
        for (uint32_t i = 0; i < surfaceSampleDataA.size(); i++)
        {
            if (i < sizeB && !SampleDataEqual(surfaceSampleDataA[i], surfaceSampleDataB[i]))
            {
                equal = false;
            }
        }

        sizeB = surfaceZeroValueSampleDataB.size();
        for (uint32_t i = 0; i < surfaceZeroValueSampleDataA.size(); i++)
        {
            if (i < sizeB && !ZeroValueSampleDataEqual(surfaceZeroValueSampleDataA[i], surfaceZeroValueSampleDataB[i]))
            {
                equal = false;
            }
        }

        sizeB = volumeSampleDataB.size();
        for (uint32_t i = 0; i < volumeSampleDataA.size(); i++)
        {
            if (i < sizeB && !SampleDataEqual(volumeSampleDataA[i], volumeSampleDataB[i]))
            {
                equal = false;
            }
        }

        sizeB = volumeZeroValueSampleDataB.size();
        for (uint32_t i = 0; i < volumeZeroValueSampleDataA.size(); i++)
        {
            if (i < sizeB && !ZeroValueSampleDataEqual(volumeZeroValueSampleDataA[i], volumeZeroValueSampleDataB[i]))
            {
                equal = false;
            }
        }

        return equal;
    }

   private:
    void exportSamplesToObj(std::ofstream &objFile, const SampleDataContainer &sampleContainer, bool pointsOnly = true)
    {
        std::vector<SampleData> subSampledData;
        subSampledData.reserve(sampleContainer.size());
        for (uint32_t i = 0; i < sampleContainer.size(); i++)
        {
            subSampledData.push_back(sampleContainer[i]);
        }

        for (auto &sample : subSampledData)
        {
            objFile << "v " << sample.position.x << "\t" << sample.position.y << "\t" << sample.position.z << std::endl;
            if (!pointsOnly)
            {
                pgl_vec3f direction = sample.direction;
                Vector3 dir(direction.x, direction.y, direction.z);
                Point3 samplePosition(sample.position.x, sample.position.y, sample.position.z);
                Point3 pos2 = samplePosition + dir * sample.distance;
                objFile << "v " << pos2[0] << "\t" << pos2[1] << "\t" << pos2[2] << std::endl;
                objFile << "v " << sample.position.x << "\t" << sample.position.y << "\t" << sample.position.z << std::endl;
            }
        }

        for (auto &sample : subSampledData)
        {
            pgl_vec3f direction = sample.direction;
            Vector3 dir(direction.x, direction.y, direction.z);
            objFile << "vn " << dir[0] << "\t" << dir[1] << "\t" << dir[2] << std::endl;
            if (!pointsOnly)
            {
                objFile << "vn " << dir[0] << "\t" << dir[1] << "\t" << dir[2] << std::endl;
                objFile << "vn " << dir[0] << "\t" << dir[1] << "\t" << dir[2] << std::endl;
            }
        }

        if (!pointsOnly)
        {
            for (uint32_t i = 0; i < subSampledData.size(); i++)
            {
                objFile << "f " << i * 3 + 1 << "\t" << i * 3 + 2 << "\t" << i * 3 + 3 << std::endl;
            }
        }
    }
};

}  // namespace openpgl
