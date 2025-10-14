#pragma once

#include <fstream>

#include <thrust/device_vector.h>

#include "../include/openpgl/data.h"
#include "../include/openpgl/cuda/SampleStorageCUDADesc.h"
#include "Common.h"

namespace openpgl {
namespace OPENPGL_KERNEL_NS {

struct SampleStorageCUDA {
    SampleStorageCUDAAlloc alloc;
    thrust::device_vector<SampleStorageCUDAAlloc> allocDevice;
    thrust::device_vector<PGLSampleData> samplesSurface;
    thrust::device_vector<PGLSampleData> samplesVolume;

    SampleStorageCUDA() {
        alloc.capacity = 0;
        alloc.sizeSurface = 0;
        alloc.sizeVolume = 0;
        allocDevice.resize(1);
        allocDevice[0] = alloc;
    }

    SampleStorageCUDA(size_t size) : SampleStorageCUDA() {
        reserve(size);
    }

    SampleStorageCUDA(const std::string &filename) : SampleStorageCUDA() {
        std::filebuf fb;
        fb.open(filename, std::ios::in | std::ios::binary);
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

        thrust::host_vector<PGLSampleData> host_data;

        size_t num_surface_samples;
        is.read(reinterpret_cast<char *>(&num_surface_samples), sizeof(size_t));
        alloc.sizeSurface = num_surface_samples;
        host_data.reserve(num_surface_samples);
        for (size_t n = 0; n < num_surface_samples; n++)
        {
            SampleData dsd;
            is.read(reinterpret_cast<char *>(&dsd), sizeof(SampleData));
            host_data.push_back(dsd);
        }
        samplesSurface = host_data;

        host_data.clear();

        size_t num_volume_samples;
        is.read(reinterpret_cast<char *>(&num_volume_samples), sizeof(size_t));
        alloc.sizeVolume = num_volume_samples;
        host_data.reserve(num_volume_samples);
        for (size_t n = 0; n < num_volume_samples; n++)
        {
            SampleData dsd;
            is.read(reinterpret_cast<char *>(&dsd), sizeof(SampleData));
            host_data.push_back(dsd);
        }
        samplesVolume = host_data;

        alloc.capacity = std::max(alloc.sizeSurface, alloc.sizeVolume);
        samplesSurface.resize(alloc.capacity);
        samplesVolume.resize(alloc.capacity);

        allocDevice[0] = alloc;

        fb.close();
    }

    void getGPUDesc(SampleStorageCUDADesc* desc) {
        *desc = SampleStorageCUDADesc {
            .alloc = data(allocDevice),
            .samplesSurface = data(samplesSurface),
            .samplesVolume = data(samplesVolume)
        };
    }

    void store(const std::string &fileName) {
        alloc = allocDevice[0];

        std::filebuf fb;
        fb.open(fileName, std::ios::out | std::ios::binary);
        if (!fb.is_open())
            throw std::runtime_error("error: couldn't open file");
        std::ostream os(&fb);

        const char* str = SAMPLE_DATA_STORAGE_FILE_HEADER_STRING;
        os.write(str, strlen(str) + 1);

        size_t num_surface_samples = std::min(alloc.sizeSurface, alloc.capacity);
        os.write(reinterpret_cast<const char *>(&num_surface_samples), sizeof(size_t));
        thrust::host_vector<PGLSampleData> host_data = samplesSurface;
        for (size_t n = 0; n < num_surface_samples; n++)
            os.write(reinterpret_cast<const char *>(&host_data[n]), sizeof(SampleData));

        size_t num_volume_samples = std::min(alloc.sizeVolume, alloc.capacity);
        os.write(reinterpret_cast<const char *>(&num_volume_samples), sizeof(size_t));
        host_data = samplesVolume;
        for (size_t n = 0; n < num_volume_samples; n++)
            os.write(reinterpret_cast<const char *>(&host_data[n]), sizeof(SampleData));
    }

    void reset() {
        alloc.sizeSurface = 0;
        alloc.sizeVolume = 0;
        allocDevice[0] = alloc;
    }

    void reserve(uint32_t size) {
        if (alloc.capacity < size) {
            samplesSurface.resize(size);
            samplesVolume.resize(size);
            alloc.capacity = size;
        }
        reset();
    }
};

}
}