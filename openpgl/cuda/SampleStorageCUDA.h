#pragma once

#include <fstream>

#include <thrust/device_vector.h>

#include "../include/openpgl/data.h"
#include "../include/openpgl/cuda/SampleStorageCUDADesc.h"
#include "Common.h"

#include "../data/SampleDataStorage.h"

namespace openpgl {
namespace OPENPGL_KERNEL_NS {

struct HostSampleData {
    std::vector<PGLSampleData> surface;
    std::vector<PGLSampleData> volume;
};

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
        HostSampleData hData;

        size_t num_surface_samples;
        is.read(reinterpret_cast<char *>(&num_surface_samples), sizeof(size_t));
        alloc.sizeSurface = num_surface_samples;
        hData.surface.reserve(num_surface_samples);
        for (size_t n = 0; n < num_surface_samples; n++)
        {
            SampleData dsd;
            is.read(reinterpret_cast<char *>(&dsd), sizeof(SampleData));
            hData.surface.push_back(dsd);
        }

        size_t num_volume_samples;
        is.read(reinterpret_cast<char *>(&num_volume_samples), sizeof(size_t));
        alloc.sizeVolume = num_volume_samples;
        hData.volume.reserve(num_volume_samples);
        for (size_t n = 0; n < num_volume_samples; n++)
        {
            SampleData dsd;
            is.read(reinterpret_cast<char *>(&dsd), sizeof(SampleData));
            hData.volume.push_back(dsd);
        }

        fb.close();

        uploadAppend(hData);
    }

    void fillGPUDesc(SampleStorageCUDADesc* desc) {
        SampleStorageCUDADesc hDesc {
            .alloc = data(allocDevice),
            .samplesSurface = data(samplesSurface),
            .samplesVolume = data(samplesVolume)
        };

        cudaMemcpy(desc, &hDesc, sizeof(*desc), cudaMemcpyHostToDevice);
    }

    void uploadAppend(HostSampleData& hostSampleData) {
        alloc = allocDevice[0]; // sync in-flights ops

        uint32_t offsetSurface = std::min(alloc.sizeSurface, alloc.capacity);
        uint32_t offsetVolume =  std::min(alloc.sizeVolume,  alloc.capacity);

        alloc.sizeSurface = offsetSurface + hostSampleData.surface.size();
        alloc.sizeVolume  = offsetVolume  + hostSampleData.volume.size();

        reserve(std::max(alloc.sizeSurface, alloc.sizeVolume));

        cudaMemcpy(
            data(samplesSurface) + offsetSurface, hostSampleData.surface.data(),
            hostSampleData.surface.size() * sizeof(PGLSampleData), cudaMemcpyHostToDevice);
        cudaMemcpy(
            data(samplesVolume)  + offsetVolume,  hostSampleData.volume.data(),
            hostSampleData.volume.size()  * sizeof(PGLSampleData), cudaMemcpyHostToDevice);

        allocDevice[0] = alloc;
    }

    void download(HostSampleData& hData) {
        alloc = allocDevice[0];  // sync in-flights ops
        uint32_t sizeSurface = std::min(alloc.sizeSurface, alloc.capacity);
        uint32_t sizeVolume  = std::min(alloc.sizeVolume,  alloc.capacity);
        hData.surface.resize(sizeSurface);
        hData.volume.resize(sizeVolume);

        cudaMemcpy(
            hData.surface.data(), data(samplesSurface),
            sizeSurface * sizeof(PGLSampleData), cudaMemcpyDeviceToHost);
        cudaMemcpy(
            hData.volume.data(),  data(samplesVolume),
            sizeVolume *  sizeof(PGLSampleData), cudaMemcpyDeviceToHost);
    }

    void transferToCPU(SampleDataStorage& sds) {
        HostSampleData hData;
        download(hData);
        sds.addSamples(hData.surface.data(), hData.surface.size());
        sds.addSamples(hData.volume.data(),  hData.volume.size());
    }

    void transferFromCPU(SampleDataStorage& sds) {
        HostSampleData hData;

        hData.surface.reserve(sds.sizeSurface());
        hData.volume.reserve(sds.sizeVolume());
        // tbb concurrent vector 
        for (int i = 0; i < sds.sizeSurface(); i++)
            hData.surface[i] = sds.getSampleSurface(i);
        for (int i = 0; i < sds.sizeVolume(); i++)
            hData.volume[i] = sds.getSampleVolume(i);
        uploadAppend(hData);
    }


    void store(const std::string &fileName) {
        HostSampleData hData;
        download(hData);

        std::filebuf fb;
        fb.open(fileName, std::ios::out | std::ios::binary);
        if (!fb.is_open())
            throw std::runtime_error("error: couldn't open file");
        std::ostream os(&fb);

        const char* str = SAMPLE_DATA_STORAGE_FILE_HEADER_STRING;
        os.write(str, strlen(str) + 1);

        size_t sizeSurface = hData.surface.size(), sizeVolume = hData.volume.size();
        os.write(reinterpret_cast<const char *>(&sizeSurface), sizeof(size_t));
        for (const auto& sample : hData.surface)
            os.write(reinterpret_cast<const char *>(&sample), sizeof(sample));

        os.write(reinterpret_cast<const char *>(&sizeVolume), sizeof(size_t));
        for (const auto& sample : hData.volume)
            os.write(reinterpret_cast<const char *>(&sample), sizeof(sample));
    }

    void reset() {
        alloc.sizeSurface = 0;
        alloc.sizeVolume = 0;
        allocDevice[0] = alloc;
    }

    void reserve(uint32_t size) {
        if (alloc.capacity < size) {
            alloc = allocDevice[0];
            samplesSurface.resize(size);
            samplesVolume.resize(size);
            alloc.capacity = size;
            allocDevice[0] = alloc;
        }
    }
};

}
}