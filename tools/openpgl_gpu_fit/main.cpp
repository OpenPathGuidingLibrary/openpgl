#include <iostream>
#include <memory>
#include <sstream>

#include "openpgl/cpp/Field.h"
#include "openpgl/cpp/Device.h"
#include "openpgl/cpp/SampleStorage.h"

#include "gpu_fit.h" // Include the header for our CUDA function

using namespace openpgl::gpu::cuda;

int main() {
    auto field = GPUFieldCreate();

    std::vector<SamplesDevice*> samples;

    printf("Uploading samples...");
    fflush(stdout);
    int max_it = 128;
    for (int i = 1; i < max_it ; i++) {
        std::ostringstream ss;
        ss << "input/cbox-emissive-simple_" << i << ".samples";

        std::unique_ptr<openpgl::cpp::SampleStorage> sampleStorage;
        try { sampleStorage = std::make_unique<openpgl::cpp::SampleStorage>(ss.str()); }
        catch(const std::runtime_error& e)
        {
            printf(" stopped early.\n");
            break;
        }

        samples.push_back(SamplesDeviceCreate(*sampleStorage));

        if (i == max_it - 1)
            printf(" done.\n");
    }

    for (auto samplesDevice : samples)
        GPUFieldUpdate(field, samplesDevice);

    for (auto samplesDevice : samples)
        SamplesDeviceDestroy(samplesDevice);

    GPUFieldDestroy(field);

    return 0;
}

