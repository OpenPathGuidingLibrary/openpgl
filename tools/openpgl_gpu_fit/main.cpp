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

    for (int i = 1; i < 128; i++) {
        std::ostringstream ss;
        ss << "input/cbox-emissive-simple_" << i << ".samples";

        std::unique_ptr<openpgl::cpp::SampleStorage> sampleStorage;
        try { sampleStorage = std::make_unique<openpgl::cpp::SampleStorage>(ss.str()); }
        catch(const std::runtime_error& e)
        {
            std::cerr << e.what() << '\n';
            break;
        }
        
        GPUFieldUpdate(field, *sampleStorage);
    }

    GPUFieldDestroy(field);

    return 0;
}

