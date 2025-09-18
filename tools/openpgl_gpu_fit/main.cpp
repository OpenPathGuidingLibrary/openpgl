#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include "openpgl/cpp/Field.h"
#include "openpgl/cpp/Device.h"
#include "openpgl/cpp/SampleStorage.h"
#include "../../openpgl/include/openpgl/breadcrump.h"

#include "gpu_fit.h" // Include the header for our CUDA function
#include "timer.h"
using namespace openpgl;
using namespace openpgl::gpu;

bool compSDumpTree(const SDumpTree *a, const SDumpTree *b, Breadcrumb bc) {
    bool aLeaf = a->left == nullptr, bLeaf = b->left == nullptr;

    if (aLeaf != bLeaf) {
        std::cout << "unequal Tree node leaf state: " << bc.toString() << std::endl
                  << "   leaf: " << aLeaf << ", " << bLeaf << std::endl;
        return false;
    }

    if (aLeaf) {
        return true;
    }

    if (!(a->axis == b->axis && std::abs(a->split - b->split) <= 1e-8)) {
        std::cout << "unequal Tree node: " << bc.toString() << std::endl
                  << "   axis: " << (uint32_t)a->axis << ", " << (uint32_t)b->axis << std::endl
                  << "  pivot: " << a->split << ", " << b->split << std::endl;
        return false;
    }
    return compSDumpTree(a->left, b->left, bc.push(false)) & compSDumpTree(a->right, b->right, bc.push(true));
}

bool compSDump(const SDump* a, const SDump *b) {
    int surI = 0, volI = 0;
    return compSDumpTree(a->sur, b->sur, Breadcrumb()) & compSDumpTree(a->vol, b->vol, Breadcrumb());
}

int main() {
    auto device = cpp::Device(PGL_DEVICE_TYPE_CPU_8);
    cpp::FieldConfig config;
    config.Init(PGL_SPATIAL_STRUCTURE_KDTREE, PGL_DIRECTIONAL_DISTRIBUTION_PARALLAX_AWARE_VMM);
    auto fieldCPU = cpp::Field(&device, config);
    auto fieldGPU = cuda::GPUFieldCreate();

    std::vector<std::unique_ptr<cpp::SampleStorage>> sampleStoragesCPU;
    std::vector<cuda::SamplesDevice*> sampleStoragesGPU;

    bool inlineUpdate = true;
    bool validateCPU = true;
    auto update = [&](int i, cpp::SampleStorage* samplesCPU, cuda::SamplesDevice* samplesGPU) {
        cuda::checkUsage();
        SDump *sDumpCPU = nullptr;
        if (validateCPU) {
            cuda::checkUsage();
            CudaTimer timer;
            fieldCPU.Update(*samplesCPU);
            double time = timer.elapsed();
            printf("time: %fms\n", time*1e3);
            cuda::checkUsage();
            sDumpCPU = new SDump;
            cuda::checkUsage();
            fieldCPU.sDump(sDumpCPU);
            cuda::checkUsage();
            fieldCPU.Dump(std::string("dump/CPU_") + std::to_string(i));
        }
        cuda::checkUsage();
        CudaTimer timer;
        GPUFieldUpdate(fieldGPU, &*samplesGPU);
        printf("time: %fms\n", timer.elapsed()*1e3);

        cuda::checkUsage();

        if (!validateCPU) return;

        cuda::checkUsage();
        SDump *sDumpGPU = new SDump;
        cuda::checkUsage();
        cuda::GPUFieldSDump(fieldGPU, sDumpGPU);
        cuda::checkUsage();

        compSDump(sDumpGPU, sDumpCPU);
    };

    printf("Uploading samples...\n");
    fflush(stdout);
    int max_it = 2;
    for (int i = 1; i < max_it ; i++) {
        std::ostringstream ss;
        ss << "input/cbox-emissive-simple_" << i << ".samples";
        
        //std::unique_ptr<openpgl::cpp::SampleStorage> sampleStorageCPU;
        //try { sampleStorageCPU = std::make_unique<openpgl::cpp::SampleStorage>(ss.str()); }
        //catch(const std::runtime_error& e)
        //{
        //    printf(" stopped early.\n");
        //    break;
        //}

        cuda::checkUsage();
        auto sampleStorageCPU = std::unique_ptr<cpp::SampleStorage>(new cpp::SampleStorage(ss.str()));
        auto sampleStorageGPU = cuda::SamplesDeviceCreate(ss.str());
        cuda::checkUsage();

        if (inlineUpdate) {
            cuda::checkUsage();
            update(i, &*sampleStorageCPU, sampleStorageGPU);
            cuda::checkUsage();
            cuda::SamplesDeviceDestroy(sampleStorageGPU);
            cuda::checkUsage();
        } else {
            cuda::checkUsage();
            sampleStoragesCPU.push_back(std::move(sampleStorageCPU));
            cuda::checkUsage();
            sampleStoragesGPU.push_back(sampleStorageGPU);
            cuda::checkUsage();
        }


        if (i == max_it - 1)
            printf(" done.\n");
    }

    //for (auto i = 0; !inlineUpdate && i < sampleStoragesCPU.size(); i++)
    //    update(i);

    for (auto& sampleStorageGPU : sampleStoragesGPU)
        cuda::SamplesDeviceDestroy(sampleStorageGPU);

    gpu::cuda::GPUFieldDestroy(fieldGPU);

    return 0;
}

