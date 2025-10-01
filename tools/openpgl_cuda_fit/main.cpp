#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include "openpgl/cuda.h"

#include "openpgl/cpp/Field.h"
#include "openpgl/cpp/Device.h"
#include "openpgl/cpp/SampleStorage.h"
#include "../../openpgl/include/openpgl/breadcrump.h"

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

    if (!(a->axis == b->axis && a->split == b->split)) {
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
    auto fieldGPU = pglNewFieldCUDA();

    std::vector<std::unique_ptr<cpp::SampleStorage>> sampleStoragesCPU;
    std::vector<PGLSampleStorageCUDA> sampleStoragesGPU;

    bool inlineUpdate = true;
    bool validateCPU = true;
    auto update = [&](int i, cpp::SampleStorage* samplesCPU, PGLSampleStorageCUDA samplesGPU) {
        SDump *sDumpCPU = nullptr;
        if (validateCPU) {
            fieldCPU.Update(*samplesCPU);
            //double time = timer.elapsed();
            //printf("cpu time: %fms\n", time*1e3);
            sDumpCPU = new SDump;
            //fieldCPU.sDump(sDumpCPU);
            fieldCPU.Dump(std::string("dump/CPU_") + std::to_string(i));
        }

        pglFieldCUDAUpdate(fieldGPU, samplesGPU);

        //CudaTimer timer;
        //printf("time: %fms\n", timer.elapsed()*1e3);

        if (!validateCPU) return;

        SDump *sDumpGPU = new SDump;
        //cuda::GPUFieldSDump(fieldGPU, sDumpGPU);
        
        //compSDump(sDumpGPU, sDumpCPU);
    };

    printf("Uploading samples...\n");
    fflush(stdout);
    int max_it = 10;
    for (int i = 1; i < max_it ; i++) {
        std::ostringstream ss;
        ss << "input/cbox-emissive-simple_" << i << ".samples";
        std::string str = ss.str();
        
        //std::unique_ptr<openpgl::cpp::SampleStorage> sampleStorageCPU;
        //try { sampleStorageCPU = std::make_unique<openpgl::cpp::SampleStorage>(ss.str()); }
        //catch(const std::runtime_error& e)
        //{
        //    printf(" stopped early.\n");
        //    break;
        //}

        auto sampleStorageCPU = std::unique_ptr<cpp::SampleStorage>(new cpp::SampleStorage(str));
        auto sampleStorageGPU = pglNewSampleStorageCUDAFromFile(str.c_str());

        if (inlineUpdate) {
            update(i, &*sampleStorageCPU, sampleStorageGPU);
            pglReleaseSampleStorageCUDA(sampleStorageGPU);
        } else {
            sampleStoragesCPU.push_back(std::move(sampleStorageCPU));
            sampleStoragesGPU.push_back(sampleStorageGPU);
        }


        if (i == max_it - 1)
            printf(" done.\n");
    }

    //for (auto i = 0; !inlineUpdate && i < sampleStoragesCPU.size(); i++)
    //    update(i);

    for (auto& sampleStorageGPU : sampleStoragesGPU)
        pglReleaseSampleStorageCUDA(sampleStorageGPU);

    pglReleaseFieldCUDA(fieldGPU);

    return 0;
}

