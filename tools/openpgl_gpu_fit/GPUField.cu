#include "GPUField.h"

#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

#include "directionalkernel.cuh"

#include <cstdio>
#include <iostream>
#include <unordered_set>
#include <sstream>
#include "../../openpgl/data/BlobWriter.h"

//#define OPENPGL_GPU_CUDA
//#include "openpgl/gpu/OpenPGLGPU.h"



#include "../../openpgl/data/SampleDataStorage.h"

#include "gpu_fit.h"
#include "util.h"
#include "timer.h"

namespace embree {
    bool isvalid(float &val) { return true; }
}

namespace openpgl{
namespace gpu {
namespace cuda {

struct SamplesDevice {
    thrust::device_vector<PGLSampleData> surface, volume;
};

struct BuildSettings {
    uint32_t maxSamples;
    bool firstIteration = false;
};

void cudaCheck() {
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void checkUsage() {
//#ifndef NDEBUG
    cudaCheck();
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    float ratio = (float)free/(float)total;
    if (ratio < 0.6) {
        printf("!!! %f %llu/%llu\n", ratio, free, total);
    }
//#endif
}

HOST_DEVICE Vector3 toVec3(pgl_vec3f vec) {
    return Vector3(vec.x, vec.y, vec.z);
}


HOST_DEVICE bool isValid(float val) {
    return -std::numeric_limits<float>::max() <= val && val <= std::numeric_limits<float>::max();
}

HOST_DEVICE bool isValid(Vector3 val) {
    return isValid(val.x) && isValid(val.y) && isValid(val.z);
}

template<typename T>
T *data(thrust::device_vector<T> &vector) {
    return thrust::raw_pointer_cast(vector.data());
}

template<typename T, typename U>
HOST_DEVICE T ceilDiv(T a, U b) {
    return (a + b - 1) / b;
}

template <typename Kernel, typename... Args>
void launchSMem(const std::string& name, Kernel kernel, int num_blocks, int block_size, int smem_size, Args&&... args) {
    CudaTimer timer;
    kernel<<<num_blocks, block_size, smem_size>>>(std::forward<Args>(args)...);
    printf("%s: %fms\n", name.c_str(), 1e3*timer.elapsed());
    checkUsage();
}

template <typename Kernel, typename... Args>
void launch(const std::string& name, Kernel kernel, int num_blocks, int block_size, Args&&... args) {
    launchSMem(name, kernel, num_blocks, block_size, 0, std::forward<Args>(args)...);
}

template <typename Kernel, typename... Args>
void launchThreads(const std::string& name, Kernel kernel, int num_elements, int block_size, Args&&... args) {
    const int blocks = ceilDiv(num_elements, block_size);
    launch(name, kernel, blocks, block_size, std::forward<Args>(args)...);
}

__device__ uint32_t globalIdx() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ uint32_t numThreads() {
    return gridDim.x * blockDim.x;
}

__device__ uint32_t traverseTree(const Vector3 &position, const TreeNode *tree) {
    uint32_t n = 0;
    TreeNode node;
    for (int i = 0;; i++) {
        node = tree[n];
        if (node.isLeaf())
            break;
        auto base = node.getNodeIdx();
        auto offset = (position[node.getSplitDim()] < node.pivot) ? 0 : 1;
        n = base + offset;
    }
    return n;
}

__global__ void BinSamples(
    const TreeNode *tree, const uint32_t numSamples, const PGLSampleData *samples,
    uint32_t *leafIndices, uint32_t *leafHistogram, uint32_t *sampleOffsets
) {
    int i = globalIdx();

    if (!(i < numSamples)) return;

    const Vector3 samplePosition = toVec3(samples[i].position);
    uint32_t n = traverseTree(samplePosition, tree);
    leafIndices[i] = n;
    sampleOffsets[i] = atomicAdd(&leafHistogram[n], 1);
}

__global__ void ScatterSamples(
    const uint32_t numSamples, const PGLSampleData *srcSamples, const uint32_t *srcLeafIndices,
    const uint32_t *leafHistogram, const uint32_t *sampleOffsets, 
    PGLSampleData *dstSamples, uint32_t *dstLeafIndices
) {
    int i = globalIdx();
    if (!(i < numSamples)) return;

    const uint32_t n = srcLeafIndices[i];
    const uint32_t j = leafHistogram[n] + sampleOffsets[i];
    dstSamples[j] = srcSamples[i];
    dstLeafIndices[j] = n;
}

__global__
__launch_bounds__(768, 2)
void AggregateSamples(
    const State *state, const TreeNode *tree, const QuantizationFrame* quantizationFrame,
    const uint32_t numSamples, const PGLSampleData *samples,
    Vector3 *samplePositions, IntegerSampleStats *newSampleStats
)
{
    constexpr int numLocalNodes = 256 + 128;
    __shared__ IntegerSampleStats localStats[numLocalNodes];

    constexpr int localSamples = 3;
    bool valid[localSamples];
    Vector3 samplePosition[localSamples];
    uint n[localSamples];
    // TODO load tree into shared memory
    for (int i = 0; i < localSamples; i++) {
        int idx = globalIdx() + i * numThreads();
        valid[i] = idx < numSamples;
        if (valid[i]) {
            if (samples) {
                samplePosition[i] = toVec3(samples[idx].position);
                samplePositions[idx] = samplePosition[i];
            }
            else
                samplePosition[i] = samplePositions[idx];

            n[i] = traverseTree(samplePosition[i], tree);
        } else {
            samplePosition[i] = Vector3(0);
            n[i] = ~0;
        }
    }    
    
    FOREACH(j, 0, numLocalNodes)
        localStats[j] = IntegerSampleStats();

    for (int base = 0; base < ceilDiv(state->nodeAlloc, numLocalNodes) * numLocalNodes; base += numLocalNodes) {
        __syncthreads();

        for (int i = 0; i < localSamples; i++) {
            if (valid[i] && base <= n[i] && n[i] < base + numLocalNodes) {
                QuantizationFrame frame = quantizationFrame[n[i]];
                auto stats = IntegerSampleStats(frame, samplePosition[i]);
                localStats[n[i] - base].atomicReduce(stats);
            }
        }

        __syncthreads();

        FOREACH(j, 0, numLocalNodes) {
            if (localStats[j].numSamples > 0) {
                newSampleStats[base + j].atomicReduce(localStats[j]);
                localStats[j] = IntegerSampleStats();
            }
        }
    }
}

__global__ void SplitNodes(
    const BuildSettings buildSetings, const uint32_t numLeafIndices, const uint32_t *leafIndices, const IntegerSampleStats *newSampleStats,
    State *state, TreeNode *tree, QuantizationFrame* quantizationFrame, SampleStatistics *sampleStats, Record* records, uint32_t *finishedNodes
) {
    int i = globalIdx();
    if (!(i < numLeafIndices)) return;
    int n = leafIndices ? leafIndices[i] : i;

    TreeNode node = tree[n];
    if (!node.isLeaf()) return;

    int bits = std::numeric_limits<uint32_t>::digits;
    uint32_t finishedI = n / bits, finishedMask = 1 << (n % bits); 
    if (finishedNodes[finishedI] & finishedMask) return;

    //uint32_t s = node.getNodeIdx();
    QuantizationFrame frame = quantizationFrame[n];

    SampleStatistics mergedStats = sampleStats[n];
    mergedStats.merge(newSampleStats[i].toSampleStats(frame));

    Record record;
    if (buildSetings.firstIteration)
        record.readIdx = n;
    else
        record = records[n];

    if (mergedStats.numSamples > buildSetings.maxSamples) {
        //printf("%f %i\n", mergedStats.count, buildSetings.maxSamples);

        //uint32_t lLeafIdx = s;
        //uint32_t rLeafIdx = atomicAdd(&state[0].leafAlloc, 1);
        uint32_t childIdx = atomicAdd(&state[0].nodeAlloc, 2);

        sampleStats[childIdx + 0] = {};
        sampleStats[childIdx + 1] = {};

        records[childIdx + 0] = record;
        records[childIdx + 1] = record;

        TreeNode left, right;
        left.pivot = 0;
        left.setSplitDimAndNodeIdx(TreeNode::ELeafNode, 0);
        left.bc = node.bc.push(false);
        right.pivot = 0;
        right.setSplitDimAndNodeIdx(TreeNode::ELeafNode, 0);
        right.bc = node.bc.push(true);

        auto maxDimension = [] __device__ (const Vector3 &v) -> uint8_t {
            return v[v[1] > v[0]] > v[2] ? v[1] > v[0] : 2;
        };
        uint8_t dim = maxDimension(mergedStats.variance);
        node.pivot = mergedStats.mean[dim];
        node.setSplitDimAndNodeIdx(dim, childIdx);
        
        tree[n] = node;
        tree[childIdx + 0] = left;
        tree[childIdx + 1] = right;
        if (node.bc.isParent()) {
            printf("   cuda is2: ");
            node.bc.print();
            printf(" %i %.10f\n", (uint32_t)dim, node.pivot);
        }

        if (n != 0) {
            if (frame.nextIsRight)
                frame.aabb.lower[frame.nextSplitDim]  = frame.nextPivot;
            else
                frame.aabb.upper[frame.nextSplitDim]  = frame.nextPivot;
        }
        frame.init();
        frame.nextSplitDim = node.getSplitDim();
        frame.nextPivot  = node.pivot;
        frame.nextIsRight = false;
        quantizationFrame[childIdx + 0] = frame;
        frame.nextIsRight = true;
        quantizationFrame[childIdx + 1] = frame;

        // inform host, that a split has been performed
        atomicMax(&state[0].anySplit, 1);
    } else {
        // TODO update pivot
        sampleStats[n] = mergedStats;

        //records[n] = record;

        // write to finished mask, so this node is not processed again
        atomicOr(&finishedNodes[finishedI], finishedMask);
    }
}

template<class T>
float getSize(const thrust::device_vector<T>& vec) {
    return (vec.capacity() * sizeof(T)) / (1024.f * 1024.f * 1024.f);
}

template<typename T>
void resize(T& vector, size_t size) {
    if (vector.capacity() < size) {
        size_t newCapacity = 1.3 * size;
        vector.reserve(newCapacity);
        printf("new size (GB): %f\n", ((float)sizeof(vector[0]) * newCapacity) / (1024.f*1024.f*1024.f));
    }
    vector.resize(size);
    checkUsage();
}

template<typename T>
void printGBSize(T& vector){
    printf("size (GB): %f\n", ((float)sizeof(vector[0]) * vector.size()) / (1024.f*1024.f*1024.f));

}

GPUField::GPUField() {
    // initialize tree to single root node
    maxNumLeaves = 16*1024;
    maxNumNodes = 2 * maxNumLeaves - 1;

    resize(state, 1);
    resize(tree, maxNumNodes);
    resize(quantizationFrame, maxNumNodes);
    resize(leafStats, maxNumNodes);
    resize(sampleStats, maxNumNodes);
    
    resize(trainingData, maxNumNodes);
    resize(samplingData, maxNumNodes);

    resize(sampleStats, maxNumNodes);

    resize(finishedNodes, ceilDiv(maxNumNodes, std::numeric_limits<uint32_t>::digits));
    resize(records, maxNumNodes);


    hostState.nodeAlloc = 1;
    hostState.anySplit = 0;
    state[0] = hostState;

    TreeNode node;
    node.pivot = 0;
    node.setSplitDimAndNodeIdx(TreeNode::ELeafNode, 0);
    node.bc = Breadcrumb();
    tree[0] = node;

    leafStats[0] = {};

    computeSizes();
}

void GPUField::computeSizes() const {
    float size = 
        getSize(state) +
        getSize(tree) +
        getSize(quantizationFrame) +
        getSize(leafStats) +
        getSize(sampleStats) +

        getSize(trainingData) +
        getSize(samplingData) +

        getSize(leafIndices) +
        getSize(leafHistogram) +
        getSize(sampleOffset) +
        getSize(reorderedSamples) +
        getSize(reorderedLeafIndices) +

        getSize(sampleStats) +

        getSize(finishedNodes);// +
        //getSize(records);
    printf("total size: %f GB\n", size);
}
    
void GPUField::sDump(SDump* sDump) const {

    thrust::host_vector<TreeNode> tree = this->tree;

    std::function<void(SDumpTree*, int)> rec;
    rec = [&](SDumpTree* sDump, int idx) -> void {
        const auto& node = tree[idx];
        if (node.isLeaf()) {
            sDump->left = nullptr;
            sDump->right = nullptr;
        } else {
            sDump->axis = node.getSplitDim();
            sDump->split = node.pivot;
            
            sDump->left = new SDumpTree;
            sDump->right = new SDumpTree;
            rec(sDump->left, node.getNodeIdx() + 0);
            rec(sDump->right, node.getNodeIdx() + 1);
        }
    };

    sDump->sur = new SDumpTree;
    rec(sDump->sur, 0);
    sDump->vol = new SDumpTree;
    sDump->vol->left = nullptr;
    sDump->vol->right = nullptr;
}

void GPUField::UpdateTree(uint32_t numSamples, thrust::device_vector<PGLSampleData> &samples) {
    if (false) {
        thrust::host_vector<PGLSampleData> hSamples = samples;
        std::vector<Vector3> points;
        for (int i = 0; i < hSamples.size(); i++) {
            points.push_back(toVec3(hSamples[i].position));
        }
        write_point_cloud_to_obj(points, "dump/samples.obj");
    }
    
    resize(samplePositions, numSamples);

    resize(leafIndices, numSamples);
    // + 1 so that ranges are always be computed with [hist[i], hist[i + 1]]
    resize(leafHistogram, maxNumNodes + 1);
    resize(sampleOffset, numSamples);
    resize(reorderedSamples, numSamples);
    resize(reorderedLeafIndices, numSamples);

    computeSizes();

    checkUsage();

    CudaTimer wholeTimer;

    // first iteration: estimate scene bounds
    if (it == 0) {
        CudaTimer initTimer;
        hostState.bounds = thrust::transform_reduce(samples.begin(), samples.end(),
            [] HOST_DEVICE (const PGLSampleData &x) { 
                OPENPGL_ASSERT(isValid(toVec3(x.position)));
                auto res = BBox(toVec3(x.position));
                OPENPGL_ASSERT(is_finite(res));
                return res;
            },
            BBox(),
            [] HOST_DEVICE (const BBox &a, const BBox &b) -> BBox {
                auto res = BBox::merge(a, b);
                OPENPGL_ASSERT(is_finite(res));
                return res;
            }
        );
        checkUsage();
        BBox bounds = hostState.bounds;
        //bounds.print();
        Vector3 center = (bounds.lower + bounds.upper) / 2.f;
        bounds.lower = center + 3.f * (bounds.lower - center);
        bounds.upper = center + 3.f * (bounds.upper - center);
        hostState.bounds = bounds;
        state[0] = hostState;
        QuantizationFrame frame;
        frame.aabb = hostState.bounds;
        frame.init();
        quantizationFrame[0] = frame;
        trainingData[0] = TrainingData {
            .initialized = false,
        };
        printf("init: %fms\n", initTimer.elapsed() * 1e3f);
    }


    thrust::fill(finishedNodes.begin(), finishedNodes.end(), 0);
    checkUsage();

    //printf("  nodeAlloc: %i anySplit: %i\n", hostState.nodeAlloc, hostState.anySplit);

    BuildSettings buildSettings;
    buildSettings.maxSamples = 32000;
    buildSettings.firstIteration = true;

    CudaTimer spatialTimer;
    do {
        // clear buffers
        thrust::fill(sampleStats.begin(), sampleStats.begin() + hostState.nodeAlloc, IntegerSampleStats());
    
        // accumulate stats
        launchThreads(" AggregateSamples",
            AggregateSamples, ceilDiv(numSamples, 3), 768,
            data(state), data(tree), data(quantizationFrame),
            numSamples, data(samples),
            data(samplePositions), data(sampleStats)
        );

        hostState.anySplit = 0;
        state[0] = hostState;
        launchThreads(" SplitNodes",
            SplitNodes, hostState.nodeAlloc, 128,
            buildSettings, hostState.nodeAlloc, nullptr, data(sampleStats),
            data(state), data(tree), data(quantizationFrame), data(leafStats), data(records), data(finishedNodes)
        );
        hostState = state[0];

        printf(" nodeAlloc: %i anySplit: %i\n", hostState.nodeAlloc, hostState.anySplit);
    
        buildSettings.firstIteration = false;
    } while(hostState.anySplit);
    printf("spatial: %fms\n", spatialTimer.elapsed() * 1e3f);

    CudaTimer scatterTimer;
    {
        // bin samples according to leaf nodes
        thrust::fill(leafHistogram.begin(), leafHistogram.end(), 0);
        launchThreads(" BinSamples", BinSamples, numSamples, 128, data(tree), numSamples, data(samples), data(leafIndices), data(leafHistogram), data(sampleOffset));

        if (false) {
            cudaDeviceSynchronize();
            thrust::host_vector<uint32_t> hLeafHistogram = leafHistogram;
            uint sum = 0;
            for (int i = 0; i < hostState.nodeAlloc; i++) {
                sum += hLeafHistogram[i];
                printf("    %zu: %zu\n", i, hLeafHistogram[i]);
            }
            uint rsum = 0;
            for (int i = hostState.nodeAlloc; i < maxNumLeaves; i++) {
                rsum += hLeafHistogram[i];
            }
            assert(rsum == 0);
            assert(sum == numSamples);
        }

        thrust::exclusive_scan(leafHistogram.begin(), leafHistogram.end(), leafHistogram.begin());
        launchThreads(" ScatterSamples",
            ScatterSamples, numSamples, 128,
            numSamples, data(samples), data(leafIndices), data(leafHistogram), data(sampleOffset),
            data(reorderedSamples), data(reorderedLeafIndices)
        );
    }
    printf("scatter: %fms\n", scatterTimer.elapsed() * 1e3f);

    Factory::Configuration cfg;
    launchSMem("EMFit", EMFit, hostState.nodeAlloc, BlockDim,  5852 /*14200*/,
        cfg, data(tree), data(records), data(leafHistogram), data(leafStats),
        data(trainingData), data(samplingData), data(reorderedSamples)
    );

    printf("whole: %fms\n", wholeTimer.elapsed() * 1e3f);


    //if (false) {
    //    std::vector<BBox> boxes;
    //    for (int i = 0; i < hostState.leafAlloc; i++) {
    //        openpgl::SampleStatistics stats = leafStats[i];
    //        boxes.push_back(stats.sampleBounds);
    //    }
    //
    //    writeBoundingBoxes(boxes, std::string("dump/") + std::to_string(it) + std::string(".obj"));
    //}
    it++;

    dump(std::string("dump/") + std::to_string(it) + std::string(".dump"));
}

void GPUField::Update(thrust::device_vector<PGLSampleData> &samples) {
    printf("Update: %i surface samples, %i volume samples\n", samples.size(), 0);

    printf(" updating tree\n");
    UpdateTree(samples.size(), samples);
}

void GPUField::dump(const std::string& dumpFileName) const {
    BlobWriter writer(dumpFileName);

    thrust::host_vector<TreeNode> hTree = tree;
    thrust::host_vector<SamplingData> hSamplingData = samplingData;
    thrust::host_vector<TrainingData> hTrainingData = trainingData;
    thrust::host_vector<SampleStatistics> hSampleStatistics = leafStats;

    //writer << (uint64_t) hTree.size();

    for (int n = 0; n < hTree.size(); n++) {
        TreeNode node = hTree[n];

        if (!node.isLeaf()) continue;

        SamplingData samplingData = hSamplingData[n];
        VMM vmm = samplingData.vmm;
        TrainingData trainingData = hTrainingData[n];
        SampleStatistics sampleStatistics = hSampleStatistics[n];

        BBox bbox = sampleStatistics.sampleBounds;
        Vector3 center = bbox.center();
        Vector3 size = bbox.size();

        // write spatial
        // clang-format off
        writer << (float)center.x << (float)center.y << (float)center.z 
            << (float)size.x << (float)size.y << (float)size.z
            << (float)0.5                                          // sampling.mean()
            << (uint64_t)sampleStatistics.getNumSamples()                                        // sampling.statisticalWeight()
            << (uint64_t)n                                             // this serves as ID to identify cells from visualizer
            << (float)0.5
            << (float)0.5
            << (float)0.5 << (float)0.5 << (float)0.5 /*irradiance.z*/
            << (float)0.f << (float)0.f << (float)0.f 
            //   << region.getBsdfSamplingFraction() << region.getBsdfSamplingFraction() /*product.z*/
            << (float)1.f /*normal.x*/ << (float)1.f /*normal.y*/ << (float)1.f;        /*normal.z;*/

        // write VMM
        //writer << (uint64_t)(sizeof(uint64_t) + vmm.getNumComponents() * 5 * sizeof(float));
        writer << (uint64_t)vmm.getNumComponents();
        
        for (int k = 0; k < vmm.getNumComponents(); k++) {
            writer << vmm._kappas[k]
                << vmm._meanDirections[k].x << vmm._meanDirections[k].y << vmm._meanDirections[k].z
                << vmm._weights[k];
        }
    }
}

openpgl::gpu::cuda::GPUField *GPUFieldCreate() {
    return new openpgl::gpu::cuda::GPUField();
}

void GPUFieldDestroy(openpgl::gpu::cuda::GPUField *field) {
    delete field;
}

void GPUFieldUpdate(openpgl::gpu::cuda::GPUField *field, SamplesDevice *samples) {
    field->Update(samples->surface);
}

void GPUFieldSDump(openpgl::gpu::cuda::GPUField *field, SDump* sDump) {
    field->sDump(sDump);
}

SamplesDevice* SamplesDeviceCreate(const std::string &path) {
    SampleDataStorage* storage = SampleDataStorage::newSampleDataStorageFromFile(path);
    
    thrust::host_vector<PGLSampleData> surface(storage->sizeSurface()), volume(storage->sizeVolume());
    for (int i = 0; i < storage->sizeSurface(); i++) {
        auto s =  storage->getSampleSurface(i);
        OPENPGL_ASSERT(isValid(toVec3(s.position)));
        surface[i] = s;
    }
    for (int i = 0; i < storage->sizeVolume(); i++)
        volume[i] = storage->getSampleVolume(i);
    auto samplesDevice = new SamplesDevice {.surface = surface, .volume = volume};
    printf("samples size: %f GB\n", getSize(samplesDevice->surface) + getSize(samplesDevice->volume));

    return samplesDevice;
}

void SamplesDeviceDestroy(SamplesDevice* samplesDevice) {
    delete samplesDevice;
}

}
} 
}
