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

//#define OPENPGL_GPU_CUDA
//#include "openpgl/gpu/OpenPGLGPU.h"


namespace embree {
    bool isvalid(float &val) { return true; }
}

#include "../../openpgl/data/SampleDataStorage.h"

#include "gpu_fit.h"
#include "util.h"


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

void checkUsage() {
    cudaDeviceSynchronize();
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    float ratio = (float)free/(float)total;
    if (ratio < 0.6) {
        printf("!!! %f %llu/%llu\n", ratio, free, total);
    }
}

void checkCudaError() {
    cudaDeviceSynchronize();
    checkUsage();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

HOST_DEVICE Vector3 toVec3(pgl_vec3f vec) {
    return Vector3(vec.x, vec.y, vec.z);
}


HOST_DEVICE bool isValid(float val) {
    return std::numeric_limits<float>::min() <= val && val <= std::numeric_limits<float>::max();
}

template<typename T>
T *data(thrust::device_vector<T> &vector) {
    return thrust::raw_pointer_cast(vector.data());
}

template<typename T, typename U>
T ceilDiv(T a, U b) {
    return (a + b - 1) / b;
}

template <typename Kernel, typename... Args>
void launchThreads(Kernel kernel, int num_elements, int block_size, Args&&... args) {
    const int blocks = ceilDiv(num_elements, block_size);
    kernel<<<blocks, block_size>>>(std::forward<Args>(args)...);
    checkCudaError();
}

template <typename Kernel, typename... Args>
void launch(Kernel kernel, int num_blocks, int block_size, Args&&... args) {
    kernel<<<num_blocks, block_size>>>(std::forward<Args>(args)...);
    checkCudaError();
}

__device__ uint32_t globalIdx() {
    return blockIdx.x * blockDim.x + threadIdx.x;
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

__global__ void SplitNodes(
    const BuildSettings buildSetings, const uint32_t numLeafIndices, const uint32_t *leafIndices, const IntegerSampleStats *newSampleStats,
    State *state, TreeNode *tree, QuantizationFrame* quantizationFrame, SampleStatistics *sampleStats, Record* records, uint32_t *finishedNodes
) {
    int i = globalIdx();
    if (!(i < numLeafIndices)) return;
    const uint32_t n = leafIndices[i];
    
    int bits = std::numeric_limits<uint32_t>::digits;
    uint32_t finishedI = n / bits, finishedMask = 1 << (n % bits); 
    if (finishedNodes[finishedI] & finishedMask) return;

    TreeNode node = tree[n];
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
            printf("cuda is2: ");
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
    
    resize(trainingData, maxNumNodes);
    resize(samplingData, maxNumNodes);

    resize(reducedLeafIndices, maxNumNodes);
    resize(reducedSampleStats, maxNumNodes);
    resize(scatteredSampleStats, maxNumNodes);

    resize(finishedNodes, ceilDiv(maxNumNodes, std::numeric_limits<uint32_t>::digits));
    resize(records, maxNumNodes);


    hostState.nodeAlloc = 1;
    hostState.leafAlloc = 1;
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

        getSize(trainingData) +
        getSize(samplingData) +

        getSize(leafIndices) +
        getSize(leafHistogram) +
        getSize(sampleOffset) +
        getSize(reorderedSamples) +
        getSize(reorderedLeafIndices) +

        getSize(sampleStats) +
        getSize(reducedLeafIndices) +
        getSize(reducedSampleStats) +
        getSize(scatteredSampleStats) +

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
    {
        thrust::device_vector<int> keys = {1, 1, 1, 2, 2, 2, 3};
        thrust::device_vector<Vector3> values(keys.size(), Vector3(1.f));
        thrust::device_vector<int> keysO(keys.size());
        thrust::device_vector<Vector3> valuesO(values.size());
        auto [keysOEnd, valuesOEnd] = thrust::reduce_by_key(keys.begin(), keys.end(), values.begin(), keysO.begin(), valuesO.end());            
        printf("lol: %llu\n", keysOEnd - keysO.begin());
    }

    if (false) {
        thrust::host_vector<PGLSampleData> hSamples = samples;
        std::vector<Vector3> points;
        for (int i = 0; i < hSamples.size(); i++) {
            points.push_back(toVec3(hSamples[i].position));
        }
        write_point_cloud_to_obj(points, "dump/samples.obj");
    }
    
    resize(leafIndices, numSamples);
    // + 1 so that ranges are always be computed with [hist[i], hist[i + 1]]
    resize(leafHistogram, maxNumNodes + 1);
    resize(sampleOffset, numSamples);
    resize(reorderedSamples, numSamples);
    resize(reorderedLeafIndices, numSamples);

    resize(sampleStats, numSamples);

    resize(reducedLeafIndices, numSamples);
    resize(reducedSampleStats, numSamples);
    resize(scatteredSampleStats, numSamples);

    computeSizes();

    checkCudaError();

    // first iteration: estimate scene bounds
    if (it == 0) {
        hostState.bounds = thrust::transform_reduce(samples.begin(), samples.end(),
            [] HOST_DEVICE (const PGLSampleData &x) { return BBox(toVec3(x.position)); },
            BBox(),
            [] HOST_DEVICE (const BBox &a, const BBox &b) { return BBox::merge(a, b); }
        );
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
    }

    thrust::fill(finishedNodes.begin(), finishedNodes.end(), 0);
    checkCudaError();

    printf("  nodeAlloc: %i leafAlloc: %i anySplit: %i\n", hostState.nodeAlloc, hostState.leafAlloc, hostState.anySplit);

    BuildSettings buildSettings;
    buildSettings.maxSamples = 32000;
    buildSettings.firstIteration = true;

    size_t numLeafIndices;

    do {
        // TODO whenever a leaf split is happening, this loop repeats all over, considering all samples again.
        // We might want to partition split nodes from unsplit nodes to reduce the working set.
        // Note that this might not be a big issue:
        // In the beginning most leaves need to be split repeatedly, so there is not much to be gained.
        // Then, in later iterations, this loop will probably run at most two times

        // bin samples according to leaf nodes
        printf("   bin\n");
        thrust::fill(leafHistogram.begin(), leafHistogram.end(), 0);
        printf("    BinSamples\n");
        launchThreads(BinSamples, numSamples, 128, data(tree), numSamples, data(samples), data(leafIndices), data(leafHistogram), data(sampleOffset));

        checkCudaError();
    
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
        checkCudaError();

        thrust::exclusive_scan(leafHistogram.begin(), leafHistogram.end(), leafHistogram.begin());
        printf("    ScatterSamples\n");
        launchThreads(
            ScatterSamples, numSamples, 128,
            numSamples, data(samples), data(leafIndices), data(leafHistogram), data(sampleOffset),
            data(reorderedSamples), data(reorderedLeafIndices)
        );
        checkCudaError();

        // aggregate sample statistics
        // MEM, PERF: perform initial reduction in warps to reduce memory impact of stats
        printf("   aggregate\n");
        {
            QuantizationFrame* framePtr = data(quantizationFrame);
            thrust::transform(
                thrust::make_zip_iterator(thrust::make_tuple(reorderedLeafIndices.begin(), reorderedSamples.begin())),
                thrust::make_zip_iterator(thrust::make_tuple(reorderedLeafIndices.end(),   reorderedSamples.end())),
                sampleStats.begin(),
                [framePtr] __device__ (const thrust::tuple<uint32_t, PGLSampleData>& t) {
                    return IntegerSampleStats(
                        framePtr[thrust::get<0>(t)],
                        toVec3(thrust::get<1>(t).position)
                    );
                }
            );
        }
        checkCudaError();

        {
            thrust::host_vector<IntegerSampleStats> o = sampleStats;
            auto *ptr = data(o);
            printf("stats: %lli\n", ptr->mean[0]);
        }

        printf("%llu\n", reorderedLeafIndices.end() - reorderedLeafIndices.begin());

        numLeafIndices = reduceSampleStats();
        //// PERF: reduce_by_key forces synchronization to return number of reduced elements
        //// we might be able to avoid this, by directly using ReduceByKey from cub
        //auto [reducedLeafIndicesEnd, reducedSampleStatsEnd] = thrust::reduce_by_key(
        //    reorderedLeafIndices.begin(), reorderedLeafIndices.end(), sampleStats.begin(),
        //    reducedLeafIndices.begin(), reducedSampleStats.begin(),
        //    thrust::equal_to<uint32_t>(),
        //    [] __device__ (const IntegerSampleStats &a, const IntegerSampleStats &b) {
        //        //return IntegerSampleStats();
        //        return IntegerSampleStats(a, b);
        //    }
        //);
        //numLeafIndices = reducedLeafIndicesEnd - reducedLeafIndices.begin();
        checkCudaError();
        assert(numLeafIndices > 0);
        
        if (false) {
            for (int i = 0; i < numLeafIndices; i++) {
                int idx = reducedLeafIndices[i];
                TreeNode node = tree[idx];
                if (!node.bc.isParent()) continue;
                QuantizationFrame qframe = quantizationFrame[idx];
                std::cout << node.bc.toString() << std::endl;
                //std::cout << qframe.aabb.toString() << std::endl;
                IntegerSampleStats iStats = reducedSampleStats[i];
                //std::cout << iStats.toString(quantizationFrame[idx]) << std::endl;
                //std::cout << iStats.toSampleStats(quantizationFrame[i]).toString() << std::endl;
            }
        }
        //std::cout << iStats.toSampleStats(quantizationFrame[0]).toString();

        // Split nodes
        printf("   split\n");
        hostState.anySplit = 0;
        state[0] = hostState;
        hostState = state[0];
        launchThreads(
            SplitNodes, numLeafIndices, 128,
            buildSettings, numLeafIndices, data(reducedLeafIndices), data(reducedSampleStats),
            data(state), data(tree), data(quantizationFrame), data(leafStats), data(records), data(finishedNodes)
        );
        hostState = state[0];

        printf("  nodeAlloc: %i leafAlloc: %i anySplit: %i\n", hostState.nodeAlloc, hostState.leafAlloc, hostState.anySplit);

        buildSettings.firstIteration = false;
    } while(hostState.anySplit);
    

    printf("updating!\n");
    Factory::Configuration cfg;
    EMFit<<<numLeafIndices, 32>>>(
        cfg, data(reducedLeafIndices), data(records), data(leafHistogram), data(leafStats),
        data(trainingData), data(samplingData), data(samples)
    );
    checkUsage();
    printf("updating finished!\n");

    if (false) {
        std::vector<BBox> boxes;
        for (int i = 0; i < hostState.leafAlloc; i++) {
            openpgl::SampleStatistics stats = leafStats[i];
            boxes.push_back(stats.sampleBounds);
        }

        writeBoundingBoxes(boxes, std::string("dump/") + std::to_string(it) + std::string(".obj"));
    }
    it++;
}

void GPUField::Update(thrust::device_vector<PGLSampleData> &samples) {
    printf("Update: %i surface samples, %i volume samples\n", samples.size(), 0);

    printf(" updating tree\n");
    UpdateTree(samples.size(), samples);
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
    for (int i = 0; i < storage->sizeSurface(); i++)
        surface[i] = storage->getSampleSurface(i);
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
