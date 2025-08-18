#include <cstdio>
#include <iostream>
#include <unordered_set>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

#define OPENPGL_GPU_CUDA
#include "openpgl/gpu/OpenPGLGPU.h"

#include "gpu_fit.h"
#include "util.h"

namespace openpgl{
namespace gpu {
namespace cuda {

struct BuildSettings {
    uint32_t maxSamples;
};

struct Alloc {
    uint32_t nodeAlloc;
    uint32_t leafAlloc;
    uint32_t anySplit;
};

struct SampleStats {
    Vector3 mean = {0};
    Vector3 variance = {0};
    Vector3 lower = {std::numeric_limits<float>::infinity()};
    Vector3 upper = {-std::numeric_limits<float>::infinity()};
    float count = 0;

    SampleStats() = default;

    __device__ SampleStats(const Vector3 &position) {
        mean = position;
        variance = Vector3(0);
        lower = position;
        upper = position;
        count = 1;
    }

    __device__ SampleStats(const SampleStats &a, const SampleStats &b) {
        count = a.count + b.count;

        const float weightA = a.count / (a.count + b.count);
        const float weightB = 1.0f - weightA;
        mean = a.mean * weightA + b.mean * weightB;

        // Simple version to calculate the variance of two merged distributions
        // variance = (weightA * (varianceA + meanA * meanA) + weightB * (varianceB + meanB * meanB)) - (mean * mean);
        // Numerical more stable version to calculate the variance of two merged distributions
        const Vector3 meanDiffA = a.mean - mean;
        const Vector3 meanDiffB = b.mean - mean;
        variance = (weightA * a.variance + weightB * b.variance) + (weightA * (meanDiffA * meanDiffA) + weightB * (meanDiffB * meanDiffB));
        variance = max(variance, Vector3(0));
        
        lower = min(a.lower, b.lower);
        upper = max(a.upper, b.upper);
    }
};

struct TreeNode {
    enum
    {
        ESPlitDimX = 0,
        ESPlitDimY = 1,
        ESPlitDimZ = 2,
        ELeafNode = 3,
    };

    const static uint32_t MASK = 0xC0000000;

    float pivot;
    uint32_t splitDimAndNodeIdx{0};

    __device__ uint8_t getSplitDim() {
        return splitDimAndNodeIdx >> 30;
    }
    
    __device__ uint32_t getNodeIdx() {
        return splitDimAndNodeIdx & ~MASK;
    }

    __host__ __device__ void setSplitDimAndNodeIdx(uint8_t splitDim, uint32_t nodeIdx) {
        assert((splitDim & ~3) == 0);
        assert((nodeIdx & MASK) == 0);
        splitDimAndNodeIdx = splitDim << 30 | nodeIdx;
    }

    __device__ bool isLeaf() {
        return getSplitDim() == ELeafNode;
    }

};

template<typename T>
T *data(thrust::device_vector<T> &vector) {
    return thrust::raw_pointer_cast(vector.data());
}

template<typename T, typename U>
T ceilDiv(T a, U b) {
    return (a + b - 1) / b;
}

template <typename Kernel, typename... Args>
void launch(Kernel kernel, int num_elements, int block_size, Args&&... args) {
    const int blocks = ceilDiv(num_elements, block_size);
    kernel<<<blocks, block_size>>>(std::forward<Args>(args)...);
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

    const Vector3 samplePosition(samples[i].position);
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
    const BuildSettings buildSetings, const uint32_t numLeafIndices, const uint32_t *leafIndices, const SampleStats *newSampleStats,
    Alloc *alloc, TreeNode *tree, SampleStats *sampleStats, uint32_t *finishedNodes
) {
    int i = globalIdx();

    if (!(i < numLeafIndices)) return;

    
    const uint32_t n = leafIndices[i];
    
    int bits = std::numeric_limits<uint32_t>::digits;
    uint32_t finishedI = n / bits, finishedMask = 1 << (n % bits); 
    if (finishedNodes[finishedI] & finishedMask) return;

    TreeNode node = tree[n];
    uint32_t s = node.getNodeIdx();
    SampleStats mergedStats = SampleStats(newSampleStats[i], sampleStats[s]);

    if (mergedStats.count > buildSetings.maxSamples) {

        uint32_t lLeafIdx = s;
        uint32_t rLeafIdx = atomicAdd(&alloc[0].leafAlloc, 1);
        uint32_t childIdx = atomicAdd(&alloc[0].nodeAlloc, 2);

        sampleStats[lLeafIdx] = {};
        sampleStats[rLeafIdx] = {};

        TreeNode left, right;
        left.pivot = 0;
        left.setSplitDimAndNodeIdx(TreeNode::ELeafNode, lLeafIdx);
        right.pivot = 0;
        right.setSplitDimAndNodeIdx(TreeNode::ELeafNode, rLeafIdx);

        auto maxDimension = [] __device__ (const Vector3 &v) -> uint8_t {
            return v[v[1] > v[0]] > v[2] ? v[1] > v[0] : 2;
        };
        uint8_t dim = maxDimension(mergedStats.variance);
        node.pivot = mergedStats.mean[dim];
        node.setSplitDimAndNodeIdx(dim, childIdx);
        
        tree[n] = node;
        tree[childIdx + 0] = left;
        tree[childIdx + 1] = right;
        // inform host, that a split has been performed
        atomicMax(&alloc[0].anySplit, 1);
    } else {
        // TODO update pivot
        sampleStats[s] = mergedStats;
        // write to finished mask, so this node is not processed again
        atomicOr(&finishedNodes[finishedI], finishedMask);
    }
}

class GPUField {
    // stats for leaf nodes
    uint32_t maxNumNodes;
    uint32_t maxNumLeaves;
    uint32_t numNodes;
    uint32_t numLeaves;
    Alloc hostAlloc;
    thrust::device_vector<Alloc> alloc;
    thrust::device_vector<TreeNode> tree;
    thrust::device_vector<SampleStats> leafStats;

    int it = 0;

public:
    GPUField() {
        // initialize tree to single root node
        maxNumLeaves = 64*1024;
        maxNumNodes = 2 * maxNumLeaves - 1;

        numNodes = 1;
        numLeaves = 1;
        alloc = thrust::device_vector<Alloc>(1);
        tree = thrust::device_vector<TreeNode>(maxNumNodes);
        leafStats = thrust::device_vector<SampleStats>(maxNumLeaves);

        
        hostAlloc.nodeAlloc = 1;
        hostAlloc.leafAlloc = 1;
        hostAlloc.anySplit = 0;
        alloc[0] = hostAlloc;

        TreeNode node;
        node.pivot = 0;
        node.setSplitDimAndNodeIdx(TreeNode::ELeafNode, 0);
        tree[0] = node;

        leafStats[0] = {};
    }

    void UpdateTree(uint32_t numSamples, thrust::device_vector<PGLSampleData> &samples) {
        thrust::host_vector<PGLSampleData> hSamples = samples;
        std::vector<Vector3> points;
        for (int i = 0; i < hSamples.size(); i++) {
            points.push_back(Vector3(hSamples[i].position));
        }

        write_point_cloud_to_obj(points, "dump/samples.obj");

        // TODO reduce allocations by caching these arrays in the GPUField struct
        // TODO use aliasing between vectors to reduce memory footprint
        thrust::device_vector<uint32_t> leafIndices(numSamples);
        // + 1 so that ranges are always be computed with [hist[i], hist[i + 1]]
        thrust::device_vector<uint32_t> leafHistogram(maxNumNodes + 1);
        thrust::device_vector<uint32_t> sampleOffset(numSamples);
        thrust::device_vector<PGLSampleData> reorderedSamples(numSamples);
        thrust::device_vector<uint32_t> reorderedLeafIndices(numSamples);

        thrust::device_vector<SampleStats> sampleStats(numSamples);
        thrust::device_vector<uint32_t> reducedLeafIndices(maxNumNodes);
        thrust::device_vector<SampleStats> reducedSampleStats(maxNumNodes);
        thrust::device_vector<SampleStats> scatteredSampleStats(maxNumNodes);

        thrust::device_vector<uint32_t> finishedNodes(ceilDiv(maxNumNodes, std::numeric_limits<uint32_t>::digits));
        
        thrust::host_vector<uint32_t> hLeafHistogram(maxNumNodes + 1);


        thrust::fill(finishedNodes.begin(), finishedNodes.end(), 0);

        printf("  nodeAlloc: %i leafAlloc: %i anySplit: %i\n", hostAlloc.nodeAlloc, hostAlloc.leafAlloc, hostAlloc.anySplit);

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
            launch(BinSamples, numSamples, 128, data(tree), numSamples, data(samples), data(leafIndices), data(leafHistogram), data(sampleOffset));

            if (false) {
                cudaDeviceSynchronize();
                hLeafHistogram = leafHistogram;
                uint sum = 0;
                for (int i = 0; i < hostAlloc.nodeAlloc; i++) {
                    sum += hLeafHistogram[i];
                    printf("    %zu: %zu\n", i, hLeafHistogram[i]);
                }
                uint rsum = 0;
                for (int i = hostAlloc.nodeAlloc; i < maxNumLeaves; i++) {
                    rsum += hLeafHistogram[i];
                }
                assert(rsum == 0);
                assert(sum == numSamples);
            }

            thrust::exclusive_scan(leafHistogram.begin(), leafHistogram.end(), leafHistogram.begin());
            printf("    ScatterSamples\n");
            launch(
                ScatterSamples, numSamples, 128,
                numSamples, data(samples), data(leafIndices), data(leafHistogram), data(sampleOffset),
                data(reorderedSamples), data(reorderedLeafIndices)
            );

            // aggregate sample statistics
            // MEM, PERF: perform initial reduction in warps to reduce memory impact of stats
            printf("   aggregate\n");
            thrust::transform(
                reorderedSamples.begin(), reorderedSamples.end(), sampleStats.begin(),
                [] __device__ (const PGLSampleData &x) { 
                    return SampleStats(Vector3(x.position));
                }
            );
            // PERF: reduce_by_key forces synchronization to return number of reduced elements
            // we might be able to avoid this, by directly using ReduceByKey from cub
            auto [reducedLeafIndicesEnd, reducedSampleStatsEnd] = thrust::reduce_by_key(
                reorderedLeafIndices.begin(), reorderedLeafIndices.end(), sampleStats.begin(),
                reducedLeafIndices.begin(), reducedSampleStats.begin(),
                thrust::equal_to<uint32_t>(),
                [] __device__ (const SampleStats &a, const SampleStats &b) {
                    return SampleStats(a, b);
                }
            );
            auto numLeafIndices = reducedLeafIndicesEnd - reducedLeafIndices.begin();

            // Split nodes
            printf("   split\n");
            BuildSettings buildSettings;
            buildSettings.maxSamples = 32000;
            hostAlloc.anySplit = 0;
            alloc[0] = hostAlloc;
            launch(
                SplitNodes, numLeafIndices, 128,
                buildSettings, numLeafIndices, data(reducedLeafIndices), data(reducedSampleStats),
                data(alloc), data(tree), data(leafStats), data(finishedNodes)
            );
            hostAlloc = alloc[0];

            printf("  nodeAlloc: %i leafAlloc: %i anySplit: %i\n", hostAlloc.nodeAlloc, hostAlloc.leafAlloc, hostAlloc.anySplit);
        } while(hostAlloc.anySplit);

        if (false) {
            std::vector<std::pair<Vector3, Vector3>> boxes;
            for (int i = 0; i < hostAlloc.leafAlloc; i++) {
                SampleStats stats = leafStats[i];
                boxes.push_back({stats.lower, stats.upper});
            }
    
            writeBoundingBoxes(boxes, std::string("dump/") + std::to_string(it) + std::string(".obj"));
        }
        it++;
    }

    void Update(const openpgl::cpp::SampleStorage &samples) {
        printf("Update: %i surface samples, %i volume samples\n", samples.GetSizeSurface(), samples.GetSizeVolume());
        
        thrust::host_vector<PGLSampleData> samplesHost(samples.GetSizeSurface());
        printf(" copying samples\n");
        for (int i = 0; i < samples.GetSizeSurface(); i++)
            samplesHost[i] = samples.GetSampleSurface(i);

        printf("  upload to device\n");
        thrust::device_vector<PGLSampleData> samplesDevice = samplesHost;

        printf(" updating tree\n");
        UpdateTree(samples.GetSizeSurface(), samplesDevice);
    }
};

openpgl::gpu::cuda::GPUField *GPUFieldCreate() {
    return new openpgl::gpu::cuda::GPUField();
}

void GPUFieldDestroy(openpgl::gpu::cuda::GPUField *field) {
    delete field;
}

void GPUFieldUpdate(openpgl::gpu::cuda::GPUField *field, const openpgl::cpp::SampleStorage &sampleStorage) {
    field->Update(sampleStorage);
}

}
} 
}
