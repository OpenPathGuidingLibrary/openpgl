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

#define HOST_DEVICE __host__ __device__

namespace openpgl{
namespace gpu {
namespace cuda {

struct SamplesDevice {
    thrust::device_vector<PGLSampleData> surface, volume;
};

struct BuildSettings {
    uint32_t maxSamples;
};

struct AABB {
    Vector3 lower, upper;

    HOST_DEVICE AABB() {
        lower = {std::numeric_limits<float>::infinity()};
        upper = {-std::numeric_limits<float>::infinity()};
    }

    HOST_DEVICE AABB(const Vector3 &val) {
        lower = val;
        upper = val;
    }

    HOST_DEVICE void add(const Vector3 &val) {
        lower = min(lower, val);
        upper = max(upper, val);
    }

    HOST_DEVICE static AABB merge(const AABB& a, const AABB& b) {
        AABB aabb;
        aabb.lower = min(a.lower, b.lower);
        aabb.upper = max(a.upper, b.upper);
        return aabb;
    }

    std::string toString() const {
        std::stringstream ss;
        ss.precision(15);
        ss << "{\n  {" << lower.data[0] << ", " << lower.data[1] << ", " << lower.data[2] << "}\n  {"
            << upper.data[0] << ", " << upper.data[1] << ", " << upper.data[2] << "}\n}";
        return ss.str();
    }

    void print() const {
        printf("AABB {\n  {%f, %f, %f}\n  {%f, %f, %f}\n}\n",
            lower.vec.x, lower.vec.y, lower.vec.z,
            upper.vec.x, upper.vec.y, upper.vec.z
        );
    }
};

struct State {
    AABB bounds;
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

    HOST_DEVICE SampleStats(const Vector3 &position) {
        mean = position;
        variance = Vector3(0);
        lower = position;
        upper = position;
        count = 1;
    }

    HOST_DEVICE SampleStats(const SampleStats &a, const SampleStats &b) {
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

    std::string toString() {
        std::stringstream ss;
        ss.precision(15);
        ss << "SampleStatistics:" << std::endl;
        ss << "numSamples: " << count << std::endl;
        ss << "numZeroValueSamples: " << 0 << std::endl;
        ss << "mean: " << mean[0] << ",\t" << mean[1] << ",\t" << mean[2] << std::endl;
        ss << "variance: " << variance[0] << ",\t" << variance[1] << ",\t" << variance[2] << std::endl;
        ss << "sampleBounds: [" << lower[0] << ",\t" << lower[1] << ",\t" << lower[2] << "] \t [" << upper[0] << ",\t"
           << upper[1] << ",\t" << upper[2] << "] " << std::endl;
        return ss.str();
    }
};

struct QuantizationFrame {
    constexpr static float INTEGER_BINS = 1 << 18;
    constexpr static float INTEGER_SAMPLE_STATS_BOUND_SCALE = 1.0f + 2.f / INTEGER_BINS;

    AABB aabb;
    int nextIsRight;
    int nextSplitDim;
    float nextPivot;

    AABB scaledBounds;
    Vector3 center;
    Vector3 halfExtend;
    Vector3 invHalfExtend;

    HOST_DEVICE void init() {
        // scaling the boundary of the samples to avoid discretization problems at the boundaries
        scaledBounds = aabb;
        Vector3 center = (scaledBounds.lower + scaledBounds.upper) / 2;
        scaledBounds.lower = center + INTEGER_SAMPLE_STATS_BOUND_SCALE * (scaledBounds.lower - center);
        scaledBounds.upper = center + INTEGER_SAMPLE_STATS_BOUND_SCALE * (scaledBounds.upper - center);

        halfExtend = (scaledBounds.upper - scaledBounds.lower) * 0.5f;

        // Checking and compenstaing for sampled bounds with dimensions of zero extend (e.g. plane)
        invHalfExtend.vec.x = halfExtend.vec.x > 0.f ? 1.0f / halfExtend.vec.x : 0.f;
        invHalfExtend.vec.y = halfExtend.vec.y > 0.f ? 1.0f / halfExtend.vec.y : 0.f;
        invHalfExtend.vec.z = halfExtend.vec.z > 0.f ? 1.0f / halfExtend.vec.z : 0.f;
        this->center = scaledBounds.lower + halfExtend;

        //OPENPGL_ASSERT(embree::isvalid(invSampleBoundsHalfExtend.x));
        //OPENPGL_ASSERT(embree::isvalid(invSampleBoundsHalfExtend.y));
        //OPENPGL_ASSERT(embree::isvalid(invSampleBoundsHalfExtend.z));
    }
};

struct IntegerSampleStats {
    uint32_t numSamples;
    int64_t mean[3];
    int64_t variance[3];
    int64_t intSampleBounds[2][3];
    AABB sampleBounds;

    IntegerSampleStats() = default;

    HOST_DEVICE bool isValid() const
    {
        bool valid = true;
        //valid = valid && numSamples >= 0.0f;
        ////OPENPGL_ASSERT(valid);
        //valid = valid && isValid(mean[0]);
        //valid = valid && isValid(mean[1]);
        //valid = valid && isValid(mean[2]);
        ////OPENPGL_ASSERT(valid);
        //
        //valid = valid && isValid(variance[0]);
        //valid = valid && isValid(variance[1]);
        //valid = valid && isValid(variance[2]);
        //OPENPGL_ASSERT(valid);

        valid = valid && variance[0] >= 0;
        valid = valid && variance[1] >= 0;
        valid = valid && variance[2] >= 0;
        //OPENPGL_ASSERT(valid);

        return valid;
    }

    HOST_DEVICE IntegerSampleStats(const QuantizationFrame &frame, const Vector3 &position) {
        numSamples = 1;

        Vector3 tmpSample = ((position - frame.center) * frame.invHalfExtend);
        Vector3 tmpVariance = (tmpSample * tmpSample) * QuantizationFrame::INTEGER_BINS;
        tmpSample *= QuantizationFrame::INTEGER_BINS;

        //OPENPGL_ASSERT(embree::isvalid(tmpSample.x));
        //OPENPGL_ASSERT(embree::isvalid(tmpSample.y));
        //OPENPGL_ASSERT(embree::isvalid(tmpSample.z));

        for (int i = 0; i < 3; i++) {
            mean[i] = tmpSample.data[i];
            variance[i] = tmpVariance.data[i];
            intSampleBounds[0][i] = tmpSample.data[i];
            intSampleBounds[1][i] = tmpSample.data[i];
        }
        sampleBounds = AABB(position);

        if (!isValid()) {
            variance[0] = 0;
        }
        //OPENPGL_ASSERT(isValid());
    }    

    HOST_DEVICE IntegerSampleStats(const IntegerSampleStats &a, const IntegerSampleStats &b) {
        numSamples = a.numSamples + b.numSamples;
        for (int i = 0; i < 3; i++) {
            mean[i] = a.mean[i] + b.mean[i];
            variance[i] = a.variance[i] + b.variance[i];
            intSampleBounds[0][i] = std::min(a.intSampleBounds[0][i], b.intSampleBounds[0][i]);
            intSampleBounds[1][i] = std::max(a.intSampleBounds[1][i], b.intSampleBounds[1][i]);
        }
        sampleBounds = AABB::merge(a.sampleBounds, b.sampleBounds);
        //OPENPGL_ASSERT(isValid());

        if (!isValid()) {
            variance[0] = 0;
        }

    }    


    std::string toString(const QuantizationFrame& frame) const {
        std::stringstream ss;
        ss.precision(15);
        ss << "IntegerSampleStatistics:" << std::endl;
        ss << "numSamples: " << numSamples << std::endl;
        //ss << "numZeroValueSamples: " << numZeroValueSamples << std::endl;
        ss << "mean: " << mean[0] << ",\t" << mean[1] << ",\t" << mean[2] << std::endl;
        ss << "variance: " << variance[0] << ",\t" << variance[1] << ",\t" << variance[2] << std::endl;
        ss << "intSampleBounds: [" << intSampleBounds[0][0] << ",\t" << intSampleBounds[0][1] << ",\t" << intSampleBounds[0][2] << "] \t [" << intSampleBounds[1][0] << ",\t"
            << intSampleBounds[1][1] << ",\t" << intSampleBounds[1][2] << "] " << std::endl;
        ss << "sampleBounds: [" << sampleBounds.lower[0] << ",\t" << sampleBounds.lower[1] << ",\t" << sampleBounds.lower[2] << "] \t [" << sampleBounds.upper[0] << ",\t"
            << sampleBounds.upper[1] << ",\t" << sampleBounds.upper[2] << "] " << std::endl;


        ss << "scaledBounds: [" << frame.scaledBounds.lower.data[0] << ",\t" << frame.scaledBounds.lower.data[1] << ",\t" << frame.scaledBounds.lower.data[2] << "] \t [" << frame.scaledBounds.upper.data[0] << ",\t"
            << frame.scaledBounds.upper.data[1] << ",\t" << frame.scaledBounds.upper.data[2] << "] " << std::endl;
        ss << "center: " << frame.center.data[0] << ",\t" << frame.center.data[1] << ",\t" << frame.center.data[2] << std::endl;
        ss << "halfExtend: " << frame.halfExtend.data[0] << ",\t" << frame.halfExtend.data[1] << ",\t" << frame.halfExtend.data[2] << std::endl;
        ss << "invHalfExtend: " << frame.invHalfExtend.data[0] << ",\t" << frame.invHalfExtend.data[1] << ",\t" << frame.invHalfExtend.data[2] << std::endl;
        return ss.str();
    }

    HOST_DEVICE SampleStats toSampleStats(const QuantizationFrame &frame) const {
        SampleStats sampleStats;
        if (numSamples > 0)
        {
            Vector3 lowerCollectedSampleBound = sampleBounds.lower;
            Vector3 upperCollectedSampleBound = sampleBounds.upper;
            Vector3 collectedSampleBoundExtend = (upperCollectedSampleBound - lowerCollectedSampleBound);
            Vector3 halfCollectedSampleBoundExtend = (upperCollectedSampleBound - lowerCollectedSampleBound) * 0.5f;
            Vector3 halfBinSize = Vector3(0.5f / QuantizationFrame::INTEGER_BINS) * frame.halfExtend;

            float invNumSamples = 1.f / float(numSamples);
            Vector3 sampleMean = Vector3(mean[0], mean[1], mean[2]) * invNumSamples;
            sampleMean /= QuantizationFrame::INTEGER_BINS;
            Vector3 sampleMeanBin = sampleMean;
            sampleMean = sampleMean * frame.halfExtend;
            sampleMean += frame.center;

            // Ensuring that them sample mean position is inside the collected/measured sample bounds
            sampleMean.vec.x = sampleMean.vec.x <= lowerCollectedSampleBound.vec.x ? lowerCollectedSampleBound.vec.x + std::min(halfCollectedSampleBoundExtend.vec.x, halfBinSize.vec.x) : sampleMean.vec.x;
            sampleMean.vec.y = sampleMean.vec.y <= lowerCollectedSampleBound.vec.y ? lowerCollectedSampleBound.vec.y + std::min(halfCollectedSampleBoundExtend.vec.y, halfBinSize.vec.y) : sampleMean.vec.y;
            sampleMean.vec.z = sampleMean.vec.z <= lowerCollectedSampleBound.vec.z ? lowerCollectedSampleBound.vec.z + std::min(halfCollectedSampleBoundExtend.vec.z, halfBinSize.vec.z) : sampleMean.vec.z;
            sampleMean.vec.x = sampleMean.vec.x >= upperCollectedSampleBound.vec.x ? upperCollectedSampleBound.vec.x - std::min(halfCollectedSampleBoundExtend.vec.x, halfBinSize.vec.x) : sampleMean.vec.x;
            sampleMean.vec.y = sampleMean.vec.y >= upperCollectedSampleBound.vec.y ? upperCollectedSampleBound.vec.y - std::min(halfCollectedSampleBoundExtend.vec.y, halfBinSize.vec.y) : sampleMean.vec.y;
            sampleMean.vec.z = sampleMean.vec.z >= upperCollectedSampleBound.vec.z ? upperCollectedSampleBound.vec.z - std::min(halfCollectedSampleBoundExtend.vec.z, halfBinSize.vec.z) : sampleMean.vec.z;

            Vector3 sampleVariance = (Vector3(variance[0], variance[1], variance[2]) / QuantizationFrame::INTEGER_BINS) * invNumSamples;
            sampleVariance -= sampleMeanBin * sampleMeanBin;
            sampleVariance = Vector3(std::max(0.f, sampleVariance.vec.x), std::max(0.f, sampleVariance.vec.y), std::max(0.f, sampleVariance.vec.z));
            // sampleVariance = Vector3(std::fabs(sampleVariance.x), std::fabs(sampleVariance.y), std::fabs(sampleVariance.z));

            sampleVariance = sampleVariance * (frame.halfExtend * frame.halfExtend);
            // Ensuring that the estimated variance is in the right bounds.
            sampleVariance.vec.x = std::min(collectedSampleBoundExtend.vec.x * collectedSampleBoundExtend.vec.x, sampleVariance.vec.x);
            sampleVariance.vec.y = std::min(collectedSampleBoundExtend.vec.y * collectedSampleBoundExtend.vec.y, sampleVariance.vec.y);
            sampleVariance.vec.z = std::min(collectedSampleBoundExtend.vec.z * collectedSampleBoundExtend.vec.z, sampleVariance.vec.z);

            sampleStats.mean = sampleMean;
            sampleStats.count = numSamples;
            sampleStats.variance = sampleVariance;

            // setting the variance to zero if the measured integer sample bound is zero
            sampleStats.variance.vec.x = intSampleBounds[1][0] - intSampleBounds[0][0] <= 0 ? 0.f : sampleStats.variance.vec.x;
            sampleStats.variance.vec.y = intSampleBounds[1][1] - intSampleBounds[0][1] <= 0 ? 0.f : sampleStats.variance.vec.y;
            sampleStats.variance.vec.z = intSampleBounds[1][2] - intSampleBounds[0][2] <= 0 ? 0.f : sampleStats.variance.vec.z;

            // using the real (float) measured sample bound and not a transformed version of the integer sample bound for accuracy reasons
            sampleStats.lower = sampleBounds.lower;
            sampleStats.upper = sampleBounds.upper;

        }
        //OPENPGL_ASSERT(sampleStats.isValid());
        return sampleStats;
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

    HOST_DEVICE uint8_t getSplitDim() const {
        return splitDimAndNodeIdx >> 30;
    }
    
    HOST_DEVICE uint32_t getNodeIdx() const {
        return splitDimAndNodeIdx & ~MASK;
    }

    HOST_DEVICE void setSplitDimAndNodeIdx(uint8_t splitDim, uint32_t nodeIdx) {
        assert((splitDim & ~3) == 0);
        assert((nodeIdx & MASK) == 0);
        splitDimAndNodeIdx = splitDim << 30 | nodeIdx;
    }

    HOST_DEVICE bool isLeaf() const {
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
    const BuildSettings buildSetings, const uint32_t numLeafIndices, const uint32_t *leafIndices, const IntegerSampleStats *newSampleStats,
    State *state, TreeNode *tree, QuantizationFrame* quantizationFrame, SampleStats *sampleStats, uint32_t *finishedNodes
) {
    int i = globalIdx();

    if (!(i < numLeafIndices)) return;

    
    const uint32_t n = leafIndices[i];
    
    int bits = std::numeric_limits<uint32_t>::digits;
    uint32_t finishedI = n / bits, finishedMask = 1 << (n % bits); 
    if (finishedNodes[finishedI] & finishedMask) return;

    TreeNode node = tree[n];
    uint32_t s = node.getNodeIdx();
    QuantizationFrame frame = quantizationFrame[n];
    SampleStats mergedStats = SampleStats(newSampleStats[i].toSampleStats(frame), sampleStats[s]);

    if (mergedStats.count > buildSetings.maxSamples) {

        uint32_t lLeafIdx = s;
        uint32_t rLeafIdx = atomicAdd(&state[0].leafAlloc, 1);
        uint32_t childIdx = atomicAdd(&state[0].nodeAlloc, 2);

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
        sampleStats[s] = mergedStats;
        // write to finished mask, so this node is not processed again
        atomicOr(&finishedNodes[finishedI], finishedMask);
    }
}

template<typename T>
void resize(T& vector, size_t size) {
    if (vector.capacity() < size) {
        vector.reserve(1.3 * size);
    }
    vector.resize(size);
}

class GPUField {
    // stats for leaf nodes
    uint32_t maxNumNodes;
    uint32_t maxNumLeaves;
    State hostState;
    thrust::device_vector<State> state;
    thrust::device_vector<TreeNode> tree;
    thrust::device_vector<QuantizationFrame> quantizationFrame; 
    thrust::device_vector<SampleStats> leafStats;

    int it = 0;



    // temporary vectors used for fitting
    // TODO use aliasing between vectors to reduce memory footprint
    thrust::device_vector<uint32_t> leafIndices;
    thrust::device_vector<uint32_t> leafHistogram;
    thrust::device_vector<uint32_t> sampleOffset;
    thrust::device_vector<PGLSampleData> reorderedSamples;
    thrust::device_vector<uint32_t> reorderedLeafIndices;

    thrust::device_vector<IntegerSampleStats> sampleStats;
    thrust::device_vector<uint32_t> reducedLeafIndices;
    thrust::device_vector<IntegerSampleStats> reducedSampleStats;
    thrust::device_vector<IntegerSampleStats> scatteredSampleStats;

    thrust::device_vector<uint32_t> finishedNodes;

public:
    GPUField() {
        // initialize tree to single root node
        maxNumLeaves = 64*1024;
        maxNumNodes = 2 * maxNumLeaves - 1;

        state = thrust::device_vector<State>(1);
        tree = thrust::device_vector<TreeNode>(maxNumNodes);
        quantizationFrame = thrust::device_vector<QuantizationFrame>(maxNumNodes);
        leafStats = thrust::device_vector<SampleStats>(maxNumLeaves);

        hostState.nodeAlloc = 1;
        hostState.leafAlloc = 1;
        hostState.anySplit = 0;
        state[0] = hostState;

        TreeNode node;
        node.pivot = 0;
        node.setSplitDimAndNodeIdx(TreeNode::ELeafNode, 0);
        tree[0] = node;

        leafStats[0] = {};
    }

    void UpdateTree(uint32_t numSamples, thrust::device_vector<PGLSampleData> &samples) {
        if (false) {
            thrust::host_vector<PGLSampleData> hSamples = samples;
            std::vector<Vector3> points;
            for (int i = 0; i < hSamples.size(); i++) {
                points.push_back(Vector3(hSamples[i].position));
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
        resize(reducedLeafIndices, maxNumNodes);
        resize(reducedSampleStats, maxNumNodes);
        resize(scatteredSampleStats, maxNumNodes);

        resize(finishedNodes, ceilDiv(maxNumNodes, std::numeric_limits<uint32_t>::digits));

        // first iteration: estimate scene bounds
        if (it == 0) {
            hostState.bounds = thrust::transform_reduce(samples.begin(), samples.end(),
                [] HOST_DEVICE (const PGLSampleData &x) { return AABB(Vector3(x.position)); },
                AABB(),
                [] HOST_DEVICE (const AABB &a, const AABB &b) { return AABB::merge(a, b); }
            );
            AABB bounds = hostState.bounds;
            bounds.print();
            Vector3 center = (bounds.lower + bounds.upper) / 2;
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

        printf("  nodeAlloc: %i leafAlloc: %i anySplit: %i\n", hostState.nodeAlloc, hostState.leafAlloc, hostState.anySplit);

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

            //if (false) {
            //    cudaDeviceSynchronize();
            //    hLeafHistogram = leafHistogram;
            //    uint sum = 0;
            //    for (int i = 0; i < hostAlloc.nodeAlloc; i++) {
            //        sum += hLeafHistogram[i];
            //        printf("    %zu: %zu\n", i, hLeafHistogram[i]);
            //    }
            //    uint rsum = 0;
            //    for (int i = hostAlloc.nodeAlloc; i < maxNumLeaves; i++) {
            //        rsum += hLeafHistogram[i];
            //    }
            //    assert(rsum == 0);
            //    assert(sum == numSamples);
            //}

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
            {
                QuantizationFrame* framePtr = data(quantizationFrame);
                thrust::transform(
                    thrust::make_zip_iterator(thrust::make_tuple(reorderedLeafIndices.begin(), reorderedSamples.begin())),
                    thrust::make_zip_iterator(thrust::make_tuple(reorderedLeafIndices.end(),   reorderedSamples.end())),
                    sampleStats.begin(),
                    [framePtr] __device__ (const thrust::tuple<uint32_t, PGLSampleData>& t) {
                        return IntegerSampleStats(
                            framePtr[thrust::get<0>(t)],
                            Vector3(thrust::get<1>(t).position)
                        );
                    }
                );
            }
            // PERF: reduce_by_key forces synchronization to return number of reduced elements
            // we might be able to avoid this, by directly using ReduceByKey from cub
            auto [reducedLeafIndicesEnd, reducedSampleStatsEnd] = thrust::reduce_by_key(
                reorderedLeafIndices.begin(), reorderedLeafIndices.end(), sampleStats.begin(),
                reducedLeafIndices.begin(), reducedSampleStats.begin(),
                thrust::equal_to<uint32_t>(),
                [] __device__ (const IntegerSampleStats &a, const IntegerSampleStats &b) {
                    return IntegerSampleStats(a, b);
                }
            );
            auto numLeafIndices = reducedLeafIndicesEnd - reducedLeafIndices.begin();

            // Split nodes
            printf("   split\n");
            BuildSettings buildSettings;
            buildSettings.maxSamples = 32000;
            hostState.anySplit = 0;
            state[0] = hostState;
            launch(
                SplitNodes, numLeafIndices, 128,
                buildSettings, numLeafIndices, data(reducedLeafIndices), data(reducedSampleStats),
                data(state), data(tree), data(quantizationFrame), data(leafStats), data(finishedNodes)
            );
            hostState = state[0];

            printf("  nodeAlloc: %i leafAlloc: %i anySplit: %i\n", hostState.nodeAlloc, hostState.leafAlloc, hostState.anySplit);
        } while(hostState.anySplit);

        if (false) {
            std::vector<std::pair<Vector3, Vector3>> boxes;
            for (int i = 0; i < hostState.leafAlloc; i++) {
                SampleStats stats = leafStats[i];
                boxes.push_back({stats.lower, stats.upper});
            }
    
            writeBoundingBoxes(boxes, std::string("dump/") + std::to_string(it) + std::string(".obj"));
        }
        it++;
    }

    void Update(thrust::device_vector<PGLSampleData> &samples) {
        printf("Update: %i surface samples, %i volume samples\n", samples.size(), 0);

        printf(" updating tree\n");
        UpdateTree(samples.size(), samples);
    }
};

openpgl::gpu::cuda::GPUField *GPUFieldCreate() {
    return new openpgl::gpu::cuda::GPUField();
}

void GPUFieldDestroy(openpgl::gpu::cuda::GPUField *field) {
    delete field;
}

void GPUFieldUpdate(openpgl::gpu::cuda::GPUField *field, SamplesDevice *samples) {
    field->Update(samples->surface);
}

SamplesDevice* SamplesDeviceCreate(const openpgl::cpp::SampleStorage &sampleStorage) {
    thrust::host_vector<PGLSampleData> surface(sampleStorage.GetSizeSurface()), volume(sampleStorage.GetSizeVolume());
    for (int i = 0; i < sampleStorage.GetSizeSurface(); i++)
        surface[i] = sampleStorage.GetSampleSurface(i);
    for (int i = 0; i < sampleStorage.GetSizeVolume(); i++)
        volume[i] = sampleStorage.GetSampleVolume(i);

    return new SamplesDevice {.surface = surface, .volume = volume};
}

void SamplesDeviceDestroy(SamplesDevice* samplesDevice) {
    delete samplesDevice;
}

}
} 
}
