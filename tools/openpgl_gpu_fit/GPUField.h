
#include <cstdint>

#include <thrust/device_vector.h>

#define OPENPGL_VERSION_STRING "fasff"
#define OPENPGL_VEC_SIZE 1
#include "../../openpgl/kernel/cuda.h"
#include "../../openpgl/include/openpgl/data.h"
#include "../../openpgl/data/SampleStatistics.h"

#include "../../openpgl/include/openpgl/breadcrump.h"
#include "../../openpgl/include/openpgl/sdump.h"

#include "directional.cuh"

#define HOST_DEVICE __host__ __device__

#define QFRAME_BINS ((float)(1 << 18))
#define QFRAME_SAMPLE_STATS_BOUND_SCALE (1.0f + 2.f / QFRAME_BINS)

namespace openpgl {
namespace gpu {
namespace cuda {

struct State {
    BBox bounds;
    uint32_t nodeAlloc;
    uint32_t leafAlloc;
    uint32_t anySplit;
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

    Breadcrumb bc;

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

struct QuantizationFrame {

    BBox aabb;
    int nextIsRight;
    int nextSplitDim;
    float nextPivot;

    BBox scaledBounds;
    Vector3 center;
    Vector3 halfExtend;
    Vector3 invHalfExtend;

    HOST_DEVICE void init() {
        // scaling the boundary of the samples to avoid discretization problems at the boundaries
        scaledBounds = aabb;
        Vector3 center = (scaledBounds.lower + scaledBounds.upper) / 2.f;
        scaledBounds.lower = center + QFRAME_SAMPLE_STATS_BOUND_SCALE * (scaledBounds.lower - center);
        scaledBounds.upper = center + QFRAME_SAMPLE_STATS_BOUND_SCALE * (scaledBounds.upper - center);

        halfExtend = (scaledBounds.upper - scaledBounds.lower) * 0.5f;

        // Checking and compenstaing for sampled bounds with dimensions of zero extend (e.g. plane)
        invHalfExtend.x = halfExtend.x > 0.f ? 1.0f / halfExtend.x : 0.f;
        invHalfExtend.y = halfExtend.y > 0.f ? 1.0f / halfExtend.y : 0.f;
        invHalfExtend.z = halfExtend.z > 0.f ? 1.0f / halfExtend.z : 0.f;
        this->center = scaledBounds.lower + halfExtend;

        //OPENPGL_ASSERT(embree::isvalid(invSampleBoundsHalfExtend.x));
        //OPENPGL_ASSERT(embree::isvalid(invSampleBoundsHalfExtend.y));
        //OPENPGL_ASSERT(embree::isvalid(invSampleBoundsHalfExtend.z));
    }
};


struct IntegerSampleStats {
    uint32_t numSamples = 0;
    int64_t mean[3] = {0, 0, 0};
    int64_t variance[3] = {0, 0, 0};
    int64_t intSampleBounds[2][3] = {
        {std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::max()},
        {std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::min()}
    };

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
        Vector3 tmpVariance = (tmpSample * tmpSample) * QFRAME_BINS;
        tmpSample *= QFRAME_BINS;

        //OPENPGL_ASSERT(embree::isvalid(tmpSample.x));
        //OPENPGL_ASSERT(embree::isvalid(tmpSample.y));
        //OPENPGL_ASSERT(embree::isvalid(tmpSample.z));

        for (int i = 0; i < 3; i++) {
            mean[i] = tmpSample[i];
            variance[i] = tmpVariance[i];
            intSampleBounds[0][i] = tmpSample[i];
            intSampleBounds[1][i] = tmpSample[i];
        }

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

        ss << "scaledBounds: [" << frame.scaledBounds.lower[0] << ",\t" << frame.scaledBounds.lower[1] << ",\t" << frame.scaledBounds.lower[2] << "] \t [" << frame.scaledBounds.upper[0] << ",\t"
            << frame.scaledBounds.upper[1] << ",\t" << frame.scaledBounds.upper[2] << "] " << std::endl;
        ss << "center: " << frame.center[0] << ",\t" << frame.center[1] << ",\t" << frame.center[2] << std::endl;
        ss << "halfExtend: " << frame.halfExtend[0] << ",\t" << frame.halfExtend[1] << ",\t" << frame.halfExtend[2] << std::endl;
        ss << "invHalfExtend: " << frame.invHalfExtend[0] << ",\t" << frame.invHalfExtend[1] << ",\t" << frame.invHalfExtend[2] << std::endl;
        return ss.str();
    }

    HOST_DEVICE void print(const QuantizationFrame& frame) const {
        printf(
            "IntegerSampleStatistics:\n numSamples: %i\n mean: [%li, %li, %li]\n variance: [%li, %li, %li]\n intSampleBounds: [[%li, %li, %li], [%li, %li, %li]]\n"
            " scaledBounds: [[%.9f, %.9f, %.9f], [%.9f, %.9f, %.9f]]\n center: [%.9f, %.9f, %.9f]\n halfExtend: [%.9f, %.9f, %.9f]\n invHalfExtend: [%.9f, %.9f, %.9f]\n",
            numSamples,
            mean[0], mean[1], mean[2],
            variance[0], variance[1], variance[2],
            intSampleBounds[0][0], intSampleBounds[0][1], intSampleBounds[0][2],
            intSampleBounds[1][0], intSampleBounds[1][1], intSampleBounds[1][2],
            frame.scaledBounds.lower[0], frame.scaledBounds.lower[1], frame.scaledBounds.lower[2],
            frame.scaledBounds.upper[0], frame.scaledBounds.upper[1], frame.scaledBounds.upper[2],
            frame.center[0], frame.center[1], frame.center[2],
            frame.halfExtend[0], frame.halfExtend[1], frame.halfExtend[2],
            frame.invHalfExtend[0], frame.invHalfExtend[1], frame.invHalfExtend[2]
        );
    }

    __device__ void atomicReduce(const IntegerSampleStats &other) {
        atomicAdd(&numSamples, other.numSamples);
        for (int i = 0; i < 3; i++) {
            atomicAdd((unsigned long long int*)&mean[i], (unsigned long long int)other.mean[i]);
            atomicAdd((unsigned long long int*)&variance[i], (unsigned long long int)other.variance[i]);
            atomicMin((long long*)&intSampleBounds[0][i], (long long)other.intSampleBounds[0][i]);
            atomicMax((long long*)&intSampleBounds[1][i], (long long)other.intSampleBounds[1][i]);
        }
    }

    HOST_DEVICE SampleStatistics toSampleStats(const QuantizationFrame &frame) const {
        SampleStatistics sampleStats;
        if (numSamples > 0)
        {
            auto unpack = [&](const int64_t v[3]) {
                return frame.center + (Vector3(v[0], v[1], v[2]) / QFRAME_BINS) * frame.halfExtend;
            };
            
            Vector3 lowerCollectedSampleBound = unpack(intSampleBounds[0]);
            Vector3 upperCollectedSampleBound = unpack(intSampleBounds[1]);
            Vector3 collectedSampleBoundExtend = (upperCollectedSampleBound - lowerCollectedSampleBound);
            Vector3 halfCollectedSampleBoundExtend = (upperCollectedSampleBound - lowerCollectedSampleBound) * 0.5f;
            Vector3 halfBinSize = Vector3(0.5f / QFRAME_BINS) * frame.halfExtend;

            float invNumSamples = 1.f / float(numSamples);
            Vector3 sampleMean = Vector3(mean[0], mean[1], mean[2]) * invNumSamples;
            sampleMean /= QFRAME_BINS;
            Vector3 sampleMeanBin = sampleMean;
            sampleMean = sampleMean * frame.halfExtend;
            sampleMean += frame.center;

            // Ensuring that them sample mean position is inside the collected/measured sample bounds
            sampleMean.x = sampleMean.x <= lowerCollectedSampleBound.x ? lowerCollectedSampleBound.x + std::min(halfCollectedSampleBoundExtend.x, halfBinSize.x) : sampleMean.x;
            sampleMean.y = sampleMean.y <= lowerCollectedSampleBound.y ? lowerCollectedSampleBound.y + std::min(halfCollectedSampleBoundExtend.y, halfBinSize.y) : sampleMean.y;
            sampleMean.z = sampleMean.z <= lowerCollectedSampleBound.z ? lowerCollectedSampleBound.z + std::min(halfCollectedSampleBoundExtend.z, halfBinSize.z) : sampleMean.z;
            sampleMean.x = sampleMean.x >= upperCollectedSampleBound.x ? upperCollectedSampleBound.x - std::min(halfCollectedSampleBoundExtend.x, halfBinSize.x) : sampleMean.x;
            sampleMean.y = sampleMean.y >= upperCollectedSampleBound.y ? upperCollectedSampleBound.y - std::min(halfCollectedSampleBoundExtend.y, halfBinSize.y) : sampleMean.y;
            sampleMean.z = sampleMean.z >= upperCollectedSampleBound.z ? upperCollectedSampleBound.z - std::min(halfCollectedSampleBoundExtend.z, halfBinSize.z) : sampleMean.z;

            Vector3 sampleVariance = (Vector3(variance[0], variance[1], variance[2]) / QFRAME_BINS) * invNumSamples;
            sampleVariance -= sampleMeanBin * sampleMeanBin;
            sampleVariance = Vector3(std::max(0.f, sampleVariance.x), std::max(0.f, sampleVariance.y), std::max(0.f, sampleVariance.z));
            // sampleVariance = Vector3(std::fabs(sampleVariance.x), std::fabs(sampleVariance.y), std::fabs(sampleVariance.z));

            sampleVariance = sampleVariance * (frame.halfExtend * frame.halfExtend);
            // Ensuring that the estimated variance is in the right bounds.
            sampleVariance.x = std::min(collectedSampleBoundExtend.x * collectedSampleBoundExtend.x, sampleVariance.x);
            sampleVariance.y = std::min(collectedSampleBoundExtend.y * collectedSampleBoundExtend.y, sampleVariance.y);
            sampleVariance.z = std::min(collectedSampleBoundExtend.z * collectedSampleBoundExtend.z, sampleVariance.z);

            sampleStats.mean = sampleMean;
            sampleStats.numSamples = numSamples;
            sampleStats.variance = sampleVariance;

            // setting the variance to zero if the measured integer sample bound is zero
            sampleStats.variance.x = intSampleBounds[1][0] - intSampleBounds[0][0] <= 0 ? 0.f : sampleStats.variance.x;
            sampleStats.variance.y = intSampleBounds[1][1] - intSampleBounds[0][1] <= 0 ? 0.f : sampleStats.variance.y;
            sampleStats.variance.z = intSampleBounds[1][2] - intSampleBounds[0][2] <= 0 ? 0.f : sampleStats.variance.z;

            // using the real (float) measured sample bound and not a transformed version of the integer sample bound for accuracy reasons
            sampleStats.sampleBounds.lower = lowerCollectedSampleBound;
            sampleStats.sampleBounds.upper = upperCollectedSampleBound;
        }
        //OPENPGL_ASSERT(sampleStats.isValid());
        return sampleStats;
    }
};

struct GPUField {
    // stats for leaf nodes
    uint32_t maxNumNodes;
    uint32_t maxNumLeaves;
    State hostState;
    thrust::device_vector<State> state;
    thrust::device_vector<TreeNode> tree;
    thrust::device_vector<QuantizationFrame> quantizationFrame; 
    thrust::device_vector<SampleStatistics> leafStats;

    // directional data
    thrust::device_vector<TrainingData> trainingData;
    thrust::device_vector<SamplingData> samplingData;
    
    int it = 0;
    
    // temporary vectors used for fitting
    // TODO use aliasing between vectors to reduce memory footprint
    thrust::device_vector<Vector3> samplePositions;
    thrust::device_vector<IntegerSampleStats> sampleStats;

    thrust::device_vector<uint32_t> finishedNodes;
    thrust::device_vector<Record> records;

    thrust::device_vector<uint64_t> sortKeys;
    thrust::device_vector<uint32_t> sortIndices;
    thrust::device_vector<uint32_t> leafHistogram;
    thrust::device_vector<PGLSampleData> reorderedSamples;
        
    GPUField();
    void Update(thrust::device_vector<PGLSampleData> &samples);
    void sDump(SDump* sDump) const;

    void computeSizes() const;

    void UpdateTree(uint32_t numSamples, thrust::device_vector<PGLSampleData> &samples);
    void dump(const std::string& dumpFileName) const;
};

}
}
}