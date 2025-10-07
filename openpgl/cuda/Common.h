#pragma once

#include <inttypes.h>

#include "../../openpgl/directional/vmm/AdaptiveSplitandMergeFactory.h"
#include "../../openpgl/directional/vmm/ParallaxAwareVonMisesFisherWeightedEMFactory.h"

#include "timer.h"

#define HOST_DEVICE __host__ __device__

namespace embree {
    inline bool isvalid(float &val) { return true; }
}

namespace openpgl {
namespace OPENPGL_KERNEL_NS {

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

#define QFRAME_BINS ((float)(1 << 18))
#define QFRAME_SAMPLE_STATS_BOUND_SCALE (1.0f + 2.f / QFRAME_BINS)

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
    uint32_t pad = 0;
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


struct SamplesDevice {
    thrust::device_vector<PGLSampleData> surface, volume;
};

struct BuildSettings {
    uint32_t maxSamples;
    bool firstIteration = false;
};

void check(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void sync() {
    check(cudaDeviceSynchronize());
}

void checkUsage() {
//#ifndef NDEBUG
    sync();
    size_t free, total;
    check(cudaMemGetInfo(&free, &total));
    float ratio = (float)free/(float)total;
    if (ratio < 0.6) {
        printf("!!! %f %lu/%lu\n", ratio, free, total);
    }
//#endif
}

HOST_DEVICE Vector3 toVec3(pgl_vec3f vec) {
    return Vector3(vec.x, vec.y, vec.z);
}

//
//HOST_DEVICE bool isValid(float val) {
//    return -std::numeric_limits<float>::max() <= val && val <= std::numeric_limits<float>::max();
//}

HOST_DEVICE bool isValid(Vector3 val) {
    return isValid(val.x) && isValid(val.y) && isValid(val.z);
}

template<typename T>
T *data(thrust::device_vector<T> &vector) {
    return thrust::raw_pointer_cast(vector.data());
}

template<typename T>
const T *data(const thrust::device_vector<T> &vector) {
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
    check(cudaGetLastError());
    printf("%s: %fms\n", name.c_str(), 1e3*timer.elapsed());
    sync();
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

KERNEL_FUNCTION
int requiredBits(int num) noexcept
{
    if (num == 0) return 0;
    num -= 1;
    int bits = 0;
    while (num > 0) {
        num >>= 1;
        bits++;
    }
    return bits;
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

//constexpr static int BlockDim = 768;
constexpr static int BlockDim = 512;
//constexpr static int BlockDim = 384;
using VMM = ParallaxAwareVonMisesFisherMixture<KernelCuda<BlockDim>, 32, true>;
using Factory = AdaptiveSplitAndMergeFactory<VMM>;

struct Record {
    // idx from where to read old cache data
    // if != identity, a split has occured
    size_t readIdx;

    size_t samplesBegin;
    size_t samplesEnd;
};

struct SamplingData {
    Vector3 pivot;
    VMM vmm;
};

struct TrainingData {
    uint32_t initialized;
    Factory::Statistics statistics;
    Factory::FittingStatistics fittingStatistics;
};

struct Fingerprints {
    uint32_t inSampleStatistics = 0;
    uint32_t inSamplingData = 0;
    uint32_t inTrainingData = 0;
    uint32_t inSampleData = 0;
    uint32_t outSamplingData = 0;
    uint32_t outTrainingData = 0;
    uint32_t outTrainingDataStats = 0;
    uint32_t outTrainingDataStatsSuff = 0;
    uint32_t outTrainingDataStatsSplit = 0;
    uint32_t outTrainingDataFitStats = 0;
    uint32_t outTrainingDataInit = 0;

    void print() {
        printf("Fingerprints:\n");
        printf(" inSampleStatistics:        0x%08" PRIX32 "\n", inSampleStatistics);
        printf(" inSamplingData:            0x%08" PRIX32 "\n", inSamplingData);
        printf(" inTrainingData:            0x%08" PRIX32 "\n", inTrainingData);
        printf(" outSamplingData:           0x%08" PRIX32 "\n", outSamplingData);
        printf(" outTrainingData:           0x%08" PRIX32 "\n", outTrainingData);
        printf(" outTrainingDataStats:      0x%08" PRIX32 "\n", outTrainingDataStats);
        printf(" outTrainingDataStatsSuff:  0x%08" PRIX32 "\n", outTrainingDataStatsSuff);
        printf(" outTrainingDataStatsSplit: 0x%08" PRIX32 "\n", outTrainingDataStatsSplit);
        printf(" outTrainingDataFitStats:   0x%08" PRIX32 "\n", outTrainingDataFitStats);
        printf(" outTrainingDataInit:       0x%08" PRIX32 "\n", outTrainingDataInit);
    }
};

constexpr static bool enableFingerprinting = false;


template<typename T>
__device__ constexpr size_t salign() {
    return std::max(alignof(T), alignof(int));
}

template<typename T>
__device__ T* getptr(void *ptr) {
    uintptr_t int_ptr = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t aligned_int_ptr = (int_ptr + salign<T>() - 1) & ~(salign<T>() - 1);
    return reinterpret_cast<T*>(aligned_int_ptr);
}

__device__ void coopCopy(void* dst, const void* src, const size_t size) {
    // TODO replace by CUB or vectorize further (using int64/int2/int4)
    int* dst_ = (int*)dst;
    const int* src_ = (const int*)src;
    assert(size % sizeof(int) == 0);
    const size_t size_ = size / sizeof(int);
    for (int i = threadIdx.x; i < size_; i += blockDim.x)
        dst_[i] = src_[i];
}

}
}