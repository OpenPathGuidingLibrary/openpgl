#pragma once

#include "Common.h"

namespace openpgl {
namespace OPENPGL_KERNEL_NS {

__global__ void ComputeKeys(
    const State state, const TreeNode *tree, 
    const uint32_t numSamples, const PGLSampleData *samples,
    uint64_t *keys, uint32_t *values, uint32_t *leafHistogram
) {
    int nodeBits = requiredBits(state.nodeAlloc);

    int i = globalIdx();
    if (!(i < numSamples)) return;

    const PGLSampleData sample = samples[i];
    uint32_t n = traverseTree(toVec3(sample.position), tree);
    uint32_t h0 = murmur3_32((uint8_t*)&sample, sizeof(PGLSampleData), 0);
    uint32_t h1 = murmur3_32((uint8_t*)&sample, sizeof(PGLSampleData), 1);
    keys[i] = (uint64_t)n << (64 - nodeBits) | (uint64_t)h1 << 32 << nodeBits >> nodeBits | h0;
    //keys[i] = (uint64_t)n << 32 /*| (uint64_t)h1 << 32 << nodeBits >> nodeBits */ | h0;
    values[i] = i;
    atomicAdd(&leafHistogram[n], 1);
}

__global__ void GatherSamples(
    const uint32_t numSamples, const PGLSampleData *srcSamples, const uint32_t *indices,
    PGLSampleData *dstSamples
) {
    int i = globalIdx();
    if (!(i < numSamples)) return;

    dstSamples[i] = srcSamples[indices[i]];
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
    const BuildSettings buildSetings, const uint32_t numNodes, const IntegerSampleStats *newSampleStats,
    State *state, TreeNode *tree, QuantizationFrame* quantizationFrame, SampleStatistics *sampleStats, Record* records, uint32_t *finishedNodes
) {
    int n = globalIdx();
    if (!(n < numNodes)) return;

    TreeNode node = tree[n];
    if (!node.isLeaf()) return;

    int bits = std::numeric_limits<uint32_t>::digits;
    uint32_t finishedI = n / bits, finishedMask = 1 << (n % bits); 
    if (finishedNodes[finishedI] & finishedMask) return;

    //uint32_t s = node.getNodeIdx();
    QuantizationFrame frame = quantizationFrame[n];

    SampleStatistics mergedStats = sampleStats[n];
    mergedStats.merge(newSampleStats[n].toSampleStats(frame));

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
        left.setSplitDimAndNodeIdx(TreeNode::ELeafNode, childIdx + 0);
        //left.bc = node.bc.push(false);
        right.pivot = 0;
        right.setSplitDimAndNodeIdx(TreeNode::ELeafNode, childIdx + 1);
        //right.bc = node.bc.push(true);

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
        sampleStats[n] = mergedStats;

        records[n] = record;

        // write to finished mask, so this node is not processed again
        atomicOr(&finishedNodes[finishedI], finishedMask);
    }
}

__global__ void
__launch_bounds__(512)
//__launch_bounds__(384)
EMFit(
    const Factory::Configuration cfg, const TreeNode *tree, const Record* records, const uint32_t *leafHistogram,
    const SampleStatistics* gSampleStatistics, TrainingData* gTrainingData, SamplingData* gSamplingData, SampleData* gSamples,
    Fingerprints *fp
) {
    static_assert(alignof(SampleStatistics) <= 4);
    static_assert(alignof(SamplingData) <= 4);
    static_assert(alignof(TrainingData) <= 4);

    const uint32_t n = blockIdx.x;
    if (!tree[n].isLeaf()) return;

    const Record &record = records[n];

    SHARED SampleStatistics sampleStatistics;
    SHARED SamplingData samplingData;
    SHARED TrainingData trainingData;

    coopCopy(&sampleStatistics, gSampleStatistics + n, sizeof(SampleStatistics));
    coopCopy(&samplingData, gSamplingData + record.readIdx, sizeof(SamplingData));
    coopCopy(&trainingData, gTrainingData + record.readIdx, sizeof(TrainingData));

    Factory factory;

    SampleData* samples = gSamples + leafHistogram[n];
    const size_t numSamples = leafHistogram[n + 1] - leafHistogram[n];

    SYNC; // sampleStatistics used by prepareSamples
    
    if (numSamples > 0) {

    // TODO sort samples
    factory.prepareSamples(samples, numSamples, sampleStatistics, cfg);

    if (enableFingerprinting) {
        SYNC;
        SINGLE {
            atomicXor(&fp->inSampleStatistics, murmur3_32_t(&sampleStatistics, 1));
            atomicXor(&fp->inSamplingData, murmur3_32_t(&samplingData, 1));
            atomicXor(&fp->inTrainingData, murmur3_32_t(&trainingData, 1));
            atomicXor(&fp->inSampleData, murmur3_32_t(samples, numSamples));
        }
    }
    
    // no need to sync prepared samples, since they are read by the same threads
    // SYNC; 
    
    if (trainingData.initialized) {
        //printf("update\n");
        SINGLE {
            if (record.readIdx != n) {
                const float alpha = 0.25f; // TODO expose parameter
                samplingData.vmm.decay(alpha); 
                trainingData.statistics.decay(alpha);
            }
            
            Vector3 shift = samplingData.pivot - sampleStatistics.getMean();
            trainingData.statistics.sufficientStatistics.applyParallaxShift(samplingData.vmm, shift);
            samplingData.vmm.performRelativeParallaxShift(shift);
        }
        factory.update(samplingData.vmm, trainingData.statistics, samples, numSamples, cfg, trainingData.fittingStatistics);
    } else {
        //printf("fit\n");
        SINGLE samplingData.vmm.sumOutgoingRadiance = Vector3(0.f);
        SINGLE samplingData.vmm.numOutgoingRadiance = 0.f;
        factory.fit(samplingData.vmm, trainingData.statistics, samples, numSamples, cfg, trainingData.fittingStatistics);
        SINGLE trainingData.initialized = true;
    }
    SINGLE samplingData.pivot = sampleStatistics.getMean();
    SINGLE samplingData.vmm._pivotPosition = sampleStatistics.getMean();
    factory.updateOutgoingRadiance(samplingData.vmm, samples, numSamples);

    }
        
    SYNC;

    SINGLE OPENPGL_ASSERT(samplingData.vmm.isValid());

    SINGLE if (enableFingerprinting) {
        atomicXor(&fp->outSamplingData, murmur3_32_t(&samplingData, 1));
        atomicXor(&fp->outTrainingData, murmur3_32_t(&trainingData, 1));
        atomicXor(&fp->outTrainingDataStats, murmur3_32_t(&trainingData.statistics, 1));
        atomicXor(&fp->outTrainingDataStatsSuff, murmur3_32_t(&trainingData.statistics.sufficientStatistics, 1));
        atomicXor(&fp->outTrainingDataStatsSplit, murmur3_32_t(&trainingData.statistics.splittingStatistics, 1));
        atomicXor(&fp->outTrainingDataFitStats, murmur3_32_t(&trainingData.fittingStatistics, 1));
        atomicXor(&fp->outTrainingDataInit, murmur3_32_t(&trainingData.initialized, 1));
    }
        
    coopCopy(gSamplingData + n, &samplingData, sizeof(SamplingData));
    coopCopy(gTrainingData + n, &trainingData, sizeof(TrainingData));
}

}
}
