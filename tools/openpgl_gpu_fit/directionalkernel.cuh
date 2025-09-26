#pragma once

#include "directional.cuh"
#include <inttypes.h>

namespace openpgl {
namespace gpu {
namespace cuda {
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
        // TODO vectorize further (using int64/int2/int4)
        int* dst_ = (int*)dst;
        const int* src_ = (const int*)src;
        assert(size % sizeof(int) == 0);
        const size_t size_ = size / sizeof(int);
        for (int i = threadIdx.x; i < size_; i += blockDim.x)
            dst_[i] = src_[i];
    }

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

    __global__ void
    __launch_bounds__(384)
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

        SHARED char sampleStatisticsBuffer[sizeof(SampleStatistics) + salign<SampleStatistics>()];
        SHARED char samplingDataBuffer[sizeof(SamplingData) + salign<SamplingData>()];
        SHARED char trainingDataBuffer[sizeof(TrainingData) + salign<TrainingData>()];

        SampleStatistics *sampleStatistics = getptr<SampleStatistics>(sampleStatisticsBuffer);
        SamplingData *samplingData = getptr<SamplingData>(samplingDataBuffer);
        TrainingData *trainingData = getptr<TrainingData>(trainingDataBuffer);

        coopCopy(sampleStatistics, gSampleStatistics + n, sizeof(SampleStatistics));
        coopCopy(samplingData, gSamplingData + record.readIdx, sizeof(SamplingData));
        coopCopy(trainingData, gTrainingData + record.readIdx, sizeof(TrainingData));

        Factory factory;

        SampleData* samples = gSamples + leafHistogram[n];
        const size_t numSamples = leafHistogram[n + 1] - leafHistogram[n];

        SYNC; // sampleStatistics used by prepareSamples
        
        if (numSamples > 0) {

        // TODO sort samples
        factory.prepareSamples(samples, numSamples, *sampleStatistics, cfg);

        if (enableFingerprinting) {
            SYNC;
            SINGLE {
                atomicXor(&fp->inSampleStatistics, murmur3_32_t(sampleStatistics, 1));
                atomicXor(&fp->inSamplingData, murmur3_32_t(samplingData, 1));
                atomicXor(&fp->inTrainingData, murmur3_32_t(trainingData, 1));
                atomicXor(&fp->inSampleData, murmur3_32_t(samples, numSamples));
            }
        }
        
        // no need to sync prepared samples, since they are read by the same threads
        // SYNC; 
        
        if (trainingData->initialized) {
            //printf("update\n");
            SINGLE {
                if (record.readIdx != n) {
                    const float alpha = 0.25f; // TODO expose parameter
                    samplingData->vmm.decay(alpha); 
                    trainingData->statistics.decay(alpha);
                }
                
                Vector3 shift = samplingData->pivot - sampleStatistics->getMean();
                trainingData->statistics.sufficientStatistics.applyParallaxShift(samplingData->vmm, shift);
                samplingData->vmm.performRelativeParallaxShift(shift);
            }
            factory.update(samplingData->vmm, trainingData->statistics, samples, numSamples, cfg, trainingData->fittingStatistics);
        } else {
            //printf("fit\n");
            factory.fit(samplingData->vmm, trainingData->statistics, samples, numSamples, cfg, trainingData->fittingStatistics);
            SINGLE trainingData->initialized = true;
        }
        SINGLE samplingData->pivot = sampleStatistics->getMean();

        }
            
        SYNC;

        SINGLE OPENPGL_ASSERT(samplingData->vmm.isValid());

        SINGLE if (enableFingerprinting) {
            atomicXor(&fp->outSamplingData, murmur3_32_t(samplingData, 1));
            atomicXor(&fp->outTrainingData, murmur3_32_t(trainingData, 1));
            atomicXor(&fp->outTrainingDataStats, murmur3_32_t(&trainingData->statistics, 1));
            atomicXor(&fp->outTrainingDataStatsSuff, murmur3_32_t(&trainingData->statistics.sufficientStatistics, 1));
            atomicXor(&fp->outTrainingDataStatsSplit, murmur3_32_t(&trainingData->statistics.splittingStatistics, 1));
            atomicXor(&fp->outTrainingDataFitStats, murmur3_32_t(&trainingData->fittingStatistics, 1));
            atomicXor(&fp->outTrainingDataInit, murmur3_32_t(&trainingData->initialized, 1));
        }
            
        coopCopy(gSamplingData + n, samplingData, sizeof(SamplingData));
        coopCopy(gTrainingData + n, trainingData, sizeof(TrainingData));
    }
}
}
}
