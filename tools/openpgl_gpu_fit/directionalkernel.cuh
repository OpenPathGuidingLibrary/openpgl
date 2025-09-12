#pragma once

#include "directionalkernel.cuh"

namespace openpgl {
namespace gpu {
namespace cuda {
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

    __global__ void EMFit(
        const Factory::Configuration cfg, const uint32_t *leafIndices, const Record* records, const uint32_t *leafHistogram,
        const SampleStatistics* gSampleStatistics, TrainingData* gTrainingData, SamplingData* gSamplingData, SampleData* gSamples
    ) {
        const uint32_t n = leafIndices[blockIdx.x]; 
        const Record record = records[n];

        __shared__ char sampleStatisticsBuffer[sizeof(SampleStatistics) + salign<SampleStatistics>()];
        __shared__ char samplingDataBuffer[sizeof(SamplingData) + salign<SamplingData>()];
        __shared__ char trainingDataBuffer[sizeof(TrainingData) + salign<TrainingData>()];

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
        
        // TODO sort samples
        factory.prepareSamples(samples, numSamples, *sampleStatistics, cfg);
        
        // no need to sync prepared samples, since they are read by the same threads
        // SYNC; 

        // TODO parallelize
        openpgl::Point3 sampleMean = sampleStatistics->getMean();
        if (false) {
            SINGLE {
                if (record.readIdx != n) {
                    const float alpha = 0.25f; // TODO expose parameter
                    samplingData->vmm.decay(alpha); 
                    trainingData->statistics.decay(alpha);
                }
            
                Vector3 shift = samplingData->pivot - sampleMean;
                trainingData->statistics.sufficientStatistics.applyParallaxShift(samplingData->vmm, shift);
                samplingData->vmm.performRelativeParallaxShift(shift);
            }
            factory.update(samplingData->vmm, trainingData->statistics, samples, numSamples, cfg, trainingData->fittingStatistics);
        } else {
            factory.fit(samplingData->vmm, trainingData->statistics, samples, numSamples, cfg, trainingData->fittingStatistics);
        }
        SINGLE samplingData->pivot = sampleMean;

        SYNC;

        coopCopy(gSamplingData + n, samplingData, sizeof(SamplingData));
        coopCopy(gTrainingData + n, trainingData, sizeof(TrainingData));
    }
}
}
}
