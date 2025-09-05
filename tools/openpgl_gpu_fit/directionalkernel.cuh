#pragma once

#include "directionalkernel.cuh"

namespace openpgl {
namespace gpu {
namespace cuda {
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

        __shared__ char sampleStatisticsBuffer[sizeof(SampleStatistics)];
        __shared__ char samplingDataBuffer[sizeof(SamplingData)];
        __shared__ char trainingDataBuffer[sizeof(TrainingData)];

        SampleStatistics *sampleStatistics = reinterpret_cast<SampleStatistics*>(sampleStatisticsBuffer);
        SamplingData *samplingData = reinterpret_cast<SamplingData*>(samplingDataBuffer);
        TrainingData *trainingData = reinterpret_cast<TrainingData*>(trainingDataBuffer);

        coopCopy(sampleStatistics, gSampleStatistics + n, sizeof(SampleStatistics));
        coopCopy(samplingData, gSamplingData + record.readIdx, sizeof(SamplingData));
        coopCopy(trainingData, gTrainingData + record.readIdx, sizeof(TrainingData));

        __syncthreads();

        Factory factory;

        SampleData* samples = gSamples + leafHistogram[n];
        const size_t numSamples = leafHistogram[n + 1] - leafHistogram[n];

        // TODO parallelize
        if (threadIdx.x == 0) {
            // TODO sort samples
            factory.prepareSamples(samples, numSamples, *sampleStatistics, cfg);

            openpgl::Point3 sampleMean = sampleStatistics->getMean();
            if (false) {
                if (record.readIdx != n) {
                    const float alpha = 0.25f; // TODO expose parameter
                    samplingData->vmm.decay(alpha); 
                    trainingData->statistics.decay(alpha);
                }
    
                
                Vector3 shift = samplingData->pivot - sampleMean;
                trainingData->statistics.sufficientStatistics.applyParallaxShift(samplingData->vmm, shift);
                samplingData->vmm.performRelativeParallaxShift(shift);
    
                factory.update(samplingData->vmm, trainingData->statistics, samples, numSamples, cfg, trainingData->fittingStatistics);
            } else {
                factory.fit(samplingData->vmm, trainingData->statistics, samples, numSamples, cfg, trainingData->fittingStatistics);
            }
            samplingData->pivot = sampleMean;
        }

        __syncthreads();

        coopCopy(gSamplingData + n, samplingData, sizeof(SamplingData));
        coopCopy(gTrainingData + n, trainingData, sizeof(TrainingData));
    }
}
}
}
