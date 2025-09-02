#pragma once

//#include "../../openpgl/directional/vmm/AdaptiveSplitandMergeFactory.h"
//#include "../../openpgl/directional/vmm/ParallaxAwareVonMisesFisherWeightedEMFactory.h"
#if true
#define OPENPGL_VEC_SIZE 1
#include "../../openpgl/kernel/cuda.h"
#include "../../openpgl/data/SampleStatistics.h"
#include "../../openpgl/directional/vmm/ParallaxAwareVonMisesFisherMixture.h"
#include "../../openpgl/directional/vmm/AdaptiveSplitandMergeFactory.h"

namespace openpgl {
    using VMM = ParallaxAwareVonMisesFisherMixture<Kernel, 32, true>;
    using Factory = AdaptiveSplitAndMergeFactory<VMM>;
    
    __global__ void EMFit(Factory::Configuration cfg, SampleData* samples, const size_t numSamples) {
        __shared__ char vmmBuffer[sizeof(VMM)];
        __shared__ char statisticsBuffer[sizeof(Factory::Statistics)];
        __shared__ char fittingStatisticsBuffer[sizeof(Factory::FittingStatistics)];

        VMM *vmm = reinterpret_cast<VMM*>(vmmBuffer);
        Factory::Statistics *statistics = reinterpret_cast<Factory::Statistics*>(statisticsBuffer);
        Factory::FittingStatistics *fittingStatistics = reinterpret_cast<Factory::FittingStatistics*>(fittingStatisticsBuffer);

        // TODO parallel loading of stats into shared memory

        Factory factory;

        if (threadIdx.x == 0)
            factory.update(*vmm, *statistics, samples, numSamples, cfg, *fittingStatistics);
        //using DirectionalDistributionFactory = AdaptiveSplitAndMergeFactory<>;
    
    
        //__shared__ openpgl::AdaptiveSplitandMergeFactory
        //ParallaxAwareVonMisesFisherWeightedEMFactory::
        //__shared__ 
        //SufficientStatistics;
    }
}
#endif