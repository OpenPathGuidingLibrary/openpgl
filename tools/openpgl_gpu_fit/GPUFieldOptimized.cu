#include "GPUField.h"

namespace openpgl{
namespace gpu {
namespace cuda {

size_t GPUField::reduceSampleStats() {
    // PERF: reduce_by_key forces synchronization to return number of reduced elements
    // we might be able to avoid this, by directly using ReduceByKey from cub
    auto [reducedLeafIndicesEnd, reducedSampleStatsEnd] = thrust::reduce_by_key(
        reorderedLeafIndices.begin(), reorderedLeafIndices.end(), sampleStats.begin(),
        reducedLeafIndices.begin(), reducedSampleStats.begin(),
        thrust::equal_to<uint32_t>(),
        [] __device__ (const IntegerSampleStats &a, const IntegerSampleStats &b) {
            //return IntegerSampleStats();
            return IntegerSampleStats(a, b);
        }
    );
    return reducedLeafIndicesEnd - reducedLeafIndices.begin();
}

}
}
}