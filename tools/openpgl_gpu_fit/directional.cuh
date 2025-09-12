#include "../../openpgl/directional/vmm/AdaptiveSplitandMergeFactory.h"
#include "../../openpgl/directional/vmm/ParallaxAwareVonMisesFisherWeightedEMFactory.h"
//#include "../../openpgl/directional/vmm/ParallaxAwareVonMisesFisherMixture.h"
//#include "../../openpgl/directional/vmm/AdaptiveSplitandMergeFactory.h"

namespace openpgl {
namespace gpu {
namespace cuda {
    constexpr static int BlockDim = 384;
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
        Factory::Statistics statistics;
        Factory::FittingStatistics fittingStatistics;
    };
}
}
}
