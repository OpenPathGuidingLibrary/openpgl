#pragma once

#include <cstdint>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/sort.h>

#include "../../openpgl/include/openpgl/data.h"
#include "../../openpgl/data/SampleStatistics.h"

#include "../../openpgl/include/openpgl/breadcrump.h"
#include "../../openpgl/include/openpgl/sdump.h"
#include "../../openpgl/include/openpgl/gpu/Data.h"

#include "Common.h"
#include "Kernels.h"

namespace openpgl {
namespace OPENPGL_KERNEL_NS {

struct FieldCUDA {
    Factory::Configuration dcfg;

    // stats for leaf nodes
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
        
    FieldCUDA() {
        // initialize tree to single root node
        uint32_t initCapacity = 1024;

        resize(state, 1);
        resize(tree, initCapacity);
        resize(quantizationFrame, initCapacity);
        resize(leafStats, initCapacity);
        resize(sampleStats, initCapacity);

        resize(trainingData, initCapacity);
        resize(samplingData, initCapacity);

        resize(sampleStats, initCapacity);
        
        // + 1 so that ranges are always be computed with [hist[i], hist[i + 1]]
        resize(leafHistogram, initCapacity + 1);

        resize(finishedNodes, ceilDiv(initCapacity, std::numeric_limits<uint32_t>::digits));
        resize(records, initCapacity);


        hostState.nodeAlloc = 1;
        hostState.anySplit = 0;
        state[0] = hostState;

        TreeNode node;
        node.pivot = 0;
        node.setSplitDimAndNodeIdx(TreeNode::ELeafNode, 0);
        //node.bc = Breadcrumb();
        tree[0] = node;

        leafStats[0] = {};

        computeSizes();
    }

    void computeSizes() const {
        float size = 
            getSize(state) +
            getSize(tree) +
            getSize(quantizationFrame) +
            getSize(leafStats) +

            getSize(trainingData) +
            getSize(samplingData) +

            getSize(samplePositions) +
            getSize(sampleStats) +

            getSize(finishedNodes) +
            getSize(records) +

            getSize(sortKeys) +
            getSize(sortIndices) +
            getSize(leafHistogram) +
            getSize(reorderedSamples);

        printf("total size: %f GB\n", size);
    }

    void sDump(SDump* sDump) const {
        thrust::host_vector<TreeNode> tree = this->tree;

        std::function<void(SDumpTree*, int)> rec;
        rec = [&](SDumpTree* sDump, int idx) -> void {
            const auto& node = tree[idx];
            if (node.isLeaf()) {
                sDump->left = nullptr;
                sDump->right = nullptr;
            } else {
                sDump->axis = node.getSplitDim();
                sDump->split = node.pivot;
                
                sDump->left = new SDumpTree;
                sDump->right = new SDumpTree;
                rec(sDump->left, node.getNodeIdx() + 0);
                rec(sDump->right, node.getNodeIdx() + 1);
            }
        };

        sDump->sur = new SDumpTree;
        rec(sDump->sur, 0);
        sDump->vol = new SDumpTree;
        sDump->vol->left = nullptr;
        sDump->vol->right = nullptr;
    }

    void Update(uint32_t numSamples, const thrust::device_vector<PGLSampleData> &samples) {
        if (enableFingerprinting) {
            thrust::host_vector<PGLSampleData> hSamples = samples;
            uint32_t hash = 0;
            for (int i = 0; i < hSamples.size(); i++) {
                hash ^= murmur3_32_t(&hSamples[i], 1);
            }
            printf("samples: 0x%08" PRIX32 "\n", hash);
        }

        if (numSamples == 0) return;

        printf("updating tree with %u samples\n", numSamples);
        
        resize(sortKeys, numSamples);
        resize(sortIndices, numSamples);
        resize(samplePositions, numSamples);
        resize(reorderedSamples, numSamples);

        checkUsage();

        CudaTimer wholeTimer;

        // first iteration: estimate scene bounds
        if (it == 0) {
            CudaTimer initTimer;
            hostState.bounds = thrust::transform_reduce(samples.begin(), samples.end(),
                [] HOST_DEVICE (const PGLSampleData &x) { 
                    OPENPGL_ASSERT(isValid(toVec3(x.position)));
                    auto res = BBox(toVec3(x.position));
                    OPENPGL_ASSERT(is_finite(res));
                    return res;
                },
                BBox {openpgl::Vector3(std::numeric_limits<float>::max()), openpgl::Vector3(-std::numeric_limits<float>::max())},
                [] HOST_DEVICE (const BBox &a, const BBox &b) -> BBox {
                    auto res = BBox::merge(a, b);
                    OPENPGL_ASSERT(is_finite(res));
                    return res;
                }
            );
            checkUsage();
            if (enableFingerprinting) 
                printf("bounds: 0x%08" PRIX32 "\n", murmur3_32_t(&hostState.bounds, 1));
            BBox bounds = hostState.bounds;
            //bounds.print();
            Vector3 center = (bounds.lower + bounds.upper) / 2.f;
            bounds.lower = center + 3.f * (bounds.lower - center);
            bounds.upper = center + 3.f * (bounds.upper - center);
            hostState.bounds = bounds;
            state[0] = hostState;
            QuantizationFrame frame;
            frame.aabb = hostState.bounds;
            frame.init();
            quantizationFrame[0] = frame;
            trainingData[0] = TrainingData {
                .initialized = false,
            };
            printf("init: %fms\n", initTimer.elapsed() * 1e3f);
        }


        thrust::fill(finishedNodes.begin(), finishedNodes.end(), 0);
        checkUsage();

        //printf("  nodeAlloc: %i anySplit: %i\n", hostState.nodeAlloc, hostState.anySplit);

        BuildSettings buildSettings;
        buildSettings.maxSamples = 32000;
        buildSettings.firstIteration = true;

        CudaTimer spatialTimer;
        do {
            {
                // worst case, every leaf node needs to be split
                uint32_t newSize = 2 * hostState.nodeAlloc + 1;
                resize(tree, newSize);
                resize(quantizationFrame, newSize); 
                resize(leafStats, newSize);
                resize(sampleStats, newSize);
                // + 1 so that ranges are always be computed with [hist[i], hist[i + 1]]
                resize(finishedNodes, ceilDiv(newSize, std::numeric_limits<uint32_t>::digits));
                resize(records, newSize);
            }

            // clear buffers
            thrust::fill(sampleStats.begin(), sampleStats.begin() + hostState.nodeAlloc, IntegerSampleStats());
        
            // accumulate stats
            launchThreads(" AggregateSamples",
                AggregateSamples, ceilDiv(numSamples, 3), 768,
                data(state), data(tree), data(quantizationFrame),
                numSamples, data(samples),
                data(samplePositions), data(sampleStats)
            );

            if (enableFingerprinting) {
                cudaDeviceSynchronize();
                uint32_t hash = 0;
                thrust::host_vector<TreeNode> hTree = tree;
                thrust::host_vector<IntegerSampleStats> hSampleStats = sampleStats;
                for (int i = 0; i < hostState.nodeAlloc; i++) {
                    if (hTree[i].isLeaf())
                        hash ^= murmur3_32_t(&hSampleStats[i], 1);
                }
                printf("  integer: 0x%08" PRIX32 "\n", hash);
            }

            hostState.anySplit = 0;
            state[0] = hostState;
            launchThreads(" SplitNodes",
                SplitNodes, hostState.nodeAlloc, 128,
                buildSettings, hostState.nodeAlloc, data(sampleStats),
                data(state), data(tree), data(quantizationFrame), data(leafStats), data(records), data(finishedNodes)
            );
            hostState = state[0];

            if (enableFingerprinting) {
                cudaDeviceSynchronize();
                uint32_t hash = 0;
                thrust::host_vector<TreeNode> hTree = tree;
                thrust::host_vector<SampleStatistics> hLeafStats = leafStats;
                for (int i = 0; i < hostState.nodeAlloc; i++) {
                    if (hTree[i].isLeaf())
                        hash ^= murmur3_32_t(&hLeafStats[i], 1);
                }
                printf("  plain:   0x%08" PRIX32 "\n", hash);
            }

            printf(" nodeAlloc: %i anySplit: %i\n", hostState.nodeAlloc, hostState.anySplit);
        
            buildSettings.firstIteration = false;
        } while(hostState.anySplit);
        printf("spatial: %fms\n", spatialTimer.elapsed() * 1e3f);

        resize(leafHistogram, hostState.nodeAlloc + 1);

        CudaTimer scatterTimer;
        {
            thrust::fill(leafHistogram.begin(), leafHistogram.end(), 0);
            launchThreads(" ComputeKeys", ComputeKeys, numSamples, 128,
                hostState, data(tree), numSamples, data(samples), data(sortKeys), data(sortIndices), data(leafHistogram));
            thrust::exclusive_scan(leafHistogram.begin(), leafHistogram.end(), leafHistogram.begin());
            thrust::stable_sort_by_key(sortKeys.begin(), sortKeys.end(), sortIndices.begin());
            launchThreads(" GatherSamples", GatherSamples, numSamples, 128,
                numSamples, data(samples), data(sortIndices), data(reorderedSamples));
        }
        printf("reorder: %fms\n", scatterTimer.elapsed() * 1e3f);

        resize(trainingData, hostState.nodeAlloc);
        resize(samplingData, hostState.nodeAlloc);

        thrust::device_vector<Fingerprints> fp(enableFingerprinting ? 1 : 0, Fingerprints());

        launchSMem("EMFit", EMFit, hostState.nodeAlloc, BlockDim, /*13824*/ 18432 /*29184*/ /*5852*/ /*14200*/,
            dcfg, data(tree), data(records), data(leafHistogram), data(leafStats),
            data(trainingData), data(samplingData), data(reorderedSamples), data(fp)
        );

        printf("whole: %fms\n", wholeTimer.elapsed() * 1e3f);

        
        if (enableFingerprinting) {
            sync();
            Fingerprints fp_ = fp[0];
            fp_.print();

            thrust::host_vector<SamplingData> hSamplingData = samplingData;
            uint32_t hash = 0;
            for (int i = 0; i < hostState.nodeAlloc; i++) {
                TreeNode node = tree[i];
                if (!node.isLeaf()) continue;
                SamplingData *samplingData_ = &hSamplingData[i];
                hash ^= murmur3_32_t(samplingData_, 1);
                //hash ^= sha256_hash((unsigned char*)&samplingData.vmm, sizeof(samplingData.vmm));
            }

            printf(" hash: 0x%08" PRIX32 "\n", hash);
        }

        it++;

        computeSizes();
    }

    void dump(const std::string& dumpFileName) const {
        BlobWriter writer(dumpFileName);

        thrust::host_vector<TreeNode> hTree = tree;
        thrust::host_vector<SamplingData> hSamplingData = samplingData;
        thrust::host_vector<TrainingData> hTrainingData = trainingData;
        thrust::host_vector<SampleStatistics> hSampleStatistics = leafStats;

        //writer << (uint64_t) hTree.size();

        for (int n = 0; n < hTree.size(); n++) {
            TreeNode node = hTree[n];

            if (!node.isLeaf()) continue;

            SamplingData samplingData = hSamplingData[n];
            VMM vmm = samplingData.vmm;
            //TrainingData trainingData = hTrainingData[n];
            SampleStatistics sampleStatistics = hSampleStatistics[n];

            BBox bbox = sampleStatistics.sampleBounds;
            Vector3 center = bbox.center();
            Vector3 size = bbox.size();

            // write spatial
            // clang-format off
            writer << (float)center.x << (float)center.y << (float)center.z 
                << (float)size.x << (float)size.y << (float)size.z
                << (float)0.5                                          // sampling.mean()
                << (uint64_t)sampleStatistics.getNumSamples()                                        // sampling.statisticalWeight()
                << (uint64_t)n                                             // this serves as ID to identify cells from visualizer
                << (float)0.5
                << (float)0.5
                << (float)0.5 << (float)0.5 << (float)0.5 /*irradiance.z*/
                << (float)0.f << (float)0.f << (float)0.f 
                //   << region.getBsdfSamplingFraction() << region.getBsdfSamplingFraction() /*product.z*/
                << (float)1.f /*normal.x*/ << (float)1.f /*normal.y*/ << (float)1.f;        /*normal.z;*/

            // write VMM
            //writer << (uint64_t)(sizeof(uint64_t) + vmm.getNumComponents() * 5 * sizeof(float));
            writer << (uint64_t)vmm.getNumComponents();
            
            for (int k = 0; k < vmm.getNumComponents(); k++) {
                writer << vmm._kappas[k]
                    << vmm._meanDirections[k].x << vmm._meanDirections[k].y << vmm._meanDirections[k].z
                    << vmm._distances[k]
                    << vmm._weights[k];
            }
        }
    }

    void fillFieldData(int &numNodes, void **nodes_, int &numDistributions, void **distributions_) const {
        struct Node {
            float splitPosition;
            unsigned int splitDimAndNodeIdx;
        };

        numNodes = hostState.nodeAlloc;
        numDistributions = hostState.nodeAlloc;

        thrust::host_vector<TreeNode> hostTree = tree;
        thrust::host_vector<SamplingData> hostSamplingData = samplingData;
        
        Node* nodes = new Node[numNodes];
        *nodes_ = (void*)nodes;

        for (int i = 0; i < numNodes; i++) {
            const auto node = hostTree[i];
            nodes[i] = Node {
                .splitPosition = node.pivot,
                .splitDimAndNodeIdx = node.splitDimAndNodeIdx
            };
        }

        openpgl::gpu::FlatVMM<32> *distributions = new openpgl::gpu::FlatVMM<32>[numDistributions];
        *distributions_ = (void*)distributions;

        for (int i = 0; i < numDistributions; i++) {
            auto &dst = distributions[i];
            const auto &src = hostSamplingData[i];

            for (int i = 0; i < src.vmm._numComponents; i++) {
                dst._weights[i] = src.vmm._weights[i];
                dst._kappas[i] = src.vmm._kappas[i];
                for (int j = 0; j < 3; j++)
                    dst._meanDirections[i][j] = src.vmm._meanDirections[i][j];
                dst._distances[i] = src.vmm._distances[i];
            }

            for (int j = 0; j < 3; j++)
                dst._pivotPosition[j] = src.pivot[j];
            dst._numComponents = src.vmm._numComponents;
            for (int j = 0; j < 3; j++)
                dst._outgoingRGB[j] = src.vmm.sumOutgoingRadiance[j] / src.vmm.numOutgoingRadiance;
        }
    }
};

}
}
