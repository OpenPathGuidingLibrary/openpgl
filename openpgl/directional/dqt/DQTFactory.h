#pragma once

#include <cinttypes>
#include <vector>

#include "../../data/SampleData.h"
#include "../../data/SampleStatistics.h"
#include "../../include/openpgl/types.h"
#include "../../openpgl_common.h"
#include "DQT.h"
#include "Rect.h"
#include "Traversal.h"

namespace openpgl
{

enum LeafEstimator
{
    REJECTION_SAMPLING = 0,
    PER_LEAF,
};

enum SplitMetric
{
    MEAN = 0,
    SECOND_MOMENT,
};

template <typename TDistribution>
class DirectionalQuadtreeFactory
{
   public:
    const static PGL_DIRECTIONAL_DISTRIBUTION_TYPE DIRECTIONAL_DISTRIBUTION_TYPE = PGL_DIRECTIONAL_DISTRIBUTION_QUADTREE;

    using Distribution = TDistribution;
    using Sphere2Square = typename Distribution::Sphere2Square;

    struct Configuration
    {
        // Estimator for the incoming radiance (and second moment thereof) of each leaf.
        // REJECTION_SAMPLING
        //   Accumulate samples (contribution / pdf) into quadtree leaves,
        //   then normalize with total sample count.
        // PER_LEAF
        //   Accumulate samples (contribution) into quadtree leaves,
        //   then normalize with leaf sample count * solid angle of leaf.
        //   Gives more stable results when leaves receive few samples.
        LeafEstimator leafEstimator = LeafEstimator::REJECTION_SAMPLING;
        // Metric according to which leaves are split (or merged).
        // MEAN
        //   Split according to mean of incoming radiance from leaf.
        // SECOND_MOMENT
        //   Split according to mean and variance of incoming radiance from leaf.
        //   This additionally steers splitting of leaves which internally have an uneven distribution of incoming radiance.
        SplitMetric splitMetric = SplitMetric::MEAN;
        // Nodes are split if their metric is more than a fraction (splitThreshold) of the summed metric over all nodes
        float splitThreshold = 0.1f;
        // Samples are accumulated into multiple nodes according to a footprint that is constructed from the sample center.
        // The footprint size is scaled by footprintFactor and the size of leaf in which the sample falls.
        // A footprint of 0 denotes that all radiance is accumulated into the single leaf in which the sample falls.
        float footprintFactor = 0;

        uint32_t maxLevels = 12;

        void serialize(std::ostream &stream) const
        {
            stream.write(reinterpret_cast<const char *>(this), sizeof(*this));
        };

        void deserialize(std::istream &stream)
        {
            stream.read(reinterpret_cast<char *>(this), sizeof(*this));
        };

        bool operator==(const Configuration &b) const
        {
            return true;
        }
    };

    struct StatsNode
    {
        uint32_t offsetChildren = 0;

        float sampleWeight = 0.f;
        float splitWeight = 0.f;

        float numSamples = 0.f;
        float firstMoment = 0.f;
        float secondMoment = 0.f;
    };

    struct Statistics
    {
        // TODO this is very ugly
        struct SufficientStatistics
        {
            void applyParallaxShift(const Distribution &dist, const Vector3 shift) {}
            bool isValid() const
            {
                return true;
            }
        } sufficientStatistics;

        float numSamples = 0;
        std::vector<StatsNode> nodes = {StatsNode()};

        bool isValid() const
        {
            auto valid = [](float number) {
                return !std::isnan(number) && !std::isinf(number) && number >= 0;
            };
            for (auto &node : nodes)
            {
                if (!valid(node.sampleWeight))
                    return false;
                if (!valid(node.splitWeight))
                    return false;
                if (!valid(node.numSamples))
                    return false;
                if (!valid(node.firstMoment))
                    return false;
                if (!valid(node.secondMoment))
                    return false;
            }
            return nodes[0].sampleWeight > 0 && nodes[0].splitWeight > 0;
        }

        void decay(float &alpha)
        {
            numSamples *= alpha;
            for (auto &node : nodes)
            {
                node.numSamples *= alpha;
                node.firstMoment *= alpha;
                node.secondMoment *= alpha;
            }
        }

        void serialize(std::ostream &os) const
        {
            os.write(reinterpret_cast<const char *>(&numSamples), sizeof(numSamples));
            size_t size = nodes.size();
            os.write(reinterpret_cast<const char *>(&size), sizeof(size));
            os.write(reinterpret_cast<const char *>(nodes.data()), size * sizeof(nodes[0]));
        };

        void deserialize(std::istream &is)
        {
            is.read(reinterpret_cast<char *>(&numSamples), sizeof(numSamples));
            size_t size;
            is.read(reinterpret_cast<char *>(&size), sizeof(size));
            nodes = std::vector<StatsNode>(size);
            is.read(reinterpret_cast<char *>(nodes.data()), size * sizeof(nodes[0]));
        };

        // TODO: Needs to be implmented
        bool operator==(const Statistics &b) const
        {
            return true;
        }
    };

    struct FittingStatistics
    {
        uint32_t numSamples = 0;
        uint32_t numNodes = 0;
        uint32_t numSplits = 0;
        uint32_t numMerges = 0;
    };

    void prepareSamples(SampleData *samples, const size_t numSamples, const SampleStatistics &sampleStatistics, const Configuration &cfg) const {}

    void updateFluenceEstimate(Distribution &dist, const SampleData *samples, const size_t numSamples, const size_t numZeroValueSamples,
                               const SampleStatistics &sampleStatistics) const
    {}

    void fit(Distribution &dist, Statistics &stats, const SampleData *samples, const size_t numSamples, const Configuration &cfg, FittingStatistics &fitStats)
    {
        for (uint32_t i = 0; i < 5; i++)
        {
            float decay = 0.25;
            stats.decay(decay);
            update(dist, stats, samples, numSamples, cfg, fitStats);
        }
    }

    void update(Distribution &dist, Statistics &stats, const SampleData *samples, const size_t numSamples, const Configuration &cfg, FittingStatistics &fitStats)
    {
        Context ctx = {&dist, &stats, &fitStats, samples, numSamples, &cfg};

        auto leafEstimator = ctx.cfg->leafEstimator;
        auto splitMetric = ctx.cfg->splitMetric;
        auto footprintFactor = ctx.cfg->footprintFactor;

        OPENPGL_ASSERT(footprintFactor >= 0);

        // for efficiency, updateInternal is templated over the leaf estimator, split metric and if the footprintFactor is 0
        // here we dispatch to the respective implementation, depending on the configuration options
        if (leafEstimator == LeafEstimator::REJECTION_SAMPLING && splitMetric == SplitMetric::MEAN && footprintFactor == 0)
            updateInternal<LeafEstimator::REJECTION_SAMPLING, SplitMetric::MEAN, false>(ctx);
        else if (leafEstimator == LeafEstimator::REJECTION_SAMPLING && splitMetric == SplitMetric::MEAN && footprintFactor > 0)
            updateInternal<LeafEstimator::REJECTION_SAMPLING, SplitMetric::MEAN, true>(ctx);
        else if (leafEstimator == LeafEstimator::REJECTION_SAMPLING && splitMetric == SplitMetric::SECOND_MOMENT && footprintFactor == 0)
            updateInternal<LeafEstimator::REJECTION_SAMPLING, SplitMetric::SECOND_MOMENT, false>(ctx);
        else if (leafEstimator == LeafEstimator::REJECTION_SAMPLING && splitMetric == SplitMetric::SECOND_MOMENT && footprintFactor > 0)
            updateInternal<LeafEstimator::REJECTION_SAMPLING, SplitMetric::SECOND_MOMENT, true>(ctx);
        else if (leafEstimator == LeafEstimator::PER_LEAF && splitMetric == SplitMetric::MEAN && footprintFactor == 0)
            updateInternal<LeafEstimator::PER_LEAF, SplitMetric::MEAN, false>(ctx);
        else if (leafEstimator == LeafEstimator::PER_LEAF && splitMetric == SplitMetric::MEAN && footprintFactor > 0)
            updateInternal<LeafEstimator::PER_LEAF, SplitMetric::MEAN, true>(ctx);
        else if (leafEstimator == LeafEstimator::PER_LEAF && splitMetric == SplitMetric::SECOND_MOMENT && footprintFactor == 0)
            updateInternal<LeafEstimator::PER_LEAF, SplitMetric::SECOND_MOMENT, false>(ctx);
        else if (leafEstimator == LeafEstimator::PER_LEAF && splitMetric == SplitMetric::SECOND_MOMENT && footprintFactor > 0)
            updateInternal<LeafEstimator::PER_LEAF, SplitMetric::SECOND_MOMENT, true>(ctx);
    }

   private:
    struct Context
    {
        Distribution *dist;
        Statistics *stats;
        FittingStatistics *fitStats;
        const SampleData *samples;
        const size_t numSamples;
        const Configuration *cfg;
    };

    // Internal Update Routines
    template <LeafEstimator TLeafEstimator, SplitMetric TSplitMetric, bool TIsFootprintFactorNonZero>
    void updateInternal(Context &ctx)
    {
        ctx.fitStats->numSamples = ctx.stats->numSamples += ctx.numSamples;

        // Accumulate all samples into leaves
        for (uint32_t i = 0; i < ctx.numSamples; i++)
        {
            const auto &sample = ctx.samples[i];

            OPENPGL_ASSERT(isValid(sample));

            float firstMoment = 0, secondMoment = 0;

            // TODO use if constexpr when we start using c++17
            if (TLeafEstimator == LeafEstimator::REJECTION_SAMPLING)
            {
                firstMoment = sample.weight;
                secondMoment = sample.weight * sample.weight * sample.pdf;
            }
            if (TLeafEstimator == LeafEstimator::PER_LEAF)
            {
                firstMoment = sample.weight * sample.pdf;
                secondMoment = firstMoment * firstMoment;
            }
            const pgl_vec3f pglDirection = sample.direction;
            const Vector3 direction = {pglDirection.x, pglDirection.y, pglDirection.z};
            float footprintFactor = ctx.cfg->footprintFactor;
            splat<TIsFootprintFactorNonZero>(ctx.stats->nodes.data(), footprintFactor, Sphere2Square::directionToPoint(direction), [&](StatsNode &node, float weight) {
                node.numSamples += weight;
                node.firstMoment += weight * firstMoment;
                node.secondMoment += weight * secondMoment;
            });
        }

        // Compute sampling and split weights
        traverse(
            ctx.stats->nodes.data(),
            [&](uint32_t i, Rect<float> &rect) {
                return true;  // We want to traverse all nodes
            },
            [&](uint32_t i, Rect<float> &rect) {
                auto &node = ctx.stats->nodes[i];
                if (node.offsetChildren > 0)
                {
                    node.sampleWeight = 0;
                    node.splitWeight = 0;
                    for (uint32_t i = 0; i < 4; i++)
                    {
                        StatsNode &child_node = ctx.stats->nodes[node.offsetChildren + i];
                        node.sampleWeight += child_node.sampleWeight;
                        node.splitWeight += child_node.splitWeight;
                    }
                }
                else
                {
                    float n;
                    if (TLeafEstimator == LeafEstimator::REJECTION_SAMPLING)
                        n = 1.0f / ctx.stats->numSamples;
                    if (TLeafEstimator == LeafEstimator::PER_LEAF)
                        n = Sphere2Square::area(rect) / node.numSamples;
                    float firstMoment = node.firstMoment * n;
                    float secondMoment = node.secondMoment * n;

                    node.sampleWeight = firstMoment;
                    if (TSplitMetric == SplitMetric::MEAN)
                        node.splitWeight = firstMoment;
                    if (TSplitMetric == SplitMetric::SECOND_MOMENT)
                        node.splitWeight = std::sqrt(Sphere2Square::area(rect) * secondMoment);
                }
            });

        // Build new tree according to split weights
        ctx.fitStats->numSplits = 0;
        ctx.fitStats->numMerges = 0;
        auto old_nodes = std::move(ctx.stats->nodes);
        ctx.stats->nodes = {};
        ctx.stats->nodes.emplace_back();
        buildRecursive(ctx, old_nodes, {{0, 0}, {1, 1}}, 0, 0);
        ctx.fitStats->numNodes = ctx.stats->nodes.size();

        ctx.dist->nodes.clear();
        for (auto &node : ctx.stats->nodes)
        {
            DirectionalQuadtreeNode qnode;
            qnode.offsetChildren = node.offsetChildren;
            qnode.sampleWeight = node.sampleWeight;
            ctx.dist->nodes.push_back(qnode);
        }
    }

    void buildRecursive(Context &ctx, std::vector<StatsNode> &old_nodes, Rect<float> rect, uint32_t i, uint32_t j, uint32_t level = 0)
    {
        StatsNode &old_node = old_nodes[i];
        ctx.stats->nodes[j] = old_node;
        if (level < ctx.cfg->maxLevels && old_node.splitWeight > (ctx.cfg->splitThreshold * old_nodes[0].splitWeight))
        {
            ctx.stats->nodes[j].offsetChildren = ctx.stats->nodes.size();
            for (uint32_t c = 0; c < 4; c++)
                ctx.stats->nodes.emplace_back();
            if (old_node.offsetChildren > 0)
            {
                for (uint32_t c = 0; c < 4; c++)
                    buildRecursive(ctx, old_nodes, rect.child(c), old_node.offsetChildren + c, ctx.stats->nodes[j].offsetChildren + c, level + 1);
            }
            else
            {
                // If we have encountered a leaf node that we want to split,
                // we continue with buildSplit to potentially further split the node
                ctx.fitStats->numSplits++;
                for (uint32_t c = 0; c < 4; c++)
                    buildSplit(ctx, rect.child(c), j, ctx.stats->nodes[j].offsetChildren + c, level + 1);
            }
        }
        else
        {
            ctx.stats->nodes[j].offsetChildren = 0;
            // continue traversal to quantify number of nodes that have been merged
            // TODO switch to disable this
            traverse(
                old_nodes.data(),
                [&](uint32_t i, Rect<float> &rect) {
                    if (old_nodes[i].offsetChildren > 0)
                        ctx.fitStats->numMerges++;
                    return true;
                },
                [&](uint32_t i, Rect<float> &rect) {
                },
                i, rect);
        }
    }

    void buildSplit(Context &ctx, Rect<float> rect, uint32_t p, uint32_t i, uint32_t level)
    {
        const auto &parent = ctx.stats->nodes[p];
        auto &node = ctx.stats->nodes[i];

        node.offsetChildren = 0;
        node.numSamples = parent.numSamples / 4;
        node.firstMoment = parent.firstMoment / 4;
        node.secondMoment = parent.secondMoment / 4;
        node.sampleWeight = parent.sampleWeight / 4;
        node.splitWeight = parent.splitWeight / 4;

        return;  // TODO splitting further seems to lead to overfitting

        if (level < ctx.cfg->maxLevels && node.splitWeight > (ctx.cfg->splitThreshold * ctx.stats->nodes[0].splitWeight))
        {
            ctx.fitStats->numSplits++;
            uint32_t offsetChildren = ctx.stats->nodes.size();
            node.offsetChildren = offsetChildren;
            for (uint32_t c = 0; c < 4; c++)
                ctx.stats->nodes.emplace_back();
            for (uint32_t c = 0; c < 4; c++)
                buildSplit(ctx, rect.child(c), i, offsetChildren + c, level + 1);
        }
    }
};

}  // namespace openpgl