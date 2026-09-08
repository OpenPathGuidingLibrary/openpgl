#pragma once

#include <cinttypes>
#include <sstream>
#include <vector>

#include "../../openpgl_common.h"
#include "Rect.h"
#include "Traversal.h"

namespace openpgl
{
struct DirectionalQuadtreeNode
{
    uint32_t offsetChildren = 0;
    float sampleWeight = 0;

    std::string toString() const
    {
        std::stringstream ss;
        ss << "DirectionalQuadtreeNode:" << std::endl;
        ss << "\toffsetChildren = " << offsetChildren << std::endl;
        ss << "\tsampleWeight = " << sampleWeight << std::endl;
        return ss.str();
    }
};

template <typename TSphere2Square>
struct DirectionalQuadtree
{
    enum
    {
        ParallaxCompensation = 0
    };

    using Sphere2Square = TSphere2Square;

    Point3 _pivotPosition;
    std::vector<DirectionalQuadtreeNode> nodes = {DirectionalQuadtreeNode()};

    DirectionalQuadtree() = default;

    // Public Interface
    inline bool isValid() const
    {
        return !std::isinf(nodes[0].sampleWeight) && nodes[0].sampleWeight > 0;
    }

    inline Vector3 sample(const Vector2 sample) const
    {
        OPENPGL_ASSERT(isValid());
        OPENPGL_ASSERT(0 <= sample.x && sample.x <= 1 && 0 <= sample.y && sample.y <= 1);

        float pdf;
        Vector2 point = sampleQuadtree(sample, pdf);
        return Sphere2Square::pointToDirection(point);
    }

    inline float pdf(const Vector3 direction) const
    {
        OPENPGL_ASSERT(isValid());

        Vector2 point = Sphere2Square::directionToPoint(direction);
        float pdf = std::min(FLT_MAX, pdfQuadtree(point) / Sphere2Square::jacobian(point));
        OPENPGL_ASSERT(pdf > 0);
        return pdf;
    }

    inline float samplePdf(const Vector2 sample, Vector3 &dir) const
    {
        OPENPGL_ASSERT(isValid());
        OPENPGL_ASSERT(0 <= sample.x && sample.x <= 1 && 0 <= sample.y && sample.y <= 1);

        float pdf;
        Vector2 point = sampleQuadtree(sample, pdf);
        pdf /= Sphere2Square::jacobian(point);
        OPENPGL_ASSERT(pdf > 0);
        dir = Sphere2Square::pointToDirection(point);
        return pdf;
    }

    void performRelativeParallaxShift(const Vector3 &) {};

    const std::string toString() const
    {
        std::stringstream out;

        out << "DirectionalQuadtree [" << std::endl;
        for (int i = 0; i < nodes.size(); i++)
            out << i << ": " << nodes[i].toString();
        out << "]" << std::endl;

        return out.str();
    }

    void serialize(std::ostream &os) const
    {
        os.write(reinterpret_cast<const char *>(&_pivotPosition), sizeof(_pivotPosition));
        size_t size = nodes.size();
        os.write(reinterpret_cast<const char *>(&size), sizeof(size));
        os.write(reinterpret_cast<const char *>(nodes.data()), size * sizeof(nodes[0]));
    };

    void deserialize(std::istream &is)
    {
        is.read(reinterpret_cast<char *>(&_pivotPosition), sizeof(_pivotPosition));
        size_t size;
        is.read(reinterpret_cast<char *>(&size), sizeof(size));
        nodes = std::vector<DirectionalQuadtreeNode>(size);
        is.read(reinterpret_cast<char *>(nodes.data()), size * sizeof(nodes[0]));
    };

    void decay(const float) {};

    size_t getNumComponents() const
    {
        return nodes.size();
    }

    // TODO: Needs to be implmented
    bool operator==(const DirectionalQuadtree &) const
    {
        return true;
    }

   private:
    // Internal Sampling Routines
    // TODO prepare weights so that no rescaling needed? (i.e. three thresholds per cell)
    inline Vector2 sampleQuadtree(const Vector2 sample, float &pdf) const
    {
        // perform stochastic top-down traveral, according to sampling weights of nodes
        Vector2 random = sample;

        float span = 1;
        Vector2 point(0, 0);

        const DirectionalQuadtreeNode *node = &nodes[0];
        while (node->offsetChildren > 0)
        {
            span /= 2;
            uint32_t offset = 0;
            const DirectionalQuadtreeNode *children = &nodes[node->offsetChildren];

            float weightLeft = children[0].sampleWeight + children[2].sampleWeight;
            float weightRight = children[1].sampleWeight + children[3].sampleWeight;
            OPENPGL_ASSERT(weightLeft + weightRight > 0);
            float probabilityLeft = weightLeft / (weightLeft + weightRight);

            if (!rescale(probabilityLeft, random.x))
            {
                offset += 1;
                point.x += span;
            }

            float weightTop = children[offset + 0].sampleWeight;
            float weightBottom = children[offset + 2].sampleWeight;
            OPENPGL_ASSERT(weightTop + weightBottom > 0);
            float probabilityTop = weightTop / (weightTop + weightBottom);

            if (!rescale(probabilityTop, random.y))
            {
                offset += 2;
                point.y += span;
            }

            node = &children[offset];
        };

        pdf = node->sampleWeight / (nodes[0].sampleWeight * span * span);
        return point + random * span;
    }

    inline float pdfQuadtree(Vector2 point) const
    {
        Rect<float> rect;
        auto node_idx = queryNode(nodes.data(), point, rect);
        return nodes[node_idx].sampleWeight / (nodes[0].sampleWeight * rect.area());
    }
};
}  // namespace openpgl