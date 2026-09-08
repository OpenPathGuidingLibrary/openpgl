
#pragma once

#include "../../openpgl_common.h"
#include "Rect.h"

namespace openpgl
{

inline bool rescale(float thresh, float &random)
{
    OPENPGL_ASSERT(0 <= thresh && thresh <= 1 && 0 <= random && random <= 1);

    if (random < thresh)
    {
        random /= thresh;
        return true;
    }
    else
    {
        random -= thresh;
        random /= 1 - thresh;
        return false;
    }
}

inline uint32_t rescaleChild(Vector2 &point)
{
    OPENPGL_ASSERT(0.f <= point.x && point.x <= 1.f && 0.f <= point.y && point.y <= 1.f);

    uint32_t offset = 0;
    if (point.x >= 0.5)
    {
        point.x -= 0.5;
        offset += 1;
    }
    if (point.y >= 0.5)
    {
        point.y -= 0.5;
        offset += 2;
    }
    point *= 2.0f;
    return offset;
}

// Generic query routine for quadtrees.
// returns node and rectangle which a given point falls into
template <typename TNode>
size_t queryNode(const TNode *nodes, Vector2 point, Rect<float> &rect)
{
    rect = {{0, 0}, {1, 1}};
    size_t nodeIdx = 0;
    const TNode *node = &nodes[nodeIdx];
    while (node->offsetChildren > 0)
    {
        uint32_t c = rescaleChild(point);
        rect = rect.child(c);
        nodeIdx = node->offsetChildren + c;
        node = &nodes[nodeIdx];
    }
    return nodeIdx;
}

// Generic traversal routine for quadtrees
template <typename TNode, typename F1, typename F2>
void traverse(const TNode *nodes, F1 pre, F2 post, uint32_t i = 0, Rect<float> rect = {{0, 0}, {1, 1}})
{
    if (pre(i, rect) && nodes[i].offsetChildren > 0)
    {
        for (uint32_t j = 0; j < 4; j++)
            traverse(nodes, pre, post, nodes[i].offsetChildren + j, rect.child(j));
    }
    post(i, rect);
}

// Generic splatting routine for quadtrees
template <bool TIsFootprintFactorNonZero, typename TNode, typename F>
void splat(TNode *nodes, float footprintFactor, Vector2 point, F apply)
{
    Rect<float> rect;
    size_t node_idx = queryNode(nodes, point, rect);
    if (TIsFootprintFactorNonZero)
    {
        auto size = rect.max - rect.min;
        Rect<float> filterRect = {point - footprintFactor * size / 2, point + footprintFactor * size / 2};
        filterRect = filterRect.intersect({{0, 0}, {1, 1}});
        float filterArea = filterRect.area();

        OPENPGL_ASSERT(filterArea > 0);

        traverse(
            nodes,
            [&](uint32_t i, Rect<float> rect) {
                auto &node = nodes[i];
                float nodeArea = filterRect.intersect(rect).area();
                if (nodeArea == 0)
                    return false;
                if (node.offsetChildren == 0)
                    apply(node, nodeArea / filterArea);
                return true;
            },
            [&]([[maybe_unused]] uint32_t i, [[maybe_unused]] Rect<float> rect) {
            });
    }
    else
    {
        apply(nodes[node_idx], 1);
    }
}

}  // namespace openpgl