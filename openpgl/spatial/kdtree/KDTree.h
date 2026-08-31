// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tbb/concurrent_vector.h>

#include <fstream>
#include <iostream>
#include <string>

#include "../../openpgl_common.h"
#include "KDTreeStatistics.h"
#include "data/Buffered.h"

//#define USE_TREELETS

namespace openpgl
{

struct KDNode
{
    enum
    {
        ESPlitDimX = 0,
        ESPlitDimY = 1,
        ESPlitDimZ = 2,
        ELeafNode = 3,
    };

    float splitPosition{0.0f};
    uint32_t splitDimAndNodeIdx{0};

    /////////////////////////////
    // Child node functions
    /////////////////////////////
    bool isChild() const
    {
        return (splitDimAndNodeIdx >> 30) < ELeafNode;
    }

    void setLeftChildIdx(const uint32_t &idx)
    {
        OPENPGL_ASSERT(idx < (1U << 31));
        OPENPGL_ASSERT((splitDimAndNodeIdx & (3U << 30)) != (3U << 30));
        OPENPGL_ASSERT((splitDimAndNodeIdx >> 30) != 3);
        OPENPGL_ASSERT(!isLeaf());

        splitDimAndNodeIdx = ((splitDimAndNodeIdx >> 30) << 30) | idx;
        OPENPGL_ASSERT(idx == getLeftChildIdx());
    }

    uint32_t getLeftChildIdx() const
    {
        OPENPGL_ASSERT((splitDimAndNodeIdx & (3U << 30)) != (3U << 30));
        OPENPGL_ASSERT((splitDimAndNodeIdx >> 30) != 3);
        OPENPGL_ASSERT(!isLeaf());

        return (splitDimAndNodeIdx << 2) >> 2;
    }

    /////////////////////////////
    // Inner node functions
    /////////////////////////////

    void setToInnerNode(const uint8_t &_splitDim, const float &_splitPos, const uint32_t &_leftChildIdx)
    {
        splitPosition = _splitPos;
        splitDimAndNodeIdx = 0;
        splitDimAndNodeIdx = (uint32_t(_splitDim) << 30);
        splitDimAndNodeIdx = ((splitDimAndNodeIdx >> 30) << 30) | _leftChildIdx;

        OPENPGL_ASSERT(_splitDim == getSplitDim());
        OPENPGL_ASSERT(_leftChildIdx == getLeftChildIdx());
    }

    /////////////////////////////
    // Leaf node functions
    /////////////////////////////

    void setLeaf()
    {
        splitDimAndNodeIdx = (3U << 30);
        OPENPGL_ASSERT(isLeaf());
    }

    bool isLeaf() const
    {
        return (splitDimAndNodeIdx >> 30) == 3;
    }

    void setChildNodeIdx(const uint32_t &idx)
    {
        OPENPGL_ASSERT(idx < (1U << 31));
        OPENPGL_ASSERT((splitDimAndNodeIdx & (3U << 30)) != (3U << 30));
        OPENPGL_ASSERT((splitDimAndNodeIdx >> 30) != 3);
        OPENPGL_ASSERT(!isLeaf());

        splitDimAndNodeIdx = ((splitDimAndNodeIdx >> 30) << 30) | idx;
        OPENPGL_ASSERT(idx == getLeftChildIdx());
    }

    void setDataNodeIdx(const uint32_t &idx)
    {
        OPENPGL_ASSERT(idx < (1U << 31));  // checks if the idx is in the right range
        setLeaf();
        splitDimAndNodeIdx = ((splitDimAndNodeIdx >> 30) << 30) | idx;
        OPENPGL_ASSERT(isLeaf());
        OPENPGL_ASSERT(getDataIdx() == idx);
    }

    uint32_t getDataIdx() const
    {
        OPENPGL_ASSERT(isLeaf());
        return (splitDimAndNodeIdx << 2) >> 2;
    }

    /////////////////////////////
    // Split dimension functions
    /////////////////////////////

    uint8_t getSplitDim() const
    {
        return (splitDimAndNodeIdx >> 30);
    }

    void setSplitDim(const uint8_t &splitAxis)
    {
        OPENPGL_ASSERT(splitAxis < ELeafNode);
        splitDimAndNodeIdx = (uint32_t(splitAxis) << 30);
        OPENPGL_ASSERT(splitAxis == getSplitDim());
    }

    float getSplitPivot() const
    {
        return splitPosition;
    }

    void setSplitPivot(const float &pos)
    {
        splitPosition = pos;
    }

    void serialize(std::ostream &stream) const
    {
        stream.write(reinterpret_cast<const char *>(&splitPosition), sizeof(float));
        stream.write(reinterpret_cast<const char *>(&splitDimAndNodeIdx), sizeof(uint32_t));
    }

    void deserialize(std::istream &stream)
    {
        stream.read(reinterpret_cast<char *>(&splitPosition), sizeof(float));
        stream.read(reinterpret_cast<char *>(&splitDimAndNodeIdx), sizeof(uint32_t));
    }

    bool operator==(const KDNode &b) const
    {
        bool equal = true;
        if (splitPosition != b.splitPosition || isLeaf() != b.isLeaf())
        {
            equal = false;
        }
        if (!isLeaf())
        {
            if (getSplitDim() != b.getSplitDim() || getLeftChildIdx() != b.getLeftChildIdx())
            {
                equal = false;
            }
        }
        else
        {
            if (getDataIdx() != b.getDataIdx())
            {
                equal = false;
            }
        }
        return equal;
    }
};

struct KDTreeLet
{
    KDNode nodes[8];
};

struct KDTree
{
#ifdef USE_TREELETS
    using NodesType = KDTreeLet;
#else
    using NodesType = KDNode;
#endif
    KDTree() = default;

    KDTree(const KDTree &) = delete;

    ~KDTree()
    {
        if (m_nodesPtr)
            delete[] m_nodesPtr;

        if (m_treeLets)
        {
            delete[] m_treeLets;
            m_treeLets = nullptr;
            m_numTreeLets = 0;
        }
    }

    inline void init(const BBox &bounds, size_t numNodesReseve = 0)
    {
        m_bounds = bounds;
        m_nodes.clear();
        if (numNodesReseve > 0)
        {
            m_nodes.reserve(numNodesReseve);
        }
        clear();
        m_isInit = true;
    }

    inline void clear()
    {
        m_nodes.resize(1);
        m_nodes[0].setLeaf();
        m_nodes[0].setDataNodeIdx(0);
    }

    KDNode &getRoot()
    {
        return getNode(0);
    }

    KDNode &getNode(const size_t &idx)
    {
        // OPENPGL_ASSERT( m_isInit );
        OPENPGL_ASSERT(m_nodes.size() > idx);
        return m_nodes[idx];
    }

    const KDNode &getRoot() const
    {
        return getNode(0);
    }

    const KDNode &getNode(const size_t &idx) const
    {
        OPENPGL_ASSERT(m_nodes.size() > idx);
        return m_nodes[idx];
    }

    size_t getNumNodes() const
    {
        return m_nodes.size();
    }

    uint32_t addChildrenPair()
    {
        auto firstChildItr = m_nodes.grow_by(2);
        return std::distance(m_nodes.begin(), firstChildItr);
    }

    const BBox &getBounds() const
    {
        return m_bounds;
    }

    void rearrangeNodeForCompare(const KDNode &node, int idx, std::vector<KDNode> &newNodes, std::vector<uint32_t> &dataStorageIndices) const
    {
        if (!node.isLeaf())
        {
            uint32_t leftIdx = node.getLeftChildIdx();
            KDNode nodeLeft = m_nodes[leftIdx];
            KDNode nodeRight = m_nodes[leftIdx + 1];
            uint32_t newIdx = newNodes.size();
            newNodes[idx].setLeftChildIdx(newIdx);
            newNodes.push_back(nodeLeft);
            newNodes.push_back(nodeRight);
            rearrangeNodeForCompare(nodeLeft, newIdx, newNodes, dataStorageIndices);
            rearrangeNodeForCompare(nodeRight, newIdx + 1, newNodes, dataStorageIndices);
        }
        else
        {
            uint32_t dataIdx = node.getDataIdx();
            uint32_t newDataIdx = dataStorageIndices.size();
            dataStorageIndices.push_back(dataIdx);
            newNodes[idx].setDataNodeIdx(newDataIdx);
        }
    }

    void rearrangeNodesForCompare(std::vector<KDNode> &newNodes, std::vector<uint32_t> &dataStorageIndices) const
    {
        if (m_nodes.size() > 0)
        {
            KDNode root = m_nodes[0];
            newNodes.push_back(root);
            rearrangeNodeForCompare(root, 0, newNodes, dataStorageIndices);
        }
    }

    void finalize()
    {
        if (m_nodesPtr)
        {
            delete[] m_nodesPtr;
            m_nodesPtr = nullptr;
        }
        size_t nNodes = m_nodes.size();
        if (nNodes > 0)
        {
            m_numNodes = nNodes;
            m_nodesPtr = new KDNode[nNodes];
            for (int n = 0; n < nNodes; n++)
            {
                m_nodesPtr[n] = m_nodes[n];
            }
#ifdef USE_TREELETS
            buildTreeLets();
#endif
        }
    }

    void buildTreeLets()
    {
        if (m_treeLets)
        {
            delete[] m_treeLets;
            m_treeLets = nullptr;
            m_numTreeLets = 0;
        }

        std::vector<KDTreeLet> treeLets;
        treeLets.push_back(KDTreeLet());
        insertNode(m_nodesPtr[0], 0, 0, 0, treeLets);

        m_numTreeLets = treeLets.size();
        m_treeLets = new KDTreeLet[m_numTreeLets];
        for (int i = 0; i < m_numTreeLets; i++)
        {
            m_treeLets[i] = treeLets[i];
        }
    }

    uint32_t insertNode(const KDNode &node, uint32_t nodeIdx, uint32_t treeLetIdx, uint32_t treeDepth, std::vector<KDTreeLet> &treeLets)
    {
        uint32_t treeLetLevel = treeDepth % 3;
        uint32_t globalNodeId = (treeLetIdx * 8) + nodeIdx;
        treeLets[treeLetIdx].nodes[nodeIdx] = node;
        if (!node.isLeaf())
        {
            if (treeLetLevel == 0)
            {
                uint32_t childIdx = node.getLeftChildIdx();
                uint32_t newChildIdx = insertNode(m_nodesPtr[childIdx], 1, treeLetIdx, treeDepth + 1, treeLets);
                insertNode(m_nodesPtr[childIdx + 1], 2, treeLetIdx, treeDepth + 1, treeLets);
                treeLets[treeLetIdx].nodes[nodeIdx].setLeftChildIdx(newChildIdx);
                OPENPGL_ASSERT(node.getSplitDim() == treeLets[treeLetIdx].nodes[nodeIdx].getSplitDim());
                OPENPGL_ASSERT(node.getSplitPivot() == treeLets[treeLetIdx].nodes[nodeIdx].getSplitPivot());
                OPENPGL_ASSERT(newChildIdx == treeLets[treeLetIdx].nodes[nodeIdx].getLeftChildIdx());
            }
            else if (treeLetLevel == 1)
            {
                if (nodeIdx == 1)
                {
                    uint32_t childIdx = node.getLeftChildIdx();
                    uint32_t newChildIdx = insertNode(m_nodesPtr[childIdx], 3, treeLetIdx, treeDepth + 1, treeLets);
                    insertNode(m_nodesPtr[childIdx + 1], 4, treeLetIdx, treeDepth + 1, treeLets);
                    treeLets[treeLetIdx].nodes[nodeIdx].setLeftChildIdx(newChildIdx);
                    OPENPGL_ASSERT(node.getSplitDim() == treeLets[treeLetIdx].nodes[nodeIdx].getSplitDim());
                    OPENPGL_ASSERT(node.getSplitPivot() == treeLets[treeLetIdx].nodes[nodeIdx].getSplitPivot());
                    OPENPGL_ASSERT(newChildIdx == treeLets[treeLetIdx].nodes[nodeIdx].getLeftChildIdx());
                }
                else if (nodeIdx == 2)
                {
                    uint32_t childIdx = node.getLeftChildIdx();
                    uint32_t newChildIdx = insertNode(m_nodesPtr[childIdx], 5, treeLetIdx, treeDepth + 1, treeLets);
                    insertNode(m_nodesPtr[childIdx + 1], 6, treeLetIdx, treeDepth + 1, treeLets);
                    treeLets[treeLetIdx].nodes[nodeIdx].setLeftChildIdx(newChildIdx);
                    OPENPGL_ASSERT(node.getSplitDim() == treeLets[treeLetIdx].nodes[nodeIdx].getSplitDim());
                    OPENPGL_ASSERT(node.getSplitPivot() == treeLets[treeLetIdx].nodes[nodeIdx].getSplitPivot());
                    OPENPGL_ASSERT(newChildIdx == treeLets[treeLetIdx].nodes[nodeIdx].getLeftChildIdx());
                }
                else
                {
                    OPENPGL_ASSERT(false);
                }
            }
            else
            {
                OPENPGL_ASSERT(node.getSplitDim() == treeLets[treeLetIdx].nodes[nodeIdx].getSplitDim());
                uint32_t childIdx = node.getLeftChildIdx();
                treeLets.push_back(KDTreeLet());
                treeLets.push_back(KDTreeLet());
                OPENPGL_ASSERT(node.getSplitDim() == treeLets[treeLetIdx].nodes[nodeIdx].getSplitDim());
                uint32_t leftTreeLetIdx = treeLets.size() - 2;
                insertNode(m_nodesPtr[childIdx], 0, leftTreeLetIdx, treeDepth + 1, treeLets);
                OPENPGL_ASSERT(node.getSplitDim() == treeLets[treeLetIdx].nodes[nodeIdx].getSplitDim());
                insertNode(m_nodesPtr[childIdx + 1], 0, leftTreeLetIdx + 1, treeDepth + 1, treeLets);
                OPENPGL_ASSERT(node.getSplitDim() == treeLets[treeLetIdx].nodes[nodeIdx].getSplitDim());
                treeLets[treeLetIdx].nodes[nodeIdx].setLeftChildIdx(leftTreeLetIdx);
                OPENPGL_ASSERT(node.getSplitDim() == treeLets[treeLetIdx].nodes[nodeIdx].getSplitDim());

                OPENPGL_ASSERT(node.getSplitDim() == treeLets[treeLetIdx].nodes[nodeIdx].getSplitDim());
                OPENPGL_ASSERT(node.getSplitPivot() == treeLets[treeLetIdx].nodes[nodeIdx].getSplitPivot());
                OPENPGL_ASSERT(leftTreeLetIdx == treeLets[treeLetIdx].nodes[nodeIdx].getLeftChildIdx());
            }
        }
        return globalNodeId;
    }
#ifndef USE_TREELETS
    uint32_t getDataIdxAtPos(const Vector3 &pos) const
    {
        OPENPGL_ASSERT(m_isInit);
        OPENPGL_ASSERT(embree::inside(m_bounds, pos));

        uint32_t nodeIdx = 0;
        while (!m_nodesPtr[nodeIdx].isLeaf())
        {
            uint8_t splitDim = m_nodesPtr[nodeIdx].getSplitDim();
            float pivot = m_nodesPtr[nodeIdx].getSplitPivot();

            nodeIdx = m_nodesPtr[nodeIdx].getLeftChildIdx();
            nodeIdx += pos[splitDim] >= pivot ? 1 : 0;
        }
        return m_nodesPtr[nodeIdx].getDataIdx();
    }

#else
    uint32_t getDataIdxAtPos(const Vector3 &pos) const
    {
        OPENPGL_ASSERT(m_isInit);
        OPENPGL_ASSERT(embree::inside(m_bounds, pos));

        uint32_t treeIdx = 0;
        uint32_t nodeIdx = 0;
        uint32_t depth = 0;
        KDTreeLet treeLet = m_treeLets[treeIdx];

        while (!treeLet.nodes[nodeIdx].isLeaf())
        {
            uint8_t splitDim = treeLet.nodes[nodeIdx].getSplitDim();
            uint32_t childIdx = treeLet.nodes[nodeIdx].getLeftChildIdx();
            float pivot = treeLet.nodes[nodeIdx].getSplitPivot();

            if (depth % 3 == 2)
            {
                nodeIdx = 0;
                treeIdx = childIdx;
                treeIdx += pos[splitDim] >= pivot ? 1 : 0;
                treeLet = m_treeLets[treeIdx];
            }
            else
            {
                nodeIdx = childIdx - (treeIdx * 8);
                nodeIdx += pos[splitDim] >= pivot ? 1 : 0;
            }
            depth++;
        }
        return treeLet.nodes[nodeIdx].getDataIdx();
    }
#endif
    uint32_t getMaxNodeDepth(const KDNode &node) const
    {
        if (node.isLeaf())
        {
            return 1;
        }
        else
        {
            const uint32_t leftNodeId = node.getLeftChildIdx();
            const uint32_t leftMaxNodeDepth = getMaxNodeDepth(m_nodes[leftNodeId]);
            const uint32_t rightMaxNodeDepth = getMaxNodeDepth(m_nodes[leftNodeId + 1]);
            return 1 + std::max(leftMaxNodeDepth, rightMaxNodeDepth);
        }
    }

    uint32_t getMaxTreeDepth() const
    {
        if (m_nodes.size() > 0)
        {
            return getMaxNodeDepth(m_nodes[0]);
        }
        return 0;
    }

    std::string toString() const
    {
        std::stringstream ss;
        ss.precision(5);
        ss << "KDTree::" << std::endl;
        ss << "  isInit: " << m_isInit << std::endl;
        ss << "  bounds: " << m_bounds << std::endl;
        ss << "  maxDepth: " << getMaxTreeDepth() << std::endl;
        ss << "  numNodes: " << m_nodes.size() << std::endl;
        return ss.str();
    }

    void exportKDTreeStructureToObj(std::string objFileName) const
    {
        std::ofstream objFile;
        objFile.open(objFileName.c_str());

        BBox rootBBox = m_bounds;
        const KDNode &root = getRoot();
        uint32_t vertexOffset = 0;
        exportKDNodeToObj(objFile, root, rootBBox, vertexOffset);
        objFile.close();
    }

    void exportKDNodeToObj(std::ofstream &objFile, const KDNode &node, BBox bbox, uint32_t &vertexIDOffset) const
    {
        if (!node.isLeaf())
        {
            const uint32_t leftNodeId = node.getLeftChildIdx();
            const uint8_t splitDim = node.getSplitDim();
            const float splitPosition = node.getSplitPivot();
            BBox bboxLeft = bbox;
            bboxLeft.upper[splitDim] = splitPosition;
            BBox bboxRight = bbox;
            bboxRight.lower[splitDim] = splitPosition;

            const KDNode &nodeLeft = m_nodes[leftNodeId];
            const KDNode &nodeRight = m_nodes[leftNodeId + 1];

            exportKDNodeToObj(objFile, nodeLeft, bboxLeft, vertexIDOffset);
            exportKDNodeToObj(objFile, nodeRight, bboxRight, vertexIDOffset);
        }
        else
        {
            const uint32_t leafNodeId = node.getDataIdx();
            objFile << "# KDLeafNode" << leafNodeId << std::endl;

            objFile << "v " << bbox.lower[0] << "\t" << bbox.lower[1] << "\t" << bbox.lower[2] << std::endl;
            objFile << "v " << bbox.lower[0] << "\t" << bbox.upper[1] << "\t" << bbox.lower[2] << std::endl;
            objFile << "v " << bbox.upper[0] << "\t" << bbox.upper[1] << "\t" << bbox.lower[2] << std::endl;
            objFile << "v " << bbox.upper[0] << "\t" << bbox.lower[1] << "\t" << bbox.lower[2] << std::endl;

            objFile << "v " << bbox.lower[0] << "\t" << bbox.lower[1] << "\t" << bbox.upper[2] << std::endl;
            objFile << "v " << bbox.lower[0] << "\t" << bbox.upper[1] << "\t" << bbox.upper[2] << std::endl;
            objFile << "v " << bbox.upper[0] << "\t" << bbox.upper[1] << "\t" << bbox.upper[2] << std::endl;
            objFile << "v " << bbox.upper[0] << "\t" << bbox.lower[1] << "\t" << bbox.upper[2] << std::endl;

            objFile << "f " << vertexIDOffset + 1 + 0 << "\t" << vertexIDOffset + 1 + 1 << "\t" << vertexIDOffset + 1 + 2 << "\t" << vertexIDOffset + 1 + 3 << std::endl;
            objFile << "f " << vertexIDOffset + 1 + 7 << "\t" << vertexIDOffset + 1 + 6 << "\t" << vertexIDOffset + 1 + 5 << "\t" << vertexIDOffset + 1 + 4 << std::endl;

            objFile << "f " << vertexIDOffset + 1 + 0 << "\t" << vertexIDOffset + 1 + 3 << "\t" << vertexIDOffset + 1 + 7 << "\t" << vertexIDOffset + 1 + 4 << std::endl;
            objFile << "f " << vertexIDOffset + 1 + 5 << "\t" << vertexIDOffset + 1 + 6 << "\t" << vertexIDOffset + 1 + 2 << "\t" << vertexIDOffset + 1 + 1 << std::endl;

            objFile << "f " << vertexIDOffset + 1 + 5 << "\t" << vertexIDOffset + 1 + 1 << "\t" << vertexIDOffset + 1 + 0 << "\t" << vertexIDOffset + 1 + 4 << std::endl;
            objFile << "f " << vertexIDOffset + 1 + 7 << "\t" << vertexIDOffset + 1 + 3 << "\t" << vertexIDOffset + 1 + 2 << "\t" << vertexIDOffset + 1 + 6 << std::endl;

            vertexIDOffset += 8;
            // std::cout << "BBox: lower = [" << bbox.lower[0] << ",\t"<< bbox.lower[1] << ",\t"<< bbox.lower[2] << "] \t upper = ["<< bbox.upper[0] << ",\t"<< bbox.upper[1] <<
            // ",\t"<< bbox.upper[2] << std::endl;
        }
    }

    template <typename TRegion, typename TRange>
    void exportSampleBoundsToObj(std::string objFileName, const tbb::concurrent_vector<std::pair<TRegion, TRange> > &dataStorage) const
    {
        std::ofstream objFile;
        objFile.open(objFileName.c_str());

        BBox rootBBox = m_bounds;
        const KDNode &root = getRoot();
        uint32_t vertexOffset = 0;
        exportSampleBoundToObj<TRegion, TRange>(objFile, root, rootBBox, vertexOffset, dataStorage);
        objFile.close();
    }

    template <typename TRegion, typename TRange>

    void exportSampleBoundToObj(std::ofstream &objFile, const KDNode &node, BBox bbox, uint32_t &vertexIDOffset,
                                const tbb::concurrent_vector<std::pair<TRegion, TRange> > &dataStorage) const
    {
        if (!node.isLeaf())
        {
            const uint32_t leftNodeId = node.getLeftChildIdx();
            const uint8_t splitDim = node.getSplitDim();
            const float splitPosition = node.getSplitPivot();
            BBox bboxLeft = bbox;
            bboxLeft.upper[splitDim] = splitPosition;
            BBox bboxRight = bbox;
            bboxRight.lower[splitDim] = splitPosition;

            const KDNode &nodeLeft = m_nodes[leftNodeId];
            const KDNode &nodeRight = m_nodes[leftNodeId + 1];

            exportSampleBoundToObj<TRegion, TRange>(objFile, nodeLeft, bboxLeft, vertexIDOffset, dataStorage);
            exportSampleBoundToObj<TRegion, TRange>(objFile, nodeRight, bboxRight, vertexIDOffset, dataStorage);
        }
        else
        {
            const uint32_t leafNodeId = node.getDataIdx();

            const std::pair<TRegion, TRange> &regionAndRange = dataStorage[leafNodeId];
            BBox sampleBound = regionAndRange.first.sampleStatistics.sampleBounds;
            objFile << "# SampleBound" << leafNodeId << std::endl;

            objFile << "v " << sampleBound.lower[0] << "\t" << sampleBound.lower[1] << "\t" << sampleBound.lower[2] << std::endl;
            objFile << "v " << sampleBound.lower[0] << "\t" << sampleBound.upper[1] << "\t" << sampleBound.lower[2] << std::endl;
            objFile << "v " << sampleBound.upper[0] << "\t" << sampleBound.upper[1] << "\t" << sampleBound.lower[2] << std::endl;
            objFile << "v " << sampleBound.upper[0] << "\t" << sampleBound.lower[1] << "\t" << sampleBound.lower[2] << std::endl;

            objFile << "v " << sampleBound.lower[0] << "\t" << sampleBound.lower[1] << "\t" << sampleBound.upper[2] << std::endl;
            objFile << "v " << sampleBound.lower[0] << "\t" << sampleBound.upper[1] << "\t" << sampleBound.upper[2] << std::endl;
            objFile << "v " << sampleBound.upper[0] << "\t" << sampleBound.upper[1] << "\t" << sampleBound.upper[2] << std::endl;
            objFile << "v " << sampleBound.upper[0] << "\t" << sampleBound.lower[1] << "\t" << sampleBound.upper[2] << std::endl;

            objFile << "f " << vertexIDOffset + 1 + 0 << "\t" << vertexIDOffset + 1 + 1 << "\t" << vertexIDOffset + 1 + 2 << "\t" << vertexIDOffset + 1 + 3 << std::endl;
            objFile << "f " << vertexIDOffset + 1 + 7 << "\t" << vertexIDOffset + 1 + 6 << "\t" << vertexIDOffset + 1 + 5 << "\t" << vertexIDOffset + 1 + 4 << std::endl;

            objFile << "f " << vertexIDOffset + 1 + 0 << "\t" << vertexIDOffset + 1 + 3 << "\t" << vertexIDOffset + 1 + 7 << "\t" << vertexIDOffset + 1 + 4 << std::endl;
            objFile << "f " << vertexIDOffset + 1 + 5 << "\t" << vertexIDOffset + 1 + 6 << "\t" << vertexIDOffset + 1 + 2 << "\t" << vertexIDOffset + 1 + 1 << std::endl;

            objFile << "f " << vertexIDOffset + 1 + 5 << "\t" << vertexIDOffset + 1 + 1 << "\t" << vertexIDOffset + 1 + 0 << "\t" << vertexIDOffset + 1 + 4 << std::endl;
            objFile << "f " << vertexIDOffset + 1 + 7 << "\t" << vertexIDOffset + 1 + 3 << "\t" << vertexIDOffset + 1 + 2 << "\t" << vertexIDOffset + 1 + 6 << std::endl;

            vertexIDOffset += 8;
        }
    }

    void serialize(std::ostream &stream) const
    {
        stream.write(reinterpret_cast<const char *>(&m_isInit), sizeof(bool));
        stream.write(reinterpret_cast<const char *>(&m_bounds), sizeof(BBox));
        size_t num_nodes = m_nodes.size();
        stream.write(reinterpret_cast<const char *>(&num_nodes), sizeof(size_t));
        for (size_t n = 0; n < num_nodes; n++)
        {
            m_nodes[n].serialize(stream);
        }
    }

    void deserialize(std::istream &stream)
    {
        stream.read(reinterpret_cast<char *>(&m_isInit), sizeof(bool));
        stream.read(reinterpret_cast<char *>(&m_bounds), sizeof(BBox));
        size_t num_nodes = 0;
        stream.read(reinterpret_cast<char *>(&num_nodes), sizeof(size_t));
        m_nodes.reserve(num_nodes);
        m_numNodes = num_nodes;
        m_nodesPtr = new KDNode[num_nodes];
        for (size_t n = 0; n < num_nodes; n++)
        {
            KDNode node;
            node.deserialize(stream);
            m_nodes.push_back(node);
            m_nodesPtr[n] = node;
        }
#ifdef USE_TREELETS
        if (num_nodes > 0)
            buildTreeLets();
#endif
    }

    void deserializeIR(BufferedReader& r) {
        r.read(&m_bounds);
        uint32_t numNodes;
        r.read(&numNodes);
        m_numNodes = numNodes;

        m_nodes.clear();
        m_nodes.reserve(numNodes);

        for (int i = 0; i < numNodes; i++) {
            KDNode node;
            r.read(&node.splitPosition);
            r.read(&node.splitDimAndNodeIdx);
            m_nodes.push_back(node);
        }

        if (numNodes > 0) m_isInit = true;

        finalize();
    }

    bool operator==(const KDTree &b) const
    {
        bool equal = true;
        if (m_isInit != b.m_isInit || m_bounds.lower.x != b.m_bounds.lower.x || m_bounds.lower.y != b.m_bounds.lower.y || m_bounds.lower.z != b.m_bounds.lower.z ||
            m_bounds.upper.x != b.m_bounds.upper.x || m_bounds.upper.y != b.m_bounds.upper.y || m_bounds.upper.z != b.m_bounds.upper.z || m_numTreeLets != b.m_numTreeLets)
        {
            equal = false;
        }

        if (m_nodes.size() == b.m_nodes.size())
        {
            for (int n = 0; n < m_nodes.size(); n++)
            {
                if (!m_nodes[n].operator==(b.m_nodes[n]))
                {
                    equal = false;
                }
            }
        }
        else
        {
            equal = false;
        }
        return equal;
    }

    int calculateMaxDepth(const KDNode &node) const
    {
        if (node.isLeaf())
        {
            return 1;
        }
        else
        {
            int childIdxLeft = node.getLeftChildIdx();

            int depthLeft = calculateMaxDepth(m_nodes[childIdxLeft]);
            int depthRight = calculateMaxDepth(m_nodes[childIdxLeft + 1]);
            return 1 + std::max(depthLeft, depthRight);
        }
    }

    KDTreeStatistics getStatistics() const
    {
        KDTreeStatistics treeStats;
        treeStats.maxDepth = calculateMaxDepth(m_nodes[0]);
        treeStats.numberOfNodes = m_nodes.size();
        treeStats.sizePerNode = sizeof(KDNode);
        treeStats.numberOfReservedNodes = m_nodes.capacity();
        treeStats.sizeAllNodesUsed = m_nodes.size() * sizeof(KDNode);
        treeStats.sizeAllNodesReserved = m_nodes.capacity() * sizeof(KDNode);
        return treeStats;
    }

    void sDumpRec(SDumpTree *sDump, uint32_t idx) const {
        const KDNode& node = m_nodesPtr[idx];
        if (node.isLeaf()) {
            sDump->left = nullptr;
            sDump->right = nullptr;
        } else {
            sDump->axis = node.getSplitDim();
            sDump->split = node.getSplitPivot();
            sDump->left = new SDumpTree;
            sDump->right = new SDumpTree;
            sDumpRec(sDump->left, node.getLeftChildIdx() + 0);
            sDumpRec(sDump->right, node.getLeftChildIdx() + 1);
        }
    }

    void sDump(SDumpTree* sDump) const {
        if (!m_isInit) {
            sDump->left = nullptr;
            sDump->right = nullptr;
            return;
        }

        sDumpRec(sDump, 0);
    }

   public:
    bool m_isInit{false};

    // bounds of the spatial region covered by the KDTree
    BBox m_bounds;

    // node storage used during build
    tbb::concurrent_vector<KDNode> m_nodes;

    // node storage used during querying
    KDNode *m_nodesPtr{nullptr};
    size_t m_numNodes{0};

    KDTreeLet *m_treeLets{nullptr};
    int m_numTreeLets{0};
};

}  // namespace openpgl