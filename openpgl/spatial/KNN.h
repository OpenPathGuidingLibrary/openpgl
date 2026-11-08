// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "../openpgl_common.h"

// Nearest neighbor queries
/* include nanoflann API */
#include <embreeSrc/common/math/transcendental.h>
#include <tbb/parallel_for.h>

#include <functional>
#include <nanoflann/include/nanoflann.hpp>
#include <queue>

#define NUM_KNN 4
#define NUM_KNN_NEIGHBOURS 8
#define DEBUG_SAMPLE_APPROXIMATE_CLOSEST_REGION_IDX 0

#define KNN_IS_SIMD

namespace openpgl
{

inline uint32_t draw(float *sample, uint32_t size)
{
    size = std::min<uint32_t>(NUM_KNN, size);
    uint32_t selected = *sample * size;
    *sample = (*sample - float(selected) / size) * size;
    OPENPGL_ASSERT(*sample >= 0.f && *sample < 1.0f);
    return std::min(selected, size - 1);
}

template <typename RegionNeighbours>
uint32_t sampleApproximateClosestRegionIdxRef(const RegionNeighbours &nh, const openpgl::Point3 &p, float sample)
{
    uint32_t selected = draw(&sample, nh.size);

    using E = std::pair<uint32_t, float>;
    E candidates[NUM_KNN_NEIGHBOURS];

    for (int i = 0; i < nh.size; i++)
    {
        auto tup = nh.get(i);
        const uint32_t primID = std::get<0>(tup);
        const float xd = std::get<1>(tup) - p.x, yd = std::get<2>(tup) - p.y, zd = std::get<3>(tup) - p.z;
        float d = xd * xd + yd * yd + zd * zd;

        // we use the three least significant bits of the mantissa to store the array ids,
        // so we have to do the same to get the same results
        uint32_t mask = (1 << 3) - 1;
        uint32_t *df = (uint32_t *)&d;
        *df = (*df & ~mask) | (i & mask);

        candidates[i] = {primID, d};
    }

    std::sort(std::begin(candidates), std::begin(candidates) + nh.size, [&](E &a, E &b) {
        return a.second < b.second;
    });

    return candidates[selected].first;
}
template <int Vecsize>
struct RegionNeighbours
{};

template <>
struct RegionNeighbours<4>
{
    embree::vuint<4> ids[2];
    embree::Vec3<embree::vfloat<4>> points[2];
    uint32_t size;

    inline void set(uint32_t i, uint32_t id, float x, float y, float z)
    {
        ids[i / 4][i % 4] = id;
        points[i / 4].x[i % 4] = x;
        points[i / 4].y[i % 4] = y;
        points[i / 4].z[i % 4] = z;
    }

    inline std::tuple<uint32_t, float, float, float> get(uint32_t i) const
    {
        return {ids[i / 4][i % 4], points[i / 4].x[i % 4], points[i / 4].y[i % 4], points[i / 4].z[i % 4]};
    }

    inline embree::vfloat<4> prepare(const uint32_t i, const embree::Vec3<embree::vfloat<4>> &p) const
    {
        // While we only need the first two mantissa bits here,
        // we want to keep the output consistent with the 8- and 16-wide implementation
        const embree::vfloat<4> ids = asFloat(embree::vint<4>(0, 1, 2, 3) + 4 * i);
        const embree::vfloat<4> mask = asFloat(embree::vint<4>(~7));

        const embree::Vec3<embree::vfloat<4>> d = points[i] - p;
        embree::vfloat<4> distances = embree::dot(d, d);
        distances = distances & mask | ids;
        distances = select(this->ids[i] != ~0, distances, embree::vfloat<4>(std::numeric_limits<float>::infinity()));

        return sort_ascending(distances);
    }

    inline uint32_t sampleApproximateClosestRegionIdx(const openpgl::Point3 &p, float *sample) const
    {
        uint32_t selected = draw(sample, size);
        const embree::Vec3<embree::vfloat<4>> _p(p[0], p[1], p[2]);

        const embree::vfloat<4> d0 = prepare(0, _p);
        const embree::vfloat<4> d1 = prepare(1, _p);

        uint32_t i0 = 0, i1 = 0;
        for (uint32_t i = 0; i < selected; i++)
        {
            if (d0[i0] < d1[i1])
                i0++;
            else
                i1++;
        }

        if (d0[i0] < d1[i1])
            return ids[0][asInt(d0)[i0] & 3];
        else
            return ids[1][asInt(d1)[i1] & 3];
    }

    inline uint32_t sampleApproximateClosestRegionIdxIS(const openpgl::Point3 &p, float *sample) const
    {
        const embree::Vec3<embree::vfloat<4>> _p(p[0], p[1], p[2]);
        embree::Vec3<embree::vfloat<4>> d[2];
        d[0] = this->points[0] - _p;
        d[1] = this->points[1] - _p;

        embree::vfloat<4> dist[2];
        dist[0] = embree::dot(d[0], d[0]);
        dist[1] = embree::dot(d[1], d[1]);

        const float maxDist = std::max(embree::reduce_max(dist[0]), embree::reduce_max(dist[1]));
        const float sigma = std::sqrt(maxDist) / 4.0f;
        dist[0] = embree::fastapprox::exp(-0.5f * dist[0] / (sigma * sigma));
        dist[1] = embree::fastapprox::exp(-0.5f * dist[1] / (sigma * sigma));
#ifdef KNN_IS_SIMD
        embree::vfloat<4> cdfs[2];
        cdfs[0] = vinclusive_prefix_sum(dist[0]);
        cdfs[1] = vinclusive_prefix_sum(dist[1]);

        const float sumDist0 = cdfs[0][3];
        const float sumDist1 = cdfs[1][3];
        const float sumDist = sumDist0 + sumDist1;
        float searched = *sample * sumDist;
        size_t idx = 0;
        if (searched > sumDist0)
        {
            searched -= sumDist0;
            idx = 1;
        }
        const size_t sidx = embree::select_min(cdfs[idx] >= searched, cdfs[idx]);
        const float sumCDF = sidx > 0 ? cdfs[idx][sidx - 1] : 0.f;
        *sample = std::min(1 - FLT_EPSILON, (searched - sumCDF) / dist[idx][sidx]);
        return this->ids[idx][sidx];
#else
        const float sumDist0 = embree::reduce_add(dist[0]);
        const float sumDist1 = embree::reduce_add(dist[1]);

        float sumDist = sumDist0 + sumDist1;
        size_t idx = 0;
        float sumCDF = 0.0f;
        float searched = *sample * sumDist;
        if (searched > sumDist0)
        {
            sumCDF = sumDist0;
            idx = 1;
        }
        float cdf = 0.f;
        size_t sidx = 0;
        while (true)
        {
            cdf = dist[idx][sidx];
            if (sumCDF + cdf >= searched || sidx + 1 >= 4)
            {
                break;
            }
            else
            {
                sumCDF += cdf;
                sidx++;
            }
        }

        *sample = std::min(1 - FLT_EPSILON, (searched - sumCDF) / cdf);
        return this->ids[idx][sidx];
#endif
    }
};

#if defined(__AVX__)
template <>
struct RegionNeighbours<8>
{
    embree::vuint<8> ids;
    embree::Vec3<embree::vfloat<8>> points;
    uint32_t size;

    inline void set(uint32_t i, uint32_t id, float x, float y, float z)
    {
        ids[i] = id;
        points.x[i] = x;
        points.y[i] = y;
        points.z[i] = z;
    }

    inline std::tuple<uint32_t, float, float, float> get(uint32_t i) const
    {
        return {ids[i], points.x[i], points.y[i], points.z[i]};
    }

    inline uint32_t sampleApproximateClosestRegionIdx(const openpgl::Point3 &p, float *sample) const
    {
        uint32_t selected = draw(sample, size);

        const embree::vfloat<8> ids = asFloat(embree::vint<8>(0, 1, 2, 3, 4, 5, 6, 7));
        const embree::vfloat<8> mask = asFloat(embree::vint<8>(~7));

        const embree::Vec3<embree::vfloat<8>> _p(p[0], p[1], p[2]);
        const embree::Vec3<embree::vfloat<8>> d = points - _p;
        embree::vfloat<8> distances = embree::dot(d, d);
        distances = distances & mask | ids;
        distances = select(this->ids != ~0, distances, embree::vfloat<8>(std::numeric_limits<float>::infinity()));
        distances = sort_ascending(distances);

        return this->ids[asInt(distances)[selected] & 7];
    }

    inline uint32_t sampleApproximateClosestRegionIdxIS(const openpgl::Point3 &p, float *sample) const
    {
        const embree::Vec3<embree::vfloat<8>> _p(p[0], p[1], p[2]);
        embree::Vec3<embree::vfloat<8>> d;
        d = this->points - _p;
        embree::vfloat<8> dist = embree::dot(d, d);
        const float maxDist = embree::reduce_max(dist);
        const float sigma = std::sqrt(maxDist) / 4.0f;
        dist = embree::fastapprox::exp(-0.5f * dist / (sigma * sigma));

#ifdef KNN_IS_SIMD
        const embree::vfloat<8> cdfs = vinclusive_prefix_sum(dist);
        const float sumDist = cdfs[7];
        const float searched = *sample * sumDist;
        const size_t idx = embree::select_min(cdfs >= searched, cdfs);
        const float sumCDF = idx > 0 ? cdfs[idx - 1] : 0.f;
        *sample = std::min(1 - FLT_EPSILON, (searched - sumCDF) / dist[idx]);
        return this->ids[idx];
#else
        const float sumDist = embree::reduce_add(dist);

        size_t idx = 0;
        float sumCDF = 0.0f;
        float searched = *sample * sumDist;
        float cdf = 0.f;
        while (true)
        {
            cdf = dist[idx];
            if (sumCDF + cdf >= searched || idx + 1 >= 4)
            {
                break;
            }
            else
            {
                sumCDF += cdf;
                idx++;
            }
        }

        *sample = std::min(1 - FLT_EPSILON, (searched - sumCDF) / cdf);
        return this->ids[idx];
#endif
    }
};

template <>
struct RegionNeighbours<16> : public RegionNeighbours<8>
{};
#endif

template <int Vecsize>
struct KNearestRegionsSearchTree
{
    struct Point
    {
        OPENPGL_ALIGNED_STRUCT_(16)
        embree::Vec3fa p;  //!< position
    };

    struct Neighbour
    {
        unsigned int primID;
        float d;

        bool operator<(Neighbour const &n1) const
        {
            return d < n1.d;
        }
    };

    using This = KNearestRegionsSearchTree<Vecsize>;
    using Index = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, This>, This, 3>;
    using coord_t = float;

    using RN = RegionNeighbours<Vecsize>;

    KNearestRegionsSearchTree() = default;

    KNearestRegionsSearchTree(const KNearestRegionsSearchTree &) = delete;

    ~KNearestRegionsSearchTree()
    {
        alignedFree(points);
        alignedFree(neighbours);
    }

    template <typename TRegionStorageContainer>
    void buildRegionSearchTree(const TRegionStorageContainer &regionStorage)
    {
        num_points = regionStorage.size();
        if (points)
        {
            alignedFree(points);
        }
        points = (Point *)alignedMalloc(num_points * sizeof(Point), 32);

        for (size_t i = 0; i < num_points; i++)
        {
            const auto &region = regionStorage[i].first;
            const openpgl::Point3 distributionPivot = region.regionPivot;
            points[i].p = embree::Vec3f(distributionPivot[0], distributionPivot[1], distributionPivot[2]);
        }

        index = std::unique_ptr<Index>(new Index(3, *this, 10));

        _isBuild = true;
        _isBuildNeighbours = false;
    }

    void buildRegionNeighbours()
    {
        OPENPGL_ASSERT(_isBuild);

        if (neighbours)
        {
            alignedFree(neighbours);
        }
        neighbours = (RN *)alignedMalloc(num_points * sizeof(RN), 32);

        tbb::parallel_for(tbb::blocked_range<int>(0, num_points), [&](tbb::blocked_range<int> r) {
            for (int n = r.begin(); n < r.end(); ++n)
            {
                Point &point = points[n];

                const float query_pt[3] = {point.p.x, point.p.y, point.p.z};

                size_t num_results = NUM_KNN_NEIGHBOURS;
                unsigned int ret_index[NUM_KNN_NEIGHBOURS];
                float ret_dist_sqr[NUM_KNN_NEIGHBOURS];

                num_results = index->knnSearch(&query_pt[0], num_results, &ret_index[0], &ret_dist_sqr[0]);

                bool selfIsIn = false;

                auto &nh = neighbours[n];
                nh.size = num_results;
                uint32_t i = 0;
                for (; i < num_results; i++)
                {
                    size_t idx = ret_index[i];
                    selfIsIn = selfIsIn || idx == (uint32_t)n;
                    nh.set(i, idx, points[idx].p.x, points[idx].p.y, points[idx].p.z);
                }
                for (; i < NUM_KNN_NEIGHBOURS; i++)
                {
                    nh.set(i, ~0, 0, 0, 0);
                }

                OPENPGL_ASSERT(selfIsIn);
#ifdef OPENPGL_SHOW_PRINT_OUTS
                if (!selfIsIn)
                {
                    std::cout << "No closest region found" << std::endl;
                }
#endif
            }
        });

        _isBuildNeighbours = true;
    }

    uint32_t sampleClosestRegionIdx(const openpgl::Point3 &p, float *sample) const
    {
        OPENPGL_ASSERT(_isBuild);

        const float query_pt[3] = {p.x, p.y, p.z};

        size_t num_results = NUM_KNN;
        unsigned int ret_index[NUM_KNN];
        float ret_dist_sqr[NUM_KNN];

        num_results = index->knnSearch(&query_pt[0], num_results, &ret_index[0], &ret_dist_sqr[0]);

        if (num_results == 0)
        {
#ifdef OPENPGL_SHOW_PRINT_OUTS
            std::cout << "No closest region found" << std::endl;
#endif
            return -1;
        }

        return ret_index[draw(sample, num_results)];
    }

    uint32_t sampleApproximateClosestRegionIdx(unsigned int regionIdx, const openpgl::Point3 &p, float *sample) const
    {
        OPENPGL_ASSERT(_isBuildNeighbours);

#if DEBUG_SAMPLE_APPROXIMATE_CLOSEST_REGION_IDX
        uint32_t ref = sampleApproximateClosestRegionIdxRef(neighbours[regionIdx], p, *sample);
#endif
        uint32_t out = neighbours[regionIdx].sampleApproximateClosestRegionIdx(p, sample);
#if DEBUG_SAMPLE_APPROXIMATE_CLOSEST_REGION_IDX
        OPENPGL_ASSERT(ref == out);
#endif

        return out;
    }

    uint32_t sampleApproximateClosestRegionIdxIS(unsigned int regionIdx, const openpgl::Point3 &p, float *sample) const
    {
        OPENPGL_ASSERT(_isBuildNeighbours);
        uint32_t out = neighbours[regionIdx].sampleApproximateClosestRegionIdxIS(p, sample);
        return out;
    }

    bool isBuild() const
    {
        return _isBuild;
    }

    bool isBuildNeighbours() const
    {
        return _isBuildNeighbours;
    }

    uint32_t numRegions() const
    {
        return num_points;
    }

    void serialize(std::ostream &stream) const
    {
        stream.write(reinterpret_cast<const char *>(&_isBuild), sizeof(bool));
        if (_isBuild)
        {
            stream.write(reinterpret_cast<const char *>(&num_points), sizeof(uint32_t));
            for (uint32_t n = 0; n < num_points; n++)
            {
                stream.write(reinterpret_cast<const char *>(&points[n]), sizeof(Point));
            }
        }
    }

    void reset()
    {
        if (points)
        {
            alignedFree(points);
            points = nullptr;
            num_points = 0;
        }

        if (neighbours)
        {
            alignedFree(neighbours);
            neighbours = nullptr;
        }

        _isBuildNeighbours = false;
        _isBuild = false;
    }

    void deserialize(std::istream &stream)
    {
        stream.read(reinterpret_cast<char *>(&_isBuild), sizeof(bool));
        if (_isBuild)
        {
            stream.read(reinterpret_cast<char *>(&num_points), sizeof(uint32_t));
            points = (Point *)alignedMalloc(num_points * sizeof(Point), 32);
            for (uint32_t n = 0; n < num_points; n++)
            {
                Point p;
                stream.read(reinterpret_cast<char *>(&p), sizeof(Point));
                points[n] = p;
            }

            index = std::unique_ptr<Index>(new Index(3, *this, 10));
        }
    }

    std::string toString() const
    {
        std::stringstream ss;
        ss << "KNearestRegionsSearchTree:" << std::endl;
        ss << "  num_points: " << num_points << std::endl;
        ss << "  _isBuild: " << _isBuild << std::endl;
        return ss.str();
    }

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const
    {
        return num_points;
    }

    inline coord_t kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        return points[idx].p[dim];
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX & /*bb*/) const
    {
        return false;
    }

   private:
    Point *points = nullptr;
    uint32_t num_points{0};

    std::unique_ptr<Index> index;

    RN *neighbours = nullptr;

    bool _isBuild{false};
    bool _isBuildNeighbours{false};
};

}  // namespace openpgl