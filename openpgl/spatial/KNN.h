// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "../openpgl_common.h"

// Nearest neighbor queries
/* include embree API */
#include <embree3/rtcore.h>
#include <functional>
#include <queue>

//using namespace embree;

#define NUM_KNN 4

inline  void* myalignedMalloc(size_t size, size_t align)
  {
    if (size == 0)
      return nullptr;

    assert((align & (align-1)) == 0);
    void* ptr = _mm_malloc(size,align);

    if (size != 0 && ptr == nullptr)
      throw std::bad_alloc();

    return ptr;
  }

inline  void myalignedFree(void* ptr)
  {
    if (ptr)
      _mm_free(ptr);
  }



namespace openpgl
{

struct KNearestRegionsSearchTree
{

    struct Point
    {
        ALIGNED_STRUCT_(16)
        embree::Vec3fa p;                      //!< position
    };

    struct Neighbour
    {
        unsigned int primID;
        float d;

        bool operator<(Neighbour const& n1) const { return d < n1.d; }
    };

    struct KNNResult
    {
        KNNResult(const Point *_points)
        {
            points = _points;
        }

        ~KNNResult()
        {

        }

        unsigned int k;
        const Point* points = nullptr;
        std::priority_queue<Neighbour, std::vector<Neighbour>> knn;
    };

    KNearestRegionsSearchTree()
    {
        embree_device = rtcNewDevice("");
    }


    template<typename TRegionStorageContainer, typename TRegion>
    void buildRegionSearchTree(const TRegionStorageContainer &regionStorage)
    {
        num_points = regionStorage.size();
        if (embree_points)
        {
            myalignedFree(embree_points);
        }
        embree_points = (Point*) myalignedMalloc(num_points*sizeof(Point), 16);

        for (size_t i = 0; i < num_points; i++)
        {
            const TRegion region = regionStorage[i].first;
            openpgl::SampleStatistics combinedStats = region.sampleStatistics;
            openpgl::Point3 distributionPivot = combinedStats.mean;
            embree_points[i].p = embree::Vec3f(distributionPivot[0], distributionPivot[1], distributionPivot[2]);
        }

        if (embree_scene){
            rtcReleaseScene (embree_scene);
        }
        embree_scene = rtcNewScene(embree_device);
        RTCGeometry geom = rtcNewGeometry(embree_device, RTC_GEOMETRY_TYPE_USER);
        rtcAttachGeometry(embree_scene, geom);
        rtcSetGeometryUserPrimitiveCount(geom, num_points);
        rtcSetGeometryUserData(geom, embree_points);
        rtcSetGeometryBoundsFunction(geom, pointBoundsFunc, nullptr);
        rtcCommitGeometry(geom);
        rtcReleaseGeometry(geom);
        rtcCommitScene(embree_scene);
        _isBuild = true;
    }

    uint32_t sampleClosestRegionIdx(const openpgl::Point3 &p, float* sample) const
    {
        embree::Vec3fa q = embree::Vec3fa(p[0], p[1], p[2]);
        KNNResult result(embree_points);
        result.k = NUM_KNN;
        knnQuery(embree_scene, q, std::numeric_limits<float>::max(), &result);

        if (!result.knn.empty())
        {
            int randomNeighbor = (*sample * NUM_KNN);
            const float sampleProb = 1.f / float(NUM_KNN);
            *sample = (*sample - randomNeighbor * sampleProb) * embree::rcp(sampleProb);
            OPENPGL_ASSERT(*sample >= 0.f && *sample < 1.0f);
            unsigned int primID;
            for (size_t k = 0; k <= randomNeighbor; k++)
            {
                primID = result.knn.top().primID;
                result.knn.pop();
            }
            OPENPGL_ASSERT(primID < num_points);
            return primID;
        }
        else
        {
            std::cout << "No closest region found" << std::endl;
            return -1;
        }
    }

    static bool pointQueryFunc(struct RTCPointQueryFunctionArguments* args)
    {
        assert(args->query);
        RTCPointQuery* query = (RTCPointQuery*)args->query;
        const unsigned int primID = args->primID;
        const embree::Vec3f q(query->x, query->y, query->z);

        assert(args->userPtr);
        KNNResult* result = (KNNResult*)args->userPtr;
        assert(result->points);
        Point* points = (Point*) result->points;
        const Point& point = points[primID];
        const float d = distance(point.p, q);

        if (d < query->radius && (result->knn.size() < result->k || d < result->knn.top().d))
        {
            Neighbour neighbour;
            neighbour.primID = primID;
            neighbour.d = d;

            if (result->knn.size() == result->k)
            result->knn.pop();

            result->knn.push(neighbour);

            if (result->knn.size() == result->k)
            {
            const float R = result->knn.top().d;
            query->radius = R;
            return true;
            }
        }
        return false;
    }

    static void pointBoundsFunc(const struct RTCBoundsFunctionArguments* args)
    {
        const Point* points = (const Point*) args->geometryUserPtr;
        RTCBounds* bounds_o = args->bounds_o;
        const Point& point = points[args->primID];
        bounds_o->lower_x = point.p.x;
        bounds_o->lower_y = point.p.y;
        bounds_o->lower_z = point.p.z;
        bounds_o->upper_x = point.p.x;
        bounds_o->upper_y = point.p.y;
        bounds_o->upper_z = point.p.z;
    }

    void knnQuery(RTCScene scene, embree::Vec3f const& q, float radius, KNNResult* result) const
    {
        RTCPointQuery query;
        query.x = q.x;
        query.y = q.y;
        query.z = q.z;
        query.radius = radius;
        query.time = 0.f;
        RTCPointQueryContext context;
        rtcInitPointQueryContext(&context);
        rtcPointQuery(scene, &query, &context, pointQueryFunc, (void*)result);
    }

    bool isBuild() const
    {
        return _isBuild;
    }

    uint32_t numRegions() const
    {
        return num_points;
    }

    void serialize(std::ostream& stream) const
    {
        stream.write(reinterpret_cast<const char*>(&_isBuild), sizeof(bool));
        if(_isBuild)
        {
            stream.write(reinterpret_cast<const char*>(&num_points), sizeof(uint32_t));
            for (uint32_t n = 0; n < num_points; n++)
            {
                stream.write(reinterpret_cast<const char*>(&embree_points[n]), sizeof(Point));
            }
        }
    }

    void deserialize(std::istream& stream)
    {
        stream.read(reinterpret_cast<char*>(&_isBuild), sizeof(bool));
        if(_isBuild)
        {
            stream.read(reinterpret_cast<char*>(&num_points), sizeof(uint32_t));
            embree_points = (Point*) myalignedMalloc(num_points*sizeof(Point), 16);
            for (uint32_t n = 0; n < num_points; n++)
            {
                Point p;
                stream.read(reinterpret_cast<char*>(&p), sizeof(Point));
                embree_points[n] = p;
            }
            if (embree_scene){
                rtcReleaseScene (embree_scene);
            }
            embree_scene = rtcNewScene(embree_device);
            RTCGeometry geom = rtcNewGeometry(embree_device, RTC_GEOMETRY_TYPE_USER);
            rtcAttachGeometry(embree_scene, geom);
            rtcSetGeometryUserPrimitiveCount(geom, num_points);
            rtcSetGeometryUserData(geom, embree_points);
            rtcSetGeometryBoundsFunction(geom, pointBoundsFunc, nullptr);
            rtcCommitGeometry(geom);
            rtcReleaseGeometry(geom);
            rtcCommitScene(embree_scene);
        }
        else
        {
            embree_scene = nullptr;
            embree_points = nullptr;
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

private:
    // Embree
    RTCDevice embree_device = nullptr;
    RTCScene embree_scene = nullptr;
    Point* embree_points= nullptr;
    uint32_t num_points {0};

    bool _isBuild{false};

};

}