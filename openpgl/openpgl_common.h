// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "include/openpgl/common.h"

#define OPENPGL_DEBUG_SAM

#define USE_EMBREE_PARALLEL
#define USE_INTEGER_ARITHMETIC_STATS
#define USE_PARALLEL_PIVOT_SPLIT

#define ONE_OVER_FOUR_PI 0.07957747154594767f
#define M_PI_F 3.14159265358979323846f /* pi */

#include <embreeSrc/common/math/bbox.h>
#include <embreeSrc/common/math/constants.h>
#include <embreeSrc/common/math/emath.h>
#include <embreeSrc/common/math/vec2.h>
#include <embreeSrc/common/math/vec3.h>

#include <algorithm>
#include <cmath>

#if defined(__WIN32__) || (defined(__MACOSX__) && !defined(__INTEL_COMPILER))

inline void sincosf(const float theta, float *sin, float *cos)
{
    embree::sincosf(theta, sin, cos);
}
#elif defined(__MACOSX__) && defined(__INTEL_COMPILER)

inline void sincosf(const float theta, float *sin, float *cos)
{
    *sin = sinf(theta);
    *cos = cosf(theta);
}

#endif

namespace openpgl
{
template <int NumVectors, int VectorSize>
void serializeFloatVectors(std::ostream &stream, const embree::vfloat<VectorSize> *vectors)
{
    for (int i = 0; i < NumVectors; i++)
    {
        for (int j = 0; j < VectorSize; j++)
        {
            stream.write(reinterpret_cast<const char *>(&vectors[i][j]), sizeof(float));
        }
    }
}

template <int NumVectors, int VectorSize>
void deserializeFloatVectors(std::istream &stream, embree::vfloat<VectorSize> *vectors)
{
    for (int i = 0; i < NumVectors; i++)
    {
        for (int j = 0; j < VectorSize; j++)
        {
            stream.read(reinterpret_cast<char *>(&vectors[i][j]), sizeof(float));
        }
    }
}

template <int NumVectors, int VectorSize>
void serializeIntVectors(std::ostream &stream, const embree::vint<VectorSize> *vectors)
{
    for (int i = 0; i < NumVectors; i++)
    {
        for (int j = 0; j < VectorSize; j++)
        {
            stream.write(reinterpret_cast<const char *>(&vectors[i][j]), sizeof(int32_t));
        }
    }
}

template <int NumVectors, int VectorSize>
void deserializeIntVectors(std::istream &stream, embree::vint<VectorSize> *vectors)
{
    for (int i = 0; i < NumVectors; i++)
    {
        for (int j = 0; j < VectorSize; j++)
        {
            stream.read(reinterpret_cast<char *>(&vectors[i][j]), sizeof(int32_t));
        }
    }
}

template <int NumVectors, int VectorSize>
void serializeVec2Vectors(std::ostream &stream, const embree::Vec2<embree::vfloat<VectorSize> > *vectors)
{
    for (int i = 0; i < NumVectors; i++)
    {
        for (int j = 0; j < VectorSize; j++)
        {
            stream.write(reinterpret_cast<const char *>(&vectors[i].x[j]), sizeof(float));
        }
    }
    for (int i = 0; i < NumVectors; i++)
    {
        for (int j = 0; j < VectorSize; j++)
        {
            stream.write(reinterpret_cast<const char *>(&vectors[i].y[j]), sizeof(float));
        }
    }
}

template <int NumVectors, int VectorSize>
void deserializeVec2Vectors(std::istream &stream, embree::Vec2<embree::vfloat<VectorSize> > *vectors)
{
    for (int i = 0; i < NumVectors; i++)
    {
        for (int j = 0; j < VectorSize; j++)
        {
            stream.read(reinterpret_cast<char *>(&vectors[i].x[j]), sizeof(float));
        }
    }
    for (int i = 0; i < NumVectors; i++)
    {
        for (int j = 0; j < VectorSize; j++)
        {
            stream.read(reinterpret_cast<char *>(&vectors[i].y[j]), sizeof(float));
        }
    }
}

template <int NumVectors, int VectorSize>
void serializeVec3Vectors(std::ostream &stream, const embree::Vec3<embree::vfloat<VectorSize> > *vectors)
{
    for (int i = 0; i < NumVectors; i++)
    {
        for (int j = 0; j < VectorSize; j++)
        {
            stream.write(reinterpret_cast<const char *>(&vectors[i].x[j]), sizeof(float));
        }
    }
    for (int i = 0; i < NumVectors; i++)
    {
        for (int j = 0; j < VectorSize; j++)
        {
            stream.write(reinterpret_cast<const char *>(&vectors[i].y[j]), sizeof(float));
        }
    }
    for (int i = 0; i < NumVectors; i++)
    {
        for (int j = 0; j < VectorSize; j++)
        {
            stream.write(reinterpret_cast<const char *>(&vectors[i].z[j]), sizeof(float));
        }
    }
}

template <int NumVectors, int VectorSize>
void deserializeVec3Vectors(std::istream &stream, embree::Vec3<embree::vfloat<VectorSize> > *vectors)
{
    for (int i = 0; i < NumVectors; i++)
    {
        for (int j = 0; j < VectorSize; j++)
        {
            stream.read(reinterpret_cast<char *>(&vectors[i].x[j]), sizeof(float));
        }
    }
    for (int i = 0; i < NumVectors; i++)
    {
        for (int j = 0; j < VectorSize; j++)
        {
            stream.read(reinterpret_cast<char *>(&vectors[i].y[j]), sizeof(float));
        }
    }
    for (int i = 0; i < NumVectors; i++)
    {
        for (int j = 0; j < VectorSize; j++)
        {
            stream.read(reinterpret_cast<char *>(&vectors[i].z[j]), sizeof(float));
        }
    }
}
}  // namespace openpgl

namespace openpgl
{
template <int imm>
__forceinline embree::vfloat4 vshift_left(const embree::vfloat4 &v)
{
    return embree::asFloat(_mm_slli_si128(asInt(v), imm));
}

__forceinline embree::vfloat4 vinclusive_prefix_sum(const embree::vfloat4 &v)
{
    embree::vfloat4 x = v;
    x += vshift_left<4>(x);
    x += vshift_left<8>(x);
    return x;
}

//__forceinline vint8   asInt  (const vfloat8& a) { return _mm256_castps_si256(a); }
#if defined(__AVX__)
template <int imm>
__forceinline embree::vfloat8 vshift_left(const embree::vfloat8 &v)
{
    return embree::asFloat(_mm256_slli_si256(asInt(v), imm));
}

__forceinline embree::vfloat8 vinclusive_prefix_sum(const embree::vfloat8 &v)
{
    embree::vfloat8 x = v;
    x += vshift_left<4>(x);
    x += vshift_left<8>(x);
    x += embree::vfloat8(0.0f, 0.0f, 0.0f, 0.0f, x[3], x[3], x[3], x[3]);
    return x;
}
#endif

#if defined(__AVX512F__)
template <int k>
__m512i _mm512_slli_si512(__m512i x)
{
    const __m512i ZERO = _mm512_setzero_si512();
    return _mm512_alignr_epi32(x, ZERO, 16 - k);
}

template <int imm>
__forceinline embree::vfloat16 vshift_left(const embree::vfloat16 &v)
{
    return embree::asFloat(_mm512_slli_si512<imm>(asInt(v)));
}

// From the arxiv paper: Parallel Prefix Sum with SIMD
__forceinline embree::vfloat16 vinclusive_prefix_sum(const embree::vfloat16 &v)
{
    embree::vfloat16 x = v;
    x += vshift_left<1>(x);
    x += vshift_left<2>(x);
    x += vshift_left<4>(x);
    x += vshift_left<8>(x);
    return x;
}
#endif
}  // namespace openpgl

namespace openpgl
{

inline void *alignedMalloc(size_t size, size_t align)
{
    if (size == 0)
        return nullptr;

    assert((align & (align - 1)) == 0);
    void *ptr = _mm_malloc(size, align);

    if (size != 0 && ptr == nullptr)
        throw std::bad_alloc();

    return ptr;
}

inline void alignedFree(void *ptr)
{
    if (ptr)
        _mm_free(ptr);
}
}  // namespace openpgl

#define OPENPGL_ALIGNED_STRUCT_(align)              \
    void *operator new(size_t size)                 \
    {                                               \
        return openpgl::alignedMalloc(size, align); \
    }                                               \
    void operator delete(void *ptr)                 \
    {                                               \
        openpgl::alignedFree(ptr);                  \
    }                                               \
    void *operator new[](size_t size)               \
    {                                               \
        return openpgl::alignedMalloc(size, align); \
    }                                               \
    void operator delete[](void *ptr)               \
    {                                               \
        openpgl::alignedFree(ptr);                  \
    }

namespace openpgl
{
typedef embree::Vec2<float> Vector2;
typedef embree::Vec3<float> Vector3;
typedef embree::Vec2<float> Point2;
typedef embree::Vec3<float> Point3;

typedef embree::Vec3<int64_t> Point3i;
typedef embree::Vec3<int64_t> Vector3i;

typedef embree::BBox<Vector3> BBox;
typedef embree::BBox<Vector3i> BBoxi;

inline float dot(Vector2 &a, Vector2 &b)
{
    return embree::dot(a, b);
}

inline float dot(Vector3 &a, Vector3 &b)
{
    return embree::dot(a, b);
}
}  // namespace openpgl

// #define OPENPGL_DISABLE_ASSERTS

#ifndef OPENPGL_DISABLE_ASSERTS
#include <assert.h>
#define OPENPGL_ASSERT(cond) assert(cond);
// #define OPENPGL_ASSERT_MSG(cond, msg) SAssertEx(cond, msg);
#else
#define OPENPGL_ASSERT(cond)
// #define OPENPGL_ASSERT_MSG(cond, msg)
#endif

namespace openpgl
{
template <int VecSize>
inline float sum(const embree::vfloat<VecSize> &v)
{
    float sum = 0.0f;
    for (int i = 0; i < VecSize; i++)
    {
        sum += v[i];
    }
    return sum;
}

inline Vector2 toSphericalCoordinates(const Vector3 &v)
{
    Vector2 result(std::acos(v.z), std::atan2(v.y, v.x));
    if (result.y < 0)
        result.y += 2.0f * M_PI_F;
    return result;
}

inline Vector3 sphericalDirection(const float &cosTheta, const float &sinTheta, const float &cosPhi, const float &sinPhi)
{
    return Vector3(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
};

inline Vector3 sphericalDirection(const float &theta, const float &phi)
{
    const float cosTheta = std::cos(theta);
    const float sinTheta = std::sin(theta);
    const float cosPhi = std::cos(phi);
    const float sinPhi = std::sin(phi);

    return sphericalDirection(cosTheta, sinTheta, cosPhi, sinPhi);
};

inline Vector3 squareToUniformSphere(const Vector2 sample)
{
    float z = 1.0f - 2.0f * sample.y;
    float r = std::sqrt(std::max(0.f, (1.0f - z * z)));
    float sinPhi, cosPhi;
    sincosf(2.0f * M_PI_F * sample.x, &sinPhi, &cosPhi);
    return Vector3(r * cosPhi, r * sinPhi, z);
}

}  // namespace openpgl

#include <chrono>
namespace openpgl
{
class Timer
{
   private:
    using clock = std::chrono::high_resolution_clock;
    using time_point = clock::time_point;

   public:
    Timer()
    {
        reset();
    }

    void reset()
    {
        start = clock::now();
    }

    double elapsed()
    {
        time_point end = clock::now();
        std::chrono::duration<double, std::micro> diff = end - start;
        return diff.count();
    }

   private:
    time_point start;
};
}  // namespace openpgl