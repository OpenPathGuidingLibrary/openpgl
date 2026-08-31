
// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../openpgl.h"

/* we consider floating point numbers in that range as valid input numbers */
#define FLT_LARGE 1.844E18f

namespace openpgl
{
namespace cpp
{
struct Vector3f : public pgl_vec3f
{
    Vector3f(const pgl_vec3f &v)
    {
        this->x = v.x;
        this->y = v.y;
        this->z = v.z;
    }

    Vector3f(float x, float y, float z)
    {
        this->x = x;
        this->y = y;
        this->z = z;
    }

    Vector3f(float v)
    {
        this->x = v;
        this->y = v;
        this->z = v;
    }
};

struct Vector2f : public pgl_vec2f
{
    Vector2f(const pgl_vec2f &v)
    {
        this->x = v.x;
        this->y = v.y;
    }

    Vector2f(float x, float y)
    {
        this->x = x;
        this->y = y;
    }

    Vector2f(float v)
    {
        this->x = v;
        this->y = v;
    }
};

struct Point3f : public pgl_point3f
{
    Point3f(const pgl_point3f &p)
    {
        this->x = p.x;
        this->y = p.y;
        this->z = p.z;
    }
    Point3f(float x, float y, float z)
    {
        this->x = x;
        this->y = y;
        this->z = z;
    }

    Point3f(float v)
    {
        this->x = v;
        this->y = v;
        this->z = v;
    }
};

struct Point3i : public pgl_point3i
{
    Point3i(const pgl_point3i &p)
    {
        this->x = p.x;
        this->y = p.y;
        this->z = p.z;
    }
    Point3i(int32_t x, int32_t y, int32_t z)
    {
        this->x = x;
        this->y = y;
        this->z = z;
    }

    Point3i(int32_t v)
    {
        this->x = v;
        this->y = v;
        this->z = v;
    }
};

struct Point2f : public pgl_point2f
{
    Point2f(const pgl_point2f &p)
    {
        this->x = p.x;
        this->y = p.y;
    }
    Point2f(float x, float y)
    {
        this->x = x;
        this->y = y;
    }

    Point2f(float v)
    {
        this->x = v;
        this->y = v;
    }
};

struct Point2i : public pgl_point2i
{
    Point2i(const pgl_point2i &p)
    {
        this->x = p.x;
        this->y = p.y;
    }
    Point2i(int32_t x, int32_t y)
    {
        this->x = x;
        this->y = y;
    }

    Point2i(int32_t v)
    {
        this->x = v;
        this->y = v;
    }
};

struct Box3f : public pgl_box3f
{
    Box3f(Point3f lower, Point3f upper)
    {
        this->lower = lower;
        this->upper = upper;
    }
};

/* for a 3d vector of type pgl_vec3f.
 *
 * @param x
 * @param y
 * @param z
 * @return pgl_vec3f
 */
OPENPGL_INLINE pgl_vec3f Vector3(float x, float y, float z)
{
    pgl_vec3f vec3{x, y, z};
    return vec3;
}

/**
 * @brief Wrapper function to simulate a C++ constructor
 * for a 2d vector of type pgl_vec2f.
 *
 * @param x
 * @param y
 * @return pgl_vec2f
 */
OPENPGL_INLINE pgl_vec2f Vector2(float x, float y)
{
    pgl_vec2f vec2{x, y};
    return vec2;
}

/**
 * @brief Wrapper function to simulate a C++ constructor
 * for a 3d position of type pgl_point3f.
 *
 * @param x
 * @param y
 * @param z
 * @return pgl_point3f
 */
OPENPGL_INLINE pgl_point3f Point3(float x, float y, float z)
{
    pgl_point3f point3{x, y, z};
    return point3;
}

/**
 * @brief Wrapper function to simulate a C++ constructor
 * for a 2d position of type pgl_point2f.
 *
 * @param x
 * @param y
 * @return pgl_point2f
 */
OPENPGL_INLINE pgl_vec2f Point2(float x, float y)
{
    pgl_point2f point2{x, y};
    return point2;
}

/**
 * @brief Wrapper function to simulate a C++ constructor
 * for a bounding box of type pgl_box3f.
 *
 * @param lower
 * @param upper
 * @return pgl_box3f
 */
OPENPGL_INLINE pgl_box3f Box3(pgl_point3f lower, pgl_point3f upper)
{
    pgl_box3f box3{lower, upper};
    return box3;
}

/**
 * @brief Wrapper function to simulate a C++ constructor
 * for a bounding box of type pgl_box3f.
 *
 * @param lower_x
 * @param lower_y
 * @param lower_z
 * @param upper_x
 * @param upper_y
 * @param upper_z
 * @return pgl_box3f
 */
OPENPGL_INLINE pgl_box3f Box3(float lower_x, float lower_y, float lower_z, float upper_x, float upper_y, float upper_z)
{
    pgl_box3f box3{pgl_point3f{lower_x, lower_y, lower_z}, pgl_point3f{upper_x, upper_y, upper_z}};
    return box3;
}

OPENPGL_INLINE bool IsValid(const pgl_vec3f &vec)
{
    return ((vec.x > -FLT_LARGE) & (vec.x < +FLT_LARGE)) && ((vec.y > -FLT_LARGE) & (vec.y < +FLT_LARGE)) && ((vec.z > -FLT_LARGE) & (vec.z < +FLT_LARGE));
}

OPENPGL_INLINE bool IsZero(const pgl_vec3f &vec)
{
    return vec.x == 0.f && vec.y == 0.f && vec.z == 0.f;
}

OPENPGL_INLINE float Max(const pgl_vec3f &vec)
{
    return std::max(vec.x, std::max(vec.y, vec.z));
}

OPENPGL_INLINE float Average(const pgl_vec3f &vec)
{
    return (vec.x + vec.y + vec.z) / 3.f;
}

OPENPGL_INLINE float Average(const pgl_vec2f &vec)
{
    return (vec.x + vec.y) * 0.5f;
}
/*
   OPENPGL_INLINE pgl_vec3f operator+(const pgl_vec3f& v1, const pgl_vec3f& v2)
   {
      return {v1.x + v2.x, v1.y + v2.y, v1.z + v2.z};
   }

   OPENPGL_INLINE pgl_vec3f operator+(const float f, const pgl_vec3f& v)
   {
      return {v.x + f, v.y + f, v.z + f};
   }

   OPENPGL_INLINE pgl_vec3f operator+(const pgl_vec3f& v, const float f)
   {
      return {v.x + f, v.y + f, v.z + f};
   }

   OPENPGL_INLINE pgl_vec3f operator-(const pgl_vec3f& v1, const pgl_vec3f& v2)
   {
      return {v1.x - v2.x, v1.y - v2.y, v1.z - v2.z};
   }

   OPENPGL_INLINE pgl_vec3f operator-(const float f, const pgl_vec3f& v)
   {
      return {v.x - f, v.y - f, v.z - f};
   }

   OPENPGL_INLINE pgl_vec3f operator-(const pgl_vec3f& v, const float f)
   {
      return {v.x - f, v.y - f, v.z - f};
   }

   OPENPGL_INLINE pgl_vec3f operator*(const pgl_vec3f& v1, const pgl_vec3f& v2)
   {
      return {v1.x * v2.x, v1.y * v2.y, v1.z * v2.z};
   }

   OPENPGL_INLINE pgl_vec3f operator*(const float f, const pgl_vec3f& v)
   {
      return {v.x * f, v.y * f, v.z * f};
   }

   OPENPGL_INLINE pgl_vec3f operator*(const pgl_vec3f& v, const float f)
   {
      return {v.x * f, v.y * f, v.z * f};
   }

   OPENPGL_INLINE pgl_vec3f operator/(const pgl_vec3f& v1, const pgl_vec3f& v2)
   {
      return {v1.x / v2.x, v1.y / v2.y, v1.z / v2.z};
   }

   OPENPGL_INLINE pgl_vec3f operator/(const float f, const pgl_vec3f& v)
   {
      return {v.x / f, v.y / f, v.z / f};
   }

   OPENPGL_INLINE pgl_vec3f operator/(const pgl_vec3f& v, const float f)
   {
      return {v.x / f, v.y / f, v.z / f};
   }
*/
OPENPGL_INLINE pgl_direction CompressDirection(const pgl_vec3f &dir)
{
    return dir;
}

OPENPGL_INLINE pgl_vec3f DecompressDirection(const pgl_direction cdir)
{
    return cdir;
}

OPENPGL_INLINE pgl_spectrum CompressSpectrum(const pgl_vec3f &specRGB)
{
    return specRGB;
}

OPENPGL_INLINE pgl_vec3f DecompressSpectrum(const pgl_spectrum &cspecRGB)
{
    return cspecRGB;
}

}  // namespace cpp
}  // namespace openpgl

#undef FLT_LARGE