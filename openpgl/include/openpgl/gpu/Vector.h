
OPENPGL_GPU_CALLABLE inline void pgl_sincosf(float x, float *sin, float *cos)
{
#ifdef SYCL_LANGUAGE_VERSION
    *sin = ::sycl::sincos(x, cos);
#else
#if defined(WIN32)
    *sin = sinf(x);
    *cos = cosf(x);
#else
    sincosf(x, sin, cos);
#endif
#endif
}

#if !defined(OPENPGL_GPU_CUDA) && !defined(OPENPGL_GPU_SYCL)
struct float2
{
    float x;
    float y;
};

struct float3
{
    float x;
    float y;
    float z;
};

#endif

inline OPENPGL_GPU_CALLABLE float rsqrt(float v)
{
    return 1.0f / std::sqrt(v);
}

#if defined(OPENPGL_GPU_SYCL)
typedef ::sycl::mfloat2 Vector2;
typedef ::sycl::mfloat3 Vector3;
typedef ::sycl::mfloat2 Point2;
typedef ::sycl::mfloat3 Point3;
typedef ::sycl::mfloat3 Normal3;
using namespace sycl;
#define sqrtf ::sycl::native::sqrt
#else  // defined(OPENPGL_GPU_CUDA)

union Vector2
{
    float2 vec;
    float data[2];

    OPENPGL_GPU_CALLABLE Vector2() = default;

    OPENPGL_GPU_CALLABLE Vector2(float x, float y)
    {
        vec = {x, y};
    }

    OPENPGL_GPU_CALLABLE inline float &operator[](std::size_t idx)
    {
        return data[idx];
    }
    OPENPGL_GPU_CALLABLE inline const float &operator[](std::size_t idx) const
    {
        return data[idx];
    }

    OPENPGL_GPU_CALLABLE inline const Vector2 &operator*=(const Vector2 &b)
    {
        this->vec.x *= b.vec.x;
        this->vec.y *= b.vec.y;
        return *this;
    }

    OPENPGL_GPU_CALLABLE inline const Vector2 &operator*=(const float b)
    {
        this->vec.x *= b;
        this->vec.y *= b;
        return *this;
    }

    OPENPGL_GPU_CALLABLE inline const Vector2 &operator/=(const Vector2 &b)
    {
        this->vec.x /= b.vec.x;
        this->vec.y /= b.vec.y;
        return *this;
    }

    OPENPGL_GPU_CALLABLE inline const Vector2 &operator/=(const float b)
    {
        this->vec.x /= b;
        this->vec.y /= b;
        return *this;
    }
};

OPENPGL_GPU_CALLABLE inline const Vector2 operator*(Vector2 lhs, const Vector2 &rhs)
{
    return lhs *= rhs;
}

OPENPGL_GPU_CALLABLE inline const Vector2 operator*(Vector2 lhs, const float f)
{
    return lhs *= f;
}

OPENPGL_GPU_CALLABLE inline const Vector2 operator/(Vector2 lhs, const Vector2 &rhs)
{
    return lhs /= rhs;
}

OPENPGL_GPU_CALLABLE inline const Vector2 operator/(Vector2 lhs, const float f)
{
    return lhs /= f;
}

OPENPGL_GPU_CALLABLE inline float dot(const Vector2 a, const Vector2 b)
{
    return a[0] * b[0] + a[1] * b[1];
}

OPENPGL_GPU_CALLABLE inline float length(const Vector2 &a)
{
    return sqrtf(dot(a, a));
}

OPENPGL_GPU_CALLABLE inline Vector2 normalize(const Vector2 &a)
{
    return a * rsqrt(dot(a, a));
}

union Vector3
{
    float3 vec;
    float data[3];

    OPENPGL_GPU_CALLABLE Vector3() = default;

    OPENPGL_GPU_CALLABLE Vector3(float x, float y, float z)
    {
        vec = {x, y, z};
    }

    OPENPGL_GPU_CALLABLE inline float &operator[](std::size_t idx)
    {
        return data[idx];
    }
    OPENPGL_GPU_CALLABLE inline const float &operator[](std::size_t idx) const
    {
        return data[idx];
    }

    OPENPGL_GPU_CALLABLE inline const Vector3 &operator*=(const Vector3 &b)
    {
        this->vec.x *= b.vec.x;
        this->vec.y *= b.vec.y;
        this->vec.z *= b.vec.z;
        return *this;
    }
    OPENPGL_GPU_CALLABLE inline const Vector3 &operator*=(const float b)
    {
        this->vec.x *= b;
        this->vec.y *= b;
        this->vec.z *= b;
        return *this;
    }

    OPENPGL_GPU_CALLABLE inline const Vector3 &operator/=(const Vector3 &b)
    {
        this->vec.x /= b.vec.x;
        this->vec.y /= b.vec.y;
        this->vec.z /= b.vec.z;
        return *this;
    }
    OPENPGL_GPU_CALLABLE inline const Vector3 &operator/=(const float b)
    {
        this->vec.x /= b;
        this->vec.y /= b;
        this->vec.z /= b;
        return *this;
    }

    OPENPGL_GPU_CALLABLE inline const Vector3 &operator+=(const Vector3 &b)
    {
        this->vec.x += b.vec.x;
        this->vec.y += b.vec.y;
        this->vec.z += b.vec.z;
        return *this;
    }
    OPENPGL_GPU_CALLABLE inline const Vector3 &operator+=(const float b)
    {
        this->vec.x += b;
        this->vec.y += b;
        this->vec.z += b;
        return *this;
    }

    OPENPGL_GPU_CALLABLE inline const Vector3 &operator-=(const Vector3 &b)
    {
        this->vec.x -= b.vec.x;
        this->vec.y -= b.vec.y;
        this->vec.z -= b.vec.z;
        return *this;
    }
    OPENPGL_GPU_CALLABLE inline const Vector3 &operator-=(const float b)
    {
        this->vec.x -= b;
        this->vec.y -= b;
        this->vec.z -= b;
        return *this;
    }
};

OPENPGL_GPU_CALLABLE inline const Vector3 operator*(Vector3 lhs, const Vector3 &rhs)
{
    return lhs *= rhs;
}

OPENPGL_GPU_CALLABLE inline const Vector3 operator*(Vector3 lhs, const float f)
{
    return lhs *= f;
}

OPENPGL_GPU_CALLABLE inline const Vector3 operator/(Vector3 lhs, const Vector3 &rhs)
{
    return lhs /= rhs;
}

OPENPGL_GPU_CALLABLE inline const Vector3 operator/(Vector3 lhs, const float f)
{
    return lhs /= f;
}

OPENPGL_GPU_CALLABLE inline const Vector3 operator+(Vector3 lhs, const Vector3 &rhs)
{
    return lhs += rhs;
}

OPENPGL_GPU_CALLABLE inline const Vector3 operator+(Vector3 lhs, const float f)
{
    return lhs += f;
}

OPENPGL_GPU_CALLABLE inline const Vector3 operator-(Vector3 lhs, const Vector3 &rhs)
{
    return lhs -= rhs;
}

OPENPGL_GPU_CALLABLE inline const Vector3 operator-(Vector3 lhs, const float f)
{
    return lhs -= f;
}


OPENPGL_GPU_CALLABLE inline float dot(const Vector3 &a, const Vector3 &b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

OPENPGL_GPU_CALLABLE inline float length(const Vector3 &a)
{
    return sqrtf(dot(a, a));
}

OPENPGL_GPU_CALLABLE inline Vector3 normalize(const Vector3 &a)
{
    return a * rsqrt(dot(a, a));
}

OPENPGL_GPU_CALLABLE inline Vector3 cross(const Vector3 &a, const Vector3 &b)
{
    return Vector3(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]);
}

typedef Vector2 Point2;
typedef Vector3 Point3;
typedef Vector3 Normal3;
#endif
