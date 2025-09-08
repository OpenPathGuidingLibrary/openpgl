#include <embreeSrc/common/math/bbox.h>
#include <embreeSrc/common/math/constants.h>
#include <embreeSrc/common/math/emath.h>
#include <embreeSrc/common/math/vec2.h>
#include <embreeSrc/common/math/vec3.h>

#ifdef OPENPGL_VEC_SIZE
constexpr static int VectorSize = 1;
#endif

// not cuda
#define KERNEL_FUNCTION __device__ __host__

#define FOREACH(var, start, end) \
    for (int var = start + threadIdx.x; var < end; var += blockDim.x)

namespace openpgl {
    using Vector2 = embree::Vec2<float>;
    using Vector3 = embree::Vec3<float>;
    using Point2 = embree::Vec2<float>;
    using Point3 = embree::Vec3<float>;
    
    using Point3i = embree::Vec3<int64_t>;
    using Vector3i = embree::Vec3<int64_t>;
    
    using BBox = embree::BBox<Vector3>;
    using BBoxi = embree::BBox<Vector3i>;
    
#ifdef OPENPGL_VEC_SIZE
    template<int VectorSize>
    struct KernelCuda { };
    using Kernel = KernelCuda<VectorSize>;

    using vfloat = float;
    using vint = int;
    using vbool = bool;
#endif

    KERNEL_FUNCTION inline bool isValid(float &val) {
        return embree::isvalid(val);
    }

    KERNEL_FUNCTION inline float dot(Vector2 &a, Vector2 &b)
    {
        return embree::dot(a, b);
    }

    KERNEL_FUNCTION inline float dot(Vector3 &a, Vector3 &b)
    {
        return embree::dot(a, b);
    }
}
