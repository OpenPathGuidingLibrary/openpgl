#define OPENPGL_KERNEL_NS cpu

#include <embreeSrc/common/math/bbox.h>
#include <embreeSrc/common/math/constants.h>
#include <embreeSrc/common/math/emath.h>
#include <embreeSrc/common/math/vec2.h>
#include <embreeSrc/common/math/vec3.h>

namespace openpgl {
    using Vector2 = embree::Vec2<float>;
    using Vector3 = embree::Vec3<float>;
    using Point2 = embree::Vec2<float>;
    using Point3 = embree::Vec3<float>;
    
    using Point3i = embree::Vec3<int64_t>;
    using Vector3i = embree::Vec3<int64_t>;
    
    using BBox = embree::BBox<Vector3>;
    using BBoxi = embree::BBox<Vector3i>;
}

#ifdef OPENPGL_VEC_SIZE
constexpr static int VectorSize = OPENPGL_VEC_SIZE;
#endif

// not cuda
#define KERNEL_FUNCTION
#define SHARED_FUNCTION

#define FOREACH(var, start, end) \
    for (int var = start; var < end; var++)

#define FOREACH_COALESCED(var, end) \
    FOREACH(var, 0, end)

#define SINGLE
#define SHARED
#define SYNC

namespace openpgl {
namespace OPENPGL_KERNEL_NS {
//namespace OPENPGL_KERNEL_NS {    
#ifdef OPENPGL_VEC_SIZE
    template<int VectorSize>
    struct KernelCPU {
        static constexpr int BlockDim = 1;
    };
    using Kernel = KernelCPU<VectorSize>;

    using vfloat = embree::vfloat<VectorSize>;
    using vint = embree::vint<VectorSize>;
    using vbool = embree::vbool<VectorSize>;
#endif

    inline bool isValid(float &val) {
        return embree::isvalid(val);
    }

#ifdef OPENPGL_VEC_SIZE
    inline bool isValid(vfloat &val) {
        return embree::isvalid(val);
    }
#endif

    inline float dot(Vector2 &a, Vector2 &b)
    {
        return embree::dot(a, b);
    }

    inline float dot(Vector3 &a, Vector3 &b)
    {
        return embree::dot(a, b);
    }

//#define KAHAN
    template<int Offset, typename T, int BlockDim, int Pitch = 1>
    struct Accumulator {
        const static int OffsetEnd = 0;

#ifdef KAHAN
        T compensation[Pitch];
#endif

        T (&target)[Pitch];

        Accumulator(T &target, T init = {}) : Accumulator(*reinterpret_cast<T(*)[1]>(&target), init) {
            static_assert(Pitch == 1);
        }

        Accumulator(T (&target)[Pitch], T init = {}) : target(target) {
#ifdef KAHAN
            for (int i = 0; i < Pitch; i++)
                compensation[i] = init;
#endif
        }

        KERNEL_FUNCTION inline void accumulate(int pitch, T val) {
#ifndef KAHAN
            target[pitch] += val;
#else
            T& c = compensation[pitch];
            T& sum = target[pitch];
                
            // c is zero the first time around.
            //var y = input[i] - c
            T y = val - c;
            
            // Alas, sum is big, y small, so low-order digits of y are lost.         
            //var t = sum + y
            T t = sum + y;

            // (t - sum) cancels the high-order part of y;
            // subtracting y recovers negative (low part of y)
            //c = (t - sum) - y
            c = (t - sum) - y;

            // Algebraically, c should always be zero. Beware
            // overly-aggressive optimizing compilers!
            //sum = t
            sum = t;
#endif
        }

        KERNEL_FUNCTION inline void accumulate(T val) {
            accumulate(0, val);
        }

        KERNEL_FUNCTION inline void resolve() { }
    };

    template<typename T>
    T broadcast(T val) {
        return val;
    }

}
}