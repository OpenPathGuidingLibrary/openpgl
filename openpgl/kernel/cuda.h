#define OPENPGL_KERNEL_NS gpu

#include <embreeSrc/common/math/bbox.h>
#include <embreeSrc/common/math/constants.h>
#include <embreeSrc/common/math/emath.h>
#include <embreeSrc/common/math/vec2.h>
#include <embreeSrc/common/math/vec3.h>

#ifdef OPENPGL_VEC_SIZE
constexpr static int VectorSize = 1;
#endif

// not cuda
#define KERNEL_FUNCTION __device__
#define SHARED_FUNCTION __device__ __host__

#define FOREACH(var, start, end) \
    for (int var = (start) + threadIdx.x; var < (end); var += blockDim.x)

#define FOREACH_COALESCED(var, end) \
    FOREACH(var, 0, (((end) + warpSize - 1) / warpSize) * warpSize)

#define SINGLE if(threadIdx.x == 0)
#define SHARED __shared__
#define SYNC __syncthreads()

extern __shared__ float shared[];

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

namespace embree {
    KERNEL_FUNCTION inline bool isvalid(const Vec3<float> &val) {
        return isvalid(val.x) && isvalid(val.y) && isvalid(val.z);
    }

    KERNEL_FUNCTION inline bool isvalid(const Vec2<float> &val) {
        return isvalid(val.x) && isvalid(val.y);
    }
}

namespace openpgl {
namespace OPENPGL_KERNEL_NS {
    
#ifdef OPENPGL_VEC_SIZE
    //template<int VectorSize>
    template<int BlockDim_>
    struct KernelCuda { 
        static constexpr int BlockDim = BlockDim_;
    };
    //using Kernel = KernelCuda;

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

    static constexpr int WARP_SIZE = 32;
    static constexpr unsigned WARP_MASK = 0xffffffff;

    KERNEL_FUNCTION inline int warpIdx() {
        return threadIdx.x / WARP_SIZE;
    }

    KERNEL_FUNCTION inline int laneIdx() {
        return threadIdx.x % WARP_SIZE;
    }

    template<typename T>
    struct Comp {
        T value;
        T comp;

        KERNEL_FUNCTION Comp(T value) : value(value), comp(0) {}

        KERNEL_FUNCTION Comp(T value, T comp) : value(value), comp(comp) {}

        KERNEL_FUNCTION Comp twoSum(T a, T b) {
            T s = a + b;
            T al = s - b;
            T bl = s - al;
            T da = a - al;
            T db = b - bl;
            T t = da + db;
            return Comp(s, t);
        }

        KERNEL_FUNCTION Comp fast2sum(T a, T b) {
            T s = a + b;
            T z = s - a;
            T t = b - z;
            return Comp(s, t);
        }

        KERNEL_FUNCTION Comp& operator+= (const Comp& rhs) {
            auto s = twoSum(value, rhs.value);
            auto d = twoSum(comp, rhs.comp);
            *this = twoSum(s.value, s.comp + d.value);
            return *this;
        }
    };


    // all of this reduction code assumes that every lane participates!
    KERNEL_FUNCTION inline float warpShuffleDownSync(float var, unsigned int delta) {
        return __shfl_down_sync(WARP_MASK, var, delta);
    }

    KERNEL_FUNCTION inline Vector2 warpShuffleDownSync(Vector2 var, unsigned int delta) {
        return Vector2(
            __shfl_down_sync(WARP_MASK, var.x, delta),
            __shfl_down_sync(WARP_MASK, var.y, delta)
        );
    }

    KERNEL_FUNCTION inline Vector3 warpShuffleDownSync(Vector3 var, unsigned int delta) {
        return Vector3(
            __shfl_down_sync(WARP_MASK, var.x, delta),
            __shfl_down_sync(WARP_MASK, var.y, delta),
            __shfl_down_sync(WARP_MASK, var.z, delta)
        );
    }

    template<typename T>
    KERNEL_FUNCTION inline Comp<T> warpShuffleDownSync(Comp<T> var, unsigned int delta) {
        return Comp(
            warpShuffleDownSync(var.value, delta),
            warpShuffleDownSync(var.comp, delta)
        );
    }

    template<typename T>
    KERNEL_FUNCTION void warpReduce(T &val) {
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
            val += warpShuffleDownSync(val, offset);
    }

    // This is a convenience class to handle parallel accumulation across multiple warps in a work group
    // each warp has a dedicated accumulator in shared memory (the first warp reuses the target variable)
    // the results need to be explicitly merged through resolve()
    // TODO see if coalesced_threads(); from cooperative groups allows to merge with more branching code
    template<int Offset, typename T, int BlockDim, int Pitch = 1>
    struct Accumulator {
        using Acc = T;

        const static int NumWarps = BlockDim / WARP_SIZE;
        const static int Size = sizeof(Acc) * Pitch * NumWarps;
        const static int OffsetEnd = Offset + Size;

        // TODO VERIFY init & reference should get optimized away by the compiler
        const T init;
        T (&target)[Pitch];
        
        KERNEL_FUNCTION Accumulator(T &target, T init) : Accumulator(*reinterpret_cast<T(*)[1]>(&target), init) {
            static_assert(Pitch == 1);
        }
        
        KERNEL_FUNCTION Accumulator(T (&target)[Pitch], T init) : init(init), target(target) {
            static_assert(NumWarps <= WARP_SIZE); // Needed for correct resolving
            for (int i = 0; i < Pitch; i++)
                if (laneIdx() == 0)
                    getTarget(warpIdx(), i) = init;
        }

        KERNEL_FUNCTION inline Acc& getTarget(int idx, int pitch) {
            const int offset = Offset + sizeof(Acc) * (NumWarps * pitch + idx);
            return *(Acc*)&((char*)shared)[offset];
        }

        //KERNEL_FUNCTION inline void init(int pitch = 0) {
        //    if (laneIdx() == 0) getTarget(pitch) = {};
        //}
        
        KERNEL_FUNCTION inline void accumulate(int pitch, T val) {
            assert(embree::isvalid(val));
            Acc acc(val);
            warpReduce(acc);
            if (laneIdx() == 0) {
                assert(embree::isvalid(val));
                assert(embree::isvalid(getTarget(warpIdx(), pitch)));
                getTarget(warpIdx(), pitch) += acc;
                assert(embree::isvalid(getTarget(warpIdx(), pitch)));
            }
        }

        KERNEL_FUNCTION inline void accumulate(T val) {
            accumulate(0, val);
        }

        // call __syncthreads before resolving
        KERNEL_FUNCTION inline void resolve() {
            if (warpIdx() != 0) return;

            for (int i = 0; i < Pitch; i++) {
                Acc val = threadIdx.x < NumWarps ? getTarget(threadIdx.x, i) : Acc(init);
                assert(embree::isvalid(val));
                warpReduce(val);
                SINGLE {
                    assert(embree::isvalid(val));
                    target[i] += val;
                }
            }
        }
    };

    template <int val>
    struct PrintConst;

    template<typename T>
    KERNEL_FUNCTION T broadcast(T val) {
        __shared__ T global;
        // this sync might be needed so that thread 0 does not race
        // to the next broadcast before other threads have read the variable
        // although in many cases there are already lots of sync points in between
        SYNC;
        SINGLE global = val;
        SYNC;
        return global;
    }
}
}
