#ifndef OPENPGL_GPU_H
#define OPENPGL_GPU_H

#include <string>
#include <iostream>
#include <typeindex>
#include <map>
#include "../common.h"
#include "../cpp/Field.h"
#include "Device.h"
#include "Data.h"

#if !defined(__CUDACC__)
#include <tbb/parallel_for.h>
#else
//#include <tbb/parallel_for.h>
#include <omp.h>
#endif

#include <fstream>
#include <chrono>
#include <map>
#include <typeindex>
#include <math.h>
#include <array>

namespace openpgl
{
namespace gpu
{
#if defined(OPENPGL_GPU_SYCL_SUPPORT)
namespace sycl
{
#define OPENPGL_GPU_SYCL
#include "Common.h"
#include "Vector.h"
#include "SampleDataStorage.h"
#include "PathSegmentStorage.h"
#include "VMMPhaseFunctions.h"
#include "Distribution.h"
#include "Code.h"
#undef OPENPGL_GPU_SYCL
}
#endif
#if defined(OPENPGL_GPU_CUDA_SUPPORT)
namespace cuda
{
#define OPENPGL_GPU_CUDA
#include "Common.h"
#include "Vector.h"
#include "SampleDataStorage.h"
#include "PathSegmentStorage.h"
#include "VMMPhaseFunctions.h"
#include "Distribution.h"
#include "Code.h"
#undef OPENPGL_GPU_CUDA
}
#endif
#if false
namespace cpu
{
#define OPENPGL_GPU_CPU
#include "Common.h"
#include "Vector.h"
#include "SampleDataStorage.h"
#include "PathSegmentStorage.h"
#include "VMMPhaseFunctions.h"
#include "Distribution.h"
#include "Code.h"
#undef OPENPGL_GPU_CPU
}
#endif
}  // namespace gpu
}  // namespace openpgl

#endif  // OPENPGL_GPU_H