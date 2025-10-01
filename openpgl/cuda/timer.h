#pragma once

#include <chrono>
#ifdef __CUDACC__
#include <cuda.h>
#endif

namespace openpgl{
namespace OPENPGL_KERNEL_NS {

/*
 * @class Timer
 * @brief A simple RAII timer to measure the execution time of a scope.
 *
 * This timer automatically starts when it's created and stops when it goes
 * out of scope (is destructed). It then prints the elapsed time.
 */
class CudaTimer {
public:
    // Constructor: records the starting time point upon creation.
    CudaTimer() {
#ifdef __CUDACC__
        cudaDeviceSynchronize();
#endif
        m_StartTimePoint = std::chrono::high_resolution_clock::now();
    }

    // Stops the timer, calculates the duration, and prints it.
    double elapsed() {
#ifdef __CUDACC__
        cudaDeviceSynchronize();
#endif
        auto endTimePoint = std::chrono::high_resolution_clock::now();

        // Calculate the duration between start and end.
        auto duration = endTimePoint - m_StartTimePoint;
        
        // Use double for better precision when representing seconds with nanosecond resolution.
        // A float has ~7 decimal digits of precision, which is insufficient.
        // A double has ~15-17, which is more than enough for 9 decimal places.
        return std::chrono::duration_cast<std::chrono::duration<double>>(duration).count();
    }

private:
    // Member variable to store the starting time point.
    // std::chrono::high_resolution_clock is the most precise clock available.
    std::chrono::time_point<std::chrono::high_resolution_clock> m_StartTimePoint;
};

}
}