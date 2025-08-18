find_package(ZLIB)

add_executable(openpgl_gpu_cpu openpgl_gpu.cpp)
target_link_libraries(openpgl_gpu_cpu PRIVATE openpgl)
target_link_libraries(openpgl_gpu_cpu PRIVATE TBB::${OPENPGL_TBB_COMPONENT} ZLIB::ZLIB)
target_include_directories(openpgl_gpu_cpu PRIVATE ${CMAKE_BINARY_DIR}/openpgl/include/openpgl/gpu)
target_include_directories(openpgl_gpu_cpu PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../../third-party/)
target_compile_definitions(openpgl_gpu_cpu PRIVATE OPENPGL_GPU_CPU_SUPPORT)
add_dependencies(openpgl_gpu_cpu openpgl)

INSTALL(TARGETS openpgl_gpu_cpu RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

if(OPENPGL_SYCL_SUPPORT)
    add_executable(openpgl_gpu_sycl openpgl_gpu.cpp)
    target_link_libraries(openpgl_gpu_sycl PRIVATE openpgl)
    target_link_libraries(openpgl_gpu_sycl PRIVATE TBB::${OPENPGL_TBB_COMPONENT} ZLIB::ZLIB)
    target_include_directories(openpgl_gpu_sycl PRIVATE ${CMAKE_BINARY_DIR}/openpgl/include/openpgl/gpu)
    target_include_directories(openpgl_gpu_sycl PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../../third-party/)
    target_compile_definitions(openpgl_gpu_sycl PRIVATE OPENPGL_GPU_SYCL_SUPPORT)
    target_link_options(openpgl_gpu_sycl PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda)
    target_compile_options(openpgl_gpu_sycl PRIVATE --std=c++17 -fsycl -fsycl-targets=nvptx64-nvidia-cuda)
    add_dependencies(openpgl_gpu_sycl openpgl)

    INSTALL(TARGETS openpgl_gpu_sycl RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    if(NOT DEFINED CMAKE_CUDA_STANDARD)
        set(CMAKE_CUDA_STANDARD 11)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    endif()

    add_executable(openpgl_gpu_cuda openpgl_gpu.cu)

    set_target_properties(openpgl_gpu_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    
    target_link_libraries(openpgl_gpu_cuda PRIVATE openpgl)
    target_link_libraries(openpgl_gpu_cuda PRIVATE TBB::${OPENPGL_TBB_COMPONENT} ZLIB::ZLIB)
    target_include_directories(openpgl_gpu_cuda PRIVATE ${CMAKE_BINARY_DIR}/openpgl/include/openpgl/gpu)
    target_include_directories(openpgl_gpu_cuda PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../../third-party/)
    target_compile_definitions(openpgl_gpu_cuda PRIVATE OPENPGL_GPU_CUDA_SUPPORT)
    #target_compile_options(openpgl_gpu_cuda PRIVATE --ftz=true --prec-div=false --prec-sqrt=false --fmad=true)
    #target_compile_options(openpgl_gpu_cuda PRIVATE --ftz=false --prec-div=true --prec-sqrt=true --fmad=false --extended-lambda)
    target_compile_options(openpgl_gpu_cuda PRIVATE -lineinfo --std=c++17;--use_fast_math;--expt-relaxed-constexpr;--extended-lambda;--forward-unknown-to-host-compiler)
    add_dependencies(openpgl_gpu_cuda openpgl)

    INSTALL(TARGETS openpgl_gpu_cuda RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    
endif(CUDAToolkit_FOUND)