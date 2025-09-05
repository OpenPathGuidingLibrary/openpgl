find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    if(NOT DEFINED CMAKE_CUDA_STANDARD)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    endif()

    set(CMAKE_CUDA_ARCHITECTURES 89)

    add_executable(openpgl_cuda_fit
        main.cpp
        GPUField.cu
        GPUFieldOptimized.cu
    )
    
    #set_source_files_properties(gpu_fit.cu
    #    PROPERTIES
    #    COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Debug>:-G;-g;-O0>>
    #)
    if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
        set_source_files_properties(GPUField.cu
            PROPERTIES
            COMPILE_OPTIONS -G;-g;-O0
        )
    endif()

    target_link_libraries(openpgl_cuda_fit PRIVATE openpgl)
    target_link_libraries(openpgl_cuda_fit PRIVATE TBB::${OPENPGL_TBB_COMPONENT})
    target_include_directories(openpgl_cuda_fit PRIVATE ${CMAKE_BINARY_DIR}/openpgl/include/openpgl/gpu)
    target_include_directories(openpgl_cuda_fit PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../../third-party/)
    target_compile_definitions(openpgl_cuda_fit PRIVATE OPENPGL_GPU_CUDA_SUPPORT)
    #target_compile_options(openpgl_gpu_cuda PRIVATE --ftz=true --prec-div=false --prec-sqrt=false --fmad=true)
    #target_compile_options(openpgl_gpu_cuda PRIVATE --ftz=false --prec-div=true --prec-sqrt=true --fmad=false --extended-lambda)
    target_compile_options(openpgl_cuda_fit PRIVATE -lineinfo 
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;--expt-relaxed-constexpr;>
    )
    add_dependencies(openpgl_cuda_fit openpgl)

    INSTALL(TARGETS openpgl_cuda_fit RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    
endif(CUDAToolkit_FOUND)