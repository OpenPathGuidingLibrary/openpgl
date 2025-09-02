## Copyright 2009-2020 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

########################################
# restrict

include (CheckCXXSourceCompiles)

check_cxx_source_compiles (
"int * __restrict__ ptr;
int main() { }"
HAVE_PRE_POST_BAR_RESTRICT)

check_cxx_source_compiles (
"int * __restrict ptr;
int main() { }"
HAVE_PRE_BAR_RESTRICT)

if (HAVE_PRE_POST_BAR_RESTRICT)
  list (APPEND OPENPGL_DEFINITIONS "OPENPGL_RESTRICT=__restrict__")
elseif (HAVE_PRE_BAR_RESTRICT)
  list (APPEND OPENPGL_DEFINITIONS "OPENPGL_RESTRICT=__restrict")
else ()
  list (APPEND OPENPGL_DEFINITIONS "OPENPGL_RESTRICT")
endif ()

add_executable (soac ./cmd/soac.cpp)
target_compile_definitions(soac PRIVATE ${OPENPGL_DEFINITIONS})
add_executable (openpgl::soac ALIAS soac)
set_target_properties (soac PROPERTIES OUTPUT_NAME soac)

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/openpgl/gpu)

add_custom_command (OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/include/openpgl/gpu/PathSegmentStorage_soa.h
    COMMAND soac ${CMAKE_CURRENT_SOURCE_DIR}/soa/PathSegmentStorage.soa > ${CMAKE_CURRENT_BINARY_DIR}/include/openpgl/gpu/PathSegmentStorage_soa.h
    DEPENDS soac ${CMAKE_CURRENT_SOURCE_DIR}/soa/PathSegmentStorage.soa)
set (OPENPGL_SOA_GENERATED ${OPENPGL_SOA_GENERATED} ${CMAKE_CURRENT_BINARY_DIR}/include/openpgl/gpu/PathSegmentStorage_soa.h)

add_custom_command (OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/include/openpgl/gpu/SampleDataStorage_soa.h
    COMMAND soac ${CMAKE_CURRENT_SOURCE_DIR}/soa/SampleDataStorage_flat.soa > ${CMAKE_CURRENT_BINARY_DIR}/include/openpgl/gpu/SampleDataStorage_soa.h
    DEPENDS soac ${CMAKE_CURRENT_SOURCE_DIR}/soa/SampleDataStorage_flat.soa)
set (OPENPGL_SOA_GENERATED ${OPENPGL_SOA_GENERATED} ${CMAKE_CURRENT_BINARY_DIR}/include/openpgl/gpu/SampleDataStorage_soa.h)

message("OPENPGL_SOA_GENERATED = ${OPENPGL_SOA_GENERATED}")

add_custom_target (openpgl_soa_generated DEPENDS ${OPENPGL_SOA_GENERATED})

#set_property (TARGET soac PROPERTY FOLDER "cmd")


set(DEVICE_SRC)
if(OPENPGL_ISA_SSE4 OR OPENPGL_ISA_NEON)
  set(DEVICE_SRC ${DEVICE_SRC} api/deviceCPU4.cpp)
endif()
if(OPENPGL_ISA_AVX2 OR OPENPGL_ISA_NEON2X)
  set(DEVICE_SRC ${DEVICE_SRC} api/deviceCPU8.cpp)
endif()
if(OPENPGL_ISA_AVX512)
  set(DEVICE_SRC ${DEVICE_SRC} api/deviceCPU16.cpp)
  set(OPENPGL_SUPPORT_DEVICE_TYPE_CPU_16 ON)
endif()

set(OPENPGL_SRC
    api/api.cpp
    ${DEVICE_SRC}
    #directional/ISurfaceSamplingDistribution.cpp
    #directional/IVolumeSamplingDistribution.cpp
    directional/vmm/VMMPhaseFunctions.cpp
    ../third-party/embreeSrc/common/simd/sse.cpp
    ../third-party/embreeSrc/common/sys/alloc.cpp
    ../third-party/embreeSrc/common/sys/barrier.cpp
    ../third-party/embreeSrc/common/sys/condition.cpp
    ../third-party/embreeSrc/common/sys/mutex.cpp
    ../third-party/embreeSrc/common/sys/estring.cpp
    ../third-party/embreeSrc/common/sys/sysinfo.cpp
    ../third-party/embreeSrc/common/sys/thread.cpp
)

if(OPENPGL_BUILD_STATIC)
add_library(${PROJECT_NAME} STATIC
  ${OPENPGL_SRC}
)
target_compile_definitions(${PROJECT_NAME} PRIVATE BUILD_STATIC)
else()
add_library(${PROJECT_NAME} SHARED
  ${OPENPGL_SRC}
)

target_compile_definitions(${PROJECT_NAME} PRIVATE BUILD_SHARED)
endif()

add_dependencies(${PROJECT_NAME} openpgl_soa_generated)

if(OPENPGL_EF_RADIANCE_CACHES)
target_compile_definitions(${PROJECT_NAME} PRIVATE OPENPGL_RADIANCE_CACHES)
endif()

if(OPENPGL_EF_IMAGE_SPACE_GUIDING_BUFFER)
target_compile_definitions(${PROJECT_NAME} PRIVATE OPENPGL_IMAGE_SPACE_GUIDING_BUFFER)
endif()

if(OPENPGL_DIRECTION_COMPRESSION)
target_compile_definitions(${PROJECT_NAME} PRIVATE PGL_USE_DIRECTION_COMPRESSION)
endif()

if(OPENPGL_RADIANCE_COMPRESSION)
target_compile_definitions(${PROJECT_NAME} PRIVATE PGL_USE_COLOR_COMPRESSION)
endif()

#target_compile_definitions(${PROJECT_NAME} PRIVATE OPENPGL_DEBUG_MODE)

target_compile_definitions(${PROJECT_NAME} PRIVATE OPENPGL_VERSION_MAJOR=${PROJECT_VERSION_MAJOR} OPENPGL_VERSION_MINOR=${PROJECT_VERSION_MINOR} OPENPGL_VERSION_PATCH=${PROJECT_VERSION_PATCH})
target_compile_definitions(${PROJECT_NAME} PRIVATE OPENPGL_VERSION_STRING="${PROJECT_VERSION}")
target_compile_definitions(${PROJECT_NAME} PRIVATE OPENPGL_BUILD)
if(OPENPGL_ISA_SSE4 OR OPENPGL_ISA_NEON)
  target_compile_definitions(${PROJECT_NAME} PRIVATE OPENPGL_DEVICE_TYPE_CPU_4)
endif()
if(OPENPGL_ISA_AVX2 OR OPENPGL_ISA_NEON2X)
  target_compile_definitions(${PROJECT_NAME} PRIVATE OPENPGL_DEVICE_TYPE_CPU_8)
endif()
if(OPENPGL_ISA_AVX512)
  target_compile_definitions(${PROJECT_NAME} PRIVATE OPENPGL_DEVICE_TYPE_CPU_16)
endif()

message(STATUS "Platform: ${CMAKE_SYSTEM_NAME}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "Arch: ${CMAKE_SYSTEM_PROCESSOR}")

if(WIN32)
  # Here we chack for MSVC, or Clang pretending to be MSVC via Clang-CL
  if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_SIMULATE_ID STREQUAL "MSVC"))
    set(OPENPGL_RELEASE_OPTIONS /Ox /Oi)  
    set(OPENPGL_COMMON_OPTIONS /fp:precise)
    #set(OPENPGL_RELEASE_OPTIONS ${OPENPGL_RELEASE_OPTIONS} -ftree-vectorize -mfpmath=sse -funsafe-math-optimizations -fno-rounding-math -fno-signaling-nans -fno-math-errno -fomit-frame-pointer )

    if(OPENPGL_ISA_SSE4 OR OPENPGL_ISA_AVX2 OR OPENPGL_ISA_AVX512)
      set(OPENPGL_COMMON_OPTIONS ${OPENPGL_COMMON_OPTIONS} /D__SSE__ /D__SSE2__ /D__SSE3__ /D__SSE4_1__ /D__SSE4_2__)
    endif()
    if(OPENPGL_ISA_SSE4)
      set_source_files_properties(api/deviceCPU4.cpp PROPERTIES COMPILE_FLAGS "/D__SSE__ /D__SSE2__ /D__SSE3__ /D__SSE4_1__ /D__SSE4_2__")
    endif()
    if(OPENPGL_ISA_AVX2)
      set_source_files_properties(api/deviceCPU8.cpp PROPERTIES COMPILE_FLAGS "/D__SSE__ /D__SSE2__ /D__SSE3__ /D__SSE4_1__ /D__SSE4_2__ /arch:AVX /arch:AVX2")
    endif()
    if(OPENPGL_ISA_AVX512)
      set_source_files_properties(api/deviceCPU16.cpp PROPERTIES COMPILE_FLAGS "/D__SSE__ /D__SSE2__ /D__SSE3__ /D__SSE4_1__ /D__SSE4_2__ /arch:AVX /arch:AVX2 /arch:AVX512")
    endif()
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
      if(OPENPGL_ISA_NEON)
       set_source_files_properties(api/deviceCPU4.cpp PROPERTIES COMPILE_FLAGS "/D__SSE4_2__  /D__SSE4_1__")
      endif()
      if(OPENPGL_ISA_NEON2X)
        set_source_files_properties(api/deviceCPU8.cpp PROPERTIES COMPILE_FLAGS "/D__AVX2__ /D__AVX__ /D__SSE4_2__  /D__SSE4_1__  /D__BMI__ /D__BMI2__ /D__LZCNT__")
      endif()
    endif()
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM" OR CMAKE_CXX_COMPILER_ID STREQUAL "dpcpp")
    set(OPENPGL_RELEASE_OPTIONS -O3)
    set(OPENPGL_COMMON_OPTIONS -Wall) 
    list(APPEND OPENPGL_COMMON_OPTIONS -ffp-model=precise)

    if(OPENPGL_ISA_SSE4 OR OPENPGL_ISA_AVX2 OR OPENPGL_ISA_AVX512)
      list(APPEND OPENPGL_COMMON_OPTIONS -msse2 -msse4.1)
    endif()

    if(OPENPGL_ISA_SSE4)
      set_source_files_properties(api/deviceCPU4.cpp PROPERTIES COMPILE_FLAGS "-msse2 -msse4.1")
    endif()
    if(OPENPGL_ISA_AVX2)
      set_source_files_properties(api/deviceCPU8.cpp PROPERTIES COMPILE_FLAGS "-msse2 -msse4.1 -mf16c -mavx2 -mfma -mlzcnt -mbmi -mbmi2")
    endif()
    if(OPENPGL_ISA_AVX512)
      set_source_files_properties(api/deviceCPU16.cpp PROPERTIES COMPILE_FLAGS "-msse2 -msse4.1 -mavx512f -mavx512dq -mavx512cd -mavx512bw -mavx512vl -mf16c -mavx2 -mfma -mlzcnt -mbmi -mbmi2")
    endif()
  endif()
else()
  set(OPENPGL_RELEASE_OPTIONS -O3)
  set(OPENPGL_COMMON_OPTIONS -Wall)                                         # enables most warnings
  list(APPEND OPENPGL_COMMON_OPTIONS -Wformat -Wformat-security)            # enables string format vulnerability warnings
  #SET(OPENPGL_RELEASE_OPTIONS ${OPENPGL_RELEASE_OPTIONS} -std=c++11)        # enables C++11 features
  list(APPEND OPENPGL_COMMON_OPTIONS -fvisibility=hidden)                   # makes all symbols hidden by default
  list(APPEND OPENPGL_COMMON_OPTIONS -fvisibility-inlines-hidden)           # makes all inline symbols hidden by default
  list(APPEND OPENPGL_COMMON_OPTIONS -fno-strict-aliasing)                  # disables strict aliasing rules
  list(APPEND OPENPGL_COMMON_OPTIONS -fno-tree-vectorize)                   # disable auto vectorizer

  if (OPENPGL_ARM)
    list(APPEND OPENPGL_COMMON_OPTIONS -fsigned-char)                       # treat 'char' as 'signed char'
    list(APPEND OPENPGL_COMMON_OPTIONS -flax-vector-conversions)            # allow lax vector type conversions
  endif()

  if (CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM" OR CMAKE_CXX_COMPILER_ID STREQUAL "dpcpp")
    list(APPEND OPENPGL_COMMON_OPTIONS -fsycl -fsycl-targets=spir64,nvptx64-nvidia-cuda -fdebug-info-for-profiling -gline-tables-only)                   # GPU support
    list(APPEND OPENPGL_COMMON_OPTIONS  -ffp-model=precise)
  endif()

  #set(OPENPGL_RELEASE_OPTIONS ${OPENPGL_RELEASE_OPTIONS} -fno-strict-aliasing -Wno-padded -ftree-vectorize -mfpmath=sse -funsafe-math-optimizations -fno-rounding-math -fno-signaling-nans -fno-math-errno -fomit-frame-pointer )
  #set(OPENPGL_RELEASE_OPTIONS ${OPENPGL_RELEASE_OPTIONS} -fno-strict-aliasing -Wno-padded -ftree-vectorize -mfpmath=sse)

  if(OPENPGL_ISA_SSE4 OR OPENPGL_ISA_AVX2 OR OPENPGL_ISA_AVX512)
    list(APPEND OPENPGL_COMMON_OPTIONS -msse2 -msse4.1)
  endif()
  if(OPENPGL_ISA_SSE4)
    set_source_files_properties(api/deviceCPU4.cpp PROPERTIES COMPILE_FLAGS "-msse2 -msse4.1")
  endif()
  if(OPENPGL_ISA_AVX2)
    set_source_files_properties(api/deviceCPU8.cpp PROPERTIES COMPILE_FLAGS "-msse2 -msse4.1 -mf16c -mavx2 -mfma -mlzcnt -mbmi -mbmi2")
  endif()
  if(OPENPGL_ISA_AVX512)
    set_source_files_properties(api/deviceCPU16.cpp PROPERTIES COMPILE_FLAGS "-msse2 -msse4.1 -mavx512f -mavx512dq -mavx512cd -mavx512bw -mavx512vl -mf16c -mavx2 -mfma -mlzcnt -mbmi -mbmi2")
  endif()
  if(OPENPGL_ISA_NEON)
    set_source_files_properties(api/deviceCPU4.cpp PROPERTIES COMPILE_FLAGS "-D__SSE4_2__  -D__SSE4_1__")
  endif()
  if(OPENPGL_ISA_NEON2X)
    set_source_files_properties(api/deviceCPU8.cpp PROPERTIES COMPILE_FLAGS "-D__AVX2__ -D__AVX__ -D__SSE4_2__  -D__SSE4_1__  -D__BMI__ -D__BMI2__ -D__LZCNT__")
  endif()
endif()

target_compile_options(${PROJECT_NAME} PRIVATE ${OPENPGL_COMMON_OPTIONS})
target_compile_options(${PROJECT_NAME} PRIVATE "$<$<CONFIG:Release>:${OPENPGL_RELEASE_OPTIONS}>")
target_compile_options(${PROJECT_NAME} PRIVATE "$<$<CONFIG:RelWithDebInfo>:${OPENPGL_RELEASE_OPTIONS}>")
target_compile_options(${PROJECT_NAME} PRIVATE "$<$<CONFIG:MinSizeRel>:${OPENPGL_RELEASE_OPTIONS}>")

# CMAKE 3.8 feature
#target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)

#message(STATUS "CMAKE_CURRENT_LIST_DIR = ${CMAKE_CURRENT_LIST_DIR}")
#message(STATUS "CMAKE_CURRENT_SOURCE_DIR = ${CMAKE_CURRENT_SOURCE_DIR}")
#message(STATUS "CMAKE_CURRENT_BINARY_DIR = ${CMAKE_CURRENT_BINARY_DIR}")

target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../third-party/>
    ${CMAKE_CURRENT_LIST_DIR}
    ${TBB_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} PRIVATE TBB::${OPENPGL_TBB_COMPONENT})
if (OPENPGL_EF_IMAGE_SPACE_GUIDING_BUFFER)
  target_include_directories(${PROJECT_NAME}
    PRIVATE
      OpenImageDenoise
      #ZLIB
  )
  target_link_libraries(${PROJECT_NAME} PRIVATE OpenImageDenoise)
  #target_link_libraries(${PROJECT_NAME} PRIVATE ZLIB::ZLIB)
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME ${OPENPGL_LIBRARY_NAME})
## Configure OpenPGL installation ##

openpgl_install_library(${PROJECT_NAME})

install(DIRECTORY
  ${PROJECT_SOURCE_DIR}/${PROJECT_NAME}/include/${PROJECT_NAME}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING
  PATTERN "*.h"
)

install(FILES
  ${PROJECT_SOURCE_DIR}/LICENSE.txt
  ${PROJECT_SOURCE_DIR}/third-party-programs.txt
  ${PROJECT_SOURCE_DIR}/third-party-programs-Embree.txt
  ${PROJECT_SOURCE_DIR}/third-party-programs-oneTBB.txt
  ${PROJECT_SOURCE_DIR}/third-party-programs-ISPC.txt
  ${PROJECT_SOURCE_DIR}/third-party-programs-OIDN.txt
  ${PROJECT_SOURCE_DIR}/README.md
  ${PROJECT_SOURCE_DIR}/CHANGELOG.md
  DESTINATION ${CMAKE_INSTALL_DOCDIR}/${PROJECT_NAME})

## Version header ##
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}/version.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/${PROJECT_NAME}/version.h
  @ONLY
)

## Defines header ##

if(OPENPGL_EF_RADIANCE_CACHES)
  set(OPENPGL_RADIANCE_CACHES ON)
endif()

if(OPENPGL_EF_IMAGE_SPACE_GUIDING_BUFFER)
  set(OPENPGL_IMAGE_SPACE_GUIDING_BUFFER ON)
endif()

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}/defines.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/${PROJECT_NAME}/defines.h
  @ONLY
)

target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/${PROJECT_NAME}>
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/${PROJECT_NAME}/version.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/${PROJECT_NAME}/defines.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
)

install(FILES ${OPENPGL_SOA_GENERATED}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}/gpu/
)

