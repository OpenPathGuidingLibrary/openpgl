## Copyright 2009-2020 Intel Corporation
## SPDX-License-Identifier: Apache-2.0


set(DEVICE_SRC)
if(OPENPGL_ISA_SSE4 OR OPENPGL_ISA_NEON)
  set(DEVICE_SRC ${DEVICE_SRC} api/deviceCPU4.cpp)
endif()
if(OPENPGL_ISA_AVX2 OR OPENPGL_ISA_NEON2X)
  set(DEVICE_SRC ${DEVICE_SRC} api/deviceCPU8.cpp)
endif()
if(OPENPGL_ISA_AVX512)
  set(DEVICE_SRC ${DEVICE_SRC} api/deviceCPU16.cpp)
endif()

set(OPENPGL_SRC
    api/api.cpp
    ${DEVICE_SRC}
    directional/ISurfaceSamplingDistribution.cpp
    directional/IVolumeSamplingDistribution.cpp
    directional/vmm/VMMPhaseFunctions.cpp
    ../third-party/embreeSrc/common/simd/sse.cpp
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


#target_compile_definitions(${PROJECT_NAME} PRIVATE OPENPGL_DEBUG_MODE)

target_compile_definitions(${PROJECT_NAME} PRIVATE OPENPGL_VERSION_MAJOR=${PROJECT_VERSION_MAJOR} OPENPGL_VERSION_MINOR=${PROJECT_VERSION_MINOR} OPENPGL_VERSION_PATCH=${PROJECT_VERSION_PATCH})
target_compile_definitions(${PROJECT_NAME} PRIVATE OPENPGL_VERSION_STRING="${PROJECT_VERSION}")
if(OPENPGL_ISA_SSE4 OR OPENPGL_ISA_NEON)
  target_compile_definitions(${PROJECT_NAME} PRIVATE OPENPGL_DEVICE_TYPE_CPU_4)
endif()
if(OPENPGL_ISA_AVX2 OR OPENPGL_ISA_NEON2X)
  target_compile_definitions(${PROJECT_NAME} PRIVATE OPENPGL_DEVICE_TYPE_CPU_8)
endif()
if(OPENPGL_ISA_AVX512)
  target_compile_definitions(${PROJECT_NAME} PRIVATE OPENPGL_DEVICE_TYPE_CPU_16)
endif()

if(WIN32)
  set(OPENPGL_RELEASE_OPTIONS /Ox /Oi)
  #set(OPENPGL_RELEASE_OPTIONS ${OPENPGL_RELEASE_OPTIONS} -ftree-vectorize -mfpmath=sse -funsafe-math-optimizations -fno-rounding-math -fno-signaling-nans -fno-math-errno -fomit-frame-pointer )

  if(OPENPGL_ISA_SSE4 OR OPENPGL_ISA_AVX2 OR OPENPGL_ISA_AVX512)
    set(OPENPGL_RELEASE_OPTIONS ${OPENPGL_RELEASE_OPTIONS} /D__SSE__ /D__SSE2__ /D__SSE3__ /D__SSE4_1__ /D__SSE4_2__)
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
else()
  set(OPENPGL_RELEASE_OPTIONS -O3 -Wall)
  #set(OPENPGL_RELEASE_OPTIONS ${OPENPGL_RELEASE_OPTIONS} -fno-strict-aliasing -Wno-padded -ftree-vectorize -mfpmath=sse -funsafe-math-optimizations -fno-rounding-math -fno-signaling-nans -fno-math-errno -fomit-frame-pointer )
  #set(OPENPGL_RELEASE_OPTIONS ${OPENPGL_RELEASE_OPTIONS} -fno-strict-aliasing -Wno-padded -ftree-vectorize -mfpmath=sse)

  if(OPENPGL_ISA_SSE4 OR OPENPGL_ISA_AVX2 OR OPENPGL_ISA_AVX512)
    set(OPENPGL_RELEASE_OPTIONS ${OPENPGL_RELEASE_OPTIONS} -msse2 -msse4.1)
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


if(OPENPGL_USE_OMP_THREADING)
  set(OPENPGL_RELEASE_OPTIONS -fopenmp)
endif()

target_compile_options(${PROJECT_NAME} PUBLIC "$<$<CONFIG:RELEASE>:${OPENPGL_RELEASE_OPTIONS}>")

# CMAKE 3.8 feature
#target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)

if(OPENPGL_USE_OMP_THREADING)
  add_compile_definitions(OPENPGL_USE_OMP_THREADING)
endif()

#add_compile_options()

if (NOT ${OPENPGL_TBB_ROOT} STREQUAL "")
    set(TBB_FIND_PACKAGE_OPTION "NO_DEFAULT_PATH")
    set(TBB_ROOT ${OPENPGL_TBB_ROOT})
    list(APPEND CMAKE_PREFIX_PATH ${OPENPGL_TBB_ROOT})
endif()

FIND_PACKAGE(TBB REQUIRED ${OPENPGL_TBB_COMPONENT})

FIND_PACKAGE(embree 3.6 REQUIRED)

get_target_property(embree_include_dir embree INTERFACE_INCLUDE_DIRECTORIES)

message(STATUS "CMAKE_CURRENT_LIST_DIR = ${CMAKE_CURRENT_LIST_DIR}")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR = ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "CMAKE_CURRENT_BINARY_DIR = ${CMAKE_CURRENT_BINARY_DIR}")

target_include_directories(${PROJECT_NAME}
  PUBLIC
    #$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../third-party/>
    #$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include/>

    #$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../third-party/>
    ${CMAKE_CURRENT_LIST_DIR}
    ${embree_include_dir}
    ${TBB_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} PRIVATE embree)
target_link_libraries(${PROJECT_NAME} PRIVATE TBB::${OPENPGL_TBB_COMPONENT})

## Configure OpenPGL installation ##

openpgl_install_library(${PROJECT_NAME})

install(DIRECTORY
  ${PROJECT_SOURCE_DIR}/${PROJECT_NAME}/include/${PROJECT_NAME}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING
  PATTERN "*.h"
  #PATTERN "*.isph"
)

install(FILES
  ${PROJECT_SOURCE_DIR}/LICENSE.txt
  ${PROJECT_SOURCE_DIR}/third-party-programs.txt
  ${PROJECT_SOURCE_DIR}/third-party-programs-Embree.txt
  ${PROJECT_SOURCE_DIR}/third-party-programs-TBB.txt
  ${PROJECT_SOURCE_DIR}/README.md
  ${PROJECT_SOURCE_DIR}/CHANGELOG.md
  DESTINATION ${CMAKE_INSTALL_DOCDIR}/${PROJECT_NAME})

## Version header ##

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}/version.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/${PROJECT_NAME}/version.h
  @ONLY
)

target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/${PROJECT_NAME}>
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/${PROJECT_NAME}/version.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
)
