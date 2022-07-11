## Copyright 2009-2020 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

## Global CMake options ##

cmake_minimum_required(VERSION 3.1)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(GNUInstallDirs)

## Establish project ##

project(openpgl VERSION 0.3.1 LANGUAGES C CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)
include(openpgl_macros)

## openpgl specific configuration ##

option(OPENPGL_BUILD_PYTHON "Build python wrapper 'pyopenpgl'."  OFF)
option(OPENPGL_USE_OMP_THREADING "Use omp instead of tbb for threading." OFF)
option(OPENPGL_BUILD_STATIC "Build OpenPGL as static library." OFF)

try_compile(COMPILER_SUPPORTS_ARM_NEON "${CMAKE_BINARY_DIR}" "${PROJECT_SOURCE_DIR}/cmake/check_arm_neon.cpp")

if(COMPILER_SUPPORTS_ARM_NEON)
  option(OPENPGL_ISA_NEON "Build with support for NEON." ON)
  option(OPENPGL_ISA_NEON2X "Build with support for NEON double pumped." ON)
else()
  option(OPENPGL_ISA_SSE4 "Build with support for SSE4." ON)
  option(OPENPGL_ISA_AVX2 "Build with support for AVX2." ON)
  option(OPENPGL_ISA_AVX512 "Build with support for AVX512." OFF)
endif()

if(NOT (OPENPGL_ISA_NEON OR
        OPENPGL_ISA_NEON2X OR
        OPENPGL_ISA_SSE4 OR
        OPENPGL_ISA_AVX2 OR
        OPENPGL_ISA_AVX512))
  message(FATAL_ERROR "Must have at least one ISA enabled.")
endif()

SET(OPENPGL_TBB_COMPONENT "tbb" CACHE STRING "The TBB component/library name.")

find_package(embree 3.6.1 REQUIRED)

openpgl_create_embree_target()

## Build library and  ##
add_subdirectory(openpgl)

if (BUILD_PYTHON)
    add_subdirectory(python)
endif()

## Configure CMake find_package() config files ##

include(CMakePackageConfigHelpers)

configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
  "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rkcommon
)

write_basic_package_version_file(
  "${PROJECT_NAME}ConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
DESTINATION
  ${CMAKE_INSTALL_LIBDIR}/cmake/openpgl-${PROJECT_VERSION}
)

# Must be last
include(CPack)
