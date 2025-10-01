## Copyright 2009-2020 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

## Global CMake options ##

cmake_minimum_required(VERSION 3.5)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

enable_language(CXX)
include(GNUInstallDirs)

## Establish project ##

project(openpgl VERSION 0.7.1 LANGUAGES C CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)
include(openpgl_macros)

add_compile_options(
  -frounding-math
  -ftrapping-math
  -fno-fast-math
)

##############################################################
# Platform detection and defaults
##############################################################
SET(OPENPGL_ARM OFF)
IF (APPLE AND CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND (CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64" OR CMAKE_OSX_ARCHITECTURES MATCHES "arm64"))
  MESSAGE(STATUS "Building for Apple silicon")
  SET(OPENPGL_ARM ON)
ELSEIF(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "ARM64")
  MESSAGE(STATUS "Building for AArch64")
  SET(OPENPGL_ARM ON)
ENDIF()

##############################################################
# Open PGL specific configuration 
##############################################################

option(OPENPGL_BUILD_STATIC "Build OpenPGL as static library." ON)

option(OPENPGL_BUILD_TOOLS "Build tool applications." OFF)
option(OPENPGL_BUILD_CHECK_TOOL "Build check tool application." OFF)

try_compile(COMPILER_SUPPORTS_ARM_NEON "${CMAKE_BINARY_DIR}" "${PROJECT_SOURCE_DIR}/cmake/check_arm_neon.cpp")

OPTION(OPENPGL_EF_RADIANCE_CACHES "Enables experimental feature (ir)radiance caches." OFF)
OPTION(OPENPGL_EF_IMAGE_SPACE_GUIDING_BUFFER "Enables experimental feature ImageSpaceGuidignBuffer." OFF)

OPTION(OPENPGL_DIRECTION_COMPRESSION "Using 32-Bit compression to represent directions." OFF)
OPTION(OPENPGL_RADIANCE_COMPRESSION "Using RGBE 32-Bit compression to represent radiance data (linear RGB)." OFF)

option(OPENPGL_SYCL_SUPPORT "Build OpenPGL with SYCL support." OFF)

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

SET(OPENPGL_LIBRARY_NAME  "openpgl" CACHE STRING "Name of the Open PGL library file (default is openpgl)")

SET(OPENPGL_TBB_COMPONENT "tbb" CACHE STRING "The TBB component/library name.")


if (NOT ${OPENPGL_TBB_ROOT} STREQUAL "")
    set(TBB_FIND_PACKAGE_OPTION "NO_DEFAULT_PATH")
    set(TBB_ROOT ${OPENPGL_TBB_ROOT})
    list(APPEND CMAKE_PREFIX_PATH ${OPENPGL_TBB_ROOT})
endif()

FIND_PACKAGE(TBB REQUIRED ${OPENPGL_TBB_COMPONENT})

if (OPENPGL_EF_IMAGE_SPACE_GUIDING_BUFFER)
  FIND_PACKAGE(OpenImageDenoise 2.2.0 REQUIRED)
  #FIND_PACKAGE(ZLIB REQUIRED)
endif()

FIND_PACKAGE(CUDAToolkit)

## Global compiler setup
if(WIN32)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /DEPENDENTLOADFLAG:0x2000")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /DEPENDENTLOADFLAG:0x2000")
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /Qoption,link,/DEPENDENTLOADFLAG:0x2000")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /Qoption,link,/DEPENDENTLOADFLAG:0x2000")
  endif()
endif(WIN32)

## Build library and  ##
add_subdirectory(openpgl)

add_subdirectory(tools)

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
