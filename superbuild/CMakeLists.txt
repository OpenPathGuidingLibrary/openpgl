## Copyright 2021 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

## Global settings ##

cmake_minimum_required(VERSION 3.5)

set(CMAKE_MODULE_PATH
  ${CMAKE_MODULE_PATH}
  ${CMAKE_CURRENT_SOURCE_DIR}/dependencies
)

if(NOT CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX
      "${CMAKE_BINARY_DIR}/install"
      CACHE STRING "Final install location." FORCE)
endif()

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

## ======================================================================== ##
## Project setup.
## ======================================================================== ##

project(OpenPGL_superbuild)

include(ExternalProject)
include(GNUInstallDirs)
include(ProcessorCount)

## Setup top-level variables ##

set(installDir ${CMAKE_INSTALL_PREFIX})

get_filename_component(INSTALL_DIR_ABSOLUTE
  ${installDir} ABSOLUTE BASE_DIR ${CMAKE_CURRENT_BINARY_DIR})

ProcessorCount(PROCESSOR_COUNT)

if(NOT PROCESSOR_COUNT EQUAL 0)
  set(BUILD_JOBS ${PROCESSOR_COUNT} CACHE STRING "Number of build jobs '-j <n>'")
else()
  set(BUILD_JOBS 4 CACHE STRING "Number of build jobs '-j <n>'")
endif()

if (${CMAKE_VERSION} VERSION_GREATER 3.11.4)
  set(PARALLEL_JOBS_OPTS -j ${BUILD_JOBS})
endif()

set(DEFAULT_BUILD_COMMAND ${CMAKE_COMMAND} --build . --config Release ${PARALLEL_JOBS_OPTS})

# We must replace separators so that ExternalProject_Add can work with the
# paths we generate.
macro(add_to_prefix_path PATH)
  list(APPEND CMAKE_PREFIX_PATH ${PATH})
  string(REPLACE ";" "|" CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}")
endmacro()

## ======================================================================== ##
## Superbuild options.
## ======================================================================== ##

option(BUILD_DEPENDENCIES_ONLY "Do not build Open PGL itself, only its dependencies" OFF)
option(BUILD_STATIC "Building Open PGL as a static library"  OFF)
option(BUILD_TOOLS "Building Open PGL tools as well"  OFF)

option(BUILD_TBB "Build Intel Threading Building Blocks or search in environment?" ON)
option(BUILD_TBB_FROM_SOURCE "Build Intel Threading Building Blocks from source?" OFF)
option(BUILD_TBB_SHARED "Build Intel Threading Building Blocks as shared library?" ON)

option(BUILD_OIDN "Build OpenImageDenoise as a part of the superbuild." ON)
option(BUILD_OIDN_FROM_SOURCE "Build OpenImageDenoise or use pre-built version?" ON)
option(DOWNLOAD_ISPC "Download ISPC or use the one found in the system environment?" ON)

## ======================================================================== ##
## Build dependencies.
##
## Use the BUILD_* options to enable or disable building in superbuild. If 
## These options are OFF, then required dependencies must be available in the 
## environment.
##
## For most of the dependencies, you may override the *_URL and *_HASH
## variables to select different versions.
## ======================================================================== ##


if (BUILD_TBB)
  set(TBB_VERSION "2021.13.0" CACHE STRING "TBB version to download")
  if(APPLE)
    set(BUILD_TBB_FROM_SOURCE ON)
  endif()
  if (BUILD_TBB_FROM_SOURCE)
    if (TBB_VERSION VERSION_LESS 2021)
      message(FATAL_ERROR "Only TBB 2021 and later are supported when building TBB from source")
    endif()

    string(REGEX REPLACE "(^[0-9]+\.[0-9]+\.[0-9]+$)" "v\\1" TBB_ARCHIVE ${TBB_VERSION})
    set(_TBB_URL "https://github.com/oneapi-src/oneTBB/archive/refs/tags/${TBB_ARCHIVE}.zip")
    set(_TBB_HASH "f8dba2602f61804938d40c24d8f9b1f1cc093cd003b24901d5c3cc75f3dbb952")
  else()
    if (TBB_VERSION VERSION_LESS 2021)
      set(TBB_BASE_URL "https://github.com/oneapi-src/oneTBB/releases/download/v${TBB_VERSION}/tbb-${TBB_VERSION}")
    else()
      set(TBB_BASE_URL "https://github.com/oneapi-src/oneTBB/releases/download/v${TBB_VERSION}/oneapi-tbb-${TBB_VERSION}")
    endif()
    if (APPLE)
      set(_TBB_URL "${TBB_BASE_URL}-mac.tgz")
      set(_TBB_HASH "e9aa143de91f65ea15b62ceeba5845bbc914c882d32bf9c9a405e6b8b5a70ade")
      set(TBB_LIB_SUBDIR "")
    elseif(WIN32)
      set(_TBB_URL "${TBB_BASE_URL}-win.zip")
      set(_TBB_HASH "0a7b19b52ec9d8eaa10cbbe396e955130815489c544642acecda8ef8d2d7f64d")
      set(TBB_LIB_SUBDIR "intel64/vc14")
    else()
      set(_TBB_URL "${TBB_BASE_URL}-lin.tgz")
      set(_TBB_HASH "f5c9304710051f0193a07fb91b6d6ada5a3e0a6d623951ee176b1897816ecf4c")
      set(TBB_LIB_SUBDIR "intel64/gcc4.8")
    endif()
  endif()
  set(TBB_URL "${_TBB_URL}" CACHE STRING "URL of the TBB archive.")
  set(TBB_HASH "${_TBB_HASH}" CACHE STRING "SHA256 hash of the TBB archive.")
  include(dep_tbb)
endif()

if (DOWNLOAD_ISPC AND BUILD_OIDN_FROM_SOURCE)
  set(ISPC_VERSION "1.24.0" CACHE STRING "Which version of ISPC to download?")
  mark_as_advanced(CLEAR ISPC_VERSION)
  if (ISPC_VERSION STREQUAL "1.24.0")
    if (APPLE)
      if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm64|aarch64")
        set(ISPC_HASH "7f3891d0157aed3cab159fbc5235235b62797053db9387f5a61c8d0a22369ae0")
      else()
        set(ISPC_HASH "986eb172fe9db3e8da560e9d0d788832991638fab61ca80587d87eb175ffb520")
      endif()
    elseif (WIN32)
      set(ISPC_HASH "a7c21cb2434f5364acbdf0933af6de49198458ed6f0b62012e03c3325c972649")
    else()
      if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm64|aarch64")
        set(ISPC_HASH "890ad5f31581091bf4ae9f235a7bb6fd9cabcbd6cf1670104ad61f1ad2da4c76")
      else()
        set(ISPC_HASH "79885d4482356dba7c01da312b6449fd575b097845ebcb90344dac5c178b43f5")
      endif()
    endif()
  endif()
  include(dep_ispc)
endif()

if (BUILD_OIDN)
  set(OIDN_VERSION "2.3.0" CACHE STRING "Which version of OpenImageDenoise to build?")
  mark_as_advanced(CLEAR OIDN_VERSION)
  mark_as_advanced(CLEAR BUILD_OIDN_FROM_SOURCE)
  if (OIDN_VERSION STREQUAL "2.3.0")
    if (BUILD_OIDN_FROM_SOURCE)
      set(OIDN_HASH "cce3010962ec84e0ba1acd8c9055a3d8de402fedb1b463517cfeb920a276e427")
    else()
      if (APPLE)
        if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm64|aarch64")
          set(OIDN_HASH "9dc42edba6461082e940c8c8cf19e6528d3d40f7f45c695fab33671b39414627")
        else()
          set(OIDN_HASH "8b3285d2d8c431c061a2cf0c5bf90fd519b5ad5fb6db828b07dc6662daa9e07b")
        endif()
      elseif (WIN32)
        set(OIDN_HASH "272781c9398eec7aeef9cd1be5b5aad7bb3debfc574308ce930b8e55ee51effd")
      else()
        set(OIDN_HASH "02693c3eb3888eeb0c59c9bb1142b28b899ac8dd02bad4397beaffb5f8bef863")
      endif()
    endif()
  endif()
  include(dep_oidn)
endif()

## ======================================================================== ##
## Build OpenPGL
## ======================================================================== ##

set(COMPONENT_NAME openpgl)

set(COMPONENT_PATH ${INSTALL_DIR_ABSOLUTE})

if (NOT BUILD_DEPENDENCIES_ONLY)

  set(OPENPGL_ARGS
    -DCMAKE_PREFIX_PATH:PATH=${CMAKE_PREFIX_PATH}
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_INSTALL_PREFIX=${COMPONENT_PATH}
    -DCMAKE_INSTALL_INCLUDEDIR=${CMAKE_INSTALL_INCLUDEDIR}
    -DCMAKE_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}
    -DCMAKE_INSTALL_DOCDIR=${CMAKE_INSTALL_DOCDIR}
    -DCMAKE_INSTALL_BINDIR=${CMAKE_INSTALL_BINDIR}
  )

  if(OPENPGL_EF_RADIANCE_CACHES)
    set(OPENPGL_ARGS ${OPENPGL_ARGS}
      -DOPENPGL_EF_RADIANCE_CACHES=${OPENPGL_EF_RADIANCE_CACHES}
    )
  endif()

  if(OPENPGL_EF_IMAGE_SPACE_GUIDING_BUFFER)
    set(OPENPGL_ARGS ${OPENPGL_ARGS}
      -DOPENPGL_EF_IMAGE_SPACE_GUIDING_BUFFER=${OPENPGL_EF_IMAGE_SPACE_GUIDING_BUFFER}
    )
  endif()

  if(OPENPGL_DIRECTION_COMPRESSION)
    set(OPENPGL_ARGS ${OPENPGL_ARGS}
      -DOPENPGL_DIRECTION_COMPRESSION=${OPENPGL_DIRECTION_COMPRESSION}
    )
  endif()

  if(OPENPGL_RADIANCE_COMPRESSION)
    set(OPENPGL_ARGS ${OPENPGL_ARGS}
      -DOPENPGL_RADIANCE_COMPRESSION=${OPENPGL_RADIANCE_COMPRESSION}
    )
  endif()

  if(BUILD_STATIC)
    set(OPENPGL_ARGS ${OPENPGL_ARGS}
      -DOPENPGL_BUILD_STATIC=${BUILD_STATIC}
    )
  endif()
  
  if(BUILD_TOOLS)
    set(OPENPGL_ARGS ${OPENPGL_ARGS}
      -DOPENPGL_BUILD_TOOLS=ON
    )
  endif()

if(OPENPGL_EF_RADIANCE_CACHES)
  set(OPENPGL_ARGS ${OPENPGL_ARGS}
    -DOPENPGL_EF_RADIANCE_CACHES=ON
  )
endif()

if(OPENPGL_EF_IMAGE_SPACE_GUIDING_BUFFER)
  set(OPENPGL_ARGS ${OPENPGL_ARGS}
    -DOPENPGL_EF_IMAGE_SPACE_GUIDING_BUFFER=ON
  )
endif()

if(OPENPGL_DIRECTION_COMPRESSION)
  set(OPENPGL_ARGS ${OPENPGL_ARGS}
    -DOPENPGL_DIRECTION_COMPRESSION=ON
  )
endif()

if(OPENPGL_RADIANCE_COMPRESSION)
  set(OPENPGL_ARGS ${OPENPGL_ARGS}
    -DOPENPGL_RADIANCE_COMPRESSION=ON
  )
endif()

  if(OPENPGL_TBB_ROOT)
    set(OPENPGL_ARGS ${OPENPGL_ARGS}
      -DTBB_ROOT=${OPENPGL_TBB_ROOT}
      -DOPENPGL_TBB_ROOT=${OPENPGL_TBB_ROOT}
    )
  else()
    set(OPENPGL_ARGS ${OPENPGL_ARGS}
      -DTBB_ROOT=${CMAKE_INSTALL_PREFIX}
      -DOPENPGL_TBB_ROOT=${CMAKE_INSTALL_PREFIX}
    )
  endif()

  ExternalProject_Add(ep_openpgl
    PREFIX openpgl
    DOWNLOAD_COMMAND ""
    STAMP_DIR openpgl/stamp
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..
    BINARY_DIR openpgl/build
    LIST_SEPARATOR | # CMAKE_PREFIX_PATH uses alternate separator.
    CMAKE_ARGS
      ${OPENPGL_ARGS}
    BUILD_COMMAND ${DEFAULT_BUILD_COMMAND}
    BUILD_ALWAYS ON
  )

  ExternalProject_Add_StepDependencies(ep_openpgl
  configure
    $<$<BOOL:${BUILD_TBB}>:dep_tbb>
    $<$<BOOL:${BUILD_OIDN}>:oidn>    
  )
endif()
