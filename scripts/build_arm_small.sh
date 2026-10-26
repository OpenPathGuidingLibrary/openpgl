#!/bin/bash
## Copyright 2019-2020 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

mkdir build
cd build

cmake --version

cmake \
  -G Ninja \
  -DCMAKE_INSTALL_LIBDIR=lib \
  -DBUILD_OIDN=OFF \
  -DDOWNLOAD_ISPC=OFF \
  -DBUILD_TOOLS=ON \
  -DBUILD_TBB_FROM_SOURCE=ON \
  -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
  "$@" ../superbuild

cmake --build .
