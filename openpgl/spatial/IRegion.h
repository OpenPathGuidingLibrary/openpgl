// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../data/SampleData.h"
#include "../openpgl_common.h"

namespace openpgl
{

struct IRegion
{
    virtual ~IRegion(){};

#ifdef OPENPGL_RADIANCE_CACHES
    virtual Vector3 getOutgoingRadiance(const Vector3 dir) const = 0;
#endif
   public:
    bool valid{true};
    bool initialized{false};
};
}  // namespace openpgl