//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/attributes/dim.hpp"
#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"

namespace vpux {

std::optional<MemDimArr> deduceLegalOutputMemDims(MemShapeRef inMemShape, MemShapeRef outMemShape, MemDim inMemDim);

}  // namespace vpux
