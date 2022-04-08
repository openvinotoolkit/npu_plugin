//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <llvm/ADT/Optional.h>
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/utils/core/error.hpp"

namespace vpux {
namespace VPU {

VPU::M2iColorFmt IEtoM2iColorFmt(IE::ColorFmt fmt);
long getM2iLineStride(NDTypeInterface ndType, size_t dimW);
bool isM2iLineStrideSupported(long lineStride);

}  // namespace VPU
}  // namespace vpux
