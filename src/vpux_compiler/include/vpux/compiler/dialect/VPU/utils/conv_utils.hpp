//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/core/type_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/utils/logging.hpp"

namespace vpux {
namespace VPU {

bool isNCEConvSupported(VPU::ArchKind arch, NDTypeInterface inputType, NDTypeInterface filterType,
                        NDTypeInterface outputType, ArrayRef<int64_t> dilations, int64_t KY, int64_t KX, int64_t SY,
                        int64_t SX, PadInfo pads, bool checkLayout, bool checkChannelAlignment, LogCb logCb);

}  // namespace VPU
}  // namespace vpux
