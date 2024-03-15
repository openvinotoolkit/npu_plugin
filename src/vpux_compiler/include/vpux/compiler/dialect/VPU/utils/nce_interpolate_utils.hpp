//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

namespace vpux::VPU {

VPU::NCEInterpolateModeAttr getNCEInterpolateModeAttr(IE::InterpolateModeAttr origModeAttr);

bool isSupportedNCEInterpolateScales(ArrayRef<double> scales, vpux::LogCb logCb = emptyLogCb);
std::optional<SmallVector<double>> getNCEInterpolateScales(NDTypeInterface inputType, NDTypeInterface outputType,
                                                           IE::InterpolateCoordModeAttr coordModeAttr);

SmallVector<int64_t> getNCEInterpolateFactors(ArrayRef<double> scales, VPU::NCEInterpolateModeAttr modeAttr,
                                              IE::InterpolateCoordModeAttr coordModeAttr);
SmallVector<int64_t> getNCEInterpolatePadsBegin(ArrayRef<double> scales, VPU::NCEInterpolateModeAttr modeAttr,
                                                IE::InterpolateCoordModeAttr coordModeAttr);
SmallVector<int64_t> getNCEInterpolatePadsEnd(ArrayRef<double> scales, VPU::NCEInterpolateModeAttr modeAttr,
                                              IE::InterpolateCoordModeAttr coordModeAttr);
SmallVector<int64_t> getNCEInterpolateKernelSize(ArrayRef<double> scales, VPU::NCEInterpolateModeAttr modeAttr,
                                                 IE::InterpolateCoordModeAttr coordModeAttr);
SmallVector<int64_t> getNCEInterpolateStrides(ArrayRef<double> scales, VPU::NCEInterpolateModeAttr modeAttr,
                                              IE::InterpolateCoordModeAttr coordModeAttr);

}  // namespace vpux::VPU
