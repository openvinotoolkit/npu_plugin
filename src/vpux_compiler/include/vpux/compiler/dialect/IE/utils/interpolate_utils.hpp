//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/ops.hpp"

namespace vpux {
namespace IE {

mlir::FailureOr<SmallVector<int64_t>> extractIntVector(mlir::Location loc, const mlir::Value value,
                                                       const Optional<mlir::ArrayAttr>& attr);
mlir::FailureOr<SmallVector<double>> extractFPVector(mlir::Location loc, const mlir::Value value,
                                                     const Optional<mlir::ArrayAttr>& attr);

void applyInterpPads(MutableArrayRef<int64_t> outShape, ArrayRef<int64_t> padsBegin, ArrayRef<int64_t> padsEnd);
mlir::FailureOr<SmallVector<int64_t>> propagateShape(mlir::Location loc, mlir::FailureOr<SmallVector<int64_t>> axes,
                                                     ArrayRef<int64_t> origShape,
                                                     mlir::FailureOr<ArrayRef<int64_t>> padsBegin,
                                                     mlir::FailureOr<ArrayRef<int64_t>> padsEnd,
                                                     vpux::IE::InterpolateCalcMode calcMode,
                                                     mlir::FailureOr<ArrayRef<int64_t>> sizes,
                                                     mlir::FailureOr<ArrayRef<double>> scales, vpux::Logger log);

SmallVector<int64_t> getDefaultInterpolateAxes(IE::InterpolateOpAdaptor interpolate);
mlir::FailureOr<SmallVector<int64_t>> calcOutputShapes(mlir::Location loc, IE::InterpolateOpAdaptor interpolate,
                                                       vpux::Logger log);

// For some specific Interpolate that shape size of scaling axis is always equal 1.
// Those Interpolate can be considered as Broadcast. For example:
// - InterpolateMode: Any; InterpolateCoordMode: Any; Pad_Begin and Pad_End is null or zero
// - Input shape: 1x16x1x1; Output shape: 1x16x32x32; Scales: [1, 1, 32, 32]
// It can be convert to NEAREST with ASYMMETRIC mode that will benefit from SEP feature.
bool isBroadCastInterpolate(IE::InterpolateOp op);

}  // namespace IE
}  // namespace vpux
