//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

#include "vpux/compiler/dialect/VPU/utils/type_infer.hpp"
#include "vpux/compiler/utils/attributes.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::ReduceL1Op::inferReturnTypes(mlir::MLIRContext* ctx,
                                                            std::optional<mlir::Location> optLoc,
                                                            mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                            mlir::OpaqueProperties, mlir::RegionRange /*regions*/,
                                                            mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::ReduceL1OpAdaptor reduceL1(operands, attrs);
    if (mlir::failed(reduceL1.verify(loc))) {
        return mlir::failure();
    }

    const auto input = reduceL1.getInput();
    const auto keepDims = reduceL1.getKeepDims();

    auto axesValue = parseIntArrayAttr<int64_t>(reduceL1.getAxesValue());

    return VPU::inferReduceReturnTypes(loc, input, keepDims, axesValue, inferredReturnTypes);
}

//
// fold
//

mlir::OpFoldResult vpux::VPU::ReduceL1Op::fold(FoldAdaptor) {
    if (getInput().getType() == getOutput().getType()) {
        return getInput();
    }

    return nullptr;
}
