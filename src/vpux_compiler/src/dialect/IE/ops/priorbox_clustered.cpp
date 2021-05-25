//
// Copyright 2020 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::PriorBoxClusteredOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueRange operands, mlir::DictionaryAttr attrs,
        mlir::RegionRange, SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::PriorBoxClusteredOpAdaptor priorBoxClustered(operands, attrs);
    if (mlir::failed(priorBoxClustered.verify(loc))) {
        return mlir::failure();
    }

    const auto numPriors = static_cast<int64_t>(priorBoxClustered.widths().size());

    auto outputSizeConst = priorBoxClustered.output_size().getDefiningOp<ConstantInterface>();
    if (outputSizeConst == nullptr) {
        return errorAt(loc, "Only constant input is supported for output_size");
    }

    const auto outputSize = outputSizeConst.getContent().getValues<int64_t>();
    if (outputSize.size() != 2) {
        return errorAt(loc, "output_size of priorbox should be 2");
    }

    SmallVector<int64_t> outShape{2, 4 * numPriors};
    outShape[1] *= outputSize[0];
    outShape[1] *= outputSize[1];

    inferredReturnShapes.emplace_back(outShape, mlir::Float32Type::get(ctx));
    return mlir::success();
}
