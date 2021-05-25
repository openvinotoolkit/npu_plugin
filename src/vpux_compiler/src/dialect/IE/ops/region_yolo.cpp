
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

mlir::LogicalResult vpux::IE::RegionYoloOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueRange operands, mlir::DictionaryAttr attrs,
        mlir::RegionRange, SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::RegionYoloOpAdaptor regionYolo(operands, attrs);
    if (mlir::failed(regionYolo.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = regionYolo.input().getType().cast<mlir::ShapedType>();

    SmallVector<int64_t> outputShape;
    if (regionYolo.do_softmax().getValue()) {
        for (int64_t i = 0; i < regionYolo.axis().getInt(); i++) {
            outputShape.push_back(inType.getShape()[i]);
        }

        size_t flat_dim = 1;
        for (int64_t i = regionYolo.axis().getInt(); i < regionYolo.end_axis().getInt() + 1; i++) {
            flat_dim *= inType.getShape()[i];
        }
        outputShape.push_back(flat_dim);

        for (size_t i = regionYolo.end_axis().getInt() + 1; i < inType.getShape().size(); i++) {
            outputShape.push_back(inType.getShape()[i]);
        }
    } else {
        outputShape.push_back(inType.getShape()[0]);
        outputShape.push_back((regionYolo.classes().getInt() + regionYolo.coord().getInt() + 1) *
                              (int64_t)regionYolo.mask().size());
        outputShape.push_back(inType.getShape()[2]);
        outputShape.push_back(inType.getShape()[3]);
    }

    inferredReturnShapes.emplace_back(outputShape, inType.getElementType());
    return mlir::success();
}
