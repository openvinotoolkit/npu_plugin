//
// Copyright 2021 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include "vpux/compiler/dialect/IE/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::CTCGreedyDecoderSeqLenOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueRange operands, mlir::DictionaryAttr attrs,
        mlir::RegionRange, SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::CTCGreedyDecoderSeqLenOpAdaptor ctc(operands, attrs);
    if (mlir::failed(ctc.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = ctc.input().getType().cast<mlir::ShapedType>();
    const auto inShape = inType.getShape();

    if (inShape.size() != 3) {
        return errorAt(loc, "First input tensor should have 3 dimensions");
    }

    const auto outElemType = ctc.sequenceLength().getType().cast<mlir::ShapedType>().getElementType();

    SmallVector<int64_t> outputShape{inShape[0], inShape[1]};
    SmallVector<int64_t> outputLengthShape{inShape[0]};

    inferredReturnShapes.emplace_back(outputShape, outElemType);
    inferredReturnShapes.emplace_back(outputLengthShape, outElemType);

    return mlir::success();
}
