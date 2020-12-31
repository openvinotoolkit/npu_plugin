
// Copyright 2020 Intel Corporation.
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
