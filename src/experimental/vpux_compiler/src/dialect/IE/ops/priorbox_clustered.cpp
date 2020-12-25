//
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

mlir::LogicalResult vpux::IE::PriorBoxClusteredOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueRange operands, mlir::DictionaryAttr attrs,
        mlir::RegionRange, SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::PriorBoxClusteredOpAdaptor priorBoxClustered(operands, attrs);
    if (mlir::failed(priorBoxClustered.verify(loc))) {
        return mlir::failure();
    }

    const auto numPriors = static_cast<int64_t>(priorBoxClustered.widths().size());

    auto outputSize = priorBoxClustered.output_size().getDefiningOp<mlir::ConstantOp>();
    if (outputSize == nullptr) {
        return mlir::failure();
    }

    const auto denseElementArray = outputSize.value().dyn_cast<mlir::DenseElementsAttr>();
    if (denseElementArray == nullptr) {
        return mlir::failure();
    }

    const auto elementsRange = denseElementArray.getValues<int64_t>();
    VPUX_THROW_UNLESS(denseElementArray.size() == 2, "output_size of priorbox should be 2");

    mlir::SmallVector<int64_t, 2> outShape{2, 4 * numPriors};
    outShape[1] *= *elementsRange.begin();
    outShape[1] *= *(elementsRange.begin() + 1);

    inferredReturnShapes.emplace_back(outShape, mlir::Float32Type::get(ctx));
    return mlir::success();
}
