//
// Copyright Intel Corporation.
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
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/quantization.hpp"

#include "vpux/utils/core/range.hpp"

#include <ngraph/coordinate.hpp>

using namespace vpux;

mlir::LogicalResult vpux::IE::SliceOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::SliceOpAdaptor slice(operands, attrs);
    if (mlir::failed(slice.verify(loc))) {
        return mlir::failure();
    }

    const auto sizes = parseIntArrayAttr<int64_t>(slice.static_sizes());
    const auto inType = slice.source().getType().cast<mlir::ShapedType>();

    auto elemType = inType.getElementType();
    if (const auto perAxisQType = elemType.dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>()) {
        const Shape offsets(sizes.size(), 0);
        elemType = tileScalesAndZP(perAxisQType, ShapeRef{sizes}, offsets);
    }

    inferredReturnShapes.emplace_back(sizes, elemType);

    return mlir::success();
}
