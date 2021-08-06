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

#include "vpux/compiler/dialect/IERT/ops.hpp"

#include "vpux/compiler/core/attributes/strides.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::Value vpux::IERT::SubViewOp::getViewSource() {
    return source();
}

mlir::LogicalResult vpux::IERT::SubViewOp::inferReturnTypes(mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> loc,
                                                            mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                            mlir::RegionRange /*regions*/,
                                                            mlir::SmallVectorImpl<mlir::Type>& inferredTypes) {
    const auto location = loc.hasValue() ? loc.getValue() : mlir::UnknownLoc::get(ctx);
    IERT::SubViewOpAdaptor subViewOp(operands, attrs);
    if (mlir::failed(subViewOp.verify(location))) {
        return errorAt(location, "IERT::SubViewOp op verification failed");
    }

    const auto origType = subViewOp.source().getType().dyn_cast<mlir::MemRefType>();
    if (origType == nullptr) {
        return errorAt(location, "IERT::SubViewOp operand must have memref type");
    }

    const auto tileOffsetsAttr = subViewOp.static_offsets();
    if (tileOffsetsAttr == nullptr) {
        return errorAt(location, "IERT::SubViewOp needs static_offsets attribute");
    }

    const auto tileShapeAttr = subViewOp.static_sizes();
    if (tileShapeAttr == nullptr) {
        return errorAt(location, "IERT::SubViewOp needs static_sizes attribute");
    }

    const auto tileShape = parseIntArrayAttr<int64_t>(tileShapeAttr.cast<mlir::ArrayAttr>());
    const auto tileOffsets = parseIntArrayAttr<int64_t>(tileOffsetsAttr.cast<mlir::ArrayAttr>());

    const auto tileType = getTileType(origType, Shape(tileShape), Shape(tileOffsets));
    inferredTypes.push_back(tileType);

    return mlir::success();
}

//
// fold
//

mlir::OpFoldResult vpux::IERT::SubViewOp::fold(ArrayRef<mlir::Attribute> operands) {
    if (source().getType() == result().getType()) {
        return source();
    }

    if (const auto origContent = operands[0].dyn_cast_or_null<Const::ContentAttr>()) {
        const auto offset = Shape(parseIntArrayAttr<int64_t>(static_offsets()));
        const auto shape = Shape(parseIntArrayAttr<int64_t>(static_sizes()));
        return origContent.subview(offset, shape);
    }

    return nullptr;
}
