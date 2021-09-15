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
#include "vpux/compiler/utils/types.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IERT::ImplicitReorderOp::inferReturnTypes(
        mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> loc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange /*regions*/, mlir::SmallVectorImpl<mlir::Type>& inferredTypes) {
    const auto location = loc.hasValue() ? loc.getValue() : mlir::UnknownLoc::get(ctx);
    IERT::ImplicitReorderOpAdaptor reorderOp(operands, attrs);
    if (mlir::failed(reorderOp.verify(location))) {
        return errorAt(location, "IERT::ImplicitReorderOp verification failed");
    }

    const auto order = DimsOrder::fromPermutationAffineMap(reorderOp.dstOrder().getValue());
    const auto origType = reorderOp.source().getType().dyn_cast<mlir::ShapedType>();
    auto resultType = changeDimsOrder(origType, order);
    inferredTypes.emplace_back(resultType);

    return mlir::success();
}

mlir::Value vpux::IERT::ImplicitReorderOp::getViewSource() {
    return source();
}
