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

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

//
// build
//

void vpux::IERT::ViewOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                               mlir::MemRefType new_type) {
    build(builder, state, input, new_type);
}

//
// ViewLikeOpInterface
//

mlir::Value vpux::IERT::ViewOp::getViewSource() {
    return source();
}

//
// InferTypeOpInterface
//

mlir::LogicalResult vpux::IERT::ViewOp::inferReturnTypes(mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc,
                                                         mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                         mlir::RegionRange /*regions*/,
                                                         mlir::SmallVectorImpl<mlir::Type>& inferredTypes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IERT::ViewOpAdaptor viewOp(operands, attrs);
    if (mlir::failed(viewOp.verify(loc))) {
        return mlir::failure();
    }

    const auto origType = viewOp.source().getType().dyn_cast<mlir::MemRefType>();
    if (origType == nullptr) {
        return errorAt(loc, "IERT::ViewOp operand must have MemRef type");
    }

    auto newType = viewOp.new_type().getValue();

    /*mlir::MemRefType::Builder memRefBuilder(origType);
    auto elems = origType.getNumElements();
    elems = elems / (getElemTypeSize(newElemType).count()/8);
    llvm::ArrayRef<int64_t> newShape = makeArrayRef(elems);
    memRefBuilder.setShape(newShape);
    memRefBuilder.setElementType(newElemType);
    mlir::MemRefType newType = memRefBuilder;*/

    // auto newType = newType; // .dyn_cast<mlir::MemRefType>()

    inferredTypes.push_back(newType);

    return mlir::success();
}

//
// fold
//

/*mlir::OpFoldResult vpux::IERT::ViewOp::fold(ArrayRef<mlir::Attribute> operands) {
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
*/
