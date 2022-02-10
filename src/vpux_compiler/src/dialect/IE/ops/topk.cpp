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
#include "vpux/compiler/dialect/IE/utils/to_ngraph.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::TopKOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::TopKOpAdaptor topK(operands, attrs);
    if (mlir::failed(topK.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = topK.input().getType().cast<mlir::ShapedType>();
    const auto inputShape = inType.getShape();

    auto kConst = topK.k().getDefiningOp<Const::DeclareOp>();
    if (kConst == nullptr) {
        return errorAt(loc, "Only constant input is supported for k");
    }

    const auto kContent = kConst.content();
    if (!kContent.isSplat()) {
        return errorAt(loc, "K input must be scalar");
    }

    SmallVector<int64_t> outShape;
    for (size_t i = 0; i < inputShape.size(); ++i) {
        outShape.push_back(inputShape[i]);
    }
    int64_t axis = topK.axis().getInt();
    const auto inRank = inType.getRank();
    if (axis < 0) {
        axis += inRank;
    }
    outShape[axis] = kContent.getSplatValue<int64_t>();

    inferredReturnShapes.emplace_back(outShape, inType.getElementType());
    inferredReturnShapes.emplace_back(outShape, topK.element_type().getValue());

    return mlir::success();
}

std::unique_ptr<ngraph::Node> vpux::IE::TopKOp::toNgraph(ngraph::OutputVector &outputs)
{
    const mlir::Type elType = element_type();
    mlir::MLIRContext* ctx = elType.getContext();

    return std::make_unique<opset_latest::TopK>(outputs.at(0), outputs.at(1), axis(), exportTopKMode(mode()),
        exportTopKSortType(sort()), exportElemType(ctx, elType));
}
