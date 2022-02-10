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
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/IE/utils/to_ngraph.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::LessOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::LessOpAdaptor less(operands, attrs);
    if (mlir::failed(less.verify(loc))) {
        return mlir::failure();
    }

    const auto in1Type = less.input1().getType().cast<mlir::ShapedType>();
    const auto in2Type = less.input2().getType().cast<mlir::ShapedType>();

    const auto outShapeRes =
            IE::broadcastEltwiseShape(in1Type.getShape(), in2Type.getShape(), less.auto_broadcast().getValue(), loc);

    if (mlir::succeeded(outShapeRes)) {
        inferredReturnShapes.emplace_back(outShapeRes.getValue(), in1Type.getElementType());
    }

    return outShapeRes;
}

std::unique_ptr<ngraph::Node> vpux::IE::LessOp::toNgraph(ngraph::OutputVector &outputs)
{
    const ngraph::op::AutoBroadcastType autoBroadCastType = exportBroadcastType(auto_broadcast());

    return std::make_unique<opset_latest::Less>(outputs.at(0), outputs.at(1),
        ngraph::op::AutoBroadcastSpec(autoBroadCastType));
}
