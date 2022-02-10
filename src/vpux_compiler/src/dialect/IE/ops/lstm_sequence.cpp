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

using namespace vpux;

mlir::LogicalResult vpux::IE::LSTMSequenceOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::LSTMSequenceOpAdaptor lstm(operands, attrs);
    if (mlir::failed(lstm.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = lstm.initialHiddenState().getType().cast<mlir::ShapedType>();
    auto outHVShape = inType.getShape().vec();
    outHVShape.insert(outHVShape.cbegin() + 2, lstm.sequenceLength().getInt());

    inferredReturnShapes.emplace_back(outHVShape, inType.getElementType());         // outputHiddenValues
    inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType());  // outputHiddenState
    inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType());  // outputCellState

    return mlir::success();
}

std::unique_ptr<ngraph::Node> vpux::IE::LSTMSequenceOp::toNgraph(ngraph::OutputVector &outputs)
{
    const auto lstm_direction = direction();
    return std::make_unique<opset_latest::LSTMSequence>(outputs.at(0), outputs.at(1), outputs.at(2), outputs.at(3),
        outputs.at(4), outputs.at(5), outputs.at(6), sequenceLength(), exportRNNSequenceDirection(lstm_direction));
}
