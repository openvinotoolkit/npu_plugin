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

using namespace vpux;

namespace {

MVCNN::RoundMode converVPUXRoundModeToMVCNN(vpux::IE::RoundMode vpux_mode) {
    MVCNN::RoundMode mvcnn_mode;
    switch (vpux_mode) {
    case IE::RoundMode::HALF_TO_EVEN:
        mvcnn_mode = MVCNN::RoundMode::RoundMode_HALF_TO_EVEN;
        break;
    case IE::RoundMode::HALF_AWAY_FROM_ZERO:
        mvcnn_mode = MVCNN::RoundMode::RoundMode_HALF_AWAY_FROM_ZERO;
        break;
    default:
        VPUX_THROW("Unsupported RoundMode {0}", vpux_mode);
    }
    return mvcnn_mode;
}

}  // namespace

mlir::LogicalResult vpux::IE::RoundOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::RoundOpAdaptor round(operands, attrs);
    if (mlir::failed(round.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = round.input().getType().cast<mlir::ShapedType>();
    inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType());

    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::IE::RoundOp::serialize(EMU::BlobWriter& writer) {
    const auto roundMode = converVPUXRoundModeToMVCNN(mode());
    const auto round = MVCNN::CreateRoundParams(writer, roundMode);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_RoundParams);
    builder.add_nested_params(round.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
}
