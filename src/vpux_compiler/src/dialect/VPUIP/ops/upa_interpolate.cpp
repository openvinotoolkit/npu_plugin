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

#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/utils/subspaces.hpp"

#include "vpux/utils/core/enums.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

namespace {

const EnumMap<IE::InterpolateMode, int> supportedInterpModeMap = {
        {IE::InterpolateMode::nearest, 0},      //
        {IE::InterpolateMode::linear, 1},       //
        {IE::InterpolateMode::linear_onnx, 3},  //
};

const EnumMap<IE::InterpolateNearestMode, int> nearestModeMap = {
        {IE::InterpolateNearestMode::round_prefer_floor, 0},  //
        {IE::InterpolateNearestMode::round_prefer_ceil, 1},   //
        {IE::InterpolateNearestMode::floor, 2},               //
        {IE::InterpolateNearestMode::ceil, 3},                //
        {IE::InterpolateNearestMode::simple, 4},              //
};

const EnumMap<IE::InterpolateCoordMode, int> coordTransformModeMap = {
        {IE::InterpolateCoordMode::half_pixel, 0},            //
        {IE::InterpolateCoordMode::pytorch_half_pixel, 1},    //
        {IE::InterpolateCoordMode::asymmetric, 2},            //
        {IE::InterpolateCoordMode::tf_half_pixel_for_nn, 3},  //
        {IE::InterpolateCoordMode::align_corners, 4},         //
};

}  // namespace

void vpux::VPUIP::InterpolateUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                          mlir::Value output, vpux::IE::InterpolateModeAttr mode,
                                          vpux::IE::InterpolateCoordModeAttr coord_mode,
                                          vpux::IE::InterpolateNearestModeAttr nearest_mode, mlir::UnitAttr antialias) {
    build(builder, state, input, output, mlir::ValueRange{}, mlir::ValueRange{}, mode, coord_mode, nearest_mode,
          antialias, nullptr, nullptr);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::InterpolateUPAOp::serialize(VPUIP::BlobWriter& writer) {
    MVCNN::InterpolateParamsBuilder builder(writer);

    const auto interpolateModeIter = supportedInterpModeMap.find(mode());
    VPUX_THROW_UNLESS(interpolateModeIter != supportedInterpModeMap.end(), "Unsupported interpolate mode {0}", mode());
    builder.add_interpolationMode(interpolateModeIter->second);

    const auto coordModeIter = coordTransformModeMap.find(coord_mode());
    VPUX_THROW_UNLESS(coordModeIter != coordTransformModeMap.end(), "Unsupported coordinate transformation mode {0}",
                      coord_mode());
    builder.add_coordTransformMode(coordModeIter->second);

    const auto nearestModeIter = nearestModeMap.find(nearest_mode());
    VPUX_THROW_UNLESS(nearestModeIter != nearestModeMap.end(), "Unsupported nearest mode {0}", nearest_mode());
    builder.add_nearestMode(nearestModeIter->second);

    builder.add_align_corners(coord_mode() == IE::InterpolateCoordMode::align_corners);
    builder.add_antialias(antialias());

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_InterpolateParams});
}
