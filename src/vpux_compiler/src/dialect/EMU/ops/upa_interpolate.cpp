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

#include "vpux/compiler/dialect/EMU/ops.hpp"
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

