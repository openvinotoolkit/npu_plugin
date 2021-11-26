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

#include "vpux/compiler/core/attributes/dim.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::EMU::verifyOp(ConvolutionUPAOp op) {
    // There are two uPA tasks which perform convolution: ConvUPA and SWConvUPA.
    // SWConvUPA supports parallel execution on multiple uPA units.
    // However, it does not have group support, so group convolutions go to ConvUPA.
    // ConvUPA expects NCHW order, SWConvUPA expects YXOI.
    const auto expectedFilterLayout = (op.groups() > 1) ? DimsOrder::OIYX : DimsOrder::YXOI;
    const auto filterLayout = DimsOrder::fromValue(op.filter());

    if (filterLayout != expectedFilterLayout) {
        return errorAt(op, "filter layout must be {0}, got {1}", expectedFilterLayout, filterLayout);
    }

    return mlir::success();
}
