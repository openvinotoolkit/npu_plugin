//
// Copyright 2020 Intel Corporation.
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

using namespace vpux;
using namespace mlir;

namespace vpux {
namespace VPUIP {

VPUIP::BlobWriter::SpecificTask SW_Kernel::serialize(vpux::VPUIP::BlobWriter& writer) {
    return writer.createSW_KernelTask(*this);
}

void SW_Kernel::build(mlir::OpBuilder& builder , mlir::OperationState& opState,
                      mlir::ValueRange inputs, mlir::ValueRange results, mlir::SymbolRefAttr kernelFunction,
                      mlir::IntegerAttr tileIndex) {
    // looks this is a result types
    build(builder, opState, results.getTypes(), kernelFunction, inputs, results, tileIndex, mlir::ValueRange{}, mlir::ValueRange{});
}


}  // namespace VPUIP
}  // namespace vpux