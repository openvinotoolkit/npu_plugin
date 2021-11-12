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

VPUIP::BlobWriter::SpecificTask SW_KernelOp::serialize(vpux::VPUIP::BlobWriter& writer) {
    return writer.createSW_KernelTask(*this);
}

void SW_KernelOp::build(mlir::OpBuilder& builder, mlir::OperationState& opState, mlir::ValueRange inputs,
                        mlir::ValueRange results, mlir::SymbolRefAttr kernelFunction, mlir::IntegerAttr tileIndex) {
    // looks this is a result types
    build(builder, opState, results.getTypes(), kernelFunction, inputs, results, tileIndex, mlir::ValueRange{},
          mlir::ValueRange{});
}

mlir::LogicalResult SW_KernelOp::inferReturnTypes(mlir::MLIRContext* /*ctx*/, mlir::Optional<mlir::Location> /*optLoc*/,
                                                  mlir::ValueRange operands, mlir::DictionaryAttr /*attrs*/,
                                                  mlir::RegionRange /*regions*/,
                                                  mlir::SmallVectorImpl<mlir::Type>& inferredTypes) {
    auto firstOperandType = operands[0].getType();
    for (auto&& operand : operands) {
        VPUX_THROW_UNLESS(firstOperandType == operand.getType(),
                          "operands of different type not yet suported: {0} vs {1}", firstOperandType,
                          operand.getType());
    }
    inferredTypes.push_back(firstOperandType);

    return mlir::success();
}

}  // namespace VPUIP
}  // namespace vpux