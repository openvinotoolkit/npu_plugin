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

#include "vpux/compiler/dialect/IERT/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/nce_sparsity.hpp"

namespace vpux {
namespace VPU {

mlir::Value createActivationWindowTensor(mlir::OpBuilder& builder, mlir::Location loc, ArrayRef<uint8_t> fakeSparsity);

std::vector<int32_t> createWeightsTableData(mlir::Value opInput, mlir::Value opOutput, mlir::Value weights,
                                            mlir::Value activationWindow, Const::ContentAttr bias, int64_t OC,
                                            vpux::VPU::PPETaskAttr ppeTaskAttr, VPU::ArchKind _arch);
mlir::Value createWeightsTableTensor(mlir::OpBuilder& builder, mlir::Location loc, ArrayRef<int32_t> weightsTable,
                                     int64_t OC);
Optional<SmallVector<int32_t>> createInstructionListTableData(mlir::Value opOutput, vpux::IE::PostOp postOp);
mlir::Value createInstructionListTableTensor(mlir::OpBuilder& builder, mlir::Location loc,
                                             Optional<SmallVector<int32_t>> instructionListTable);

}  // namespace VPU
}  // namespace vpux
