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

#include "vpux/compiler/dialect/VPU/const_utils.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/custom_pwl_table.hpp"
#include "vpux/utils/core/numeric.hpp"

namespace vpux {
namespace VPU {

mlir::Value createActivationWindowTensor(mlir::OpBuilder& builder, mlir::Location loc, ArrayRef<uint8_t> fakeSparsity) {
    const auto elemType = getUInt8Type(builder.getContext());
    const auto fakeSparsityShape = NCESparsity::inferActivationWindowShape(static_cast<int64_t>(fakeSparsity.size()));

    const auto dataStorageType = mlir::RankedTensorType::get(fakeSparsityShape.raw(), elemType);
    const auto dataAttr = mlir::DenseElementsAttr::get(dataStorageType, fakeSparsity);

    auto dataConstOp = builder.create<Const::DeclareOp>(loc, dataStorageType, Const::ContentAttr::get(dataAttr));
    return dataConstOp.output();
}

std::vector<int32_t> createWeightsTableData(mlir::Value opInput, mlir::Value opOutput, mlir::Value weights,
                                            mlir::Value activationWindow, Const::ContentAttr bias, int64_t OC,
                                            vpux::VPU::PPETaskAttr ppeTaskAttr, VPU::ArchKind _arch) {
    const auto weightPtrOffset = 0;
    const auto sparsityPtrOffset = 0;
    const auto weightPtrStep = VPU::NCESparsity::getWeightPtrStep(weights, activationWindow);

    const auto inElemType = opInput.getType().cast<vpux::NDTypeInterface>().getElementType();
    const auto outElemType = opOutput.getType().cast<vpux::NDTypeInterface>().getElementType();
    const auto weightsElemType = weights ? weights.getType().cast<vpux::NDTypeInterface>().getElementType() : nullptr;

    return VPU::NCESparsity::getWeightsTable(inElemType, outElemType, weightPtrOffset, weightPtrStep, sparsityPtrOffset,
                                             _arch, OC, weightsElemType, bias, ppeTaskAttr);
}

mlir::Value createWeightsTableTensor(mlir::OpBuilder& builder, mlir::Location loc, ArrayRef<int32_t> weightsTable,
                                     int64_t OC) {
    const auto elemType = getSInt32Type(builder.getContext());
    const auto weightTableShape = NCESparsity::inferWeightsTableShape(OC);

    const auto dataStorageType = mlir::RankedTensorType::get(weightTableShape.raw(), elemType);
    const auto dataAttr = mlir::DenseElementsAttr::get(dataStorageType, weightsTable);

    auto dataConstOp = builder.create<Const::DeclareOp>(loc, dataStorageType, Const::ContentAttr::get(dataAttr));
    return dataConstOp.output();
}

Optional<SmallVector<int32_t>> createInstructionListTableData(mlir::Value opOutput, vpux::IE::PostOp postOp) {
    const auto outElemType = opOutput.getType().cast<vpux::NDTypeInterface>().getElementType();

    if (postOp == nullptr || outElemType == nullptr) {
        return ::llvm::None;
    }

    const auto pwlTable = findCustomPWLTable(postOp.name().getValue(), outElemType);

    if (!pwlTable.hasValue()) {
        return ::llvm::None;
    }

    const auto pwlTableRange = pwlTable.getValue().range;
    const auto pwlTableShift = pwlTable.getValue().shift;
    const auto pwlTableBias = pwlTable.getValue().bias;

    const size_t vectorSize = pwlTableRange.size() + pwlTableShift.size() + pwlTableBias.size();
    // We need a NOP to terminate each chain of 16 instructions.
    const size_t nopCount = vectorSize >> 4;
    const size_t instructionListTableSize = alignVal<int32_t>(vectorSize + nopCount, 16);

    return VPU::NCESparsity::getInstructionListTable(pwlTableRange, pwlTableShift, pwlTableBias,
                                                     instructionListTableSize);
}

mlir::Value createInstructionListTableTensor(mlir::OpBuilder& builder, mlir::Location loc,
                                             Optional<SmallVector<int32_t>> instructionList) {
    if (!instructionList.hasValue()) {
        return nullptr;
    }
    const auto instructionListArrayRef = makeArrayRef(instructionList.getValue());
    const auto elemType = getSInt32Type(builder.getContext());
    const auto weightTableShape = Shape{1, 1, 1, (long int)instructionListArrayRef.size()};

    const auto dataStorageType = mlir::RankedTensorType::get(weightTableShape.raw(), elemType);
    const auto dataAttr = mlir::DenseElementsAttr::get(dataStorageType, instructionListArrayRef);

    auto dataConstOp = builder.create<Const::DeclareOp>(loc, dataStorageType, Const::ContentAttr::get(dataAttr));
    return dataConstOp.output();
}

}  // namespace VPU
}  // namespace vpux
