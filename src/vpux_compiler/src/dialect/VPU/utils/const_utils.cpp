//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"

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
                                            vpux::VPU::PPETaskAttr ppeTaskAttr, VPU::ArchKind _arch,
                                            vpux::IE::PostOp postOpAttr) {
    const auto weightPtrOffset = 0;
    const auto sparsityPtrOffset = 0;
    const auto weightPtrStep = VPU::NCESparsity::getWeightPtrStep(weights, activationWindow);
    const auto sparsityPtrStep = 0;

    const auto inElemType = opInput.getType().cast<vpux::NDTypeInterface>().getElementType();
    const auto outElemType = opOutput.getType().cast<vpux::NDTypeInterface>().getElementType();
    const auto weightsElemType = weights ? weights.getType().cast<vpux::NDTypeInterface>().getElementType() : nullptr;

    return VPU::NCESparsity::getWeightsTable(inElemType, outElemType, weightPtrOffset, weightPtrStep, sparsityPtrOffset,
                                             sparsityPtrStep, _arch, OC, weightsElemType, bias, ppeTaskAttr,
                                             postOpAttr);
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

Optional<SmallVector<int32_t>> createInstructionListTableData(mlir::Value opOutput, vpux::IE::PostOp postOp,
                                                              VPU::ArchKind _arch) {
    const auto outElemType = opOutput.getType().cast<vpux::NDTypeInterface>().getElementType();

    if (postOp == nullptr) {
        return None;
    }

    if (_arch == VPU::ArchKind::VPUX37XX) {
        return None;
    }

    const auto pwlTable = findCustomPWLTable(postOp, outElemType);

    if (!pwlTable.hasValue()) {
        return None;
    }

    const auto& pwlTableRange = pwlTable.getValue().range;
    const auto& pwlTableShift = pwlTable.getValue().shift;
    const auto& pwlTableBias = pwlTable.getValue().bias;

    const size_t vectorSize = pwlTableRange.size() + pwlTableShift.size() + pwlTableBias.size();
    // We need a NOOP to terminate each chain of 16 instructions.
    const size_t noopCount = vectorSize / 16;
    const size_t instructionListTableSize = alignVal<size_t>(vectorSize + noopCount, 16);

    return VPU::NCESparsity::getInstructionListTable(pwlTableRange, pwlTableShift, pwlTableBias,
                                                     static_cast<int32_t>(instructionListTableSize));
}

mlir::Value createInstructionListTableTensor(mlir::OpBuilder& builder, mlir::Location loc,
                                             const Optional<SmallVector<int32_t>>& instructionList) {
    if (!instructionList.hasValue()) {
        return nullptr;
    }
    const auto instructionListArrayRef = makeArrayRef(instructionList.getValue());
    const auto elemType = getSInt32Type(builder.getContext());
    const auto instructionListTableShape = Shape{1, 1, 1, static_cast<int64_t>(instructionListArrayRef.size())};

    const auto dataStorageType = mlir::RankedTensorType::get(instructionListTableShape.raw(), elemType);
    const auto dataAttr = mlir::DenseElementsAttr::get(dataStorageType, instructionListArrayRef);

    auto dataConstOp = builder.create<Const::DeclareOp>(loc, dataStorageType, Const::ContentAttr::get(dataAttr));
    return dataConstOp.output();
}

}  // namespace VPU
}  // namespace vpux
