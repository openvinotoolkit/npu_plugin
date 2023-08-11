//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mlir/IR/BuiltinTypes.h>
#include "vpux/compiler/dialect/ELF/utils.hpp"
#include "vpux/compiler/dialect/VPU37XX/api/vpu_nnrt_api.h"
#include "vpux/compiler/dialect/VPUMI37XX/ops.hpp"
#include "vpux/utils/core/checked_cast.hpp"

#include <kernels/inc/common_types.h>

using namespace vpux;

//
// KernelParamsOp
//

void vpux::VPUMI37XX::KernelParamsOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    std::vector<uint8_t> inputDimsVector, outputDimsVector;
    std::vector<uint8_t> inputStridesVector, outputStridesVector;
    const auto inputMemrefVals = inputs();
    const auto outputMemrefVals = outputs();

    auto insertDimsIntoVector = [](std::vector<uint8_t>& dimsVector, mlir::Value val) {
        const auto shape = getShape(val);
        const auto inOrderDims = DimsOrder::fromValue(val);
        const auto memShape = inOrderDims.toMemoryOrder(shape);

        for (auto& memDim : memShape | reversed) {
            auto dim = checked_cast<int32_t>(memDim);
            ArrayRef<uint8_t> valueAsArray(reinterpret_cast<const uint8_t*>(&dim), sizeof(dim));
            dimsVector.insert(dimsVector.end(), valueAsArray.begin(), valueAsArray.end());
        }
    };

    auto insertStridesIntoVector = [](std::vector<uint8_t>& stridesVector, mlir::Value val) {
        const auto strides = getMemStrides(val);
        for (auto&& stride : strides | reversed) {
            ArrayRef<uint8_t> valueAsArray(reinterpret_cast<const uint8_t*>(&stride), sizeof(stride));
            stridesVector.insert(stridesVector.end(), valueAsArray.begin(), valueAsArray.end());
        }
    };

    // input Dims & Strides
    for (const auto inputMemrefVal : inputMemrefVals) {
        insertDimsIntoVector(inputDimsVector, inputMemrefVal);
        insertStridesIntoVector(inputStridesVector, inputMemrefVal);
    }

    // output Dims & Strides
    for (const auto outputMemrefVal : outputMemrefVals) {
        insertDimsIntoVector(outputDimsVector, outputMemrefVal);
        insertStridesIntoVector(outputStridesVector, outputMemrefVal);
    }

    auto params = kernel_params();

    auto dense_elem_data = params.getValues<uint8_t>();

    auto data_vector = std::vector<uint8_t>(dense_elem_data.begin(), dense_elem_data.end());

    // serialize actual kernel params
    binDataSection.appendData(data_vector.data(), data_vector.size());

    // serialize IO dims/strides
    binDataSection.appendData(inputDimsVector.data(), inputDimsVector.size());
    binDataSection.appendData(inputStridesVector.data(), inputStridesVector.size());
    binDataSection.appendData(outputDimsVector.data(), outputDimsVector.size());
    binDataSection.appendData(outputStridesVector.data(), outputStridesVector.size());
}

size_t vpux::VPUMI37XX::KernelParamsOp::getBinarySize() {
    auto params = kernel_params();
    auto dense_elem_data = params.getValues<uint8_t>();
    auto data_vector = std::vector<uint8_t>(dense_elem_data.begin(), dense_elem_data.end());

    const auto inputMemrefVals = inputs();
    const auto outputMemrefVals = outputs();

    auto inputDimsSize = 0;
    auto inputStridesSize = 0;

    for (const auto inputMemrefVal : inputMemrefVals) {
        inputDimsSize += sizeof(int32_t) * getShape(inputMemrefVal).size();
        inputStridesSize += sizeof(int64_t) * getMemStrides(inputMemrefVal).size();
    }

    auto outputDimsSize = 0;
    auto outputStridesSize = 0;

    for (const auto outputMemrefVal : outputMemrefVals) {
        outputDimsSize += sizeof(int32_t) * getShape(outputMemrefVal).size();
        outputStridesSize += sizeof(int64_t) * getMemStrides(outputMemrefVal).size();
    }

    return data_vector.size() + inputDimsSize + outputDimsSize + inputStridesSize + outputStridesSize;
}

size_t vpux::VPUMI37XX::KernelParamsOp::getParamsStructSize() {
    auto params = kernel_params();
    auto dense_elem_data = params.getValues<uint8_t>();
    auto data_vector = std::vector<uint8_t>(dense_elem_data.begin(), dense_elem_data.end());

    return data_vector.size();
}

mlir::FailureOr<uint64_t> vpux::VPUMI37XX::KernelParamsOp::getOffsetOfWithinOperation(mlir::Value val) {
    for (auto inputsIt : inputs() | indexed) {
        if (val == inputsIt.value()) {
            return inputsIt.index() * sizeof(sw_params::MemRefData) + offsetof(sw_params::MemRefData, dataAddr);
        }
    }

    for (auto outputsIt : outputs() | indexed) {
        if (val == outputsIt.value()) {
            return (inputs().size() + outputsIt.index()) * sizeof(sw_params::MemRefData) +
                   offsetof(sw_params::MemRefData, dataAddr);
        }
    }

    return mlir::failure();
}

// The parameter structs for the sw layers must be 64Byte aligned as an ActShave requirement
size_t vpux::VPUMI37XX::KernelParamsOp::getAlignmentRequirements() {
    return ELF::VPUX_SHAVE_ALIGNMENT;
}

vpux::VPURT::BufferSection vpux::VPUMI37XX::KernelParamsOp::getMemorySpace() {
    return vpux::VPURT::BufferSection::DDR;
}

vpux::ELF::SectionFlagsAttr vpux::VPUMI37XX::KernelParamsOp::getAccessingProcs() {
    return (ELF::SectionFlagsAttr::VPU_SHF_PROC_SHAVE);
}

vpux::ELF::SectionFlagsAttr vpux::VPUMI37XX::KernelParamsOp::getUserProcs() {
    return (ELF::SectionFlagsAttr::SHF_NONE);
}
