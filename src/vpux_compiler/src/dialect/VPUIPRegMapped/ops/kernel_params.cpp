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

#include <host_parsed_inference.h>
#include <mlir/IR/BuiltinTypes.h>
#include "vpux/compiler/dialect/VPUIPRegMapped/ops.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include <kernels/inc/common_types.h>
#include <iostream>

using namespace vpux;

//
// KernelParamsOp
//

struct __attribute__((packed)) SoftmaxParams {
    struct sw_params::MemRefData input;
    struct sw_params::MemRefData output;
    int32_t axis;
};

void vpux::VPUIPRegMapped::KernelParamsOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {

    SoftmaxParams params;

    params.axis = 0;

    params.input.numDims = checked_cast<uint32_t>(getShape(input()).size());
    params.output.numDims = checked_cast<uint32_t>(getShape(output()).size());

    // order
    const auto inOrderInput = DimsOrder::fromValue(input());
    params.input.dimsOrder = inOrderInput.code();

    const auto inOrderOutput = DimsOrder::fromValue(output());
    params.output.dimsOrder = inOrderOutput.code();

    params.input.dataType = 0;  // not defined
    params.input.location = sw_params::NN_CMX;
    params.output.dataType = 0;  // not defined
    params.output.location = sw_params::NN_CMX;

    uint8_t *data = reinterpret_cast<uint8_t*>(&params);

    binDataSection.appendData(data, sizeof(SoftmaxParams));

    // commented out how serialization should actually work
    // auto params = kernel_params();
    
    // auto dense_elem_data = params.getValues<uint8_t>();

    // auto data_vector = std::vector<uint8_t>(dense_elem_data.begin(), dense_elem_data.end());

    // binDataSection.appendData(data_vector.data(), data_vector.size());
}

size_t vpux::VPUIPRegMapped::KernelParamsOp::getBinarySize() {
    auto params = kernel_params();
    auto dense_elem_data = params.getValues<uint8_t>();
    auto data_vector = std::vector<uint8_t>(dense_elem_data.begin(), dense_elem_data.end());
    return data_vector.size();
    }
