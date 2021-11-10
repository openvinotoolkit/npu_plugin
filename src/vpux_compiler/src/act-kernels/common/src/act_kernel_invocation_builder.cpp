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

#include "vpux/compiler/act_kernels/act_kernel_invocation_builder.h"

#include <kernels/inc/common_types.h>

#include <vpux/compiler/dialect/VPUIP/ops.hpp>

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Operation.h>

using namespace vpux;

void InvocationBuilder::addArg(mlir::Value operand) {
    // TODO: add support for non int constants
    if (operand.getType().isa<mlir::IntegerType>()) {
        auto intValue =
                operand.getDefiningOp()->getAttrs().begin()->second.dyn_cast_or_null<mlir::IntegerAttr>().getInt();
        storeSimple(_storage, intValue);
    } else if (operand.getType().isa<mlir::MemRefType>()) {
        addMemrefArg(operand);
    } else {
        _log.warning("Act Shave Invocation: cannot store arg of type {0}", operand.getType());
    }
}

llvm::SmallVector<uint8_t> InvocationBuilder::store() const {
    SmallVector<uint8_t> serialStorage(_storage.begin(), _storage.end());

    auto patchBase = _win_e_offset + serialStorage.size() + mvds::nce2p7::ACT_KERNEL_DATA_WINDOW;
    serialStorage.insert(serialStorage.end(), _arrayStorage.begin(), _arrayStorage.end());
    for (auto&& field : _deferredPointers) {
        field.patch(serialStorage, patchBase);
    }
    return serialStorage;
}

void InvocationBuilder::addMemrefArg(mlir::Value value) {
    auto tensor = value.getDefiningOp<VPUIP::DeclareTensorOp>();

    if (tensor == nullptr) {
        _log.trace("ACT shave: Cannot create invocation for {0}", value);
        return;
    }

    auto dimsPatcher = [](sw_params::MemRefData& memrefData, uint32_t updateTo) {
        memrefData.dimsAddr = updateTo;
    };
    auto stridesParcher = [](sw_params::MemRefData& memrefData, uint32_t updateTo) {
        memrefData.stridesAddr = updateTo;
    };
    auto getAddress = [](VPUIP::DeclareTensorOp& tensor) {
        return tensor.dataIndex() + tensor.leadingOffset().getValueOr(0);
    };

    sw_params::MemRefData memrefData{};

    auto shape = value.getType().cast<mlir::ShapedType>();

    memrefData.numDims = checked_cast<uint32_t>(shape.getShape().size());

    // dims
    createPatchPoint<sw_params::MemRefData>(dimsPatcher);
    for (auto& dim : shape.getShape()) {
        storeSimple(_arrayStorage, checked_cast<int32_t>(dim));
    }

    // order
    const auto inOrder = DimsOrder::fromValue(value);
    memrefData.dimsOrder = inOrder.code();

    // strides
    const auto strides = getStrides(shape);

    createPatchPoint<sw_params::MemRefData>(stridesParcher);
    for (auto& stride : strides) {
        storeSimple(_arrayStorage, stride);
    }

    // data addr
    memrefData.dataAddr = checked_cast<uint32_t>(mvds::nce2p7::ACT_KERNEL_CMX_WINDOW + getAddress(tensor));

    memrefData.dataType = 0;  // TODO: to be defined

    memrefData.location = sw_params::NN_CMX;

    storeSimple(_storage, memrefData);
}
