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

#pragma once

#include <kernels/inc/common_types.h>
#include "Nce2p7.h"

namespace vpux {

/**
 * Helper builder for creation activation-shaves invocation in memory arguments
 */
class InvocationBuilder {

    llvm::SmallVector<char,  128> _storage;         // keeps basic elements
    llvm::SmallVector<char , 128> _arrayStorage;    // keeps arrays elements

    Logger _log;

    // keeps offset to patchable field within _storage structure that need to be
    // updated after _storage and _arrayStorage gets concatenated
    struct PatchPoint {
        std::function<void(MutableArrayRef<uint8_t>, size_t)> patchCallback;
        size_t offset;

        void patch(MutableArrayRef<uint8_t> resialStorage, size_t patchBase) const {
            patchCallback(resialStorage, patchBase + offset);
        }
    };
    llvm::SmallVector<PatchPoint , 128> _deferredPointers;

public:

    InvocationBuilder(Logger log) : _log(log){}

    void addArg(mlir::Value operand) {
        // TODO: add support for non int constants
        if (operand.getType().isa<mlir::IntegerType>()) {
            auto intValue = operand.getDefiningOp()->getAttrs().begin()->second.dyn_cast_or_null<mlir::IntegerAttr>().getInt();
            storeSimple(_storage, intValue);
        } else if(operand.getType().isa<mlir::MemRefType>()){
            storeAsMemref(operand);
        } else {
            _log.warning("Act Shave Invocation: cannot store arg of type {0}", operand.getType());
        }
    }

    llvm::SmallVector<uint8_t> store() const {
        llvm::SmallVector<uint8_t, 128> serialStorage(_storage.begin(), _storage.end());

        auto patchBase = serialStorage.size() + mvds::nce2p7::ACT_KERNEL_DATA_WINDOW;
        serialStorage.insert(serialStorage.end(), _arrayStorage.begin(), _arrayStorage.end());
        for (auto && field : _deferredPointers) {
            field.patch(serialStorage, patchBase);
        }
        return serialStorage;
    }

protected:

    template <class T>
    static void storeSimple(llvm::SmallVectorImpl<char> & storage, const T& anyValue) {
        ArrayRef<char> valueAsArray(reinterpret_cast<const char*>(&anyValue), sizeof(anyValue));
        storage.insert(storage.end(), valueAsArray.begin(), valueAsArray.end());
    }

    /**
     * create a patch entry , that can be fuether updated
     * @tparam U structure type
     * @tparam T field type
     */
    template <class U, class T>
    void createPatchPoint(const T& patcher) {
        auto fieldPatcher = [offset = _storage.size(), this, patcher] (MutableArrayRef<uint8_t> serialStorage, size_t updateTo) {
            auto& base = reinterpret_cast<U&>(*(serialStorage.begin() + offset));
            patcher(base, updateTo);
        };
        _deferredPointers.push_back({fieldPatcher, _arrayStorage.size()});
    }

    // memref storage - works for OpOperand and OpResult
    void storeAsMemref(const mlir::Value & value) {
        auto tensor = value.getDefiningOp<VPUIP::DeclareTensorOp>();

        if (tensor == nullptr) {
            _log.trace("ACT shave: Cannot create invocation for {0}", value);
            return;
        }

        auto dimsPatcher = [] (sw_params::MemRefData &memrefData, uint32_t updateTo) {
            memrefData.dimsAddr = updateTo;
        };
        auto stridesParcher = [] (sw_params::MemRefData &memrefData, uint32_t updateTo) {
            memrefData.stridesAddr = updateTo;
        };
        auto getAddress = [](VPUIP::DeclareTensorOp & tensor) {
            return tensor.dataIndex() + tensor.leadingOffset().getValueOr(0);
        };

        sw_params::MemRefData memrefData{};

        auto shape = value.getType().cast<mlir::ShapedType>();

        memrefData.numDims = shape.getShape().size();

        // dims
        createPatchPoint<sw_params::MemRefData>(dimsPatcher);
        for (auto &&dim : shape.getShape()) {
            storeSimple(_arrayStorage, checked_cast<int32_t>(dim));
        }

        // order
        const auto inOrder = DimsOrder::fromValue(value);
        memrefData.dimsOrder = inOrder.code();

        // strides
        const auto stridesReqs = StrideReqs::simple(checked_cast<size_t>(shape.getRank()));
        const auto memStrides = stridesReqs.calcStrides(inOrder, shape);
        const auto strides = inOrder.toLogicalOrder(memStrides);

        createPatchPoint<sw_params::MemRefData>(stridesParcher);
        for (auto &&stride : strides) {
            storeSimple(_arrayStorage, stride);
        }

        // data addr
        memrefData.dataAddr = mvds::nce2p7::ACT_KERNEL_CMX_WINDOW + getAddress(tensor);

        memrefData.dataType = 0; // TODO: to be defined

        memrefData.location = sw_params::NN_CMX;

        storeSimple(_storage, memrefData);
    }
};
}  // namespace vpux
