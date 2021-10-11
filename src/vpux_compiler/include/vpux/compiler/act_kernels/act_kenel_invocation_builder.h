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

class InvocationBuilder {

    llvm::SmallVector<char,   128> _storage;         // keeps basic elements
    llvm::SmallVector<char , 128> _arrayStorage;    // keeps actual static arrays elements

    // keeps offsets within _storage structure that need to be
    // updated after _arrayStorage gets concatenated
    // and keeps offsets to array_storage
    struct PatchPoint {
        std::function<void(size_t)> patchCallback;
        size_t offset;

        void patch(size_t patchBase) const {
            patchCallback(patchBase + offset);
        }
    };
    llvm::SmallVector<PatchPoint , 128> _patchData;


    mutable llvm::SmallVector<uint8_t,   128> _finalstorage;


private:

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
        //auto offset = fieldOffset(object, patcher);
        auto fieldPatcher = [offset = _storage.size(), this, patcher] (uint32_t updateTo) {
            auto& base = reinterpret_cast<U&>(*(_finalstorage.begin() + offset));
            patcher(base, updateTo);
        };
        _patchData.push_back({fieldPatcher, _arrayStorage.size()});
    }

public:
    template<class T>
    void addArg(const T & ) {
    }
    void addArg(mlir::Value & operands) {
        // TODO: add checks for type
        // TODO: add support for non int constants
        auto intValue = operands.getDefiningOp()->getAttrs().begin()->second.dyn_cast_or_null<mlir::IntegerAttr>().getInt();

        storeSimple(_storage, intValue);
    }

    void addArg(const mlir::OpOperand &arg) {
        storeAsMemref(arg.get());
    }

    void addArg(const mlir::OpResult &result) {
        storeAsMemref(result);
    }

    ArrayRef<uint8_t> store() const {
        _finalstorage.resize(0);
        _finalstorage.insert(_finalstorage.end(), _storage.begin(), _storage.end());
        auto patchBase = _finalstorage.size();
        _finalstorage.insert(_finalstorage.end(), _arrayStorage.begin(), _arrayStorage.end());
        for (auto && field : _patchData) {
            field.patch(patchBase);
        }

        return _finalstorage;
    }

protected:

    // memref storage - works for OpOperand and OpResult
    void storeAsMemref(const mlir::Value & value) {
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

        auto tensor = value.getDefiningOp<VPUIP::DeclareTensorOp>();

        // data addr
        memrefData.dataAddr = mvds::nce2p7::ACT_KERNEL_CMX_WINDOW + getAddress(tensor);

        memrefData.dataType = 0; // TODO: to be defined

        memrefData.location = sw_params::NN_CMX;

        storeSimple(_storage, memrefData);
    }
};
}  // namespace vpux