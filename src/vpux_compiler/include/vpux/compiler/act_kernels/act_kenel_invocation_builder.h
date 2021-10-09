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
    llvm::SmallVector<std::pair<std::function<void(uint32_t)>, uint32_t> , 128> _offsetsToUpdate;


    mutable llvm::SmallVector<uint8_t,   128> _finalstorage;


private:

    template <class T>
    static void storeSimple(llvm::SmallVectorImpl<char> & storage, const T& anyValue) {
        ArrayRef<char> valueAsArray(reinterpret_cast<const char*>(&anyValue), sizeof(anyValue));
        storage.insert(storage.end(), valueAsArray.begin(), valueAsArray.end());
    }

    /**
     * registers an array wich is designated by given pointer
     * @tparam U structure type
     * @tparam T field type
     * @param object given object of structure U
     * @param objectField given field in given object
     * @param anyarr and initializer of array
     */
    template <class U, class T, class V>
    void registerArrayFor(const T& patcher,  const V & anyarr ) {
        //auto offset = fieldOffset(object, patcher);
        auto fieldPatcher = [offset = _storage.size(), this, patcher] (uint32_t updateTo) {
            auto& base = reinterpret_cast<U&>(*(_finalstorage.begin() + offset));
            patcher(base, updateTo);
        };
        _offsetsToUpdate.push_back({fieldPatcher, _arrayStorage.size()});
        for (auto &&y : anyarr) {
            storeSimple(_arrayStorage, y);
        }
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
        auto offsetToArrays = _finalstorage.size();
        _finalstorage.insert(_finalstorage.end(), _arrayStorage.begin(), _arrayStorage.end());
        for (auto && offset : _offsetsToUpdate) {
            offset.first(offsetToArrays + offset.second);
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
        registerArrayFor<sw_params::MemRefData>(dimsPatcher, shape.getShape());

        // order
        const auto inOrder = DimsOrder::fromValue(value);
        memrefData.dimsOrder = inOrder.code();

        // strides
        const auto stridesReqs = StrideReqs::simple(checked_cast<size_t>(shape.getRank()));
        const auto memStrides = stridesReqs.calcStrides(inOrder, shape);
        const auto strides = inOrder.toLogicalOrder(memStrides);

        registerArrayFor<sw_params::MemRefData>(stridesParcher, strides);

        auto tensor = value.getDefiningOp<VPUIP::DeclareTensorOp>();

        // data addr
        memrefData.dataAddr = mvds::nce2p7::ACT_KERNEL_CMX_WINDOW + getAddress(tensor);

        memrefData.dataType = 0; // TODO: to be defined

        memrefData.location = sw_params::UPA_CMX;

        storeSimple(_storage, memrefData);
    }
};
}  // namespace vpux