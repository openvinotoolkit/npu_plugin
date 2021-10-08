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

namespace vpux {

class InvocationBuilder {

    llvm::SmallVector<uint8_t,   128> _storage;         // keeps basic elements
    llvm::SmallVector<uint8_t , 128> _arrayStorage;    // keeps actual static arrays elements

    // keeps offsets within _storage structure that need to be
    // updated after _arrayStorage gets concatinated
    // and keeps offsets to array_storage
    llvm::SmallVector<std::pair<uint32_t, uint32_t> , 128> _offsetsToUpdate;


    mutable llvm::SmallVector<uint8_t,   128> _finalstorage;


private:
    template <class U, class T>
    static uint32_t fieldOffset(const U & base, const T & field) {
        auto A = reinterpret_cast<const uint8_t*>(&base);
        auto B = reinterpret_cast<const uint8_t*>(&field);
        VPUX_THROW_UNLESS(B >= A && B < A + sizeof(U), "field not part of given object");

        return B - A;
    }

    template <class T>
    static void storeSimple(llvm::SmallVectorImpl<uint8_t> & storage, const T& anyValue) {
        ArrayRef<uint8_t> valueAsArray(reinterpret_cast<const uint8_t*>(&anyValue), sizeof(anyValue));
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
    void registerArrayFor(const U & object, const T& objectField,  const V & anyarr ) {
        auto offset = fieldOffset(object, objectField);
        _offsetsToUpdate.push_back({_storage.size() + offset, _arrayStorage.size()});
        for (auto &&y : anyarr) {
            storeSimple(_arrayStorage, y);
        }
    }

public:
    template<class T>
    void addArg(const T &) {
    }

    void addArg(const mlir::OpOperand &arg) {
        auto createMemRefData = [this](const mlir::OpOperand & tensor){
            sw_params::MemRefData memrefData{};

            auto shape = tensor.get().getType().cast<mlir::ShapedType>();
            memrefData.numDims = shape.getShape().size();

            std::vector<int > i = {1,2,3};
            registerArrayFor(memrefData, memrefData.dimsAddr, i);
            registerArrayFor(memrefData, memrefData.dimsOrder, i);

            return memrefData;
        };
        storeSimple(_storage, createMemRefData(arg));
    }

    void addArg(const mlir::OpResult &result) {
        auto createMemRefData = [this](const mlir::OpResult & tensor){
            sw_params::MemRefData memrefData{};

            auto shape = tensor.getType().cast<mlir::ShapedType>();
            memrefData.numDims = shape.getShape().size();
            std::vector<int > i = {1,2,3};
            registerArrayFor(memrefData, memrefData.dimsAddr, i);
            registerArrayFor(memrefData, memrefData.dimsOrder, i);


            return memrefData;
        };

        storeSimple(_storage, createMemRefData(result));
    }


    ArrayRef<uint8_t> store() const {
        _finalstorage.resize(0);
        _finalstorage.insert(_finalstorage.end(), _storage.begin(), _storage.end());
        _finalstorage.insert(_finalstorage.end(), _arrayStorage.begin(), _arrayStorage.end());
        auto arraysOffset = _storage.size();
        for (auto && offset : _offsetsToUpdate) {
            *(_finalstorage.begin() + offset.first) = offset.second + arraysOffset;
        }
        return _storage;
    }

};
}  // namespace vpux