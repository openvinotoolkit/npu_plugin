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
    llvm::SmallVector<uint8_t, 128> _storage;
public:
    template<class T>
    void addArg(const T &) {
    }

    void addArg(const mlir::OpOperand &arg) {
        auto createMemRefData = [](const mlir::OpOperand & tensor){
            sw_params::MemRefData memrefData{};

            auto shape = tensor.get().getType().cast<mlir::ShapedType>();
            memrefData.numDims = shape.getShape().size();
            //memrefData.dataType

            return memrefData;
        };
        auto memref = createMemRefData(arg);
        ArrayRef<uint8_t> memrefAsArray(reinterpret_cast<uint8_t*>(&memref), sizeof(memref));

        _storage.insert(_storage.end(), memrefAsArray.begin(), memrefAsArray.end());
    }


    ArrayRef<uint8_t> store() const {
        return _storage;
    }

};
}  // namespace vpux