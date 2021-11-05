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

#pragma once

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/attributes/strides.hpp"
#include "vpux/compiler/dialect/VPUIP/schema.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"

#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/optional.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

#include <flatbuffers/flatbuffers.h>

#include <unordered_map>

namespace vpux {
namespace EMU {

class BlobWriter final {
public:
    using Task = flatbuffers::Offset<MVCNN::Task>;
    using TaskList = flatbuffers::Offset<MVCNN::TaskList>;

    struct SpecificTask {
        flatbuffers::Offset<void> obj;
        MVCNN::SpecificTask type;
    };

    struct SoftwareLayerParams {
        flatbuffers::Offset<void> obj;
        MVCNN::SoftwareLayerParams type;
    };

    using TensorReference = flatbuffers::Offset<MVCNN::TensorReference>;
    using Barrier = flatbuffers::Offset<MVCNN::Barrier>;

    using IndirectDataReference = flatbuffers::Offset<MVCNN::IndirectDataReference>;

    using BinaryData = flatbuffers::Offset<MVCNN::BinaryData>;

    using String = flatbuffers::Offset<flatbuffers::String>;

    template <typename T>
    using Vector = flatbuffers::Offset<flatbuffers::Vector<T>>;

public:
    explicit BlobWriter(Logger log): _log(log) {
    }

public:
    Task createTask(mlir::Operation* op);

public:
    SpecificTask createUPALayerTask(mlir::Operation* op, const SoftwareLayerParams& params);

public:
    TensorReference createTensor(StringRef name, mlir::ShapedType type, ArrayRef<uint16_t> mult,
                                 ArrayRef<uint8_t> shift, int8_t postShift, ArrayRef<uint8_t> zeroPoints);
    TensorReference createTensor(StringRef name, mlir::ShapedType type);
    TensorReference createTensor(mlir::Value val, StringRef name);
    TensorReference getTensor(mlir::Value val) const;

public:
    BinaryData createBinaryData(ArrayRef<uint64_t> content, mlir::ShapedType type);

public:
    static MVCNN::DType createDType(mlir::Type type);

    Vector<uint32_t> createDims(ShapeRef shape);
    Vector<uint32_t> createDims(mlir::ShapedType type);

    EMU::BlobWriter::Vector<float> createStrides(StridesRef strides, Bit elemSize);
    Vector<float> createStrides(mlir::ShapedType type);

    static MVCNN::order3 createOrder3(mlir::ArrayAttr attr);

public:
    auto createString(StringRef str) {
        return _impl.CreateString(str.data(), str.size());
    }

    template <typename T>
    auto createVector(ArrayRef<T> arr) {
        return _impl.CreateVector(arr.data(), arr.size());
    }

    template <class Range>
    auto createVector(const Range& range) {
        const auto vec = to_small_vector(range);
        return _impl.CreateVector(vec.data(), vec.size());
    }

    template <typename T>
    auto createVectorOfStructs(ArrayRef<T> arr) {
        return _impl.CreateVectorOfStructs(arr.data(), arr.size());
    }

public:
    auto& impl() {
        return _impl;
    }

    operator flatbuffers::FlatBufferBuilder&() {
        return impl();
    }

private:
    void setAliasForSerializedTensors(mlir::Operation* op);

private:
    using TaskMap = std::unordered_map<mlir::Operation*, Task>;
    using TensorReferenceMap = mlir::DenseMap<mlir::Value, TensorReference>;

private:
    Logger _log;
    flatbuffers::FlatBufferBuilder _impl;
    TaskMap _tasks;
    TensorReferenceMap _tensors;
};

}  // namespace EMU
}  // namespace vpux
