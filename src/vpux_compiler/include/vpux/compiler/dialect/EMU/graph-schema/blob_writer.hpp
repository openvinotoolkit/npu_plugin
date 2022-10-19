//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/attributes/strides.hpp"
#include "vpux/compiler/dialect/VPUIP/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/schema.hpp"
#include "vpux/compiler/dialect/VPURT/attributes.hpp"

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
    explicit BlobWriter(Logger log, VPU::ArchKind architecture): _log(log), _architecture(architecture) {
    }

public:
    Task createTask(mlir::Operation* op);

public:
    SpecificTask createUPALayerTask(mlir::Operation* op, const SoftwareLayerParams& params);

public:
    TensorReference createTensor(StringRef name, NDTypeInterface type, ArrayRef<int64_t> mult, ArrayRef<int64_t> shift,
                                 int64_t postShift, ArrayRef<uint8_t> zeroPoints,
                                 VPURT::BufferSection locale = VPURT::BufferSection::DDR,
                                 const uint32_t localeIndex = 0);
    TensorReference createTensor(StringRef name, NDTypeInterface type,
                                 VPURT::BufferSection locale = VPURT::BufferSection::DDR,
                                 const uint32_t localeIndex = 0);
    TensorReference createTensor(mlir::Value val, StringRef name,
                                 VPURT::BufferSection locale = VPURT::BufferSection::DDR,
                                 const uint32_t localeIndex = 0);
    TensorReference getTensor(mlir::Value val) const;

public:
    BinaryData createBinaryData(ArrayRef<uint64_t> content, mlir::ShapedType type);

public:
    Vector<uint32_t> createDims(ShapeRef shape);
    Vector<uint32_t> createDims(NDTypeInterface type);
    Vector<float> createStrides(StridesRef strides, Bit elemSize);
    Vector<float> createStrides(NDTypeInterface type);

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

    template <class UnderlyingType>
    auto arrayCast(ArrayRef<int64_t> source) {
        SmallVector<UnderlyingType> casted(source.size());
        std::transform(source.begin(), source.end(), casted.begin(), [](auto value) {
            return checked_cast<UnderlyingType>(value);
        });
        return createVector(casted);
    }

private:
    Logger _log;
    VPU::ArchKind _architecture;
    flatbuffers::FlatBufferBuilder _impl;
    TaskMap _tasks;
    TensorReferenceMap _tensors;
};

}  // namespace EMU
}  // namespace vpux
