//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#pragma once

#include "vpux/compiler/core/shape.hpp"
#include "vpux/compiler/core/strides.hpp"
#include "vpux/compiler/dialect/VPUIP/attributes/enums.hpp"

#include "vpux/utils/VPUIP/schema.hpp"
#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/optional.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <mlir/IR/Operation.h>
#include <mlir/IR/StandardTypes.h>
#include <mlir/IR/Value.h>

#include <flatbuffers/flatbuffers.h>

#include <unordered_map>

namespace vpux {
namespace VPUIP {

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

    using IndirectDataReference =
            flatbuffers::Offset<MVCNN::IndirectDataReference>;

    using BinaryData = flatbuffers::Offset<MVCNN::BinaryData>;

    using String = flatbuffers::Offset<flatbuffers::String>;

    template <typename T>
    using Vector = flatbuffers::Offset<flatbuffers::Vector<T>>;

public:
    Task createTask(mlir::Operation* op);

public:
    SpecificTask createUPALayerTask(mlir::Operation* op,
                                    const SoftwareLayerParams& params,
                                    int32_t maxShaves,
                                    bool isTrailingSWLayer);

public:
    TensorReference createTensor(StringRef name,
                                 mlir::MemRefType type,
                                 MemoryLocation location,
                                 uint64_t offset);
    TensorReference createTensor(mlir::Value val,
                                 StringRef name,
                                 MemoryLocation location,
                                 uint64_t offset);
    TensorReference getTensor(mlir::Value val) const;

    Barrier createBarrier(mlir::Value val);
    Barrier getBarrier(mlir::Value val) const;

    auto getAllBarriers() const {
        return _barriers | map_values;
    }

public:
    static MVCNN::DType createDType(mlir::Type type);

    Vector<uint32_t> createDims(ShapeRef shape);
    Vector<uint32_t> createDims(mlir::MemRefType type);

    Vector<uint32_t> createStrides(StridesRef strides, int64_t elemByteSize);
    Vector<uint32_t> createStrides(mlir::MemRefType type);

    static MVCNN::MemoryLocation createMemoryLocation(MemoryLocation location);
    IndirectDataReference createIndirectDataReference(uint64_t offset);

    BinaryData createBinaryData(mlir::DenseElementsAttr content,
                                bool csram_cacheable);

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
        const auto vec = to_vector<4>(range);
        return _impl.CreateVector(vec.data(), vec.size());
    }

public:
    auto& impl() {
        return _impl;
    }

    operator flatbuffers::FlatBufferBuilder&() {
        return impl();
    }

private:
    using TaskMap = std::unordered_map<mlir::Operation*, Task>;
    using TensorReferenceMap = mlir::DenseMap<mlir::Value, TensorReference>;
    using BarrierMap = mlir::DenseMap<mlir::Value, Barrier>;

private:
    flatbuffers::FlatBufferBuilder _impl;
    TaskMap _tasks;
    TensorReferenceMap _tensors;
    BarrierMap _barriers;
};

}  // namespace VPUIP
}  // namespace vpux
