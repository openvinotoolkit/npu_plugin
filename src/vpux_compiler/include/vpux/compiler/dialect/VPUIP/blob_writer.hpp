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

#include <vpux/compiler/act_kernels/compilation.h>
#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/attributes/strides.hpp"
#include "vpux/compiler/dialect/VPUIP/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/schema.hpp"
#include "vpux/compiler/dialect/VPURT/attributes.hpp"
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

#include <llvm/ADT/MapVector.h>
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
    using PreprocessingInfo = flatbuffers::Offset<MVCNN::preprocessingInfo>;
    using Barrier = flatbuffers::Offset<MVCNN::Barrier>;

    using IndirectDataReference = flatbuffers::Offset<MVCNN::IndirectDataReference>;

    using BinaryData = flatbuffers::Offset<MVCNN::BinaryData>;

    using KernelData = flatbuffers::Offset<MVCNN::KernelData>;
    using ActKernel = flatbuffers::Offset<MVCNN::ActKernel>;

    using KernelDataRef = flatbuffers::Offset<MVCNN::KernelDataReference>;

    using String = flatbuffers::Offset<flatbuffers::String>;

    using ActShavesKernelDataMap =
            llvm::MapVector<std::string, SerializedKernelDataDesc, std::unordered_map<std::string, size_t>>;

    template <typename T>
    using Vector = flatbuffers::Offset<flatbuffers::Vector<T>>;

public:
    explicit BlobWriter(Logger log): _log(log) {
    }

public:
    Task createTask(mlir::Operation* op);
    void setAliasForSerializedTensors(mlir::Operation* op);

public:
    SpecificTask createUPALayerTask(mlir::Operation* op, const SoftwareLayerParams& params);

    SpecificTask createSW_KernelTask(mlir::Operation* op);
    ActKernel createRuntimeKernelTask(mlir::ModuleOp module, mlir::Operation* op);

    //  compiles kernel code and returns it's data and text sections
    ActKernelDesc compileKernelData(const CompilationUnitDesc& unitDesc);
    ActKernelDesc compileManagementKernelData();

    KernelDataRef createKernelDataRef(StringRef name, uint64_t dataOffset, uint64_t dataSize,
                                      ArrayRef<uint8_t> content = None);
    KernelDataRef createKernelDataRef(const KernelDataDesc& desc);

    const ActShavesKernelDataMap& getKernelData() const;

public:
    TensorReference createTensorRef(StringRef name, mlir::ShapedType type, VPURT::BufferSection section,
                                    int64_t sectionIndex, int64_t byteOffset, ArrayRef<uint16_t> mult,
                                    ArrayRef<uint8_t> shift, int8_t postShift, ArrayRef<uint8_t> zeroPoints);
    TensorReference createTensorRef(StringRef name, mlir::ShapedType type, VPURT::BufferSection section,
                                    int64_t sectionIndex, int64_t byteOffset);
    TensorReference createTensorRef(mlir::Value val, StringRef name, VPURT::BufferSection section, int64_t sectionIndex,
                                    int64_t byteOffset);
    TensorReference getTensorRef(mlir::Value val) const;

public:
    BinaryData createBinaryData(ArrayRef<uint64_t> content, mlir::ShapedType type, bool csram_cacheable = false);

public:
    Barrier createBarrier(mlir::Value val, int64_t physicalID = 0);
    uint32_t getBarrierVirtualID(mlir::Value val) const;

public:
    static MVCNN::DType createDType(mlir::Type type);

    Vector<uint32_t> createDims(ShapeRef shape);
    Vector<uint32_t> createDims(mlir::ShapedType type);

    VPUIP::BlobWriter::Vector<float> createStrides(StridesRef strides, Bit elemSize);
    Vector<float> createStrides(mlir::ShapedType type);

    static MVCNN::MemoryLocation createMemoryLocation(VPURT::BufferSection section);
    IndirectDataReference createIndirectDataReference(int64_t dataIndex, Optional<int64_t> sparsityIndex = None,
                                                      Optional<int64_t> storageElementIndex = None,
                                                      Optional<int64_t> storageElementSize = None);

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

    static const ActShaveCompileParams& compileParams();

private:
    using TaskMap = std::unordered_map<mlir::Operation*, Task>;
    using TensorReferenceMap = mlir::DenseMap<mlir::Value, TensorReference>;
    using BarrierMap = mlir::DenseMap<mlir::Value, uint32_t>;

private:
    Logger _log;
    flatbuffers::FlatBufferBuilder _impl;
    TaskMap _tasks;
    ActShavesKernelDataMap _actKernelsData;
    TensorReferenceMap _tensors;
    BarrierMap _barriers;
};

}  // namespace VPUIP
}  // namespace vpux
