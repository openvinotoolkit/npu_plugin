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
#include "vpux/compiler/dialect/VPUIP/attributes/enums.hpp"
#include "vpux/compiler/dialect/VPUIP/schema.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include <vpux/compiler/act_kernels/act_kernel_gen.h>

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

    struct ActShaveTaskParams {
        flatbuffers::Offset<MVCNN::KernelData> text;
        flatbuffers::Offset<MVCNN::KernelData> data;
        flatbuffers::Offset<flatbuffers::Vector<uint64_t>> args;
        MVCNN::ActKernelType type;
    };

    using TensorReference = flatbuffers::Offset<MVCNN::TensorReference>;
    using Barrier = flatbuffers::Offset<MVCNN::Barrier>;

    using IndirectDataReference = flatbuffers::Offset<MVCNN::IndirectDataReference>;

    using BinaryData = flatbuffers::Offset<MVCNN::BinaryData>;

    using KernelData = flatbuffers::Offset<MVCNN::KernelData>;

    using KernelDataRef = flatbuffers::Offset<MVCNN::KernelDataReference>;

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

    KernelDataRef createInvocationArgs(mlir::Operation* op, vpux::VPUIP::MemoryLocation locale);

    SpecificTask createACTShaveTask(mlir::Operation* op);
    ActKernelDesc createKernelData(StringRef name);

    KernelDataRef createKernelDataRef(StringRef name, MemoryLocation locale,
                                      uint64_t dataOffset, uint64_t dataSize,
                                      ArrayRef<uint8_t> content = None);
    KernelDataRef createKernelDataRef(const KernelDataDesc& desc, MemoryLocation locale);

    // TODO: refactor not using string name as act-kernels definition
    template <class Key, class Value>
    class MapVectorEx {
        std::unordered_map<Key, size_t> ref;
        llvm::SmallVector<Value> data;
    public:
        class iterator {
            typename std::unordered_map<Key, size_t> :: iterator _it;
            std::reference_wrapper<llvm::SmallVector<Value>> _refData;
        public:
            iterator(llvm::SmallVector<Value> &data,
                     typename std::unordered_map<Key, size_t> :: iterator it)
                    : _it(it)
                    , _refData(data) {}
            Value & operator * () {
                return _refData.get()[_it->second];
            }
            Value * operator -> () {
                return &_refData.get()[_it->second];
            }
            bool operator == (const iterator & that) const {
                return _it == that._it;
            }
            bool operator != (const iterator & that) const {
                return _it != that._it;
            }
            iterator & operator ++ () const {
                _it++;
                return *this;
            }
        };
        typename MapVectorEx :: iterator find(StringRef name) {
            return MapVectorEx :: iterator(data, ref.find(name.data()));
        }
        const Value & operator [] (StringRef name) const {
            return data[ref[name.data()]];
        }
        Value & operator [] (StringRef name) {
            auto it = find(name.data());
            if (it != this->end()) {
                return *it;
            }
            ref[name.data()] = data.size();
            data.push_back({});
            return data[data.size() - 1];
        }
        typename MapVectorEx :: iterator end() {
            return MapVectorEx :: iterator(data, ref.end());
        }
        typename MapVectorEx :: iterator begin() {
            return MapVectorEx :: iterator(data, ref.begin());
        }
        size_t localeIndex(StringRef name) const {
            auto x = ref.find(name.data());
            return x->second;
        }
        const llvm::SmallVector<Value> & linearOrder() const {
            return data;
        }
    };
    using ActShavesKernelDataMap = MapVectorEx<std::string, KernelDataDesc>;
    const llvm::SmallVector<KernelDataDesc> & getKernelData() const;

public:
    TensorReference createTensor(StringRef name, mlir::ShapedType type, MemoryLocation locale,
                                 ArrayRef<uint32_t> localeIndex, int64_t dataIndex, ArrayRef<uint16_t> mult,
                                 ArrayRef<uint8_t> shift, ArrayRef<uint8_t> zeroPoints,
                                 Optional<int64_t> sparsityIndex = None, Optional<int64_t> storageElementIndex = None,
                                 Optional<int64_t> storageElementSize = None, Optional<int64_t> leadingOffset = None,
                                 Optional<int64_t> trailingOffset = None, Optional<double> density_rate = None,
                                 Optional<int64_t> swizzling_key = None);
    TensorReference createTensor(StringRef name, mlir::ShapedType type, MemoryLocation locale,
                                 ArrayRef<uint32_t> localeIndex, int64_t dataIndex,
                                 Optional<int64_t> sparsityIndex = None, Optional<int64_t> storageElementIndex = None,
                                 Optional<int64_t> storageElementSize = None, Optional<int64_t> leadingOffset = None,
                                 Optional<int64_t> trailingOffset = None, Optional<double> density_rate = None,
                                 Optional<int64_t> swizzling_key = None);
    TensorReference createTensor(mlir::Value val, StringRef name, MemoryLocation locale, ArrayRef<uint32_t> localeIndex,
                                 int64_t dataIndex, Optional<int64_t> sparsityIndex = None,
                                 Optional<int64_t> storageElementIndex = None,
                                 Optional<int64_t> storageElementSize = None, Optional<int64_t> leadingOffset = None,
                                 Optional<int64_t> trailingOffset = None, Optional<double> density_rate = None,
                                 Optional<int64_t> swizzling_key = None);
    TensorReference getTensor(mlir::Value val) const;

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

    static MVCNN::MemoryLocation createMemoryLocation(MemoryLocation location);
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

private:
    void setAliasForSerializedTensors(mlir::Operation* op);

private:
    using TaskMap = std::unordered_map<mlir::Operation*, Task>;
    //using CompiledActShavesMap
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
