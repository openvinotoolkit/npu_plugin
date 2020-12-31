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

#include "vpux/compiler/dialect/VPUIP/blob_writer.hpp"

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/effects.hpp"
#include "vpux/compiler/dialect/VPUIP/ops_interfaces.hpp"

#include "vpux/utils/IE/loop.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/helper_macros.hpp"
#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <algorithm>

using namespace vpux;

VPUIP::BlobWriter::Task vpux::VPUIP::BlobWriter::createTask(mlir::Operation* op) {
    _log.trace("Create BLOB Task for {0}", *op);

    auto task = mlir::cast<VPUIP::TaskOpInterface>(op);

    VPUX_THROW_UNLESS(_tasks.count(op) == 0, "Operation {0} was already serialized", *op);

    const auto curID = _tasks.size();

    String name;
    if (const auto nameLoc = op->getLoc().dyn_cast<mlir::NameLoc>()) {
        name = createString(nameLoc.getName().strref());
    }

    const auto waitBarriers = createVector(task.waitBarriers() | transformed([this](mlir::Value val) {
                                               const auto* barrier =
                                                       flatbuffers::GetTemporaryPointer(_impl, getBarrier(val));
                                               return checked_cast<uint32_t>(barrier->barrier_id());
                                           }));
    const auto updateBarriers = createVector(task.updateBarriers() | transformed([this](mlir::Value val) {
                                                 const auto* barrier =
                                                         flatbuffers::GetTemporaryPointer(_impl, getBarrier(val));
                                                 return checked_cast<uint32_t>(barrier->barrier_id());
                                             }));

    MVCNN::BarrierReferenceBuilder barriersBuilder(_impl);
    barriersBuilder.add_wait_barriers(waitBarriers);
    barriersBuilder.add_update_barriers(updateBarriers);
    const auto barriers = barriersBuilder.Finish();

    const auto specifics = task.serialize(*this);

    MVCNN::TaskBuilder builder(_impl);
    if (!name.IsNull()) {
        builder.add_name(name);
    }
    builder.add_nodeID(checked_cast<uint32_t>(curID));
    builder.add_associated_barriers(barriers);
    builder.add_task_type(specifics.type);
    builder.add_task(specifics.obj);
    const auto off = builder.Finish();

    _tasks.insert({op, off});

    return off;
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::BlobWriter::createUPALayerTask(mlir::Operation* op,
                                                                            const SoftwareLayerParams& params) {
    VPUX_THROW_UNLESS(op != nullptr, "Got NULL pointer in createUPALayerTask");

    auto layer = mlir::dyn_cast<LayerInterface>(op);
    VPUX_THROW_UNLESS(layer != nullptr, "Operation '{0}' is not a Layer", op->getName());

    auto upaTask = mlir::dyn_cast<VPUIP::UPATaskOpInterface>(op);
    VPUX_THROW_UNLESS(upaTask != nullptr, "Operation '{0}' is not a UPA Task", op->getName());

    const auto maxShaves = upaTask.maxShaves();
    const auto isTrailingSWLayer = upaTask.isTrailingSWLayer();

    const auto getTensorCb = [this](mlir::Value val) {
        return getTensor(val);
    };

    const auto inputs = createVector(layer.getInputs() | transformed(getTensorCb));
    const auto outputs = createVector(layer.getOutputs() | transformed(getTensorCb));

    MVCNN::UPALayerTaskBuilder builder(_impl);
    if (maxShaves.hasValue()) {
        builder.add_maxShaves(checked_cast<uint8_t>(maxShaves.getValue()));
    } else {
        auto resources = IERT::RunTimeResourcesOp::getFromModule(op->getParentOfType<mlir::ModuleOp>());
        VPUX_THROW_UNLESS(resources != nullptr, "Missing IERT run-time resources definition");

        auto available = resources.getAvailableExecutor(
                VPUIP::PhysicalProcessorAttr::get(op->getContext(), VPUIP::PhysicalProcessor::SHAVE_UPA));
        VPUX_THROW_UNLESS(available != nullptr, "SHAVE_UPA executor is not avaialble in run-time");

        builder.add_maxShaves(checked_cast<uint8_t>(available.count()));
    }
    builder.add_softLayerParams_type(params.type);
    builder.add_softLayerParams(params.obj);
    builder.add_inputs(inputs);
    builder.add_outputs(outputs);
    builder.add_isTrailingSWLayer(isTrailingSWLayer);
    return {builder.Finish().Union(), MVCNN::SpecificTask_UPALayerTask};
}

VPUIP::BlobWriter::TensorReference vpux::VPUIP::BlobWriter::createTensor(
        StringRef name, mlir::MemRefType type, MemoryLocation locale, Optional<uint32_t> localeIndex,
        uint64_t dataIndex, Optional<uint64_t> sparsityIndex, Optional<uint64_t> storageElementIndex,
        Optional<uint32_t> storageElementSize, Optional<uint32_t> leadingOffset, Optional<uint32_t> trailingOffset,
        Optional<float> density_rate, Optional<uint8_t> swizzling_key) {
    const auto serializedName = createString(name);

    const auto serializedDataType = createDType(type.getElementType());
    const auto serializedDims = createDims(type);
    const auto serializedStrides = createStrides(type);
    const auto dimsOrder = DimsOrder::fromType(type);
    VPUX_THROW_UNLESS(dimsOrder.hasValue(), "Can't get DimsOrder from MemRef Type '{0}'", type);

    const auto serializedDataReference =
            createIndirectDataReference(dataIndex, sparsityIndex, storageElementIndex, storageElementSize);

    const auto serializedLocale = createMemoryLocation(locale);
    const auto serializedLocaleIndex = createVector(to_small_vector(makeArrayRef({localeIndex.getValueOr(0)})));

    // TODO: get this from type.getElementType() (Quant Dialect)
    const auto quantZero = createVector(makeArrayRef<uint8_t>({0}));
    const auto quantMult = createVector(makeArrayRef<uint16_t>({1}));
    const auto quantShift = createVector(makeArrayRef<uint8_t>({0}));

    MVCNN::TensorReferenceBuilder builder(_impl);
    builder.add_name(serializedName);
    builder.add_dimensions(serializedDims);
    builder.add_strides(serializedStrides);
    builder.add_data(serializedDataReference);
    builder.add_locale(serializedLocale);
    builder.add_locale_index(serializedLocaleIndex);
    builder.add_data_dtype(serializedDataType);
    builder.add_quant_zero(quantZero);
    builder.add_quant_mult(quantMult);
    builder.add_quant_shift(quantShift);
    builder.add_order(dimsOrder->code());
    if (leadingOffset.hasValue()) {
        builder.add_leading_offset(leadingOffset.getValue());
    }
    if (trailingOffset.hasValue()) {
        builder.add_trailing_offset(trailingOffset.getValue());
    }
    if (density_rate.hasValue()) {
        builder.add_density_rate(density_rate.getValue());
    }
    if (swizzling_key.hasValue()) {
        builder.add_swizzling_key(swizzling_key.getValue());
    }
    return builder.Finish();
}

VPUIP::BlobWriter::TensorReference vpux::VPUIP::BlobWriter::createTensor(
        mlir::Value val, StringRef name, MemoryLocation locale, Optional<uint32_t> localeIndex, uint64_t dataIndex,
        Optional<uint64_t> sparsityIndex, Optional<uint64_t> storageElementIndex, Optional<uint32_t> storageElementSize,
        Optional<uint32_t> leadingOffset, Optional<uint32_t> trailingOffset, Optional<float> density_rate,
        Optional<uint8_t> swizzling_key) {
    VPUX_THROW_UNLESS(_tensors.count(val) == 0, "Value {0} was already serialized", val);

    const auto off = createTensor(name, val.getType().cast<mlir::MemRefType>(), locale, localeIndex, dataIndex,
                                  sparsityIndex, storageElementIndex, storageElementSize, leadingOffset, trailingOffset,
                                  density_rate, swizzling_key);

    _tensors.insert({val, off});

    return off;
}

VPUIP::BlobWriter::TensorReference vpux::VPUIP::BlobWriter::getTensor(mlir::Value val) const {
    const auto it = _tensors.find(val);
    VPUX_THROW_UNLESS(it != _tensors.end(), "Value {0} wasn't serialized yet", val);
    return it->second;
}

VPUIP::BlobWriter::Barrier vpux::VPUIP::BlobWriter::createBarrier(mlir::Value val) {
    VPUX_THROW_UNLESS(_barriers.count(val) == 0, "Value {0} was already serialized", val);

    const auto curID = _barriers.size();

    size_t numConsumers = 0;
    size_t numProducers = 0;
    for (auto userOp : val.getUsers()) {
        auto opEffects = mlir::cast<mlir::MemoryEffectOpInterface>(userOp);
        VPUX_THROW_UNLESS(opEffects != nullptr,
                          "Barrier Value {0} is used by Operation {1} without "
                          "MemoryEffects interface",
                          val, *userOp);

        using MemEffect = mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>;
        SmallVector<MemEffect> valEffects;

        opEffects.getEffectsOnValue(val, valEffects);
        VPUX_THROW_UNLESS(valEffects.size() == 1,
                          "Barrier Value {0} must have exactly 1 MemoryEffect "
                          "per Operation, got {1} for Operation {2}",
                          val, valEffects.size(), *userOp);

        const auto& effect = valEffects.front();
        VPUX_THROW_UNLESS(effect.getResource() == BarrierResource::get(),
                          "Barrier Value {0} has non Barrier Resource for Operation {1}", val, *userOp);

        if (effect.getEffect() == mlir::MemoryEffects::Read::get()) {
            ++numConsumers;
        } else if (effect.getEffect() == mlir::MemoryEffects::Write::get()) {
            ++numProducers;
        } else {
            VPUX_THROW("Barrier Value {0} has unsupported Effect in Operation {1}", val, *userOp);
        }
    }

    MVCNN::BarrierBuilder builder(_impl);
    builder.add_barrier_id(checked_cast<int16_t>(curID));
    builder.add_consumer_count(checked_cast<int16_t>(numConsumers));
    builder.add_producer_count(checked_cast<int16_t>(numProducers));
    const auto off = builder.Finish();

    _barriers.insert({val, off});

    return off;
}

VPUIP::BlobWriter::Barrier vpux::VPUIP::BlobWriter::getBarrier(mlir::Value val) const {
    const auto it = _barriers.find(val);
    VPUX_THROW_UNLESS(it != _barriers.end(), "Value {0} wasn't serialized yet", val);
    return it->second;
}

MVCNN::DType vpux::VPUIP::BlobWriter::createDType(mlir::Type type) {
    if (type.isF64()) {
        return MVCNN::DType_FP64;
    } else if (type.isF32()) {
        return MVCNN::DType_FP32;
    } else if (type.isF16()) {
        return MVCNN::DType_FP16;
    } else if (type.isBF16()) {
        return MVCNN::DType_FP16;
    } else if (type.isSignedInteger(8 * sizeof(int64_t))) {
        return MVCNN::DType_I64;
    } else if (type.isSignedInteger(8 * sizeof(int32_t))) {
        return MVCNN::DType_I32;
    } else if (type.isSignedInteger(8 * sizeof(int16_t))) {
        return MVCNN::DType_I16;
    } else if (type.isSignedInteger(8 * sizeof(int8_t))) {
        return MVCNN::DType_I8;
    } else if (type.isSignedInteger(4)) {
        return MVCNN::DType_I4;
    } else if (type.isSignedInteger(2)) {
        return MVCNN::DType_I2;
    } else if (type.isUnsignedInteger(8 * sizeof(uint64_t))) {
        return MVCNN::DType_U64;
    } else if (type.isUnsignedInteger(8 * sizeof(uint32_t))) {
        return MVCNN::DType_U32;
    } else if (type.isUnsignedInteger(8 * sizeof(uint16_t))) {
        return MVCNN::DType_U16;
    } else if (type.isUnsignedInteger(8 * sizeof(uint8_t))) {
        return MVCNN::DType_U8;
    } else if (type.isSignlessInteger(1)) {
        return MVCNN::DType_BIN;
    } else {
        VPUX_THROW("Unsupported element type {0}", type);
    }
}

VPUIP::BlobWriter::Vector<uint32_t> vpux::VPUIP::BlobWriter::createDims(ShapeRef shape) {
    return createVector(shape | transformed([](int64_t val) {
                            return checked_cast<uint32_t>(val);
                        }));
}

VPUIP::BlobWriter::Vector<uint32_t> vpux::VPUIP::BlobWriter::createDims(mlir::MemRefType type) {
    return createDims(getShape(type));
}

VPUIP::BlobWriter::Vector<float> vpux::VPUIP::BlobWriter::createStrides(StridesRef strides, int64_t elemByteSize) {
    Strides temp{elemByteSize};
    temp.append(strides.begin(), strides.end());

    return createVector(temp | transformed([](int64_t val) {
                            return checked_cast<float>(val);
                        }));
}

VPUIP::BlobWriter::Vector<float> vpux::VPUIP::BlobWriter::createStrides(mlir::MemRefType type) {
    return createStrides(getStrides(type), type.getElementTypeBitWidth() / CHAR_BIT);
}

MVCNN::MemoryLocation vpux::VPUIP::BlobWriter::createMemoryLocation(MemoryLocation location) {
#define CASE(_val_)             \
    case MemoryLocation::_val_: \
        return VPUX_COMBINE(MVCNN::MemoryLocation_, _val_)

    switch (location) {
        CASE(ProgrammableInput);
        CASE(ProgrammableOutput);
        CASE(VPU_DDR_Heap);
        CASE(GraphFile);
        CASE(VPU_CMX_NN);
        CASE(VPU_CMX_UPA);
        CASE(VPU_DDR_BSS);
        CASE(VPU_CSRAM);
    default:
        VPUX_THROW("Unsupported MemoryLocation {0}", location);
    }

#undef CASE
}

VPUIP::BlobWriter::IndirectDataReference vpux::VPUIP::BlobWriter::createIndirectDataReference(
        uint64_t dataIndex, Optional<uint64_t> sparsityIndex, Optional<uint64_t> storageElementIndex,
        Optional<uint32_t> storageElementSize) {
    MVCNN::IndirectDataReferenceBuilder builder(_impl);
    builder.add_data_index(dataIndex);
    if (sparsityIndex.hasValue()) {
        builder.add_sparsity_index(sparsityIndex.getValue());
    }
    if (storageElementIndex.hasValue()) {
        builder.add_storage_element_index(storageElementIndex.getValue());
    }
    if (storageElementSize.hasValue()) {
        builder.add_storage_element_size(storageElementSize.getValue());
    }
    return builder.Finish();
}

MVCNN::order3 vpux::VPUIP::BlobWriter::createOrder3(mlir::ArrayAttr attr) {
    auto vec = parseIntArrayAttr(attr);
    std::reverse(vec.begin(), vec.end());

    VPUX_THROW_UNLESS(vec.size() <= 3, "Got wrong order array : {0}", vec);

    uint8_t x = 0, y = 0, z = 0;
    if (vec.size() >= 1) {
        x = checked_cast<uint8_t>(vec[0]);
    }
    if (vec.size() >= 2) {
        y = checked_cast<uint8_t>(vec[1]);
    }
    if (vec.size() >= 3) {
        z = checked_cast<uint8_t>(vec[2]);
    }

    return MVCNN::order3(x, y, z);
}

VPUIP::BlobWriter::BinaryData vpux::VPUIP::BlobWriter::createBinaryData(mlir::DenseElementsAttr content,
                                                                        bool csram_cacheable) {
    auto type = content.getType().cast<mlir::ShapedType>();

    auto elemType = type.getElementType();
    const size_t elemTypeByteSize = elemType.getIntOrFloatBitWidth() / 8;

    const size_t totalNumElements = type.getNumElements();
    const size_t totalByteSize = totalNumElements * elemTypeByteSize;

    std::vector<uint64_t> alignedContent(alignVal(totalByteSize, sizeof(uint64_t)) / sizeof(uint64_t), 0);

    const auto rawData = content.getRawData();
    if (content.isSplat()) {
        loop_1d(LoopExecPolicy::Parallel, (totalByteSize / elemTypeByteSize), [&](int i) {
            auto dst = reinterpret_cast<uint8_t*>(alignedContent.data()) + i * elemTypeByteSize;
            std::copy_n(reinterpret_cast<const uint8_t*>(rawData.data()), elemTypeByteSize, dst);
        });
    } else {
        VPUX_THROW_UNLESS(rawData.size() == totalByteSize, "Raw Size mismatch for const content : '{0}' vs '{1}'",
                          rawData.size(), totalByteSize);
        std::copy_n(reinterpret_cast<const uint8_t*>(rawData.data()), totalByteSize,
                    reinterpret_cast<uint8_t*>(alignedContent.data()));
    }

    const auto serializedContent = createVector(alignedContent);

    MVCNN::BinaryDataBuilder builder(_impl);
    builder.add_underlying_type(MVCNN::DType::DType_U8);
    builder.add_length(totalByteSize);
    builder.add_data(serializedContent);
    builder.add_csram_cacheable(csram_cacheable);
    return builder.Finish();
}
