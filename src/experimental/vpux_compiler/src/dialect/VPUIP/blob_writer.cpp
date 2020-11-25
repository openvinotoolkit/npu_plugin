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

#include "vpux/compiler/core/dims_order.hpp"
#include "vpux/compiler/dialect/VPUIP/effects.hpp"
#include "vpux/compiler/dialect/VPUIP/ops_interfaces.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/helper_macros.hpp"
#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/core/small_vector.hpp"

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
                                                                            const SoftwareLayerParams& params,
                                                                            int32_t maxShaves, bool isTrailingSWLayer) {
    auto task = mlir::cast<VPUIP::TaskOpInterface>(op);

    const auto getTensorCb = [this](mlir::Value val) {
        return getTensor(val);
    };

    const auto inputs = createVector(task.inputTensors() | transformed(getTensorCb));
    const auto outputs = createVector(task.outputTensors() | transformed(getTensorCb));

    MVCNN::UPALayerTaskBuilder builder(_impl);
    builder.add_maxShaves(checked_cast<uint8_t>(maxShaves));
    builder.add_softLayerParams_type(params.type);
    builder.add_softLayerParams(params.obj);
    builder.add_inputs(inputs);
    builder.add_outputs(outputs);
    builder.add_isTrailingSWLayer(isTrailingSWLayer);
    return {builder.Finish().Union(), MVCNN::SpecificTask_UPALayerTask};
}

VPUIP::BlobWriter::TensorReference vpux::VPUIP::BlobWriter::createTensor(StringRef name, mlir::MemRefType type,
                                                                         MemoryLocation location, uint64_t offset) {
    const auto serializedName = createString(name);
    const auto serializedDataType = createDType(type.getElementType());
    const auto serializedDims = createDims(type);
    const auto serializedStrides = createStrides(type);
    const auto serializedLocation = createMemoryLocation(location);
    const auto serializedOffset = createIndirectDataReference(offset);

    const auto localeIndex = createVector(makeArrayRef<uint32_t>({0}));
    const auto quantZero = createVector(makeArrayRef<uint8_t>({0}));
    const auto quantMult = createVector(makeArrayRef<uint16_t>({1}));
    const auto quantShift = createVector(makeArrayRef<uint8_t>({0}));

    const auto dimsOrder = DimsOrder::fromType(type);
    VPUX_THROW_UNLESS(dimsOrder.hasValue(), "Can't get DimsOrder from MemRef Type {0}", type);

    MVCNN::TensorReferenceBuilder builder(_impl);
    builder.add_name(serializedName);
    builder.add_dimensions(serializedDims);
    builder.add_strides(serializedStrides);
    builder.add_data(serializedOffset);
    builder.add_locale(serializedLocation);
    builder.add_locale_index(localeIndex);
    builder.add_data_dtype(serializedDataType);
    builder.add_quant_zero(quantZero);
    builder.add_quant_mult(quantMult);
    builder.add_quant_shift(quantShift);
    builder.add_order(dimsOrder->code());
    return builder.Finish();
}

VPUIP::BlobWriter::TensorReference vpux::VPUIP::BlobWriter::createTensor(mlir::Value val, StringRef name,
                                                                         MemoryLocation location, uint64_t offset) {
    VPUX_THROW_UNLESS(_tensors.count(val) == 0, "Value {0} was already serialized", val);

    const auto off = createTensor(name, val.getType().cast<mlir::MemRefType>(), location, offset);

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
        SmallVector<MemEffect, 1> valEffects;

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
    return createStrides(getStrides(type), type.getElementTypeBitWidth() / 8);
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

VPUIP::BlobWriter::IndirectDataReference vpux::VPUIP::BlobWriter::createIndirectDataReference(uint64_t offset) {
    MVCNN::IndirectDataReferenceBuilder builder(_impl);
    builder.add_data_index(offset);
    return builder.Finish();
}

VPUIP::BlobWriter::BinaryData vpux::VPUIP::BlobWriter::createBinaryData(mlir::DenseElementsAttr content,
                                                                        bool csram_cacheable) {
    auto type = content.getType().cast<mlir::ShapedType>();

    auto elemType = type.getElementType();
    const size_t elemTypeByteSize = elemType.getIntOrFloatBitWidth() / 8;

    const size_t totalNumElements = type.getNumElements();
    const size_t totalByteSize = totalNumElements * elemTypeByteSize;

    std::vector<uint64_t> alignedContent(alignVal(totalByteSize, sizeof(uint64_t)));

    const auto rawData = content.getRawData();
    VPUX_THROW_UNLESS(rawData.size() == totalByteSize, "Raw Size mismatch for const content : {0} vs {1}",
                      rawData.size(), totalByteSize);

    std::copy_n(reinterpret_cast<const uint8_t*>(rawData.data()), totalByteSize,
                reinterpret_cast<uint8_t*>(alignedContent.data()));

    const auto serializedDataType = createDType(elemType);
    const auto serializedContent = createVector(alignedContent);

    MVCNN::BinaryDataBuilder builder(_impl);
    builder.add_underlying_type(serializedDataType);
    builder.add_length(checked_cast<uint64_t>(totalByteSize));
    builder.add_data(serializedContent);
    builder.add_csram_cacheable(csram_cacheable);
    return builder.Finish();
}
