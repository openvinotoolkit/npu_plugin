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
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/ops_interfaces.hpp"

#include "vpux/utils/IE/float16.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/helper_macros.hpp"
#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>

#include <algorithm>

using namespace vpux;

VPUIP::BlobWriter::Task vpux::VPUIP::BlobWriter::createTask(mlir::Operation* op) {
    _log.trace("Create BLOB Task for {0}", *op);

    auto task = mlir::dyn_cast<VPUIP::TaskOpInterface>(op);
    VPUX_THROW_UNLESS(task != nullptr, "Got non Task operation {0}", op->getName());

    VPUX_THROW_UNLESS(_tasks.count(op) == 0, "Operation {0} was already serialized", *op);

    setAliasForSerializedTensors(op);

    String name;
    if (const auto nameLoc = op->getLoc().dyn_cast<mlir::NameLoc>()) {
        name = createString(nameLoc.getName().strref());
    }

    const auto waitBarriers = createVector(task.waitBarriers() | transformed([this](mlir::Value val) {
                                               return getBarrierVirtualID(val);
                                           }));
    const auto updateBarriers = createVector(task.updateBarriers() | transformed([this](mlir::Value val) {
                                                 return getBarrierVirtualID(val);
                                             }));

    MVCNN::BarrierReferenceBuilder barriersBuilder(_impl);
    barriersBuilder.add_wait_barriers(waitBarriers);
    barriersBuilder.add_virtual_wait_barriers(waitBarriers);
    barriersBuilder.add_update_barriers(updateBarriers);
    barriersBuilder.add_virtual_update_barriers(updateBarriers);
    const auto barriers = barriersBuilder.Finish();

    const auto specifics = task.serialize(*this);
    const auto curID = _tasks.size();

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

        auto available = resources.getExecutor(
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
    const std::vector<uint16_t> basePtrsVec = {};
    const auto basePtrsFb = createVector(basePtrsVec);

    const auto serializedName = createString(name);

    const auto serializedDataType = createDType(type.getElementType());
    const auto serializedDims = createDims(type);
    const auto serializedStrides = createStrides(type);
    const auto dimsOrder = DimsOrder::fromType(type);

    const auto serializedDataReference =
            createIndirectDataReference(dataIndex, sparsityIndex, storageElementIndex, storageElementSize);

    const auto serializedLocale = createMemoryLocation(locale);
    const auto serializedLocaleIndex = createVector(to_small_vector(makeArrayRef({localeIndex.getValueOr(0)})));

    Vector<uint8_t> quantZero;
    Vector<uint16_t> quantMult;
    Vector<uint8_t> quantShift;

    auto fixedPointFP16Mult = [](float val) {
        const static int BITS = 15;
        int exp;
        auto mantissa = std::frexp(val, &exp);
        return static_cast<uint16_t>(mantissa * std::pow(2, BITS));
    };

    auto fixedPointFP16Shift = [](float val) {
        const static int BITS = 15;
        int exp;
        std::frexp(val, &exp);
        return static_cast<uint8_t>(BITS - exp);
    };

    if (const auto qType = type.getElementType().dyn_cast<mlir::quant::UniformQuantizedType>()) {
        const auto zero = checked_cast<uint8_t>(qType.getZeroPoint());
        const auto mult = fixedPointFP16Mult(static_cast<float>(qType.getScale()));
        const auto shift = fixedPointFP16Shift(static_cast<float>(qType.getScale()));
        quantZero = createVector(makeArrayRef(zero));
        quantMult = createVector(makeArrayRef(mult));
        quantShift = createVector(makeArrayRef<uint8_t>({shift}));
    } else if (const auto qType = type.getElementType().dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        quantZero = createVector(qType.getZeroPoints() | transformed([](int64_t val) {
                                     return checked_cast<uint8_t>(val);
                                 }));
        quantMult = createVector(qType.getScales() | transformed([&](double val) {
                                     return fixedPointFP16Mult(static_cast<float>(val));
                                 }));
        quantShift = createVector(qType.getScales() | transformed([&](double val) {
                                      return fixedPointFP16Shift(static_cast<float>(val));
                                  }));
    } else {
        quantZero = createVector(makeArrayRef<uint8_t>(0));
        quantMult = createVector(makeArrayRef<uint16_t>(1));
        quantShift = createVector(makeArrayRef<uint8_t>({0}));
    }

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
    builder.add_order(dimsOrder.code());
    builder.add_base_ptrs(basePtrsFb);
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

VPUIP::BlobWriter::Barrier vpux::VPUIP::BlobWriter::createBarrier(mlir::Value val, uint32_t physicalID) {
    VPUX_THROW_UNLESS(_barriers.count(val) == 0, "Value {0} was already serialized", val);

    size_t numConsumers = 0;
    size_t numProducers = 0;
    for (auto userOp : val.getUsers()) {
        auto opEffects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(userOp);
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
            if (auto nceClusterTaskOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(userOp)) {
                for (auto dpuTaskOp : nceClusterTaskOp.variants().getOps<VPUIP::DPUTaskOp>()) {
                    VPUX_THROW_UNLESS(
                            dpuTaskOp.waitBarriers().size() == 0 && dpuTaskOp.updateBarriers().size() == 0,
                            "DPUTaskOp specific waits and updates still needs to be implemented and verified.");
                    ++numConsumers;
                }
            } else {
                ++numConsumers;
            }
        } else if (effect.getEffect() == mlir::MemoryEffects::Write::get()) {
            if (auto nceClusterTaskOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(userOp)) {
                for (auto dpuTaskOp : nceClusterTaskOp.variants().getOps<VPUIP::DPUTaskOp>()) {
                    VPUX_THROW_UNLESS(
                            dpuTaskOp.waitBarriers().size() == 0 && dpuTaskOp.updateBarriers().size() == 0,
                            "DPUTaskOp specific waits and updates still needs to be implemented and verified.");
                    ++numProducers;
                }
            } else {
                ++numProducers;
            }

        } else {
            VPUX_THROW("Barrier Value {0} has unsupported Effect in Operation {1}", val, *userOp);
        }
    }

    MVCNN::BarrierBuilder builder(_impl);
    builder.add_barrier_id(checked_cast<int16_t>(physicalID));
    builder.add_consumer_count(checked_cast<int16_t>(numConsumers));
    builder.add_producer_count(checked_cast<int16_t>(numProducers));
    const auto off = builder.Finish();

    _barriers.insert({val, checked_cast<uint32_t>(_barriers.size())});

    return off;
}

uint32_t vpux::VPUIP::BlobWriter::getBarrierVirtualID(mlir::Value val) const {
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
    } else if (type.isSignedInteger(CHAR_BIT * sizeof(int64_t))) {
        return MVCNN::DType_I64;
    } else if (type.isSignedInteger(CHAR_BIT * sizeof(int32_t))) {
        return MVCNN::DType_I32;
    } else if (type.isSignedInteger(CHAR_BIT * sizeof(int16_t))) {
        return MVCNN::DType_I16;
    } else if (type.isSignedInteger(CHAR_BIT * sizeof(int8_t))) {
        return MVCNN::DType_I8;
    } else if (type.isInteger(CHAR_BIT * sizeof(uint64_t))) {
        return MVCNN::DType_U64;
    } else if (type.isInteger(CHAR_BIT * sizeof(uint32_t))) {
        return MVCNN::DType_U32;
    } else if (type.isInteger(CHAR_BIT * sizeof(uint16_t))) {
        return MVCNN::DType_U16;
    } else if (type.isInteger(CHAR_BIT * sizeof(uint8_t))) {
        return MVCNN::DType_U8;
    } else if (type.isInteger(4)) {
        return MVCNN::DType_I4;
    } else if (type.isInteger(2)) {
        return MVCNN::DType_I2;
    } else if (type.isInteger(1)) {
        return MVCNN::DType_BIN;
    } else if (type.isa<mlir::quant::QuantizedType>()) {
        return createDType(type.cast<mlir::quant::QuantizedType>().getStorageType());
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

VPUIP::BlobWriter::Vector<float> vpux::VPUIP::BlobWriter::createStrides(StridesRef strides, Bit elemSize) {
    Strides temp;
    temp.push_back(elemSize);
    temp.append(strides.begin(), strides.end());

    const auto cvtBitStrideToByteFP = [](Bit val) {
        if (val.count() % CHAR_BIT == 0) {
            return checked_cast<float>(Byte(val).count());
        }

        return checked_cast<float>(val.count()) / CHAR_BIT;
    };

    return createVector(temp | transformed(cvtBitStrideToByteFP));
}

VPUIP::BlobWriter::Vector<float> vpux::VPUIP::BlobWriter::createStrides(mlir::MemRefType type) {
    return createStrides(getStrides(type), getElemTypeSize(type));
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

VPUIP::BlobWriter::BinaryData vpux::VPUIP::BlobWriter::createBinaryData(ConstContentAttr content,
                                                                        mlir::MemRefType actualType,
                                                                        bool csram_cacheable) {
    const Byte elemTypeSize = getElemTypeSize(actualType);
    const size_t totalNumElements = actualType.getNumElements();
    const size_t totalByteSize = totalNumElements * elemTypeSize.count();

    std::vector<uint64_t> alignedContent(alignVal(totalByteSize, sizeof(uint64_t)) / sizeof(uint64_t), 0);

    const auto buf = makeMutableArrayRef(reinterpret_cast<char*>(alignedContent.data()), totalByteSize);
    content.convertTo(actualType, buf);

    const auto serializedContent = createVector(alignedContent);

    MVCNN::BinaryDataBuilder builder(_impl);
    builder.add_underlying_type(MVCNN::DType::DType_U8);
    builder.add_length(totalByteSize);
    builder.add_data(serializedContent);
    builder.add_csram_cacheable(csram_cacheable);
    return builder.Finish();
}

void vpux::VPUIP::BlobWriter::setAliasForSerializedTensors(mlir::Operation* op) {
    if (auto layer = mlir::dyn_cast<mlir::ViewLikeOpInterface>(op)) {
        const auto result = layer->getResult(0);
        const auto source = layer.getViewSource();

        VPUX_THROW_UNLESS(result.getType().isa<mlir::MemRefType>(), "Only MemRef type tensors are supported, got '{0}'",
                          result.getType());
        VPUX_THROW_UNLESS(source.getType().isa<mlir::MemRefType>(), "Only MemRef type tensors are supported, got '{0}'",
                          source.getType());

        _tensors.insert({result, getTensor(source)});
    } else if (auto multiLayer = mlir::dyn_cast<MultiViewOpInterface>(op)) {
        for (const auto result : multiLayer->getResults()) {
            VPUX_THROW_UNLESS(result.getType().isa<mlir::MemRefType>(),
                              "Only MemRef type tensors are supported, got '{0}'", result.getType());

            const auto source = multiLayer.getViewSource(result.getResultNumber());
            if (source == nullptr) {
                continue;
            }

            VPUX_THROW_UNLESS(source.getType().isa<mlir::MemRefType>(),
                              "Only MemRef type tensors are supported, got '{0}'", source.getType());

            _tensors.insert({result, getTensor(source)});
        }
    }
}
