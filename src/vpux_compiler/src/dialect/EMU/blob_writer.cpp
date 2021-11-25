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

#include "vpux/compiler/dialect/EMU/blob_writer.hpp"

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/dialect/EMU/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/strings.hpp"

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

EMU::BlobWriter::Task vpux::EMU::BlobWriter::createTask(mlir::Operation* op) {
    _log.trace("Create BLOB Task for {0}", *op);

    auto task = mlir::dyn_cast<EMUSerializeInterface>(op);
    VPUX_THROW_UNLESS(task != nullptr, "Got non Task operation {0}", op->getName());

    VPUX_THROW_UNLESS(_tasks.count(op) == 0, "Operation {0} was already serialized", *op);

    String name = createString(StringRef(stringifyLocation(op->getLoc())));

    const auto specifics = task.serialize(*this);
    const auto curID = _tasks.size();

    MVCNN::TaskBuilder builder(_impl);
    if (!name.IsNull()) {
        builder.add_name(name);
    }
    builder.add_nodeID(checked_cast<uint32_t>(curID));
    builder.add_task_type(specifics.type);
    builder.add_task(specifics.obj);
    const auto off = builder.Finish();

    _tasks.insert({op, off});

    return off;
}

EMU::BlobWriter::SpecificTask vpux::EMU::BlobWriter::createUPALayerTask(mlir::Operation* op,
                                                                        const SoftwareLayerParams& params) {
    VPUX_THROW_UNLESS(op != nullptr, "Got NULL pointer in createUPALayerTask");

    const auto getTensorCb = [this](mlir::Value val) {
        return getTensor(val);
    };

    const auto inputs = createVector(op->getOperands() | transformed(getTensorCb));
    const auto outputs = createVector(op->getResults() | transformed(getTensorCb));

    MVCNN::UPALayerTaskBuilder builder(_impl);
    builder.add_softLayerParams_type(params.type);
    builder.add_softLayerParams(params.obj);
    builder.add_inputs(inputs);
    builder.add_outputs(outputs);
    return {builder.Finish().Union(), MVCNN::SpecificTask_UPALayerTask};
}

EMU::BlobWriter::TensorReference vpux::EMU::BlobWriter::createTensor(StringRef name, mlir::ShapedType type,
                                                                     ArrayRef<uint16_t> mult, ArrayRef<uint8_t> shift,
                                                                     int8_t postShift, ArrayRef<uint8_t> zeroPoints,
                                                                     VPURT::BufferSection locale,
                                                                     const uint32_t localeIndex) {
    const auto serializedName = createString(name);

    const auto serializedDataType = VPUIP::createDType(type.getElementType());
    const auto serializedDims = createDims(type);
    const auto serializedStrides = createStrides(type);
    const auto dimsOrder = DimsOrder::fromType(type);

    Vector<uint8_t> serializedQuantZero = createVector(zeroPoints);
    Vector<uint16_t> serializedQuantMult = createVector(mult);
    Vector<uint8_t> serializedQuantShift = createVector(shift);

    const auto serializedLocale = VPUIP::createMemoryLocation(locale);
    const std::vector<uint32_t> localeIndexVec = {localeIndex};
    Vector<uint32_t> serializedLocaleIndex = _impl.CreateVector(localeIndexVec.data(), localeIndexVec.size());

    MVCNN::TensorReferenceBuilder builder(_impl);
    builder.add_name(serializedName);
    builder.add_dimensions(serializedDims);
    builder.add_strides(serializedStrides);
    builder.add_locale(serializedLocale);
    builder.add_locale_index(serializedLocaleIndex);
    builder.add_data_dtype(serializedDataType);
    builder.add_quant_zero(serializedQuantZero);
    builder.add_quant_mult(serializedQuantMult);
    builder.add_quant_shift(serializedQuantShift);
    builder.add_quant_post_shift_right(postShift);
    builder.add_order(dimsOrder.code());
    return builder.Finish();
}

EMU::BlobWriter::TensorReference vpux::EMU::BlobWriter::createTensor(StringRef name, mlir::ShapedType type,
                                                                     VPURT::BufferSection locale,
                                                                     const uint32_t localeIndex) {
    std::vector<uint8_t> zeroPoints;
    std::vector<uint16_t> mult;
    std::vector<uint8_t> shift;

    if (const auto qType = type.getElementType().dyn_cast<mlir::quant::UniformQuantizedType>()) {
        zeroPoints.push_back(checked_cast<uint8_t>(qType.getZeroPoint()));
        mult.push_back(getQuantMultFromScale(qType.getScale()));
        shift.push_back(getQuantShiftFromScale(qType.getScale()));
    } else if (const auto qType = type.getElementType().dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        auto qtype_quant_zp = qType.getZeroPoints();
        auto qtype_quant_scale = qType.getScales();
        zeroPoints.resize(qtype_quant_zp.size());
        mult.resize(qtype_quant_scale.size());
        shift.resize(qtype_quant_scale.size());

        std::transform(qtype_quant_zp.begin(), qtype_quant_zp.end(), zeroPoints.begin(), [](int64_t val) {
            return checked_cast<uint8_t>(val);
        });
        std::transform(qtype_quant_scale.begin(), qtype_quant_scale.end(), mult.begin(), getQuantMultFromScale);
        std::transform(qtype_quant_scale.begin(), qtype_quant_scale.end(), shift.begin(), getQuantShiftFromScale);
    } else {
        zeroPoints.push_back(0);
        mult.push_back(1);
        shift.push_back(0);
    }

    return createTensor(name, type, mult, shift, 0, zeroPoints, locale, localeIndex);
}

EMU::BlobWriter::TensorReference vpux::EMU::BlobWriter::createTensor(mlir::Value val, StringRef name,
                                                                     VPURT::BufferSection locale,
                                                                     const uint32_t localeIndex) {
    VPUX_THROW_UNLESS(_tensors.count(val) == 0, "Value {0} was already serialized", val);

    const auto off = createTensor(name, val.getType().cast<mlir::ShapedType>(), locale, localeIndex);

    _tensors.insert({val, off});

    return off;
}

EMU::BlobWriter::TensorReference vpux::EMU::BlobWriter::getTensor(mlir::Value val) const {
    const auto it = _tensors.find(val);
    VPUX_THROW_UNLESS(it != _tensors.end(), "Value {0} wasn't serialized yet", val);
    return it->second;
}

EMU::BlobWriter::Vector<uint32_t> vpux::EMU::BlobWriter::createDims(ShapeRef shape) {
    return createVector(shape | transformed([](int64_t val) {
                            return checked_cast<uint32_t>(val);
                        }));
}

EMU::BlobWriter::Vector<uint32_t> vpux::EMU::BlobWriter::createDims(mlir::ShapedType type) {
    return createDims(getShape(type));
}

EMU::BlobWriter::Vector<float> vpux::EMU::BlobWriter::createStrides(StridesRef strides, Bit elemSize) {
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

EMU::BlobWriter::Vector<float> vpux::EMU::BlobWriter::createStrides(mlir::ShapedType type) {
    const auto order = DimsOrder::fromType(type);

    const auto stridesReqs = StrideReqs::simple(checked_cast<size_t>(type.getRank()));
    const auto memStrides = stridesReqs.calcStrides(order, type);
    const auto strides = order.toLogicalOrder(memStrides);

    return createStrides(strides, getElemTypeSize(type));
}

EMU::BlobWriter::BinaryData vpux::EMU::BlobWriter::createBinaryData(ArrayRef<uint64_t> content, mlir::ShapedType type) {
    const Byte elemTypeSize = getElemTypeSize(type);
    const size_t totalNumElements = type.getNumElements();
    const size_t totalByteSize = totalNumElements * elemTypeSize.count();

    const auto serializedContent = createVector(content);

    MVCNN::BinaryDataBuilder builder(_impl);
    builder.add_underlying_type(MVCNN::DType::DType_U8);
    builder.add_length(totalByteSize);
    builder.add_data(serializedContent);
    return builder.Finish();
}
