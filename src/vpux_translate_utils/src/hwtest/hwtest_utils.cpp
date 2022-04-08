//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/hwtest/hwtest_utils.hpp"

#include "vpux/compiler/dialect/VPUIP/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/IE/float16.hpp"
#include "vpux/utils/core/error.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>

#include <numeric>

namespace vpux {
namespace hwtest {

namespace {

mlir::Type parseType(mlir::OpBuilder builder, mlir::Type ty, const nb::QuantParams& qp) {
    auto intTy = ty.dyn_cast<mlir::IntegerType>();
    if (qp.present && intTy) {
        ty = mlir::quant::UniformQuantizedType::get(intTy.isSigned() ? mlir::quant::QuantizationFlags::Signed : 0, ty,
                                                    builder.getF32Type(), qp.scale, qp.zeropoint, qp.low_range,
                                                    qp.high_range);
    }
    return ty;
}

mlir::Type convertToMLIRType(mlir::OpBuilder builder, nb::DType dtype) {
    auto ctx = builder.getContext();
    switch (dtype) {
    case nb::DType::U4:
        return getUInt4Type(ctx);
    case nb::DType::U8:
        return getUInt8Type(ctx);
    case nb::DType::I4:
        return getSInt4Type(ctx);
    case nb::DType::I8:
        return getSInt8Type(ctx);
    case nb::DType::I32:
        return getSInt32Type(ctx);
    case nb::DType::FP8:
        return builder.getF16Type();
    case nb::DType::FP16:
        return builder.getF16Type();
    case nb::DType::FP32:
        return builder.getF32Type();
    case nb::DType::BF16:
        return builder.getBF16Type();
    default:
        throw std::domain_error{"Expected a valid data type"};
    }
}

template <class StorageType>
mlir::DenseElementsAttr generateWeights(std::ifstream& stream, mlir::RankedTensorType type, std::size_t elementsCount) {
    if (!stream) {
        auto generatedElements = std::vector<StorageType>(elementsCount);
        // have to add at least one non-zero element to make attribute non-splat. BitPack can't
        // work with splat tensors
        generatedElements[0] = 1;
        return mlir::DenseElementsAttr::get(type, llvm::makeArrayRef<StorageType>(generatedElements));
    }

    std::vector<StorageType> buffer(elementsCount);
    const auto expectedBytesCountToRead = elementsCount * sizeof(StorageType);
    // read as bytes since FP16/BFP16 are not supported by C++ standard
    stream.read(reinterpret_cast<char*>(buffer.data()), expectedBytesCountToRead);

    const auto actualBytesCountRead = static_cast<std::size_t>(stream.gcount());
    const auto state = stream.rdstate();

    if (expectedBytesCountToRead == actualBytesCountRead) {
        return mlir::DenseElementsAttr::get(type, llvm::makeArrayRef<StorageType>(buffer));
    }

    VPUX_THROW_UNLESS((state & std::ifstream::eofbit) == 0,
                      "Failed to read {0} bytes from weights file, read {1} bytes before EOF has been reached",
                      expectedBytesCountToRead, actualBytesCountRead);
    VPUX_THROW_UNLESS(
            (state & std::ifstream::failbit) == 0,
            "Failed to read {0} bytes from weights file, read {1} bytes before logical error on i/o operation occurred",
            expectedBytesCountToRead, actualBytesCountRead);
    VPUX_THROW_UNLESS(
            (state & std::ifstream::badbit) == 0,
            "Failed to read {0} bytes from weights file, read {1} bytes before read error on i/o operation occurred",
            expectedBytesCountToRead, actualBytesCountRead);
    VPUX_THROW("Unexpected std::ifstream::rdstate value {}", state);
}

}  // namespace

mlir::DenseElementsAttr generateWeights(llvm::ArrayRef<int64_t> shape, mlir::Type type, mlir::MLIRContext* context,
                                        const char* weightsFileName) {
    VPUX_THROW_UNLESS(!shape.empty(), "generateWeights: Got empty shape");
    auto wtData_ddr_valueType = mlir::RankedTensorType::get(shape, type);
    const auto vecSize = static_cast<std::size_t>(
            std::accumulate(shape.begin(), shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>()));

    if (auto qtype = type.dyn_cast_or_null<mlir::quant::QuantizedType>()) {
        type = mlir::quant::QuantizedType::castToStorageType(qtype);
        if (qtype.getFlags() & mlir::quant::QuantizationFlags::Signed) {
            wtData_ddr_valueType = mlir::RankedTensorType::get(shape, getSInt8Type(context));
        } else {
            wtData_ddr_valueType = mlir::RankedTensorType::get(shape, getUInt8Type(context));
        }
    }

    std::ifstream stream{weightsFileName, std::ios::in | std::ios::binary};
    if (!stream) {
        std::cerr << "Warning: Unable to open weight data file " << weightsFileName << '\n';
    }

    if (type.isInteger(4)) {
        std::vector<int64_t> uintWrapperShape{shape.begin(), shape.end()};
        VPUX_THROW_UNLESS(!uintWrapperShape.empty(), "generateWeights: Got empty shape");
        // in NHWC tensor two int4 neighboring elements by C axis will be united into one uint8 element. So we have to
        // recalculate shape for uint wrapper tensor
        uintWrapperShape[Dims4D::Filter::OC.ind()] /= 2;

        const auto wrapperVecSize = static_cast<std::size_t>(std::accumulate(
                uintWrapperShape.begin(), uintWrapperShape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>()));

        auto uintWrapperValueType = mlir::RankedTensorType::get(uintWrapperShape, getUInt8Type(context));
        const auto weightsPacked = generateWeights<uint8_t>(stream, uintWrapperValueType, wrapperVecSize);
        std::vector<std::uint8_t> weightsUnpacked;
        weightsUnpacked.reserve(weightsPacked.size() * 2);
        for (const auto& elemPacked : weightsPacked.getValues<uint8_t>()) {
            const int8_t msn = (elemPacked & 0xf0) >> 4;
            const int8_t lsn = (elemPacked & 0x0f) >> 0;
            weightsUnpacked.push_back(lsn);
            weightsUnpacked.push_back(msn);
        }
        VPUX_THROW_UNLESS(
                weightsUnpacked.size() == vecSize,
                "Warning: count of elements in weights file {0} doesn't match with provided weights shape {1}",
                weightsUnpacked.size(), shape);

        if (type.isSignedInteger(4)) {
            return mlir::DenseElementsAttr::get(
                    wtData_ddr_valueType,
                    makeArrayRef(reinterpret_cast<const int8_t*>(weightsUnpacked.data()), weightsUnpacked.size()));
        } else {
            return mlir::DenseElementsAttr::get(wtData_ddr_valueType, makeArrayRef(weightsUnpacked));
        }
    } else if (type.isSignedInteger(8)) {
        return generateWeights<std::int8_t>(stream, wtData_ddr_valueType, vecSize);
    } else if (type.isInteger(8)) {
        return generateWeights<std::uint8_t>(stream, wtData_ddr_valueType, vecSize);
    } else if (type.isF16()) {
        return generateWeights<float16>(stream, wtData_ddr_valueType, vecSize);
    } else if (type.isBF16()) {
        return generateWeights<bfloat16>(stream, wtData_ddr_valueType, vecSize);
    } else if (type.isF32()) {
        return generateWeights<float>(stream, wtData_ddr_valueType, vecSize);
    } else {
        VPUX_THROW("Unexpected weights data type: {0}", type);
    }
}

std::size_t totalTensorSize(llvm::ArrayRef<int64_t> shape, mlir::Type elementType) {
    if (auto qType = elementType.dyn_cast<mlir::quant::UniformQuantizedType>()) {
        elementType = qType.getStorageType();
    }
    size_t numBits = elementType.getIntOrFloatBitWidth();

    const auto totalSize =
            std::accumulate(shape.begin(), shape.end(), static_cast<std::int64_t>(1), std::multiplies<std::int64_t>());
    const auto totalBits = totalSize * numBits;
    VPUX_THROW_UNLESS(totalBits % CHAR_BIT == 0, "Tensors size is not allligned to Byte");
    return static_cast<std::size_t>(totalBits / CHAR_BIT);
}

std::vector<int64_t> convertNBPadtoNCETaskPad(const std::array<int64_t, 4>& nb_pad) {
    std::vector<int64_t> ncetask_pad(nb_pad.size());

    ncetask_pad[PAD_NCETASK_LEFT] = nb_pad[PAD_NB_LEFT];
    ncetask_pad[PAD_NCETASK_RIGHT] = nb_pad[PAD_NB_RIGHT];
    ncetask_pad[PAD_NCETASK_TOP] = nb_pad[PAD_NB_TOP];
    ncetask_pad[PAD_NCETASK_BOTTOM] = nb_pad[PAD_NB_BOTTOM];

    return ncetask_pad;
}

mlir::Type parseInputType(mlir::OpBuilder builder, const nb::InputLayer& input) {
    return parseType(builder, convertToMLIRType(builder, input.dtype), input.qp);
}

mlir::Type parseOutputType(mlir::OpBuilder builder, const nb::OutputLayer& output) {
    return parseType(builder, convertToMLIRType(builder, output.dtype), output.qp);
}

mlir::Type parseWeightsType(mlir::OpBuilder builder, const nb::WeightLayer& weight) {
    return parseType(builder, convertToMLIRType(builder, weight.dtype), weight.qp);
}

void buildCNNOp(mlir::OpBuilder& builder, llvm::StringRef mainFuncName, llvm::ArrayRef<mlir::Type> inputs,
                llvm::ArrayRef<mlir::Type> outputs) {
    const auto mainFuncNameAttr = mlir::SymbolRefAttr::get(builder.getContext(), mainFuncName);
    auto cnnOp = builder.create<IE::CNNNetworkOp>(builder.getUnknownLoc(), mainFuncNameAttr, false);
    cnnOp.inputsInfo().emplaceBlock();
    cnnOp.outputsInfo().emplaceBlock();

    auto inputsInfoBuilder = mlir::OpBuilder::atBlockBegin(&cnnOp.inputsInfo().front(), builder.getListener());
    for (auto input : enumerate(inputs)) {
        auto inputType = input.value().cast<mlir::ShapedType>();
        if (auto quantized = inputType.getElementType().dyn_cast_or_null<mlir::quant::UniformQuantizedType>()) {
            inputType = inputType.clone(quantized.getStorageType());
        }

        const auto inputName = printToString("input_{0}", input.index());
        const auto nameAttr = builder.getStringAttr(inputName);
        const auto userTypeAttr = mlir::TypeAttr::get(inputType);
        inputsInfoBuilder.create<IE::DataInfoOp>(builder.getUnknownLoc(), nameAttr, userTypeAttr);
    }

    auto outputsInfoBuilder = mlir::OpBuilder::atBlockBegin(&cnnOp.outputsInfo().front(), builder.getListener());
    for (auto output : enumerate(outputs)) {
        auto outputType = output.value().cast<mlir::ShapedType>();
        if (auto quantized = outputType.getElementType().dyn_cast_or_null<mlir::quant::UniformQuantizedType>()) {
            outputType = outputType.clone(quantized.getStorageType());
        }

        const auto resultName = printToString("output_{0}", output.index());
        const auto nameAttr = builder.getStringAttr(resultName);
        const auto userTypeAttr = mlir::TypeAttr::get(outputType);
        outputsInfoBuilder.create<IE::DataInfoOp>(builder.getUnknownLoc(), nameAttr, userTypeAttr);
    }
}

mlir::DenseElementsAttr splitWeightsOverC(mlir::DenseElementsAttr wt_vec, ArrayRef<int64_t> wt_shape, mlir::Type dtype,
                                          mlir::MLIRContext* ctx, size_t start_C, size_t end_C) {
    auto qType = dtype.dyn_cast<mlir::quant::UniformQuantizedType>();
    if (!((dtype.isF16() || (qType && qType.getStorageType().isUnsignedInteger(8)))))
        throw std::domain_error{
                printToString("splitWeightsOverC only supports weight data type fp16 or uint8; got {0}", dtype)};

    if (dtype.isF16()) {
        float16 elementType = 0;
        return splitWeightsOverCLoop(wt_vec, wt_shape, dtype, elementType, ctx, start_C, end_C);
    } else {
        uint8_t elementType = 0;
        return splitWeightsOverCLoop(wt_vec, wt_shape, dtype, elementType, ctx, start_C, end_C);
    }
}

template <typename T>
mlir::DenseElementsAttr splitWeightsOverCLoop(mlir::DenseElementsAttr wt_vec, ArrayRef<int64_t> wt_shape,
                                              mlir::Type dtype, T, mlir::MLIRContext* ctx, size_t start_C,
                                              size_t end_C) {
    SmallVector<int64_t> original_shape(wt_shape.begin(), wt_shape.end());

    // Weights from NumericsBench in KCHW
    // For stream-over-C, need to extract weights[startC:endC]
    int64_t K = original_shape[0];
    int64_t C = original_shape[1];
    int64_t H = original_shape[2];
    int64_t W = original_shape[3];
    int64_t new_C = end_C - start_C;

    auto wt_full_itr = wt_vec.getValues<T>();
    std::vector<T> wt_full(wt_full_itr.begin(), wt_full_itr.end());
    const llvm::SmallVector<int64_t> wt_partial_shape({K, new_C, H, W});
    size_t vecSize = static_cast<size_t>(std::accumulate(wt_partial_shape.begin(), wt_partial_shape.end(),
                                                         static_cast<int64_t>(1), std::multiplies<int64_t>()));
    std::vector<T> wt_partial(vecSize);

    // std::cout << "k, c, h, w, old_offset, new_offset\n";
    for (int64_t k = 0; k < K; k++)
        for (int64_t c = 0; c < new_C; c++)
            for (int64_t h = 0; h < H; h++)
                for (int64_t w = 0; w < W; w++) {
                    // auto old_c = c+start_C;
                    auto old_offset = k * C * H * W + (c + start_C) * H * W + h * W + w;
                    auto new_offset = k * new_C * H * W + c * H * W + h * W + w;
                    wt_partial[new_offset] = wt_full[old_offset];
                    // std::cout << k << ", " << c << ", " << h << ", " << w << ", " << old_offset << ", " << new_offset
                    // << "\n";
                }

    auto wtData_ddr_valueType = mlir::RankedTensorType::get(wt_partial_shape, dtype);
    if (auto qtype = dtype.dyn_cast<mlir::quant::QuantizedType>()) {
        dtype = mlir::quant::QuantizedType::castToStorageType(qtype);
        if (qtype.getFlags() & mlir::quant::QuantizationFlags::Signed) {
            wtData_ddr_valueType = mlir::RankedTensorType::get(wt_partial_shape, getSInt8Type(ctx));
        } else {
            wtData_ddr_valueType = mlir::RankedTensorType::get(wt_partial_shape, getUInt8Type(ctx));
        }
    }

    auto wt_data_values = makeArrayRef<T>(wt_partial);
    auto wt_data_vals = mlir::DenseElementsAttr::get(wtData_ddr_valueType, wt_data_values);
    return wt_data_vals;
}

mlir::MemRefType getMemRefType(VPURT::BufferSection section, ArrayRef<int64_t> shape, mlir::Type elemType,
                               DimsOrder order) {
    return vpux::getMemRefType(ShapeRef(shape), elemType, order, VPURT::getMemoryKind(section));
}

mlir::MemRefType getMemRefType(VPURT::BufferSection section, size_t sectionIdx, ArrayRef<int64_t> shape,
                               mlir::Type elemType, DimsOrder order) {
    auto symbolAttr =
            IndexedSymbolAttr::get(elemType.getContext(), stringifyEnum(VPURT::getMemoryKind(section)), sectionIdx);
    return vpux::getMemRefType(ShapeRef(shape), elemType, order, symbolAttr);
}

mlir::MemRefType getMemRefType(VPURT::BufferSection section, ArrayRef<int64_t> shape, mlir::Type elemType,
                               DimsOrder order, StridesRef strides) {
    return vpux::getMemRefType(ShapeRef(shape), elemType, order, strides, VPURT::getMemoryKind(section));
}

mlir::MemRefType getMemRefType(VPURT::BufferSection section, size_t sectionIdx, ArrayRef<int64_t> shape,
                               mlir::Type elemType, DimsOrder order, StridesRef strides) {
    auto symbolAttr =
            IndexedSymbolAttr::get(elemType.getContext(), stringifyEnum(VPURT::getMemoryKind(section)), sectionIdx);
    return vpux::getMemRefType(ShapeRef(shape), elemType, order, strides, symbolAttr);
}

vpux::VPURT::DeclareBufferOp createDeclareTensorOp(mlir::OpBuilder& builder, VPURT::BufferSection section,
                                                   ArrayRef<int64_t> shape, mlir::Type elemType, DimsOrder order,
                                                   int64_t locale, size_t offset) {
    const auto type = getMemRefType(section, locale, shape, elemType, order);
    return builder.create<VPURT::DeclareBufferOp>(builder.getUnknownLoc(), type, section, locale, offset);
}
vpux::VPURT::DeclareBufferOp createDeclareTensorOp(mlir::OpBuilder builder, VPURT::BufferSection section,
                                                   ArrayRef<int64_t> shape, mlir::Type elemType, DimsOrder order,
                                                   StridesRef strides, int64_t locale, size_t offset) {
    const auto type = getMemRefType(section, locale, shape, elemType, order, strides);
    return builder.create<VPURT::DeclareBufferOp>(builder.getUnknownLoc(), type, section, locale, offset);
}
vpux::VPURT::DeclareBufferOp createDeclareTensorOp(mlir::OpBuilder& builder, mlir::MemRefType type,
                                                   VPURT::BufferSection section, int64_t locale, size_t offset) {
    return builder.create<VPURT::DeclareBufferOp>(builder.getUnknownLoc(), type, section, locale, offset);
}

vpux::VPURT::DeclareSparseBufferOp createDeclareSparseTensorOp(mlir::OpBuilder& builder, mlir::Value data,
                                                               mlir::Value sparsityMap) {
    return builder.create<VPURT::DeclareSparseBufferOp>(builder.getUnknownLoc(), data, sparsityMap);
}

mlir::OpResult getTensorResult(VPURT::DeclareBufferOp op) {
    return op.getOperation()->getResult(0);
}

mlir::OpResult getConstResult(vpux::Const::DeclareOp op) {
    return op.getOperation()->getResult(0);
}

vpux::VPUIP::DPUTaskOp createDPUTaskOp(mlir::OpBuilder builder, mlir::OpBuilder variantbuilder,
                                       llvm::SmallVector<int64_t> output_shape, std::vector<int64_t> padding_vec) {
    std::vector<int64_t> start_vec{0, 0, 0};
    auto start = getIntArrayAttr(builder, start_vec);
    std::vector<int64_t> end_vec{static_cast<int64_t>(output_shape[2] - 1), static_cast<int64_t>(output_shape[3] - 1),
                                 static_cast<int64_t>(output_shape[1] - 1)};
    auto end = getIntArrayAttr(builder, end_vec);
    auto pad = VPU::getPaddingAttr(builder.getContext(), padding_vec[PAD_NCETASK_LEFT], padding_vec[PAD_NCETASK_RIGHT],
                                   padding_vec[PAD_NCETASK_TOP], padding_vec[PAD_NCETASK_BOTTOM]);

    auto dpuTask = variantbuilder.create<VPUIP::DPUTaskOp>(builder.getUnknownLoc(), start, end, pad,
                                                           VPU::MPEMode::CUBOID_16x16);

    return dpuTask;
}

vpux::DimsOrder oduPermutationToLayout(const MVCNN::Permutation oduPermutation) {
    switch (oduPermutation) {
    case MVCNN::Permutation::Permutation_ZXY:
        return vpux::DimsOrder::NHWC;
    case MVCNN::Permutation::Permutation_ZYX:
        return vpux::DimsOrder::fromCode(0x1432);  // NWHC
    case MVCNN::Permutation::Permutation_YZX:
        return vpux::DimsOrder::fromCode(0x1423);  // NWCH
    case MVCNN::Permutation::Permutation_YXZ:
        return vpux::DimsOrder::fromCode(0x1243);  // NCWH
    case MVCNN::Permutation::Permutation_XZY:
        return vpux::DimsOrder::NHCW;
    case MVCNN::Permutation::Permutation_XYZ:
        return vpux::DimsOrder::NCHW;
    default:
        return vpux::DimsOrder::NHWC;
    }
}

vpux::Dim getInnermostDim(const vpux::DimsOrder& order) {
    return order.toDim(MemDim(order.numDims() - 1));
}

}  // namespace hwtest
}  // namespace vpux
