//
// Copyright 2021 Intel Corporation.
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

#include "vpux/hwtest/hwtest_utils.hpp"

#include <numeric>

#include <mlir/Dialect/Quant/QuantTypes.h>

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/IE/float16.hpp"
#include "vpux/utils/core/error.hpp"

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
    case nb::DType::FP8:
        return builder.getF16Type();
    case nb::DType::FP16:
        return builder.getF16Type();
    case nb::DType::FP32:
        return builder.getF32Type();
    case nb::DType::BF16:
        return builder.getBF16Type();
    default:
        throw std::domain_error{llvm::formatv("Expected a valid data type").str()};
    }
}

template <class StorageType>
mlir::DenseElementsAttr generateWeights(std::ifstream& stream, mlir::RankedTensorType type, std::size_t size) {
    if (!stream) {
        return mlir::DenseElementsAttr::get(type, llvm::makeArrayRef<StorageType>(std::vector<StorageType>(size)));
    }

    std::vector<StorageType> buffer(size);
    const auto expectedBytesCountToRead = buffer.size() * sizeof(StorageType);
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
            "Failed to read {0} bytes from weights file, read {1} bytes before logical error on i/o operation occured",
            expectedBytesCountToRead, actualBytesCountRead);
    VPUX_THROW_UNLESS(
            (state & std::ifstream::badbit) == 0,
            "Failed to read {0} bytes from weights file, read {1} bytes before read error on i/o operation occured",
            expectedBytesCountToRead, actualBytesCountRead);
    VPUX_THROW("Unexpected std::ifstream::rdstate value {}", state);
}

}  // namespace

mlir::DenseElementsAttr generateWeights(llvm::ArrayRef<std::int64_t> shape, mlir::Type type, mlir::MLIRContext* context,
                                        const char* weightsFileName) {
    mlir::DenseElementsAttr wt_data_vals;
    auto wtData_ddr_valueType = mlir::RankedTensorType::get(shape, type);
    const auto vecSize = static_cast<std::size_t>(
            std::accumulate(shape.begin(), shape.end(), static_cast<std::int64_t>(1), std::multiplies<std::int64_t>()));

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

    if (type.isSignedInteger(8)) {
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

std::size_t totalTensorSize(llvm::ArrayRef<std::int64_t> shape, mlir::Type elementType) {
    if (auto qType = elementType.dyn_cast<mlir::quant::UniformQuantizedType>()) {
        elementType = qType.getStorageType();
    }
    std::size_t numBytes = elementType.getIntOrFloatBitWidth() / 8;

    const auto totalSize =
            std::accumulate(shape.begin(), shape.end(), static_cast<std::int64_t>(1), std::multiplies<std::int64_t>());
    return static_cast<std::size_t>(totalSize) * numBytes;
}

std::vector<std::int64_t> convertNBPadtoNCETaskPad(const std::array<std::int64_t, 4>& nb_pad) {
    std::vector<std::int64_t> ncetask_pad(nb_pad.size());

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
    auto cnnOp = builder.create<IE::CNNNetworkOp>(builder.getUnknownLoc(), mainFuncNameAttr);
    cnnOp.inputsInfo().emplaceBlock();
    cnnOp.outputsInfo().emplaceBlock();

    auto inputsInfoBuilder = mlir::OpBuilder::atBlockBegin(&cnnOp.inputsInfo().front(), builder.getListener());
    for (auto input : enumerate(inputs)) {
        auto inputType = input.value().cast<mlir::ShapedType>();
        if (auto quantized = inputType.getElementType().dyn_cast_or_null<mlir::quant::UniformQuantizedType>()) {
            inputType = inputType.clone(quantized.getStorageType());
        }

        const auto inputName = llvm::formatv("input_{0}", input.index()).str();
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

        const auto resultName = llvm::formatv("output_{0}", output.index()).str();
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
                llvm::formatv("splitWeightsOverC only supports weight data type fp16 or uint8; got {0}", dtype).str()};

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
    const llvm::SmallVector<std::int64_t> wt_partial_shape({K, new_C, H, W});
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

unsigned round_up(unsigned x, unsigned mult) {
    return ((x + mult - 1) / mult) * mult;  // logic borrowed from MCM
}

mlir::DenseElementsAttr generateZeroPadForEltwiseMultWeights(ArrayRef<int64_t> wt_shape_padded, mlir::Type dtype,
                                                             mlir::MLIRContext* ctx) {
    auto wtData_ddr_valueType = mlir::RankedTensorType::get(wt_shape_padded, dtype);

    if (auto qtype = dtype.dyn_cast<mlir::quant::QuantizedType>()) {
        wtData_ddr_valueType = (qtype.getFlags() & mlir::quant::QuantizationFlags::Signed)
                                       ? mlir::RankedTensorType::get(wt_shape_padded, getSInt8Type(ctx))
                                       : mlir::RankedTensorType::get(wt_shape_padded, getUInt8Type(ctx));
    }

    // NOTE: This should be ZeroPoint, not 0
    size_t vecSize = static_cast<size_t>(std::accumulate(wt_shape_padded.begin(), wt_shape_padded.end(),
                                                         static_cast<int64_t>(1), std::multiplies<int64_t>()));

    mlir::DenseElementsAttr wt_data_vals;
    if (dtype.isF16()) {
        std::vector<float16> wt_vec(vecSize, 0);
        return mlir::DenseElementsAttr::get(wtData_ddr_valueType, makeArrayRef<float16>(wt_vec));
    } else if (dtype.isBF16()) {
        std::vector<bfloat16> wt_vec(vecSize, 0);
        return mlir::DenseElementsAttr::get(wtData_ddr_valueType, makeArrayRef<bfloat16>(wt_vec));
    } else {
        if (dtype.dyn_cast<mlir::quant::QuantizedType>().getFlags() & mlir::quant::QuantizationFlags::Signed) {
            std::vector<int8_t> wt_vec(vecSize, 0);
            return mlir::DenseElementsAttr::get(wtData_ddr_valueType, makeArrayRef<int8_t>(wt_vec));
        } else {
            std::vector<uint8_t> wt_vec(vecSize, 0);
            return mlir::DenseElementsAttr::get(wtData_ddr_valueType, makeArrayRef<uint8_t>(wt_vec));
        }
    }
}

mlir::MemRefType getMemRefType(mlir::OpBuilder builder, VPUIP::MemoryLocation memlocation, SmallVector<int64_t> shape,
                               mlir::Type type, SmallVector<mlir::AffineMap> affineMaps) {
    auto op_memSpaceAttr = VPUIP::MemoryLocationAttr::get(builder.getContext(), memlocation);
    return mlir::MemRefType::get(makeArrayRef(shape), type, affineMaps, op_memSpaceAttr);
}

vpux::VPUIP::DeclareTensorOp createDeclareTensorOp(mlir::OpBuilder builder, VPUIP::MemoryLocation memlocation,
                                                   SmallVector<int64_t> shape, mlir::Type type,
                                                   SmallVector<mlir::AffineMap> affineMaps, int locale, size_t offset) {
    auto op_memSpaceAttr = VPUIP::MemoryLocationAttr::get(builder.getContext(), memlocation);
    auto op_type = mlir::MemRefType::get(makeArrayRef(shape), type, affineMaps, op_memSpaceAttr);
    auto op = builder.create<VPUIP::DeclareTensorOp>(builder.getUnknownLoc(), op_type, memlocation, locale, offset);
    return op;
}

mlir::OpResult getTensorResult(VPUIP::DeclareTensorOp op) {
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
    auto pad = VPUIP::PaddingAttr::get(getIntAttr(builder, padding_vec[PAD_NCETASK_LEFT]),
                                       getIntAttr(builder, padding_vec[PAD_NCETASK_RIGHT]),
                                       getIntAttr(builder, padding_vec[PAD_NCETASK_TOP]),
                                       getIntAttr(builder, padding_vec[PAD_NCETASK_BOTTOM]), builder.getContext());

    auto dpuTask = variantbuilder.create<VPUIP::DPUTaskOp>(builder.getUnknownLoc(), start, end, pad,
                                                           VPUIP::MPEMode::CUBOID_16x16);

    return dpuTask;
}

//

void computeQuantMultShift(float scale, unsigned& shift, unsigned& mult) {
    auto bits = 15;
    int exponent;
    double mantissa = std::frexp(scale, &exponent);
    shift = bits - exponent;
    mult = static_cast<unsigned>((mantissa * pow(2, bits)));
}

size_t calcWeightsTableMultShift(const nb::TestCaseJsonDescriptor& testDesc, mlir::MemRefType input,
                                 mlir::MemRefType output, mlir::MemRefType weights) {
    size_t multshift = 0;
    // 8bit mult mask
    static const uint32_t PRELU_MULT_MASK = 0x000000FF;
    // 6bit shift mask
    static const uint32_t PRELU_SHIFT_MASK = 0x00003F00;
    static const uint32_t PRELU_SHIFT_SHIFT = 8;
    // round mode mask
    static const uint32_t ROUND_MODE_MASK = 0x0000C000;
    static const uint32_t ROUND_MODE_SHIFT = 14;
    // scale mask
    static const uint32_t SCALE_MODE_MASK = 0xFFFF0000;
    static const uint32_t SCALE_MODE_SHIFT = 16;

    float out_scale = testDesc.getOutputLayer().qp.scale;
    float in_Scale = testDesc.getInputLayer().qp.scale;
    float weights_Scale = testDesc.getWeightLayer().qp.scale;
    auto inputtype = input.getElementType();
    auto outtype = output.getElementType();
    auto isMaxPool = testDesc.getCaseType() == nb::CaseType::MaxPool;

    // For pool, no weights in config.json
    if (!isMaxPool) {
        auto wt_type = weights.getElementType();
        if (auto wt_qType = wt_type.dyn_cast<mlir::quant::UniformQuantizedType>()) {
            weights_Scale = static_cast<float>(wt_qType.getScale());
        } else {
            weights_Scale = 1.0f;
        }
    } else {
        weights_Scale = 1.0f;
    }

    if (auto in_qType = inputtype.dyn_cast<mlir::quant::UniformQuantizedType>()) {
        in_Scale = static_cast<float>(in_qType.getScale());
    }

    if (auto out_qType = outtype.dyn_cast<mlir::quant::UniformQuantizedType>()) {
        out_scale = static_cast<float>(out_qType.getScale());
    }

    float result_Scale = (in_Scale * weights_Scale) / out_scale;

    auto float_as_int = [&](float f) {
        union bit_field32 {
            float fp;
            unsigned int ui;
        };
        bit_field32 v;
        v.fp = f;
        return v.ui;
    };

    if (input.getElementType().isBF16() || input.getElementType().isF16()) {
        multshift = float_as_int(result_Scale);
    } else {
        // harcoded
        int32_t round32 = 1;
        int32_t reluMult = 0;
        unsigned mult;
        unsigned shift;
        computeQuantMultShift(result_Scale, shift, mult);
        multshift = static_cast<int64_t>(
                ((mult << SCALE_MODE_SHIFT) & SCALE_MODE_MASK) | ((round32 << ROUND_MODE_SHIFT) & ROUND_MODE_MASK) |
                ((shift << PRELU_SHIFT_SHIFT) & PRELU_SHIFT_MASK) | (reluMult & PRELU_MULT_MASK));
    }

    return multshift;
}

std::vector<int32_t> generateWeightsTablesValuesWithSparsity(const nb::TestCaseJsonDescriptor& testDesc,
                                                             mlir::MemRefType input, mlir::MemRefType output,
                                                             mlir::MemRefType weights,
                                                             mlir::MemRefType actWindow_cmx_type, std::size_t offset,
                                                             ArrayRef<int64_t> wtTbl_data_shape,
                                                             size_t weights_offset) {
    const std::size_t WT_ELEMENTS_PER_CHANNEL = 4UL;

    const size_t DATA_POINTER_IDX = 0;
    const size_t SPARSITY_POINTER_IDX = 1;
    const size_t MULTSHIFT_IDX = 2;
    const size_t BIAS_IDX = 3;
    const size_t BYTE_SIZE = 8;  // bits
    auto isMaxPool = testDesc.getCaseType() == nb::CaseType::MaxPool;

    std::vector<int32_t> weightsTableVals(wtTbl_data_shape[0] * WT_ELEMENTS_PER_CHANNEL, 0);

    if (!isMaxPool) {
        auto first_channel_offset = weights_offset;
        auto weights_outChannel = weights.getShape()[0];
        auto weights_set_size = weights.getShape()[1] * weights.getShape()[2] * weights.getShape()[3];
        size_t elementsize_bytes = 0;
        if (auto qType = weights.getElementType().dyn_cast<mlir::quant::UniformQuantizedType>()) {
            elementsize_bytes = qType.getStorageType().getIntOrFloatBitWidth() / BYTE_SIZE;

        } else {
            elementsize_bytes = (weights.getElementType().getIntOrFloatBitWidth()) / BYTE_SIZE;
        }
        auto weights_set_nbytes = weights_set_size * elementsize_bytes;

        // generate data pointers
        for (int64_t i = 0; i < weights_outChannel; ++i) {
            weightsTableVals[i * 4 + DATA_POINTER_IDX] = static_cast<int32_t>(first_channel_offset);
            first_channel_offset += weights_set_nbytes;
        }
    }

    auto actWindowShape = actWindow_cmx_type.getShape();
    auto activationWindowSizeInWords = static_cast<size_t>(std::accumulate(
            actWindowShape.begin(), actWindowShape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>()));
    auto actElementTypeBits = actWindow_cmx_type.getElementTypeBitWidth();
    auto activationWindowSizeInBytes = activationWindowSizeInWords * actElementTypeBits / 8;

    auto outputChannels = wtTbl_data_shape[0];
    auto paddedOutputChannels = outputChannels;

    paddedOutputChannels = round_up(outputChannels, 16);
    auto increment = activationWindowSizeInBytes / paddedOutputChannels;

    std::vector<int64_t> increments = std::vector<int64_t>(paddedOutputChannels, 0);
    for (unsigned i = 1; i < paddedOutputChannels; ++i) {
        increments[i] = increments[i - 1] + increment;
    }

    long int new_offset = offset;

    // TODO : later can be changed for multi-cluster test
    unsigned numClusters = 1;

    std::size_t sizeToIterate = 0;
    std::size_t totalSizeToIterate = 0;
    std::size_t k = 0;
    for (unsigned i = 0; i < numClusters; i++) {
        // Resetting offset at the beginning of the cluster
        offset = new_offset;

        // Filling cluster
        for (size_t j = 0; j < weightsTableVals.size(); j += WT_ELEMENTS_PER_CHANNEL)
            weightsTableVals[j + SPARSITY_POINTER_IDX + totalSizeToIterate] = offset + increments[k++];

        // Preparing for next iteration
        sizeToIterate = actWindowShape[0] * WT_ELEMENTS_PER_CHANNEL;
        totalSizeToIterate += sizeToIterate;
    }
    // TODO: generic dtype
    int32_t bias_value = 0;

    size_t mult_shift;
    if (isMaxPool) {
        mult_shift = calcWeightsTableMultShift(testDesc, input, output, mlir::MemRefType());
    } else {
        mult_shift = calcWeightsTableMultShift(testDesc, input, output, weights);
    }

    for (int64_t i = 0; i < outputChannels; ++i) {
        weightsTableVals[i * 4 + MULTSHIFT_IDX] = static_cast<int32_t>(mult_shift);
        weightsTableVals[i * 4 + BIAS_IDX] = bias_value;
    }

    return weightsTableVals;
}

template <typename T>
void padAvgPoolDWConvWeights(std::vector<T>& wt_vec, double scaleVal, unsigned int outputChannels,
                             unsigned int weightSetDimension, unsigned int paddingDifference) {
    unsigned j = 0;
    for (unsigned oc = 0; oc < outputChannels; ++oc) {
        for (unsigned ws = 0; ws < weightSetDimension; ++ws)
            wt_vec[j++] = scaleVal;

        for (unsigned ws = 0; ws < paddingDifference; ++ws)
            ++j;
    }
}

mlir::DenseElementsAttr generateDWConvWeightsForAvgPool(ArrayRef<int64_t> wt_shape, mlir::Type dtype, double scaleVal,
                                                        mlir::MLIRContext* ctx) {
    auto isDepthwiseConv = true;
    unsigned short kernelWidth = wt_shape[3];
    unsigned short kernelHeight = wt_shape[2];

    // Initializions are done assuming regular convolution and then eventually modified for depthwise
    auto inputChannels = wt_shape[1];
    auto outputChannels = wt_shape[0];
    if (isDepthwiseConv)
        // outputChannels = inputChannels;
        inputChannels = outputChannels;  // Backward definition NB vs MCM

    auto weightSetDimension = kernelWidth * kernelHeight * inputChannels;
    if (isDepthwiseConv)
        weightSetDimension = kernelWidth * kernelHeight;

    auto weightSetDimensionPadded = round_up(weightSetDimension, 16);
    auto paddingDifference = weightSetDimensionPadded - weightSetDimension;

    SmallVector<int64_t> wt_shape_padded({outputChannels, 1, 1, weightSetDimensionPadded});

    auto wtData_ddr_valueType = mlir::RankedTensorType::get(wt_shape_padded, dtype);

    if (auto qtype = dtype.dyn_cast<mlir::quant::QuantizedType>()) {
        if (qtype.getFlags() & mlir::quant::QuantizationFlags::Signed) {
            wtData_ddr_valueType = mlir::RankedTensorType::get(wt_shape_padded, getSInt8Type(ctx));
        } else {
            wtData_ddr_valueType = mlir::RankedTensorType::get(wt_shape_padded, getUInt8Type(ctx));
        }
    }

    // NOTE: This should be ZeroPoint, not 0
    size_t vecSize = static_cast<size_t>(std::accumulate(wt_shape_padded.begin(), wt_shape_padded.end(),
                                                         static_cast<int64_t>(1), std::multiplies<int64_t>()));

    mlir::DenseElementsAttr wt_data_vals;
    if (dtype.isF16()) {
        std::vector<float16> wt_vec(vecSize, 0);
        padAvgPoolDWConvWeights(wt_vec, scaleVal, outputChannels, weightSetDimension, paddingDifference);
        return mlir::DenseElementsAttr::get(wtData_ddr_valueType, makeArrayRef<float16>(wt_vec));
    } else if (dtype.isBF16()) {
        std::vector<bfloat16> wt_vec(vecSize, 0);
        padAvgPoolDWConvWeights(wt_vec, scaleVal, outputChannels, weightSetDimension, paddingDifference);
        return mlir::DenseElementsAttr::get(wtData_ddr_valueType, makeArrayRef<bfloat16>(wt_vec));
    } else {
        scaleVal = 1;
        if (dtype.dyn_cast<mlir::quant::QuantizedType>().getFlags() & mlir::quant::QuantizationFlags::Signed) {
            std::vector<int8_t> wt_vec(vecSize, 0);
            padAvgPoolDWConvWeights(wt_vec, scaleVal, outputChannels, weightSetDimension, paddingDifference);
            return mlir::DenseElementsAttr::get(wtData_ddr_valueType, makeArrayRef<int8_t>(wt_vec));
        } else {
            std::vector<uint8_t> wt_vec(vecSize, 0);
            padAvgPoolDWConvWeights(wt_vec, scaleVal, outputChannels, weightSetDimension, paddingDifference);
            return mlir::DenseElementsAttr::get(wtData_ddr_valueType, makeArrayRef<uint8_t>(wt_vec));
        }
    }
}

SmallVector<int64_t> getWeightsPaddedShape(SmallVector<int64_t> wt_shape, bool isDepthwiseConv) {
    unsigned short kernelWidth = wt_shape[3];
    unsigned short kernelHeight = wt_shape[2];

    // Initializions are done assuming regular convolution and then eventually modified for depthwise
    auto inputChannels = wt_shape[1];
    auto outputChannels = wt_shape[0];
    if (isDepthwiseConv)
        // outputChannels = inputChannels;
        inputChannels = outputChannels;  // Backward definition NB vs MCM

    auto weightSetDimension = kernelWidth * kernelHeight * inputChannels;
    if (isDepthwiseConv)
        weightSetDimension = kernelWidth * kernelHeight;

    auto weightSetDimensionPadded = round_up(weightSetDimension, 16);

    SmallVector<int64_t> wt_shape_padded({outputChannels, 1, 1, weightSetDimensionPadded});
    return wt_shape_padded;
}

}  // namespace hwtest
}  // namespace vpux
