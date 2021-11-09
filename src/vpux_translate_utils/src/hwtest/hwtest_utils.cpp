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

template <typename T>
std::vector<T> readFile(std::ifstream& file) {
    std::vector<T> vec;
    if (file.is_open()) {
        file.unsetf(std::ios::skipws);

        std::streampos fileSize;

        file.seekg(0, std::ios::end);
        fileSize = file.tellg();
        file.seekg(0, std::ios::beg);

        vec.reserve(fileSize);

        vec.insert(vec.begin(), std::istream_iterator<T>(file), std::istream_iterator<T>());
    }
    return vec;
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
                      "Failed to read {0} bytes from weghts file, read {1} bytes before EOF has been reached",
                      expectedBytesCountToRead, actualBytesCountRead);
    VPUX_THROW_UNLESS(
            (state & std::ifstream::failbit) == 0,
            "Failed to read {0} bytes from weghts file, read {1} bytes before logical error on i/o operation occured",
            expectedBytesCountToRead, actualBytesCountRead);
    VPUX_THROW_UNLESS(
            (state & std::ifstream::badbit) == 0,
            "Failed to read {0} bytes from weghts file, read {1} bytes before read error on i/o operation occured",
            expectedBytesCountToRead, actualBytesCountRead);
    VPUX_THROW("Unexpected std::ifstream::rdstate value {}", state);
}

mlir::Type parseType(mlir::OpBuilder builder, mlir::Type ty, const nb::QuantParams& qp) {
    auto intTy = ty.dyn_cast<mlir::IntegerType>();
    if (qp.present && intTy) {
        ty = mlir::quant::UniformQuantizedType::get(intTy.isSigned() ? mlir::quant::QuantizationFlags::Signed : 0, ty,
                                                    builder.getF32Type(), qp.scale, qp.zeropoint, qp.low_range,
                                                    qp.high_range);
    }
    return ty;
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

    if (type.isSignedInteger(4)) {
        const auto weightsU8 = readFile<std::uint8_t>(stream);
        std::vector<std::int8_t> weightsI4;
        for (const auto& elementU8 : weightsU8) {
            const int8_t msn = (elementU8 & 0xf0) >> 4;
            const int8_t lsn = (elementU8 & 0x0f) >> 0;
            weightsI4.push_back(lsn);
            weightsI4.push_back(msn);
        }
        return mlir::DenseElementsAttr::get(wtData_ddr_valueType, llvm::makeArrayRef<std::int8_t>(weightsI4));
    } else if (type.isInteger(4)) {
        const auto weightsU8 = readFile<std::uint8_t>(stream);
        std::vector<std::uint8_t> weightsU4;
        for (const auto& elementU8 : weightsU8) {
            const uint8_t msn = (elementU8 & 0xf0) >> 4;
            const uint8_t lsn = (elementU8 & 0x0f) >> 0;
            weightsU4.push_back(lsn);
            weightsU4.push_back(msn);
        }
        return mlir::DenseElementsAttr::get(wtData_ddr_valueType, llvm::makeArrayRef<std::uint8_t>(weightsU4));
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

void computeQuantMultShift(float scale, unsigned& shift, unsigned& mult) {
    auto bits = 15;
    int exponent;
    double mantissa = std::frexp(scale, &exponent);
    shift = bits - exponent;
    mult = static_cast<unsigned>((mantissa * pow(2, bits)));
}

// Based on MCM logic to generate mult-shift element in weight table
// weights_table[outChannel][2] = mult << 16 | round << 14 | shift << 8 | prelu
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

    float result_Scale = out_scale / (in_Scale * weights_Scale);

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
        int32_t round32 = 0;
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

std::vector<int32_t> generateWeightsTablesValues(const nb::TestCaseJsonDescriptor& testDesc, size_t weights_offset,
                                                 mlir::MemRefType input, mlir::MemRefType output,
                                                 mlir::MemRefType weights) {
    /*
    Each Weight table[4] represents per channel layout of weights data,
    For ex : {8448, 16777215, 1354584320, 0}
    with each index as :
    [0] -> Starting address of each weight data,
    [1] -> Sparsity map address,
    [2] -> mult << 16 | round << 14 | shift << 8 | prelu
    [3] -> bias ]
    */
    const size_t DATA_POINTER_IDX = 0;
    const size_t SPARSITY_POINTER_IDX = 1;
    const size_t MULTSHIFT_IDX = 2;
    const size_t BIAS_IDX = 3;
    const size_t BYTE_SIZE = 8;  // bits

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

    // TODO: generic dtype
    int32_t bias_value = 0;

    // TODO: calculate
    // currently hard coded
    size_t sparsity_pointer = 16777215;

    auto mult_shift = calcWeightsTableMultShift(testDesc, input, output, weights);

    // generate data pointers
    std::vector<int32_t> weightsTableVals(weights_outChannel * 4, 0);

    for (int64_t i = 0; i < weights_outChannel; ++i) {
        weightsTableVals[i * 4 + DATA_POINTER_IDX] = static_cast<int32_t>(first_channel_offset);
        first_channel_offset += weights_set_nbytes;

        weightsTableVals[i * 4 + SPARSITY_POINTER_IDX] = static_cast<int32_t>(sparsity_pointer);
        weightsTableVals[i * 4 + MULTSHIFT_IDX] = static_cast<int32_t>(mult_shift);
        weightsTableVals[i * 4 + BIAS_IDX] = bias_value;
    }

    return weightsTableVals;
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

        // Pad out to a multiple of 16
        if (weights_set_nbytes & 0xF) {
            weights_set_nbytes &= ~0xF;
            weights_set_nbytes += 0x10;
        }

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

uint16_t getWindowSize(mlir::OpBuilder builder, uint16_t kx, uint16_t sx, mlir::Type dataType) {
    // Find max mpe where if calc window <= 32
    // return window size for the max mpe
    uint16_t windowSize, maxMpeWindowSize = 64;
    int mpe = 1;
    auto ctx = builder.getContext();

    if (dataType == getUInt8Type(ctx)) {
        // mpe in [1,2,4,8,16] for uint8
        while (mpe <= 16) {
            if (sx <= kx)
                windowSize = kx + sx * (mpe - 1);
            else
                windowSize = kx * mpe;

            if (windowSize <= 32)
                maxMpeWindowSize = windowSize;

            mpe *= 2;
        }
    } else if (dataType == builder.getF16Type()) {
        // mpe in [1,2,4] for float
        while (mpe <= 4) {
            if (sx <= kx)
                windowSize = kx + sx * (mpe - 1);
            else
                windowSize = kx * mpe;

            if (windowSize <= 32)
                maxMpeWindowSize = windowSize;

            mpe *= 2;
        }
    }

    return maxMpeWindowSize;
}

std::vector<int8_t> createBitPattern(uint16_t kernelW, uint16_t kernelH, uint16_t windowsSize, uint16_t inputChannels) {
    std::vector<int8_t> bitpattern;
    bitpattern.reserve(windowsSize * kernelH * inputChannels);
    for (size_t c = 0; c < inputChannels; c++)
        for (size_t y = 0; y < kernelH; y++)
            for (size_t x = 0; x < windowsSize; x++)
                if (x < kernelW)
                    bitpattern.emplace_back(1);
                else
                    bitpattern.emplace_back(0);
    return bitpattern;
}

mlir::DenseElementsAttr getactivationWindow(mlir::OpBuilder builder, const std::vector<int64_t>& filter_size,
                                            const std::vector<int64_t>& strides, int64_t outputChannels,
                                            int64_t inputChannels, mlir::Type dtype,
                                            SmallVector<int64_t>& sparsity_shape,
                                            mlir::IntegerAttr& activationChannelLength, bool isOutFloat, bool isPool,
                                            bool isDepthWiseConv) {
    auto kernelH = filter_size[0];
    auto kernelW = filter_size[1];

    auto ctx = builder.getContext();

    if (isOutFloat) {
        dtype = builder.getF16Type();
    }

    auto windowsSize = getWindowSize(builder, kernelW, strides[0], dtype);

    std::vector<int8_t> bitpattern;
    std::vector<uint8_t> perChannelSparsity;
    SmallVector<int64_t> ndims(4);
    if (isPool || isDepthWiseConv) {
        bitpattern = std::move(createBitPattern(kernelW, kernelH, windowsSize, 1));
        perChannelSparsity.resize(static_cast<std::size_t>(std::ceil(bitpattern.size() / 128.0)) * 16);  // allocate
                                                                                                         // once
        ndims = {16 * static_cast<int64_t>(std::ceil(bitpattern.size() / 128.0)), 1, 1, inputChannels};
    } else  // isChannelMajorConvolution
    {
        bitpattern = std::move(createBitPattern(kernelW, kernelH, windowsSize, inputChannels));
        auto windowSparsitySize =
                static_cast<std::size_t>(std::ceil(windowsSize / 8.0));  // how many bytes we need per window
        auto NumberOfRowsSparistyBytes =
                static_cast<int64_t>(std::ceil((kernelH * inputChannels * windowSparsitySize) / 16.0));
        perChannelSparsity.resize(NumberOfRowsSparistyBytes * 16);  // allocate once
        ndims = {16, NumberOfRowsSparistyBytes, 1, static_cast<int64_t>(outputChannels)};
    }

    activationChannelLength = builder.getI32IntegerAttr(bitpattern.size());

    // populate sparsity

    for (size_t i = 0; i < bitpattern.size(); i++) {
        perChannelSparsity.at((i / 128) * 16 + (i % 128) / 8) |=
                bitpattern[i] << (i % 8);  // use at to check boundaries - just in case
    }

    // Create Tensor with Sparsity Data
    SmallVector<int64_t> sparsityShape = ndims;
    std::vector<uint8_t> data;
    // mv::Tensor sparsityTensor("backup", sparsityShape, mv::DType("UInt8"), mv::Order("WHCN"), data);
    // auto sparsityTensor = mv::Tensor(dpuTask->getName() + "_sparse_dw", sparsityShape, mv::DType("UInt8"),
    // mv::Order("NHWC"), data);

    // NOTE: This loop can probably be simplified without using an auxiliary tensor
    for (unsigned oc = 0; oc < sparsityShape[3]; ++oc)
        for (unsigned ic = 0; ic < sparsityShape[2]; ++ic)
            for (unsigned ky = 0; ky < sparsityShape[1]; ++ky)
                for (unsigned kx = 0; kx < sparsityShape[0]; ++kx) {
                    // sparsityTensor.at({kx, ky, ic, oc}) =
                    // static_cast<int64_t>(perChannelSparsity[ky*sparsityShape[0]
                    // + kx]);
                    data.push_back(perChannelSparsity[ky * sparsityShape[0] + kx]);
                }

    std::reverse(sparsityShape.begin(), sparsityShape.end());
    auto sparsityTensor_valueType = mlir::RankedTensorType::get(makeArrayRef(sparsityShape), getUInt8Type(ctx));
    auto sparsityTensor_values = makeArrayRef<uint8_t>(data);
    auto sparsityTensor_attr = mlir::DenseElementsAttr::get(sparsityTensor_valueType, sparsityTensor_values);
    sparsity_shape = sparsityShape;

    return sparsityTensor_attr;
}

unsigned round_up(unsigned x, unsigned mult) {
    return ((x + mult - 1) / mult) * mult;  // logic borrowed from MCM
}

std::vector<int32_t> generateWeightsTablesValuesForMaxPool(const nb::TestCaseJsonDescriptor& testDesc,
                                                           mlir::MemRefType input, mlir::MemRefType output,
                                                           mlir::MemRefType actWindow_cmx_type, std::size_t offset,
                                                           ArrayRef<int64_t> wtTbl_data_shape) {
    const std::size_t WT_ELEMENTS_PER_CHANNEL = 4UL;
    const size_t SPARSITY_POINTER_IDX = 1;
    const size_t MULTSHIFT_IDX = 2;
    const size_t BIAS_IDX = 3;

    std::vector<int32_t> weightsTableVals(wtTbl_data_shape[0] * WT_ELEMENTS_PER_CHANNEL, 0);

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

    auto mult_shift = calcWeightsTableMultShift(testDesc, input, output, mlir::MemRefType());

    for (int64_t i = 0; i < outputChannels; ++i) {
        weightsTableVals[i * 4 + MULTSHIFT_IDX] = static_cast<int32_t>(mult_shift);
        weightsTableVals[i * 4 + BIAS_IDX] = bias_value;
    }

    return weightsTableVals;
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
        throw std::domain_error{llvm::formatv("Expected a valid data type").str()};
    }
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

VPUIP::PPELayerType getPPELayerFromConfig(nb::ActivationLayer activation) {
    switch (activation.activationType) {
    case nb::ActivationType::ReLU:
    case nb::ActivationType::ReLUX: {
        return VPUIP::PPELayerType::LRELU;
    }
    case nb::ActivationType::LeakyReLU: {
        // TODO : remove throw  after adding support
        throw std::domain_error{llvm::formatv("Only relu and relux parsing supported for hwtest").str()};

        return VPUIP::PPELayerType::LRELUX;
    }
    default:
        throw std::domain_error{llvm::formatv("Only relu and relux parsing supported for hwtest").str()};
    }
}

int32_t computeclampLow(nb::InputLayer input, nb::OutputLayer output, bool flexarbINT8, bool isMaxpool,
                        nb::ActivationLayer activation) {
    auto inputdType = input.dtype;
    auto outputdType = output.dtype;
    int32_t clamp = -2147483648;

    if (!isMaxpool) {
        if (outputdType == nb::DType::U8 || outputdType == nb::DType::I8) {
            // Saturation clamp has to be computed in this case
            int32_t saturationClamp = -output.qp.zeropoint;
            if (outputdType == nb::DType::I8)
                saturationClamp -= 128;

            clamp = saturationClamp;

            if (inputdType == nb::DType::FP16 || inputdType == nb::DType::BF16)
                clamp <<= 16;
        } else if (outputdType == nb::DType::FP16 || outputdType == nb::DType::BF16) {
            if (activation.activationType == nb::ActivationType::ReLU ||
                activation.activationType == nb::ActivationType::ReLUX) {
                clamp = 0;
            }
        }
    }

    double alpha = 1.0;
    if (activation.activationType == nb::ActivationType::LeakyReLU) {
        alpha = activation.alpha;

        if (alpha > 0.0) {
            clamp = static_cast<int32_t>(clamp / alpha);
        } else {
            // no negative values
            clamp = 0;
        }
    }

    // TODO: Handle Prelu slopes
    // TODO : HAndle PWL acivations - Sigmoid, tanh

    if (flexarbINT8) {
        std::cout << "\nWarning : Output min value required for accurate clamp low value" << std::endl;
        // TODO :
        // minimum = output.minimum
        auto minimum = 0;
        if (alpha < 0.0) {
            minimum = 0;  // no negative values
        } else if (alpha != 1.0) {
            minimum /= alpha;
        }

        clamp = round(minimum / output.qp.scale);
        clamp = std::max(clamp, -128);

        // TODO : handle Mish
    }

    return clamp;
}

int32_t computeclampHigh(nb::InputLayer input, nb::OutputLayer output, bool flexarbINT8, bool isMaxpool,
                         nb::ActivationLayer activation) {
    auto inputdType = input.dtype;
    auto outputdType = output.dtype;
    int32_t clamp = 2147483647;

    if (!isMaxpool) {
        if (outputdType == nb::DType::U8 || outputdType == nb::DType::I8) {
            // Saturation clamp has to be computed in this case
            int32_t saturationClamp = -output.qp.zeropoint;
            if (outputdType == nb::DType::I8) {
                saturationClamp += 127;
            } else {
                saturationClamp += 255;
            }

            clamp = saturationClamp;

            // Ex: relu6--> minimum(relu, max=6);
            if (activation.activationType == nb::ActivationType::ReLUX) {
                double clampValue = activation.maximum;
                double outputScale = output.qp.scale;
                int32_t quantizedClampValue = static_cast<int32_t>(clampValue / outputScale);

                if (quantizedClampValue > clamp)
                    clamp = quantizedClampValue;
            }

            if (inputdType == nb::DType::FP16 || inputdType == nb::DType::BF16)
                clamp <<= 16;
        } else if (outputdType == nb::DType::FP16 || outputdType == nb::DType::BF16) {
            if (activation.activationType == nb::ActivationType::ReLUX) {
                double clampValue = activation.maximum;

                if (inputdType == nb::DType::U8 || inputdType == nb::DType::I8)
                    clamp = static_cast<int32_t>(clampValue);
                else if (inputdType == nb::DType::FP16)
                    clamp = static_cast<int32_t>(clampValue * pow(2, 16));
            }
        }
    }

    // TODO : HAndle PWL acivations - Sigmoid, tanh

    if (flexarbINT8) {
        std::cout << "\nWarning : Output max value required for accurate clamp high value" << std::endl;
        // TODO :
        // minimum = output minimum
        auto maximum = 0;
        clamp = round(maximum / output.qp.scale);
        clamp = std::min(clamp, 127);

        // TODO : handle Mish
    }

    return clamp;
}

void calculateppeParams(const nb::TestCaseJsonDescriptor& testDesc, int32_t& clampLow, int32_t& clamHigh,
                        int32_t& lreluMult, uint32_t& lreluShift) {
    auto input = testDesc.getInputLayer();
    auto activation = testDesc.getActivationLayer();
    auto output = testDesc.getOutputLayer();
    const int LEAKYRELU_BITS_MTL = 31;
    const double leakyReluHack = 1.0;
    if (activation.activationType != nb::ActivationType::None) {
        if (!(activation.activationType == nb::ActivationType::ReLU ||
              activation.activationType == nb::ActivationType::ReLUX ||
              activation.activationType == nb::ActivationType::LeakyReLU)) {
            throw std::domain_error{
                    llvm::formatv("Activation parsing supported only for relu, relux and leaky_relu ,found {0}",
                                  to_string(activation.activationType))
                            .str()};
        }

        if (activation.activationType == nb::ActivationType::LeakyReLU) {
            double leakyAlpha = 1.0;
            leakyAlpha = activation.alpha;

            if (leakyAlpha == 0.0) {
                lreluMult = 0;
            } else if (leakyAlpha != 1.0) {
                int exponent;
                double mantissa;

                mantissa = std::frexp(leakyAlpha, &exponent);
                lreluShift = static_cast<uint32_t>(LEAKYRELU_BITS_MTL - exponent);
                lreluMult = static_cast<int32_t>((mantissa * pow(2, LEAKYRELU_BITS_MTL)) * leakyReluHack);
            }
        }

        bool flexarbINT8 = (activation.activationType == nb::ActivationType::LeakyReLU);
        bool isMaxpool = testDesc.getCaseStr().find("Pool") != std::string::npos;

        clampLow = computeclampLow(input, output, flexarbINT8, isMaxpool, activation);
        clamHigh = computeclampHigh(input, output, flexarbINT8, isMaxpool, activation);
    }
}

std::size_t totalTensorSize(llvm::ArrayRef<std::int64_t> shape, mlir::Type elementType) {
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

std::vector<int32_t> getInstructionListVals(nb::ActivationType pwlType,
                                            llvm::ArrayRef<int64_t> instructionList_data_shape) {
    // NOTE : The instruction list has 5 bits of addresses so the biggest count of instructions is 11111 = 27
    // 27 of course will be aligned to 32 and will contain NOPS inside
    auto instructionListShape = instructionList_data_shape;
    size_t totalSize = static_cast<size_t>(std::accumulate(instructionListShape.begin(), instructionListShape.end(),
                                                           static_cast<int64_t>(1), std::multiplies<int64_t>()));
    std::vector<uint32_t> template_table(totalSize, 0);

    // NOTE: first 2 are hardware reserved areas
    std::size_t ADDR_OF_RESERVED = 6;
    std::size_t ADDR_OF_ADDR_FLEX = 11;
    std::size_t ADDR_OF_FIRST2_BITS = 9;
    std::size_t ADDR_OF_REST_BITS = 16;
    std::size_t ADDR_OF_VALUE = 19;
    std::size_t MASK_FIRST2_BITS = 3;
    const std::size_t ALU_HALT_OPCODE = 6;
    const std::size_t ALU_LOAD = 2;
    std::size_t first2_bits, last3_bits;
    std::vector<int> range_vector;
    std::vector<int> shift_vector;
    std::vector<int> bias_vector;
    std::function<double(double)> refFunction;

    if (pwlType == nb::ActivationType::LeakyReLU) {
        range_vector = {-128, -109, -90, -72, -54, -36, -18, 0, 128};
        shift_vector = {1, -1, 0, 0, 0, -1, -1, -4};
        bias_vector = {-119, 44, -43, -31, -19, 18, 10, 0};
    } else if (pwlType == nb::ActivationType::Mish) {
        // TODO : Handle Mish
        throw std::domain_error{llvm::formatv("Mish activation parsing not supported for hwtest").str()};
        /*
        refFunction = mish;
        const auto& quantOutHigh = outQuantParams.getMax();
        const auto& quantOutLow = outQuantParams.getMin();
        if (quantOutHigh.empty()) {
            throw std::runtime_error("populateInstructionListMap: empty output quantization parameters");
        }

        createPWLTable(quantOutLow.at(0), quantOutHigh.at(0), refFunction, range_vector, shift_vector, bias_vector);
        */
    }

    // Populate the instruction list from the table
    std::size_t k = 0;
    for (std::size_t j = 0; j < 32; j++) {
        first2_bits = j & MASK_FIRST2_BITS;
        last3_bits = j >> 2;

        if (j == 15)
            template_table[j] = (ALU_HALT_OPCODE);
        else if (j > 25)
            template_table[j] = (ALU_HALT_OPCODE);
        else {
            if (j < range_vector.size()) {
                template_table[j] = ((range_vector[j] << ADDR_OF_VALUE) | (last3_bits << ADDR_OF_REST_BITS) |
                                     (8 << ADDR_OF_ADDR_FLEX) | (first2_bits << ADDR_OF_FIRST2_BITS) |
                                     (0 << ADDR_OF_RESERVED) | ALU_LOAD);
            } else if (j < range_vector.size() + shift_vector.size() + 1) {
                if (j < 16)
                    template_table[j] = ((shift_vector[j - range_vector.size()] << ADDR_OF_VALUE) |
                                         (last3_bits << ADDR_OF_REST_BITS) | (8 << ADDR_OF_ADDR_FLEX) |
                                         (first2_bits << ADDR_OF_FIRST2_BITS) | (0 << ADDR_OF_RESERVED) | ALU_LOAD);
                else {
                    k = j - 1;
                    first2_bits = k & MASK_FIRST2_BITS;
                    last3_bits = k >> 2;
                    template_table[j] = ((shift_vector[k - range_vector.size()] << ADDR_OF_VALUE) |
                                         (last3_bits << ADDR_OF_REST_BITS) | (8 << ADDR_OF_ADDR_FLEX) |
                                         (first2_bits << ADDR_OF_FIRST2_BITS) | (0 << ADDR_OF_RESERVED) | ALU_LOAD);
                }
            } else if (j < range_vector.size() + shift_vector.size() + bias_vector.size() + 1) {
                k = j - 1;
                first2_bits = k & MASK_FIRST2_BITS;
                last3_bits = k >> 2;
                template_table[j] = ((bias_vector[k - range_vector.size() - shift_vector.size()] << ADDR_OF_VALUE) |
                                     (last3_bits << ADDR_OF_REST_BITS) | (8 << ADDR_OF_ADDR_FLEX) |
                                     (first2_bits << ADDR_OF_FIRST2_BITS) | (0 << ADDR_OF_RESERVED) | ALU_LOAD);
            }
        }
    }

    std::vector<int32_t> template_table_appropriate_type(template_table.begin(), template_table.end());
    return template_table_appropriate_type;
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

mlir::MemRefType getMemRefType(mlir::OpBuilder builder, VPUIP::MemoryLocation memlocation, SmallVector<int64_t> shape,
                               mlir::Type type, SmallVector<mlir::AffineMap> affineMaps) {
    auto op_memSpaceAttr = VPUIP::MemoryLocationAttr::get(builder.getContext(), memlocation);
    return mlir::MemRefType::get(makeArrayRef(shape), type, affineMaps, op_memSpaceAttr);
}

vpux::VPUIP::DeclareTensorOp createDeclareTensorOp(mlir::OpBuilder builder, VPUIP::MemoryLocation memlocation,
                                                   SmallVector<int64_t> shape, mlir::Type type,
                                                   SmallVector<mlir::AffineMap> affineMaps, int locale, int offset) {
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

    auto dpuTask = variantbuilder.create<VPUIP::DPUTaskOp>(builder.getUnknownLoc(), nullptr, start, end, pad,
                                                           VPUIP::MPEMode::CUBOID_16x16);

    return dpuTask;
}

std::vector<std::int64_t> convertNBPadtoNCETaskPad(const std::array<std::int64_t, 4>& nb_pad) {
    std::vector<std::int64_t> ncetask_pad(nb_pad.size());

    ncetask_pad[PAD_NCETASK_LEFT] = nb_pad[PAD_NB_LEFT];
    ncetask_pad[PAD_NCETASK_RIGHT] = nb_pad[PAD_NB_RIGHT];
    ncetask_pad[PAD_NCETASK_TOP] = nb_pad[PAD_NB_TOP];
    ncetask_pad[PAD_NCETASK_BOTTOM] = nb_pad[PAD_NB_BOTTOM];

    return ncetask_pad;
}

//
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

}  // namespace hwtest
}  // namespace vpux
