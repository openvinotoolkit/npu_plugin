//
// Copyright 2021 Intel Corporation.
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

#include <limits>
#include <numeric>

#include <mlir/Dialect/Quant/QuantTypes.h>

#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/hwtest/hwtest_utils.hpp"
#include "vpux/hwtest/test_case_json_parser.hpp"
#include "vpux/utils/core/error.hpp"

namespace vpux {
namespace hwtest {

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

void buildSimpleZMajorConvActivation(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module,
                                     mlir::OpBuilder builder, Logger& log, mlir::Type inputType, mlir::Type weightsType,
                                     mlir::Type outputType) {
    auto input = testDesc.getInputLayer();
    auto weight = testDesc.getWeightLayer();
    auto conv = testDesc.getConvLayer();
    auto activation = testDesc.getActivationLayer();
    auto output = testDesc.getOutputLayer();

    SmallVector<int64_t> in_shape(input.shape.begin(), input.shape.end());
    SmallVector<int64_t> out_shape(output.shape.begin(), output.shape.end());

    // SmallVector<int64_t> wt_data_shape(weight.shape.begin(), weight.shape.end());
    SmallVector<int64_t> wt_data_shape{weight.shape[0], weight.shape[1], weight.shape[2], weight.shape[3]};

    SmallVector<int64_t> wtTbl_data_shape{wt_data_shape[0], 1, 1, 4};
    const char* weight_file_name = "weight.dat";

    auto output_totalsize = totalTensorSize(out_shape, outputType);
    auto input_totalsize = totalTensorSize(in_shape, inputType);
    auto weightsTable_totalsize = /*always 4 bytes*/ 4 * wtTbl_data_shape[0] * wtTbl_data_shape[3];

    const auto OUTPUT_CMX_OFFSET = 0;
    const auto INPUT_CMX_OFFSET = OUTPUT_CMX_OFFSET + output_totalsize;
    const auto WEIGHTSTABLE_CMX_OFFSET = INPUT_CMX_OFFSET + input_totalsize;
    const auto WEIGHTS_CMX_OFFSET = WEIGHTSTABLE_CMX_OFFSET + weightsTable_totalsize;

    SmallVector<mlir::Type> inputTypes;
    const auto inputAffineMaps = DimsOrder::NHWC.toAffineMapsList(builder.getContext(), Shape(in_shape));
    auto memSpaceAttr_in =
            VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::ProgrammableInput);
    inputTypes.push_back(mlir::MemRefType::get(makeArrayRef(in_shape), inputType, inputAffineMaps, memSpaceAttr_in));
    auto memSpaceAttr_out =
            VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::ProgrammableOutput);
    const auto outputAffineMaps = DimsOrder::NHWC.toAffineMapsList(builder.getContext(), Shape(out_shape));
    auto outputParamType =
            mlir::MemRefType::get(makeArrayRef(out_shape), outputType, outputAffineMaps, memSpaceAttr_out);
    inputTypes.push_back(outputParamType);
    SmallVector<ArrayRef<mlir::AffineMap>> argsAffineMaps{inputAffineMaps, outputAffineMaps};

    const auto funcType = builder.getFunctionType(makeArrayRef(inputTypes), outputParamType);

    // TODO: Func should not return
    auto func = builder.create<mlir::FuncOp>(
            builder.getUnknownLoc(),
            llvm::formatv("zmajor_conv_{0}_{1}_{2}_{3}", to_string(activation.activationType), inputType, weightsType,
                          outputType)
                    .str(),
            funcType, builder.getStringAttr("private"));

    auto funcbuilder = mlir::OpBuilder::atBlockBegin(func.addEntryBlock(), builder.getListener());

    // BUild VPUIP ops
    auto funcinput = func.getArgument(0);
    auto funcoutput = func.getArgument(1);

    // weights data
    auto weight_data_ddr_memSpaceAttr =
            VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::GraphFile);
    const auto weightDataAffineMaps = DimsOrder::NHWC.toAffineMapsList(builder.getContext(), Shape(wt_data_shape));
    auto weightData_ddr_type = mlir::MemRefType::get(makeArrayRef(wt_data_shape), weightsType, weightDataAffineMaps,
                                                     weight_data_ddr_memSpaceAttr);

    auto wt_data_vals = generateWeights(wt_data_shape, weightsType, builder.getContext(), weight_file_name);
    auto wt_data_attr = Const::ContentAttr::get(wt_data_vals);
    if (auto qty = weightsType.dyn_cast<mlir::quant::QuantizedType>()) {
        wt_data_attr = wt_data_attr.quantCast(qty);
    }
    auto weight_data_ddr = funcbuilder.create<Const::DeclareOp>(builder.getUnknownLoc(), weightData_ddr_type,
                                                                wt_data_attr.reorder(DimsOrder::NHWC));

    // weights cmx tensor
    auto wtData_cmx_memSpaceAttr =
            VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::VPU_CMX_NN);
    auto wtData_cmx_type = mlir::MemRefType::get(makeArrayRef(wt_data_shape), weightsType, weightDataAffineMaps,
                                                 wtData_cmx_memSpaceAttr);
    auto wtData_cmx = funcbuilder.create<VPUIP::DeclareTensorOp>(builder.getUnknownLoc(), wtData_cmx_type,
                                                                 VPUIP::MemoryLocation::VPU_CMX_NN, /*locale index=*/0,
                                                                 /*data idx=*/WEIGHTS_CMX_OFFSET);

    // input - output cmx tensors
    auto inputcmx_memSpaceAttr =
            VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::VPU_CMX_NN);
    auto inputcmx_type =
            mlir::MemRefType::get(makeArrayRef(in_shape), inputType, inputAffineMaps, inputcmx_memSpaceAttr);
    auto inputcmx = funcbuilder.create<VPUIP::DeclareTensorOp>(builder.getUnknownLoc(), inputcmx_type,
                                                               VPUIP::MemoryLocation::VPU_CMX_NN, 0, INPUT_CMX_OFFSET);

    auto outputcmx_memSpaceAttr =
            VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::VPU_CMX_NN);
    auto outputcmx_type =
            mlir::MemRefType::get(makeArrayRef(out_shape), outputType, outputAffineMaps, outputcmx_memSpaceAttr);
    auto outputcmx = funcbuilder.create<VPUIP::DeclareTensorOp>(
            builder.getUnknownLoc(), outputcmx_type, VPUIP::MemoryLocation::VPU_CMX_NN, 0, OUTPUT_CMX_OFFSET);

    auto parent_inputcmx = funcbuilder.create<VPUIP::DeclareTensorOp>(
            builder.getUnknownLoc(), inputcmx_type, VPUIP::MemoryLocation::VPU_CMX_NN, 0, INPUT_CMX_OFFSET);
    auto parent_outputcmx = funcbuilder.create<VPUIP::DeclareTensorOp>(
            builder.getUnknownLoc(), outputcmx_type, VPUIP::MemoryLocation::VPU_CMX_NN, 0, OUTPUT_CMX_OFFSET);

    // weights table ddr tensor
    auto weightTbl_data_ddr_memSpaceAttr =
            VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::GraphFile);
    const auto weightTblAffineMaps = DimsOrder::NHWC.toAffineMapsList(builder.getContext(), Shape(wtTbl_data_shape));
    auto weightTblData_ddr_type =
            mlir::MemRefType::get(makeArrayRef(wtTbl_data_shape), builder.getIntegerType(32, /*isSigned=*/true),
                                  weightTblAffineMaps, weightTbl_data_ddr_memSpaceAttr);
    const auto wtTblData_ddr_valueType =
            mlir::RankedTensorType::get(wtTbl_data_shape, builder.getIntegerType(32, /*isSigned=*/true));

    const std::vector<int32_t> wtTbl_data_values_vec =
            generateWeightsTablesValues(testDesc, WEIGHTS_CMX_OFFSET, inputcmx_type, outputcmx_type, wtData_cmx_type);
    auto wtTbl_data_values = makeArrayRef<int32_t>(wtTbl_data_values_vec);
    auto wtTbl_data_vals = mlir::DenseElementsAttr::get(wtTblData_ddr_valueType, wtTbl_data_values);
    auto weightTbl_data_ddr =
            funcbuilder.create<Const::DeclareOp>(builder.getUnknownLoc(), weightTblData_ddr_type,
                                                 Const::ContentAttr::get(wtTbl_data_vals).reorder(DimsOrder::NHWC));

    // weights table cmx tensor
    auto wtTbl_cmx_memSpaceAttr =
            VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::VPU_CMX_NN);
    auto wtTbl_cmx_type =
            mlir::MemRefType::get(makeArrayRef(wtTbl_data_shape), builder.getIntegerType(32, /*isSigned=*/true),
                                  weightTblAffineMaps, wtTbl_cmx_memSpaceAttr);
    auto wtTbl_cmx = funcbuilder.create<VPUIP::DeclareTensorOp>(builder.getUnknownLoc(), wtTbl_cmx_type,
                                                                VPUIP::MemoryLocation::VPU_CMX_NN, /*locale index=*/0,
                                                                /*data idx=*/WEIGHTSTABLE_CMX_OFFSET);

    // barrier config
    auto barrier0 = funcbuilder.create<VPUIP::ConfigureBarrierOp>(builder.getUnknownLoc(), 0);
    auto barrier1 = funcbuilder.create<VPUIP::ConfigureBarrierOp>(builder.getUnknownLoc(), 1);

    // DMAs
    /* auto in_cmx_dma = */ funcbuilder.create<VPUIP::NNDMAOp>(
            builder.getUnknownLoc(), funcinput, inputcmx.getOperation()->getResult(0), mlir::ValueRange(),
            mlir::ValueRange(barrier0.barrier()), false);
    /* auto wt_data_cmx_dma = */ funcbuilder.create<VPUIP::NNDMAOp>(
            builder.getUnknownLoc(), weight_data_ddr.getOperation()->getResult(0),
            wtData_cmx.getOperation()->getResult(0), mlir::ValueRange(), mlir::ValueRange(barrier0.barrier()), false);
    /* auto wtTbl_cmx_dma = */ funcbuilder.create<VPUIP::NNDMAOp>(
            builder.getUnknownLoc(), weightTbl_data_ddr.getOperation()->getResult(0),
            wtTbl_cmx.getOperation()->getResult(0), mlir::ValueRange(), mlir::ValueRange(barrier0.barrier()), false);
    /* auto cmx_out_dma = */ funcbuilder.create<VPUIP::NNDMAOp>(
            builder.getUnknownLoc(), outputcmx.getOperation()->getResult(0), funcoutput,
            mlir::ValueRange(barrier1.barrier()), mlir::ValueRange(), false);

    // NCE Task
    auto strides = getIntArrayAttr(builder, conv.stride);
    std::vector<int64_t> padding_vec{conv.pad[0], conv.pad[2], conv.pad[1], conv.pad[3]};
    auto kernel_padding = getIntArrayAttr(builder, padding_vec);
    SmallVector<int64_t> kernel_vec = {wt_data_shape[2], wt_data_shape[3]};
    auto kernel_size = getIntArrayAttr(builder, kernel_vec);
    mlir::IntegerAttr actChannelLength = builder.getI32IntegerAttr(0);
    auto ppeLayer = getPPELayerFromConfig(activation);
    int32_t clampLow = std::numeric_limits<int32_t>::min();
    int32_t clampHigh = std::numeric_limits<int32_t>::max();
    int32_t lreluMult = 1;
    uint32_t lreluShift = 0;

    calculateppeParams(testDesc, clampLow, clampHigh, lreluMult, lreluShift);

    auto instructionList = mlir::Value();
    if (activation.activationType == nb::ActivationType::LeakyReLU) {
        log.info("Generating instruction table for ", to_string(activation.activationType));
        // instructionList ddr tensor
        auto instructionList_ddr_memSpaceAttr =
                VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::GraphFile);
        SmallVector<mlir::AffineMap> instructionList_ddr_affineMaps;
        std::size_t numberOfInstructions = 25;
        std::size_t alignedInstructions = round_up(numberOfInstructions, 16);
        llvm::SmallVector<int64_t> instructionList_data_shape = {1, 1, 1, static_cast<int64_t>(alignedInstructions)};
        const auto instructionListAffineMaps =
                DimsOrder::NHWC.toAffineMapsList(builder.getContext(), Shape(instructionList_data_shape));
        auto instructionList_ddr_type = mlir::MemRefType::get(
                makeArrayRef(instructionList_data_shape), builder.getIntegerType(32, /*isSigned=*/true),
                instructionListAffineMaps, instructionList_ddr_memSpaceAttr);
        /* const auto instructionList_ddr_valueType = */
        mlir::RankedTensorType::get(instructionList_data_shape, builder.getIntegerType(32, /*isSigned=*/true));

        const std::vector<int32_t> instructionList_values_vec =
                getInstructionListVals(activation.activationType, instructionList_data_shape);
        auto instructionList_data_values = makeArrayRef<int32_t>(instructionList_values_vec);
        auto instructionList_vals = mlir::DenseElementsAttr::get(instructionList_ddr_type, instructionList_data_values);
        /* auto instructionList_data_ddr = */ funcbuilder.create<Const::DeclareOp>(
                builder.getUnknownLoc(), instructionList_ddr_type,
                Const::ContentAttr::get(instructionList_vals).reorder(DimsOrder::NHWC));

        auto weights_totalsize = totalTensorSize(wt_data_shape, weightsType);
        ;
        const auto INSTRUCTIONLIST_CMX_OFFSET = WEIGHTS_CMX_OFFSET + weights_totalsize;

        // instructionList cmx tensor
        auto instructionList_cmx_memSpaceAttr =
                VPUIP::MemoryLocationAttr::get(builder.getContext(), VPUIP::MemoryLocation::VPU_CMX_NN);
        auto instructionList_cmx_type = mlir::MemRefType::get(
                makeArrayRef(instructionList_data_shape), builder.getIntegerType(32, /*isSigned=*/true),
                instructionListAffineMaps, instructionList_cmx_memSpaceAttr);
        auto instructionList_cmx =
                funcbuilder.create<VPUIP::DeclareTensorOp>(builder.getUnknownLoc(), instructionList_cmx_type,
                                                           VPUIP::MemoryLocation::VPU_CMX_NN, /*locale index=*/0,
                                                           /*data idx=*/INSTRUCTIONLIST_CMX_OFFSET);
        instructionList = instructionList_cmx.getOperation()->getResult(0);
    }

    auto nceTask = funcbuilder.create<VPUIP::NCEClusterTaskOp>(
            builder.getUnknownLoc(), outputcmx_type, inputcmx.getOperation()->getResult(0),
            wtData_cmx.getOperation()->getResult(0), wtTbl_cmx.getOperation()->getResult(0), nullptr,
            parent_inputcmx.getOperation()->getResult(0), parent_outputcmx.getOperation()->getResult(0),
            outputcmx.getOperation()->getResult(0), mlir::ValueRange(barrier0.barrier()),
            mlir::ValueRange(barrier1.barrier()), VPUIP::NCETaskType::CONV, kernel_size, strides, kernel_padding,
            actChannelLength, nullptr);

    nceTask.addPPETask(funcbuilder, ppeLayer, /*instructionList,*/ clampLow, clampHigh, lreluMult, lreluShift);

    // Create DPU task for NCE task
    nceTask.variants().emplaceBlock();
    auto variantbuilder = mlir::OpBuilder::atBlockBegin(&nceTask.variants().front(), builder.getListener());

    std::vector<int64_t> start_vec{0, 0, 0};
    auto start = getIntArrayAttr(builder, start_vec);
    std::vector<int64_t> end_vec{static_cast<int64_t>(out_shape[3] - 1), static_cast<int64_t>(out_shape[2] - 1),
                                 static_cast<int64_t>(out_shape[1] - 1)};
    auto end = getIntArrayAttr(builder, end_vec);
    auto pad = VPUIP::PaddingAttr::get(getIntAttr(builder, conv.pad[0]), getIntAttr(builder, conv.pad[1]),
                                       getIntAttr(builder, conv.pad[2]), getIntAttr(builder, conv.pad[3]),
                                       builder.getContext());

    /* auto dpuTask = */ variantbuilder.create<VPUIP::DPUTaskOp>(builder.getUnknownLoc(), nullptr, start, end, pad,
                                                                 VPUIP::MPEMode::CUBOID_16x16);

    // TODO : return empty as func does not return anything
    /* auto returnOp = */ funcbuilder.create<mlir::ReturnOp>(builder.getUnknownLoc(), funcoutput);

    // set runtime resources
    mlir::PassManager pm(builder.getContext(), mlir::OpPassManager::Nesting::Implicit);
    pm.addPass(VPUIP::createSetCompileParamsPass(VPUIP::ArchKind::MTL, VPUIP::CompilationMode(), None, log));

    VPUX_THROW_UNLESS(mlir::succeeded(pm.run(module)), "Compilation failed");

    // IE.CNNNetwork
    buildCNNOp(builder, func.getName(), {getTensorType(in_shape, inputType, DimsOrder::NHWC, nullptr)},
               {getTensorType(out_shape, outputType, DimsOrder::NHWC, nullptr)});
}

}  // namespace hwtest
}  // namespace vpux
