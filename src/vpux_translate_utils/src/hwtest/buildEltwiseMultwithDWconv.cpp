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

#include <numeric>

#include <mlir/Dialect/Quant/QuantTypes.h>

#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/hwtest/hwtest_utils.hpp"
#include "vpux/hwtest/test_case_json_parser.hpp"
#include "vpux/utils/core/error.hpp"

namespace vpux {
namespace hwtest {

void buildEltwiseMultWithDwConv(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module,
                                mlir::OpBuilder builder, Logger& log, mlir::Type inputType, mlir::Type weightsType,
                                mlir::Type outputType) {
    using namespace VPUIP;

    auto LOC_UNKNOWN = builder.getUnknownLoc();
    auto NO_BARRIER = mlir::ValueRange();

    const size_t num_func_args = 3;

    auto input = testDesc.getInputLayer();
    auto avgpool = testDesc.getPoolLayer();
    auto output = testDesc.getOutputLayer();

    /*
        Notes on shapes
        ----------------
        - if non-flat input/output shapes are used in future test cases, ImplicitReshapes are needed to flatten the
       shapes beforehand.
        - NCE DWConv as EltwiseMult requires the input/output shapes w/ the elements in the C dim...  e.g., (1,32,1,1)
        - However, the ImplicitConcat to the zero-pad weights requires the elements to be in the W dim
        - Therefore, I changed the testcase input/output shapes to (1,1,1,32) & used DeclareTensor ops to
       implicitly-reshape the input/output of NCE


        Summary of Topology
        -------------------

        input           weights             zero_pad
        (1,1,1,32)      (1,1,1,32)          (1,15,1,32)
        INPUT           INPUT               GRAPHFILE
            |               |                    |
          NNDMA           NNDMA                NNDMA
            |                \                  /
            |                   ImplicitConcat
            |                         |
        input_cmx               weights_cmx             weightstable_ddr        activation_window_ddr
        (1,1,1,32)              (1,16,1,32)             (32,1,1,4)              (32,1,1,16)
        VPU_CMX_NN              VPU_CMX_NN              GRAPHFILE               GRAPHFILE
             |                       |                        |                       |
        ImplicitReshape         ImplicitReshape             NNDMA                   NNDMA
             |                       |                        |                       |
        input_nce               weights_nce             weightstable_nce        activation_window_nce
        (1,32,1,1)              (32,1,1,16)             (32,1,1,4)              (32,1,1,16)
        VPU_CMX_NN              VPU_CMX_NN              VPU_CMX_NN               VPU_CMX_NN
           \                       \                         /                       /
            \_______________________\_______________________/_______________________/
                                                |
                                            NCEClusterTask
                                                |
                                            output_nce
                                            (1,32,1,1)
                                            VPU_CMX_NN
                                                |
                                            ImplicitReshape
                                                |
                                            output_cmx
                                            (1,1,1,32)
                                            VPU_CMX_NN
                                                |
                                              NNDMA
                                                |
                                            output
                                            (1,1,1,32)
                                            OUTPUT

    */

    // Func input/output
    SmallVector<int64_t> input_shape(input.shape.begin(), input.shape.end());
    SmallVector<int64_t> weights_shape(input.shape.begin(), input.shape.end());
    SmallVector<int64_t> output_shape(output.shape.begin(), output.shape.end());

    // Weights concat
    SmallVector<int64_t> zero_pad_shape({1, 15, 1, input_shape[3]});
    SmallVector<int64_t> weights_pad_shape({1, 16, 1, input_shape[3]});

    // NCE input/output
    SmallVector<int64_t> input_nce_shape({1, input_shape[3], 1, 1});
    SmallVector<int64_t> weights_nce_shape({input_shape[3], 1, 1, 16});
    SmallVector<int64_t> output_nce_shape(input_nce_shape.begin(), input_nce_shape.end());

    std::vector<int64_t> filter_size({1, 1});
    std::vector<int64_t> stride_vec({1, 1});
    std::vector<int64_t> padding_vec({0, 0, 0, 0});

    auto output_totalsize = totalTensorSize(output_shape, outputType);
    auto input_totalsize = totalTensorSize(input_shape, inputType);
    auto weights_nce_totalsize = totalTensorSize(weights_nce_shape, weightsType);
    auto input1_leadingoffset = totalTensorSize({weights_shape[1]}, inputType);

    const auto OUTPUT_CMX_OFFSET = 0;
    const auto INPUT0_CMX_OFFSET = OUTPUT_CMX_OFFSET + output_totalsize;
    const auto INPUT1_CMX_OFFSET = INPUT0_CMX_OFFSET + input_totalsize;
    const auto ZERO_PAD_DDR_OFFSET = 0;
    const auto ZERO_PAD_CMX_OFFSET = INPUT1_CMX_OFFSET + input1_leadingoffset;
    const auto WEIGHTS_PAD_CMX_OFFSET = INPUT1_CMX_OFFSET;
    const auto ACTIVATIONWINDOW_CMX_OFFSET = WEIGHTS_PAD_CMX_OFFSET + weights_nce_totalsize;

    const auto input_maps = DimsOrder::NHWC.toAffineMapsList(builder.getContext(), Shape(input_shape));
    const auto weights_maps = DimsOrder::NHWC.toAffineMapsList(builder.getContext(), Shape(weights_shape));
    const auto output_maps = DimsOrder::NHWC.toAffineMapsList(builder.getContext(), Shape(output_shape));

    const auto zero_pad_maps = DimsOrder::NHWC.toAffineMapsList(builder.getContext(), Shape(zero_pad_shape));
    const auto weights_pad_maps = DimsOrder::NHWC.toAffineMapsList(builder.getContext(), Shape(weights_pad_shape));

    const auto input_nce_maps = DimsOrder::NHWC.toAffineMapsList(builder.getContext(), Shape(input_nce_shape));
    const auto output_nce_maps = DimsOrder::NHWC.toAffineMapsList(builder.getContext(), Shape(output_nce_shape));

    auto memSpaceAttr_in = MemoryLocationAttr::get(builder.getContext(), MemoryLocation::ProgrammableInput);
    auto memSpaceAttr_zeros = MemoryLocationAttr::get(builder.getContext(), MemoryLocation::GraphFile);
    auto memSpaceAttr_out = MemoryLocationAttr::get(builder.getContext(), MemoryLocation::ProgrammableOutput);

    SmallVector<mlir::Type> inputTypes;
    inputTypes.reserve(num_func_args);
    auto inputParamType = mlir::MemRefType::get(makeArrayRef(input_shape), inputType, input_maps, memSpaceAttr_in);
    auto weightsParamType =
            mlir::MemRefType::get(makeArrayRef(weights_shape), weightsType, weights_maps, memSpaceAttr_in);
    auto outputParamType = mlir::MemRefType::get(makeArrayRef(output_shape), outputType, output_maps, memSpaceAttr_out);
    inputTypes.push_back(inputParamType);
    inputTypes.push_back(weightsParamType);
    inputTypes.push_back(outputParamType);

    const auto funcType = builder.getFunctionType(makeArrayRef(inputTypes), outputParamType);

    // TODO: Func should not return
    auto func = builder.create<mlir::FuncOp>(
            LOC_UNKNOWN, llvm::formatv("eltwise_mult_{0}_{1}_{2}", inputType, weightsType, outputType).str(), funcType,
            builder.getStringAttr("private"));

    auto funcbuilder = mlir::OpBuilder::atBlockBegin(func.addEntryBlock(), builder.getListener());

    // Build VPUIP ops
    auto funcinput = func.getArgument(0);
    auto funcweights = func.getArgument(1);
    auto funcoutput = func.getArgument(2);

    // Tensors - constant zero padding
    /*auto zero_pad_ddr = */ createDeclareTensorOp(funcbuilder, MemoryLocation::GraphFile, zero_pad_shape, weightsType,
                                                   zero_pad_maps, 0, ZERO_PAD_DDR_OFFSET);

    // Tensor - input cmx
    auto input_cmx = createDeclareTensorOp(funcbuilder, MemoryLocation::VPU_CMX_NN, input_shape, inputType, input_maps,
                                           0, INPUT0_CMX_OFFSET);

    // Tensors - concat input/output
    auto weights_cmx = createDeclareTensorOp(funcbuilder, MemoryLocation::VPU_CMX_NN, weights_shape, weightsType,
                                             weights_pad_maps, 0, WEIGHTS_PAD_CMX_OFFSET);
    auto zero_pad_cmx = createDeclareTensorOp(funcbuilder, MemoryLocation::VPU_CMX_NN, zero_pad_shape, weightsType,
                                              weights_pad_maps, 0, ZERO_PAD_CMX_OFFSET);
    /*auto weights_pad_cmx = */ createDeclareTensorOp(funcbuilder, MemoryLocation::VPU_CMX_NN, weights_pad_shape,
                                                      weightsType, weights_pad_maps, 0, WEIGHTS_PAD_CMX_OFFSET);

    // Tensors - NCE input/output
    auto input_nce_cmx = createDeclareTensorOp(funcbuilder, MemoryLocation::VPU_CMX_NN, input_nce_shape, inputType,
                                               input_nce_maps, 0, INPUT0_CMX_OFFSET);
    auto weights_nce_cmx = createDeclareTensorOp(funcbuilder, MemoryLocation::VPU_CMX_NN, weights_nce_shape,
                                                 weightsType, weights_pad_maps, 0, WEIGHTS_PAD_CMX_OFFSET);
    auto output_nce_cmx = createDeclareTensorOp(funcbuilder, MemoryLocation::VPU_CMX_NN, output_nce_shape, outputType,
                                                output_nce_maps, 0, OUTPUT_CMX_OFFSET);
    auto parent_input_nce_cmx = createDeclareTensorOp(funcbuilder, MemoryLocation::VPU_CMX_NN, input_nce_shape,
                                                      inputType, input_nce_maps, 0, INPUT0_CMX_OFFSET);
    auto parent_output_nce_cmx = createDeclareTensorOp(funcbuilder, MemoryLocation::VPU_CMX_NN, output_nce_shape,
                                                       outputType, output_nce_maps, 0, OUTPUT_CMX_OFFSET);

    // Tensor - output cmx
    auto output_cmx = createDeclareTensorOp(funcbuilder, MemoryLocation::VPU_CMX_NN, output_shape, outputType,
                                            output_maps, 0, OUTPUT_CMX_OFFSET);

    // Barriers
    auto barrier0 = funcbuilder.create<ConfigureBarrierOp>(builder.getUnknownLoc(), 0);
    auto barrier1 = funcbuilder.create<ConfigureBarrierOp>(builder.getUnknownLoc(), 1);
    auto BARRIER_0 = barrier0.barrier();
    auto BARRIER_1 = barrier1.barrier();

    auto wt_data_vals = generateZeroPadForEltwiseMultWeights(zero_pad_shape, weightsType, builder.getContext());
    auto wt_data_attr = Const::ContentAttr::get(wt_data_vals);
    if (auto qty = weightsType.dyn_cast<mlir::quant::QuantizedType>()) {
        wt_data_attr = wt_data_attr.quantCast(qty);
    }
    auto zero_pad_type =
            mlir::MemRefType::get(makeArrayRef(zero_pad_shape), weightsType, zero_pad_maps, memSpaceAttr_zeros);
    auto zero_pad_data =
            funcbuilder.create<Const::DeclareOp>(LOC_UNKNOWN, zero_pad_type, wt_data_attr.reorder(DimsOrder::NHWC));

    // Input DMAs
    funcbuilder.create<NNDMAOp>(LOC_UNKNOWN, funcinput, getTensorResult(input_cmx), NO_BARRIER, BARRIER_0, false);
    funcbuilder.create<NNDMAOp>(LOC_UNKNOWN, zero_pad_data, getTensorResult(zero_pad_cmx), NO_BARRIER, BARRIER_0,
                                false);
    funcbuilder.create<NNDMAOp>(LOC_UNKNOWN, funcweights, getTensorResult(weights_cmx), NO_BARRIER, BARRIER_0, false);

    // Activation Window
    SmallVector<int64_t> sparsity_shape;
    mlir::Type sparsity_type = getUInt8Type(builder.getContext());
    bool isOutFloat = (outputType.isBF16() || outputType.isF16()) ? true : false;
    mlir::IntegerAttr actChannelLength;
    auto sparsityAttr = getactivationWindow(builder, filter_size, stride_vec, output_nce_shape[1], input_nce_shape[1],
                                            sparsity_type, sparsity_shape, actChannelLength, isOutFloat, false, true);

    auto act_window_maps = DimsOrder::NHWC.toAffineMapsList(builder.getContext(), Shape(sparsity_shape));

    auto act_window_ddr_memreftype =
            getMemRefType(funcbuilder, MemoryLocation::GraphFile, sparsity_shape, sparsity_type, act_window_maps);
    auto act_window_cmx_memreftype =
            getMemRefType(funcbuilder, MemoryLocation::VPU_CMX_NN, sparsity_shape, sparsity_type, act_window_maps);

    auto act_window_totalsize = totalTensorSize(sparsity_shape, sparsity_type);
    auto act_window_totalsize_bytes = act_window_totalsize * sparsity_type.getIntOrFloatBitWidth() / 8;

    auto act_window_ddr = funcbuilder.create<Const::DeclareOp>(
            LOC_UNKNOWN, act_window_ddr_memreftype, Const::ContentAttr::get(sparsityAttr).reorder(DimsOrder::NHWC));
    auto act_window_cmx = createDeclareTensorOp(funcbuilder, MemoryLocation::VPU_CMX_NN, sparsity_shape, sparsity_type,
                                                act_window_maps, 0, ACTIVATIONWINDOW_CMX_OFFSET);

    funcbuilder.create<NNDMAOp>(LOC_UNKNOWN, getConstResult(act_window_ddr), getTensorResult(act_window_cmx),
                                NO_BARRIER, BARRIER_0, false);

    // weights table ddr
    SmallVector<int64_t> weightstable_data_shape{sparsity_shape[0], 1, 1, 4};
    auto weightstable_maps = DimsOrder::NHWC.toAffineMapsList(builder.getContext(), Shape(weightstable_data_shape));
    auto weightstable_ddr_memreftype = getMemRefType(funcbuilder, MemoryLocation::GraphFile, weightstable_data_shape,
                                                     builder.getIntegerType(32, /*isSigned=*/true), weightstable_maps);
    const auto wtTblData_ddr_valueType =
            mlir::RankedTensorType::get(weightstable_data_shape, builder.getIntegerType(32, /*isSigned=*/true));

    auto input_cmx_memreftype =
            getMemRefType(funcbuilder, MemoryLocation::VPU_CMX_NN, input_nce_shape, inputType, input_nce_maps);
    auto weights_cmx_memreftype =
            getMemRefType(funcbuilder, MemoryLocation::VPU_CMX_NN, weights_nce_shape, weightsType, weights_pad_maps);
    auto output_cmx_memreftype =
            getMemRefType(funcbuilder, MemoryLocation::VPU_CMX_NN, output_nce_shape, outputType, output_nce_maps);

    const std::vector<int32_t> weightstable_data_values_vec = generateWeightsTablesValuesWithSparsity(
            testDesc, input_cmx_memreftype, output_cmx_memreftype, weights_cmx_memreftype, act_window_cmx_memreftype,
            ACTIVATIONWINDOW_CMX_OFFSET, weightstable_data_shape, WEIGHTS_PAD_CMX_OFFSET);

    auto weightstable_data_values = makeArrayRef<int32_t>(weightstable_data_values_vec);
    auto weightstable_data_vals = mlir::DenseElementsAttr::get(wtTblData_ddr_valueType, weightstable_data_values);

    auto weightstable_data_ddr = funcbuilder.create<Const::DeclareOp>(
            LOC_UNKNOWN, weightstable_ddr_memreftype,
            Const::ContentAttr::get(weightstable_data_vals).reorder(DimsOrder::NHWC));

    // weights table cmx tensor
    const auto WEIGHTSTABLE_CMX_OFFSET = ACTIVATIONWINDOW_CMX_OFFSET + act_window_totalsize_bytes;
    auto weightstable_cmx = createDeclareTensorOp(funcbuilder, MemoryLocation::VPU_CMX_NN, weightstable_data_shape,
                                                  builder.getIntegerType(32, /*isSigned=*/true), weightstable_maps, 0,
                                                  WEIGHTSTABLE_CMX_OFFSET);

    // weights table dma ddr->cmx
    funcbuilder.create<NNDMAOp>(LOC_UNKNOWN, getConstResult(weightstable_data_ddr), getTensorResult(weightstable_cmx),
                                NO_BARRIER, BARRIER_0, false);

    // NCE Task
    auto filtersize = getIntArrayAttr(builder, filter_size);
    auto strides = getIntArrayAttr(builder, stride_vec);
    auto kernel_padding = getIntArrayAttr(builder, padding_vec);

    auto nceTask = funcbuilder.create<NCEClusterTaskOp>(
            LOC_UNKNOWN, output_cmx_memreftype, getTensorResult(input_nce_cmx), getTensorResult(weights_nce_cmx),
            getTensorResult(weightstable_cmx), getTensorResult(act_window_cmx), getTensorResult(parent_input_nce_cmx),
            getTensorResult(parent_output_nce_cmx), getTensorResult(output_nce_cmx), BARRIER_0, BARRIER_1,
            NCETaskType::DWCONV, filtersize, strides, kernel_padding, actChannelLength, /*odu_permutation=*/nullptr,
            /*weights_plt=*/mlir::Value());

    nceTask.addPPETask(funcbuilder);

    // DPU task for NCE task
    nceTask.variants().emplaceBlock();
    auto variantbuilder = mlir::OpBuilder::atBlockBegin(&nceTask.variants().front(), builder.getListener());
    createDPUTaskOp(builder, variantbuilder, output_nce_shape, padding_vec);

    // Output DMA
    funcbuilder.create<NNDMAOp>(LOC_UNKNOWN, getTensorResult(output_cmx), funcoutput, BARRIER_1, NO_BARRIER, false);

    // TODO : return empty as func does not return anything
    funcbuilder.create<mlir::ReturnOp>(LOC_UNKNOWN, funcoutput);

    // Runtime resources
    mlir::PassManager pm(builder.getContext(), mlir::OpPassManager::Nesting::Implicit);
    pm.addPass(createSetCompileParamsPass(ArchKind::MTL, CompilationMode(), None, log));

    // Compile
    VPUX_THROW_UNLESS(mlir::succeeded(pm.run(module)), "Compilation failed");

    // IE.CNNNetwork
    buildCNNOp(builder, func.getName(),
               {getTensorType(input_shape, inputType, DimsOrder::NHWC),
                getTensorType(weights_shape, weightsType, DimsOrder::NHWC)},
               {getTensorType(input_shape, outputType, DimsOrder::NHWC)});
}

}  // namespace hwtest
}  // namespace vpux
