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

#include "vpux/compiler/dialect/VPUIP/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/hwtest/hwtest_utils.hpp"
#include "vpux/hwtest/test_case_json_parser.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/numeric.hpp"

namespace vpux {
namespace hwtest {

//
//       [input]
//          |
//        (conv)
//          |
//       [output]
//

void buildSimpleZMajorConv(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                           Logger& log, mlir::Type inputType, mlir::Type weightsType, mlir::Type outputType) {
    const auto int32 = builder.getIntegerType(32, true);

    const auto input = testDesc.getInputLayer();
    const auto weights = testDesc.getWeightLayer();
    const auto conv = testDesc.getConvLayer();
    const auto output = testDesc.getOutputLayer();

    const llvm::SmallVector<std::int64_t> inputShape(input.shape.begin(), input.shape.end());
    const llvm::SmallVector<std::int64_t> outputShape(output.shape.begin(), output.shape.end());
    const llvm::SmallVector<std::int64_t> weightsShape{weights.shape[0], weights.shape[1], weights.shape[2],
                                                       weights.shape[3]};
    const llvm::SmallVector<std::int64_t> weightsTableShape{weightsShape[0], 1, 1, 4};

    const char* weightsFileName = "weights.dat";

    const auto weights_alignment = 16 * 1024;
    const auto OUTPUT_CMX_OFFSET = 0;
    const auto INPUT_CMX_OFFSET = OUTPUT_CMX_OFFSET + totalTensorSize(outputShape, outputType);
    const auto WEIGHTSTABLE_CMX_OFFSET =
            vpux::alignVal<int64_t>(INPUT_CMX_OFFSET + totalTensorSize(inputShape, inputType), weights_alignment);
    const auto WEIGHTS_CMX_OFFSET = vpux::alignVal<int64_t>(
            WEIGHTSTABLE_CMX_OFFSET + 4 * weightsTableShape[0] * weightsTableShape[3], weights_alignment);

    const auto getMemRef = [&builder](ArrayRef<std::int64_t> shape, mlir::Type elemType,
                                      vpux::VPUIP::MemoryLocation location) {
        const auto memSpaceAttr = VPUIP::MemoryLocationAttr::get(builder.getContext(), location);
        return vpux::getMemRefType(ShapeRef(shape), elemType, DimsOrder::NHWC, memSpaceAttr);
    };

    const auto outputParamType = getMemRef(outputShape, outputType, vpux::VPUIP::MemoryLocation::ProgrammableOutput);

    llvm::SmallVector<mlir::Type, 2> inputTypes;
    inputTypes.push_back(getMemRef(inputShape, inputType, vpux::VPUIP::MemoryLocation::ProgrammableInput));
    inputTypes.push_back(outputParamType);

    const auto funcType = builder.getFunctionType(llvm::makeArrayRef(inputTypes), outputParamType);

    auto function = builder.create<mlir::FuncOp>(
            builder.getUnknownLoc(), llvm::formatv("zmajor_conv_{0}_{1}_{2}", inputType, weightsType, outputType).str(),
            funcType, builder.getStringAttr("private"));

    auto functionBuilder = mlir::OpBuilder::atBlockBegin(function.addEntryBlock(), builder.getListener());

    const auto getCMXTensor = [&builder, &functionBuilder, getMemRef](const llvm::SmallVector<std::int64_t>& shape,
                                                                      mlir::Type type, std::size_t offset) {
        const auto CMXType = getMemRef(shape, type, vpux::VPUIP::MemoryLocation::VPU_CMX_NN);
        return functionBuilder.create<vpux::VPUIP::DeclareTensorOp>(builder.getUnknownLoc(), CMXType,
                                                                    vpux::VPUIP::MemoryLocation::VPU_CMX_NN, 0, offset);
    };

    auto functionInput = function.getArgument(0);
    auto functionOutput = function.getArgument(1);

    auto max_swizzling_key = [&](size_t totalByteSize, size_t cmx_address) {
        std::map<int, int> swizzling_offsets = {{0, 16}, {1, 1024}, {2, 2048}, {3, 4096}, {4, 8192}, {5, 16384}};
        auto swizzling_key = 0;
        for (auto it = swizzling_offsets.rbegin(); it != swizzling_offsets.rend(); it++) {
            swizzling_key = it->first;
            auto swizzling_offset = it->second;
            if ((totalByteSize % swizzling_offset == 0) && (cmx_address % swizzling_offset == 0))
                break;
        }

        return swizzling_key;
    };

    // Find the max swizzling key that satisfies the size and address constraint
    auto weights_swizzling_key = max_swizzling_key(totalTensorSize(weightsShape, weightsType), WEIGHTS_CMX_OFFSET);
    auto weights_table_swizzling_key =
            max_swizzling_key(totalTensorSize(weightsTableShape, int32), WEIGHTSTABLE_CMX_OFFSET);
    auto swizzling_key = std::min(weights_swizzling_key, weights_table_swizzling_key);

    log.info("weights: size={0}, addr={1}, max swizzling_key={2}", totalTensorSize(weightsShape, weightsType),
             WEIGHTS_CMX_OFFSET, weights_swizzling_key);
    log.info("weights_table: size={0}, addr={1}, max swizzling_key={2}", totalTensorSize(weightsTableShape, int32),
             WEIGHTSTABLE_CMX_OFFSET, weights_table_swizzling_key);
    log.info("using swizzling_key {0} for weights & weights_table", swizzling_key);

    const auto weightsValues = generateWeights(weightsShape, weightsType, builder.getContext(), weightsFileName);
    auto weightsAttribute = vpux::Const::ContentAttr::get(weightsValues);
    auto qty = weightsType.dyn_cast<mlir::quant::QuantizedType>();
    if (qty != nullptr) {
        weightsAttribute = weightsAttribute.quantCast(qty);
    }
    weightsAttribute = weightsAttribute.reorder(vpux::DimsOrder::NHWC);

    auto weightsCMX = getCMXTensor(weightsShape, weightsType, WEIGHTS_CMX_OFFSET);

    if (qty != nullptr && qty.getStorageType().isInteger(4)) {
        // swizzling doesn't work for int4/uint4 weights case yet
        weightsAttribute = weightsAttribute.bitPack(4);
    } else {
        weightsAttribute = weightsAttribute.swizzle(swizzling_key);
        weightsCMX.swizzlingKeyAttr(vpux::getIntAttr(builder.getContext(), swizzling_key));
    }

    const auto weightsDDRType = getMemRef(weightsShape, weightsType, vpux::VPUIP::MemoryLocation::GraphFile);

    auto weightsDDR =
            functionBuilder.create<vpux::Const::DeclareOp>(builder.getUnknownLoc(), weightsDDRType, weightsAttribute);

    auto inputCMX = getCMXTensor(inputShape, inputType, INPUT_CMX_OFFSET);
    auto outputCMX = getCMXTensor(outputShape, outputType, OUTPUT_CMX_OFFSET);

    const auto weightsTableDDRType = mlir::RankedTensorType::get(weightsTableShape, int32);
    const auto weightsTable = vpux::VPUIP::NCESparsity::getWeightsTable(
            inputType, outputType, static_cast<std::int32_t>(WEIGHTS_CMX_OFFSET),
            static_cast<std::int32_t>(weights.shape[1] * weights.shape[2] * weights.shape[3] *
                                      getElemTypeSize(weightsType).count() / 8),
            static_cast<std::int32_t>(16777215), vpux::VPUIP::ArchKind::MTL, output.shape[1], weightsType);

    const auto weightsTableDDRMemRef = getMemRef(weightsTableShape, int32, vpux::VPUIP::MemoryLocation::GraphFile);
    const auto weightsTableValues =
            mlir::DenseElementsAttr::get(weightsTableDDRType, llvm::makeArrayRef<std::int32_t>(weightsTable));
    auto weightsTableContentAttr = vpux::Const::ContentAttr::get(weightsTableValues).reorder(vpux::DimsOrder::NHWC);
    if (weightsCMX.swizzlingKeyAttr()) {
        weightsTableContentAttr = weightsTableContentAttr.swizzle(swizzling_key);
    }
    auto weightsTableDDR = functionBuilder.create<vpux::Const::DeclareOp>(
            builder.getUnknownLoc(), weightsTableDDRMemRef, weightsTableContentAttr);
    auto weightsTableCMX = getCMXTensor(weightsTableShape, int32, WEIGHTSTABLE_CMX_OFFSET);

    auto barrier0 = functionBuilder.create<vpux::VPUIP::ConfigureBarrierOp>(builder.getUnknownLoc(), 0);
    auto barrier1 = functionBuilder.create<vpux::VPUIP::ConfigureBarrierOp>(builder.getUnknownLoc(), 1);

    functionBuilder.create<vpux::VPUIP::NNDMAOp>(builder.getUnknownLoc(), functionInput,
                                                 inputCMX.getOperation()->getResult(0), mlir::ValueRange(),
                                                 mlir::ValueRange(barrier0.barrier()), false);
    functionBuilder.create<vpux::VPUIP::NNDMAOp>(builder.getUnknownLoc(), weightsDDR.getOperation()->getResult(0),
                                                 weightsCMX.getOperation()->getResult(0), mlir::ValueRange(),
                                                 mlir::ValueRange(barrier0.barrier()), false);
    functionBuilder.create<vpux::VPUIP::NNDMAOp>(builder.getUnknownLoc(), weightsTableDDR.getOperation()->getResult(0),
                                                 weightsTableCMX.getOperation()->getResult(0), mlir::ValueRange(),
                                                 mlir::ValueRange(barrier0.barrier()), false);
    functionBuilder.create<vpux::VPUIP::NNDMAOp>(builder.getUnknownLoc(), outputCMX.getOperation()->getResult(0),
                                                 functionOutput, mlir::ValueRange(barrier1.barrier()),
                                                 mlir::ValueRange(), false);

    const auto strides = getIntArrayAttr(builder.getContext(), conv.stride);
    std::vector<std::int64_t> paddings = convertNBPadtoNCETaskPad(conv.pad);
    const auto kernelPaddings = getIntArrayAttr(builder.getContext(), paddings);
    llvm::SmallVector<std::int64_t> kernel = {weightsShape[2], weightsShape[3]};
    const auto kernelSize = getIntArrayAttr(builder.getContext(), kernel);
    const auto odu_permutation =
            vpux::VPUIP::ODUPermutationAttr::get(builder.getContext(), testDesc.getODUPermutation());

    auto nceTask = functionBuilder.create<vpux::VPUIP::NCEClusterTaskOp>(
            builder.getUnknownLoc(), inputCMX.memory(), weightsCMX.memory(), weightsTableCMX.memory(), nullptr,
            inputCMX.memory(), outputCMX.memory(), outputCMX.memory(), vpux::VPUIP::NCETaskType::CONV, kernelSize,
            strides, kernelPaddings, nullptr, nullptr, odu_permutation);

    nceTask.waitBarriersMutable().append(barrier0.barrier());
    nceTask.updateBarriersMutable().append(barrier1.barrier());

    const auto start = getIntArrayAttr(builder.getContext(), std::vector<std::int64_t>{0, 0, 0});
    const auto end =
            getIntArrayAttr(builder.getContext(),
                            std::vector<std::int64_t>{outputShape[3] - 1, outputShape[2] - 1, outputShape[1] - 1});
    const auto pad = vpux::VPUIP::PaddingAttr::get(vpux::getIntAttr(builder, paddings[PAD_NCETASK_LEFT]),
                                                   vpux::getIntAttr(builder, paddings[PAD_NCETASK_RIGHT]),
                                                   vpux::getIntAttr(builder, paddings[PAD_NCETASK_TOP]),
                                                   vpux::getIntAttr(builder, paddings[PAD_NCETASK_BOTTOM]),
                                                   builder.getContext());

    nceTask.addDPUTask(functionBuilder, nullptr, start, end, pad, conv.cube_mode);

    functionBuilder.create<mlir::ReturnOp>(builder.getUnknownLoc(), functionOutput);

    mlir::PassManager pm(builder.getContext(), mlir::OpPassManager::Nesting::Implicit);
    pm.addPass(vpux::VPUIP::createSetCompileParamsPass(vpux::VPUIP::ArchKind::MTL,
                                                       vpux::VPUIP::CompilationMode::DefaultHW, None, log));

    if (conv.compress) {
        pm.addPass(VPUIP::createCompressWeightsPass(log));
    }

    VPUX_THROW_UNLESS(mlir::succeeded(pm.run(module)), "Compilation failed");

    buildCNNOp(builder, function.getName(),
               {getTensorType(ShapeRef(inputShape), inputType, vpux::DimsOrder::NHWC, nullptr)},
               {getTensorType(ShapeRef(outputShape), outputType, vpux::DimsOrder::NHWC, nullptr)});
}

}  // namespace hwtest
}  // namespace vpux
