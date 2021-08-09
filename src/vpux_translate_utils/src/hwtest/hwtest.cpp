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

#include "vpux/hwtest/hwtest.hpp"

#include <llvm/Support/ToolOutputFile.h>
#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Support/DebugStringHelper.h>
#include <mlir/Support/FileUtilities.h>

#include "vpux/compiler/backend/VPUIP.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/init.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/hwtest/hwtest_utils.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux_config.hpp"

namespace vpux {

mlir::OwningModuleRef importHWTEST(llvm::StringRef sourceJson, mlir::MLIRContext* ctx) {
    mlir::DialectRegistry registry;
    registerDialects(registry);
    ctx->appendDialectRegistry(registry);
    ctx->loadDialect<VPUIP::VPUIPDialect>();

    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(ctx), StringRef("mainModule"));
    auto log = Logger{"vpux-hwtest", LogLevel::Trace};
    auto builderLog = OpBuilderLogger{log.nest()};
    auto builder = mlir::OpBuilder(module.getBodyRegion());

    nb::TestCaseJsonDescriptor jsonDesc(sourceJson);

    if (jsonDesc.getCaseType() == nb::CaseType::activationKernelSimple) {
        hwtest::buildActKernelTest(jsonDesc, module, builder, log);
        return module;
    }

    nb::InputLayer input = jsonDesc.getInputLayer();
    nb::OutputLayer output = jsonDesc.getOutputLayer();
    auto activation = jsonDesc.getActivationLayer();

    mlir::Type input_type = hwtest::parseInputType(builder, input);
    mlir::Type output_type = hwtest::parseOutputType(builder, output);

    // ZMajor convolutions
    //
    // TODO: Replace these hardcoded loops with a configuration input.
    //
    // We'd ideally like to write something like this, but we can only produce a
    // single module as our output, and the module can only contain a single
    // network.  And we want this to be driven from the JSON hardware test
    // description anyway.
    //
    // for (auto inputType : {i8, ui8}) {for (auto weightsType : {i8, ui8}) {for
    //     (auto outputType : {i8, ui8, f16}) {buildSimpleZMajorConv(builder,
    //     log, inputType, weightsType, outputType);
    //         }
    //     }
    // }
    // for (auto outputType : {i8, ui8, f16, f32})
    //     {buildSimpleZMajorConv(builder, log, f16, f16, outputType);
    // }
    // for (auto outputType : {bf16, f32}) {buildSimpleZMajorConv(builder, log,
    //     bf16, bf16, outputType);
    // }

    // TODO:
    // This will be handled later based on op type in config json
    auto opType = jsonDesc.getCaseStr();

    bool isConv = jsonDesc.getCaseType() == nb::CaseType::ZMajorConvolution;
    bool isDepthwiseConv = jsonDesc.getCaseType() == nb::CaseType::DepthWiseConv;
    bool isEltwiseAdd = jsonDesc.getCaseType() == nb::CaseType::EltwiseAdd;
    bool isEltwiseMult = jsonDesc.getCaseType() == nb::CaseType::EltwiseMult;
    bool isMaxPool = jsonDesc.getCaseType() == nb::CaseType::MaxPool;
    bool isAvgPool = jsonDesc.getCaseType() == nb::CaseType::AvgPool;
    bool isPipeline = (opType.find("Pipeline") != std::string::npos);
    bool isRaceConditionDMA = (opType.find("RaceConditionDMA") != std::string::npos);
    bool isRaceConditionDPU = (opType.find("RaceConditionDPU") != std::string::npos);

    auto weightType = [&]() {
        nb::WeightLayer weight = jsonDesc.getWeightLayer();
        return hwtest::parseWeightsType(builder, weight);
    };

    auto weightInChannels = [&]() {
        nb::WeightLayer weight = jsonDesc.getWeightLayer();
        return weight.shape[1];
    };

    if (isConv) {
        if (weightInChannels() > 8 * 1024) {
            hwtest::buildContinuedConv(jsonDesc, module, builder, log, input_type, weightType(), output_type);
        } else {
            if (activation.activationType != nb::ActivationType::None) {
                log.info("Building conv with activation");
                hwtest::buildSimpleZMajorConvActivation(jsonDesc, module, builder, log, input_type, weightType(),
                                                        output_type);
            } else {
                hwtest::buildSimpleZMajorConv(jsonDesc, module, builder, log, input_type, weightType(), output_type);
            }
        }
    } else if (isDepthwiseConv) {
        hwtest::buildDWConv(jsonDesc, module, builder, log, input_type, weightType(), output_type);
    } else if (isEltwiseAdd) {
        hwtest::buildEltwiseAdd(jsonDesc, module, builder, log, input_type, weightType(), output_type);
    } else if (isEltwiseMult) {
        hwtest::buildEltwiseMultWithDwConv(jsonDesc, module, builder, log, input_type, weightType(), output_type);
    } else if (isMaxPool) {
        log.info("Building MaxPool");
        hwtest::buildMaxPool(jsonDesc, module, builder, log, input_type, output_type);
    } else if (isAvgPool) {
        log.info("Building AvgPool with depth wise conv");
        hwtest::buildAvgpoolWithDwConv(jsonDesc, module, builder, log, input_type, output_type);
    } else if (isPipeline) {
        hwtest::buildPipeline(jsonDesc, module, builder, log, input_type, weightType(), output_type, false);
    } else if (isRaceConditionDMA) {
        hwtest::buildRaceConditionDMATest(jsonDesc, module, builder, log, input_type, output_type);
    } else if (isRaceConditionDPU) {
        hwtest::buildRaceConditionDPUTest(jsonDesc, module, builder, log, input_type, weightType(), output_type);
    } else {
        VPUX_THROW("Unknown type: {0}", opType);
    }

    // llvm::dbgs() << "Current module: " << mlir::debugString(module);

    VPUX_THROW_UNLESS(mlir::succeeded(mlir::verify(module)),
                      "Failed to create a valid MLIR module for InferenceEngine IR");

    mlir::DefaultTimingManager tm;
    auto timing = tm.getRootScope();
    auto blob = VPUIP::exportToBlob(module, timing, log);
    std::string err;
    // dump the blob in a file
    std::unique_ptr<llvm::ToolOutputFile> outFile = mlir::openOutputFile("vpuip.blob", &err);
    outFile->os().write(reinterpret_cast<const char*>(blob.data()), blob.size());
    outFile->keep();
    log.info("Saving blob to {0}", outFile->getFilename());

    return module;
}

mlir::LogicalResult exportHWTEST(mlir::ModuleOp module, llvm::raw_ostream&) {
    auto resources = IERT::RunTimeResourcesOp::getFromModule(module);
    auto context = module->getContext();
    auto DDR = VPUIP::PhysicalMemoryAttr::get(context, VPUIP::PhysicalMemory::DDR);
    auto expectedUsedDDREntry = resources.getUsedMemory(DDR);
    VPUX_THROW_UNLESS(expectedUsedDDREntry != nullptr, "expectedUsedDDREntry == nullptr");

    const auto expectedUsedDDR = static_cast<std::uint64_t>(expectedUsedDDREntry.byteSize());

    mlir::DefaultTimingManager tm;
    auto timing = tm.getRootScope();
    auto log = Logger{"vpux-hwtest", LogLevel::Info};
    auto blob = VPUIP::exportToBlob(module, timing, log);
    auto graphFile = MVCNN::GetGraphFile(blob.data());
    const auto header = graphFile->header();
    const auto memorySizes = header->resources()->memory_sizes();
    VPUX_THROW_UNLESS(memorySizes != nullptr, "memorySizes == nullptr");

    Optional<double> actualUsedDDREntry;
    for (flatbuffers::uoffset_t memory = 0; memory < memorySizes->size(); ++memory) {
        const auto entry = memorySizes->Get(memory);
        if (entry->item() == MVCNN::PhysicalMem_DDR) {
            actualUsedDDREntry = entry->number();
        }
    }

    VPUX_THROW_UNLESS(actualUsedDDREntry.hasValue(), "There is no used DDR specified in the blob");

    // MemoryMapping::number() returns double.
    // Assuming used DDR bytes amount is big enough we may face accuracy loss during double -> std::uint64_t conversion.
    // For testing purposes, MemoryMapping::number() review should be considered to return either std::string or
    // std::uint64_t.
    const auto actualUsedDDR = static_cast<std::uint64_t>(actualUsedDDREntry.getValue());

    VPUX_THROW_UNLESS(expectedUsedDDR == actualUsedDDR, "Expected {0} bytes used by DDR, actual is {1}",
                      expectedUsedDDR, actualUsedDDR);
    return mlir::success();
}

}  // namespace vpux
