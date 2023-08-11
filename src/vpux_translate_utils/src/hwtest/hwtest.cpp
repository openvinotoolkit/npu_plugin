//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/hwtest/hwtest.hpp"

#include <llvm/Support/ToolOutputFile.h>
#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Support/DebugStringHelper.h>
#include <mlir/Support/FileUtilities.h>

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/export.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/init.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/hwtest/hwtest_utils.hpp"
#include "vpux/utils/core/error.hpp"

#include "vpux/compiler/dialect/ELF/export.hpp"

namespace {

void serialize(const uint8_t* data, size_t size, const vpux::Logger& log, mlir::StringRef fileName = "vpuip.blob") {
    std::string err;
    auto outFile = mlir::openOutputFile(fileName, &err);
    if (!outFile) {
        log.error("Failed to open file {0} to write blob: {1}", fileName, err);
        return;
    }

    outFile->os().write(reinterpret_cast<const char*>(data), size);
    outFile->keep();
    log.info("Saved blob to {0}", outFile->getFilename());
}

}  // namespace

namespace vpux {

mlir::OwningOpRef<mlir::ModuleOp> importHWTEST(llvm::StringRef sourceJson, mlir::MLIRContext* ctx) {
    mlir::DialectRegistry registry;
    registerDialects(registry);
    ctx->appendDialectRegistry(registry);
    ctx->loadDialect<VPUIP::VPUIPDialect>();
    ctx->loadDialect<VPURT::VPURTDialect>();
    ctx->loadDialect<VPUMI37XX::VPUMI37XXDialect>();

    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(ctx), StringRef("mainModule"));
    auto log = Logger{"vpux-hwtest", LogLevel::Trace};
    auto builder = mlir::OpBuilder(module.getBodyRegion());

    nb::TestCaseJsonDescriptor jsonDesc(sourceJson);

    // TODO:
    // This will be handled later based on op type in config json
    auto opType = jsonDesc.getCaseStr();

    auto mainOpJsonDesc = jsonDesc;
    if (jsonDesc.getCaseType() == nb::CaseType::RaceCondition) {
        auto underlyingOp = jsonDesc.getUnderlyingOp();
        VPUX_THROW_WHEN(underlyingOp == nullptr, "underlyingOp is nullptr for CaseType::RaceCondition");
        mainOpJsonDesc = *underlyingOp;
    }

    const SmallVector<nb::InputLayer> inputList = mainOpJsonDesc.getInputLayerList();
    auto outputs = mainOpJsonDesc.getOutputLayers();

    SmallVector<mlir::Type> input_types;
    for (std::size_t idx = 0; idx < inputList.size(); idx++) {
        input_types.push_back(hwtest::parseInputType(builder, inputList[idx]));
    }

    mlir::Type output_type = hwtest::parseOutputType(builder, outputs.front());

    const SmallVector<nb::WeightLayer> weightList = mainOpJsonDesc.getWeightLayers();

    SmallVector<mlir::Type> weightTypes;
    for (std::size_t idx = 0; idx < weightList.size(); idx++) {
        weightTypes.push_back(hwtest::parseWeightsType(builder, weightList[idx]));
    }

    switch (jsonDesc.getCaseType()) {
    case nb::CaseType::DMA: {
        hwtest::buildDMA(jsonDesc, module, builder, log, input_types.front(), output_type);
        break;
    }
    case nb::CaseType::DMAcompressAct: {
        hwtest::buildDMACompressAct(jsonDesc, module, builder, log, input_types.front(), output_type);
        break;
    }
    case nb::CaseType::ZMajorConvolution: {
        const auto weightInChannels = weightList.front().shape[1];

        if (weightInChannels > 8 * 1024) {
            hwtest::buildContinuedConv(jsonDesc, module, builder, log, input_types.front(), weightTypes.front(),
                                       output_type);
        } else {
            hwtest::buildSimpleZMajorConv(jsonDesc, module, builder, log, input_types.front(), weightTypes.front(),
                                          output_type);
        }
        break;
    }
    case nb::CaseType::SparseZMajorConvolution: {
        hwtest::buildSparseZMajorConv(jsonDesc, module, builder, log, input_types.front(), weightTypes.front(),
                                      output_type);
        break;
    }
    case nb::CaseType::DepthWiseConv: {
        hwtest::buildDWConv(jsonDesc, module, builder, log, input_types.front(), weightTypes.front(), output_type);
        break;
    }
    case nb::CaseType::DoubleZMajorConvolution: {
        hwtest::buildDoubleConv(jsonDesc, module, builder, log, input_types.front(), weightTypes.front(), output_type);
        break;
    }
    case nb::CaseType::EltwiseAdd: {
        hwtest::buildEltwiseAdd(jsonDesc, module, builder, log, input_types.front(), weightTypes.front(), output_type);
        break;
    }
    case nb::CaseType::EltwiseMult: {
        hwtest::buildEltwiseMultWithDwConv(jsonDesc, module, builder, log, input_types.front(), weightTypes.front(),
                                           output_type);
        break;
    }
    case nb::CaseType::EltwiseSparse: {
        hwtest::buildEltwiseSparse(jsonDesc, module, builder, log, input_types.front(), weightTypes.front(),
                                   output_type);
        break;
    }
    case nb::CaseType::MaxPool: {
        hwtest::buildMaxPool(jsonDesc, module, builder, log, input_types.front(), output_type);
        break;
    }
    case nb::CaseType::AvgPool: {
        hwtest::buildAvgpool(jsonDesc, module, builder, log, input_types.front(), output_type);
        break;
    }
    case nb::CaseType::DifferentClustersDPU: {
        hwtest::buildDifferentClustersDPUTest(jsonDesc, module, builder, log, input_types.front(), weightTypes.front(),
                                              output_type);
        break;
    }
    case nb::CaseType::MultiClustersDPU: {
        hwtest::buildMultiClustersDPUTest(jsonDesc, module, builder, log, input_types.front(), weightTypes.front(),
                                          output_type);
        break;
    }
    case nb::CaseType::ActShave: {
        hwtest::buildActShave(jsonDesc, module, builder, log, input_types, output_type);
        break;
    }
    case nb::CaseType::ReadAfterWriteDPUDMA: {
        hwtest::buildReadAfterWriteDPUDMATest(jsonDesc, module, builder, log, input_types.front(), weightTypes.front(),
                                              output_type);
        break;
    }
    case nb::CaseType::ReadAfterWriteDMADPU: {
        hwtest::buildReadAfterWriteDMADPUTest(jsonDesc, module, builder, log, input_types.front(), weightTypes.front(),
                                              output_type);
        break;
    }
    case nb::CaseType::ReadAfterWriteACTDMA: {
        hwtest::buildReadAfterWriteACTDMATest(jsonDesc, module, builder, log, input_types.front(), output_type);
        break;
    }
    case nb::CaseType::ReadAfterWriteDMAACT: {
        hwtest::buildReadAfterWriteDMAACTTest(jsonDesc, module, builder, log, input_types.front(), output_type);
        break;
    }
    case nb::CaseType::ReadAfterWriteDPUACT: {
        hwtest::buildReadAfterWriteDPUACTTest(jsonDesc, module, builder, log, input_types.front(), weightTypes.front(),
                                              output_type);
        break;
    }
    case nb::CaseType::ReadAfterWriteACTDPU: {
        hwtest::buildReadAfterWriteACTDPUTest(jsonDesc, module, builder, log, input_types.front(), weightTypes.front(),
                                              output_type);
        break;
    }
    case nb::CaseType::RaceConditionDMA: {
        hwtest::buildRaceConditionDMATest(jsonDesc, module, builder, log, input_types.front(), output_type);
        break;
    }
    case nb::CaseType::RaceConditionDPU: {
        hwtest::buildRaceConditionDPUTest(jsonDesc, module, builder, log, input_types.front(), weightTypes.front(),
                                          output_type);
        break;
    }
    case nb::CaseType::RaceConditionDPUDMA: {
        hwtest::buildRaceConditionDPUDMATest(jsonDesc, module, builder, log, input_types.front(), weightTypes.front(),
                                             output_type);
        break;
    }
    case nb::CaseType::RaceConditionDPUDMAACT: {
        hwtest::buildRaceConditionDPUDMAACTTest(jsonDesc, module, builder, log, input_types.front(),
                                                weightTypes.front(), output_type);
        break;
    }
    case nb::CaseType::RaceConditionDPUACT: {
        hwtest::buildRaceConditionDPUACTTest(jsonDesc, module, builder, log, input_types.front(), weightTypes.front(),
                                             output_type);
        break;
    }
    case nb::CaseType::RaceCondition: {
        hwtest::buildRaceConditionTest(jsonDesc, module, builder, log, input_types.front(), output_type);
        break;
    }
    case nb::CaseType::StorageElementTableDPU: {
        hwtest::buildSETableTest(jsonDesc, module, builder, log, input_types.front(), weightTypes.front(), output_type);
        break;
    }
    case nb::CaseType::DualChannelDMA: {
        hwtest::buildDualChannelDMATest(jsonDesc, module, builder, log, input_types.front(), output_type);
        break;
    }
    default:
        VPUX_THROW("Unknown type: {0}", opType);
        break;
    };

    // llvm::dbgs() << "Current module: " << mlir::debugString(module);

    VPUX_THROW_UNLESS(mlir::succeeded(mlir::verify(module)),
                      "Failed to create a valid MLIR module for InferenceEngine IR");

    const std::vector<std::shared_ptr<const ov::Node>> params;
    const std::vector<std::shared_ptr<const ov::Node>> results;
    if (jsonDesc.getCompilerBackend() == nb::CompilerBackend::Flatbuffer) {
        mlir::DefaultTimingManager tm;
        auto timing = tm.getRootScope();
        auto blob = VPUIP::exportToBlob(module, timing, {}, params, results, log);

        serialize(blob.data(), blob.size(), log);
    } else if (jsonDesc.getCompilerBackend() == nb::CompilerBackend::ELF) {
        mlir::PassManager pm(ctx, mlir::OpPassManager::Nesting::Implicit);

        auto getLoweringPipeline = [](vpux::VPU::ArchKind /*arch*/) {
            return buildLowerVPUIP2ELFPipeline;
        };
        auto getExportToELFfunc = [](vpux::VPU::ArchKind /*arch*/) {
            return ELF::exportToELF;
        };

        getLoweringPipeline(jsonDesc.getArchitecture())(pm, log);
        VPUX_THROW_UNLESS(mlir::succeeded(pm.run(module)), "Failed to lower test model to ELF");
        auto blob = getExportToELFfunc(jsonDesc.getArchitecture())(module, {}, params, results, log);

        serialize(blob.data(), blob.size(), log);
    } else {
        VPUX_THROW("Encountered unsupported compile backend {}", nb::to_string(jsonDesc.getCompilerBackend()));
    }

    return module;
}

}  // namespace vpux
