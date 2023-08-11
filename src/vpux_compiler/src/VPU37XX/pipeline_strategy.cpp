//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/compiler.hpp"

#include "vpux/al/config/common.hpp"
#include "vpux/al/config/compiler.hpp"

#include "vpux/compiler/VPU37XX/pipeline_strategy.hpp"
#include "vpux/compiler/VPU37XX/pipelines.hpp"

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/options_mapper.hpp"

#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/pipelines.hpp"

#include "vpux/compiler/dialect/VPU/passes.hpp"

using namespace vpux;

//
// PipelineStrategy37XX::buildPipeline
//

void PipelineStrategy37XX::buildPipeline(mlir::PassManager& pm, const Config& config, mlir::TimingScope& rootTiming,
                                         Logger log, const PrecisionInfo& prcInfo) {
    auto buildTiming = rootTiming.nest("Build compilation pipeline");

    const auto archKind = getArchKind(config);
    const auto compilationMode = getCompilationMode(config);
    const auto enableProfiling = config.get<PERF_COUNT>();
    const auto numOfDPUGroups = getNumberOfDPUGroups(config);
    const auto numOfDMAPorts = getNumberOfDMAEngines(config);
    const auto ddrHeapSize = getDDRHeapSize(config);

    pm.addPass(VPU::createInitCompilerPass(archKind, compilationMode, numOfDPUGroups, numOfDMAPorts, ddrHeapSize,
                                           log.nest()));

    if (compilationMode == VPU::CompilationMode::ReferenceSW) {
        const auto options = ReferenceSWOptions37XX::createFromString(config.get<COMPILATION_MODE_PARAMS>());
        VPUX_THROW_UNLESS(options != nullptr, "buildPipeline failed to parse COMPILATION_MODE_PARAMS");
        options->enableProfiling = enableProfiling;
        if (config.get<PLATFORM>() == InferenceEngine::VPUXConfigParams::VPUXPlatform::EMULATOR) {
            buildEMUReferenceSWModePipeline(pm, *options, log.nest());
        } else {
            buildReferenceSWModePipeline(pm, *options, log.nest());
        }
    } else if (compilationMode == VPU::CompilationMode::ReferenceHW) {
        const auto options = ReferenceHWOptions37XX::createFromString(config.get<COMPILATION_MODE_PARAMS>());
        VPUX_THROW_UNLESS(options != nullptr, "buildPipeline failed to parse COMPILATION_MODE_PARAMS");
        options->enableProfiling = enableProfiling;
        if (config.get<PLATFORM>() == InferenceEngine::VPUXConfigParams::VPUXPlatform::EMULATOR) {
            buildEMUReferenceHWModePipeline(pm, *options, log.nest());
        } else {
            buildReferenceHWModePipeline(pm, *options, log.nest());
        }
    } else if (compilationMode == VPU::CompilationMode::DefaultHW) {
        const auto options = DefaultHWOptions37XX::createFromString(config.get<COMPILATION_MODE_PARAMS>());
        VPUX_THROW_UNLESS(options != nullptr, "buildPipeline failed to parse COMPILATION_MODE_PARAMS");
        options->enableProfiling = enableProfiling;
        options->enableConvertAvgPoolToDWConv = false;
        options->enableHandleAsymmetricStrides = false;
        options->enableVerticalFusion = true;

        // floatInputPrecision:
        // In case user passes -ip fp16/fp32 and enables FORCE_HOST_QUANTIZATION feature
        // we perform quantization on host.
        options->forceHostInputQuantization = config.get<FORCE_HOST_QUANTIZATION>() && prcInfo.floatInputPrecision;

        if (config.get<PLATFORM>() == InferenceEngine::VPUXConfigParams::VPUXPlatform::EMULATOR) {
            buildEMUDefaultHWModePipeline(pm, *options, log.nest());
        } else {
            buildDefaultHWModePipeline(pm, *options, log.nest());
        }
    } else if (compilationMode == VPU::CompilationMode::ShaveCodeGen) {
        buildShaveCodeGenPipeline37XX(pm, log.nest());
    } else {
        VPUX_THROW("Unsupported compilation mode '{0}'", compilationMode);
    }

    if (isELFEnabled(config)) {
        buildLowerVPUIP2ELFPipeline(pm, log.nest());
    }
}
