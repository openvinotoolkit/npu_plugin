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

#ifndef VPUX_COMPILER_EDSL_PASSES
#define VPUX_COMPILER_EDSL_PASSES

include "mlir/Pass/PassBase.td"

def AutoTileVPUX : FunctionPass<"vpux-edsl-autotile"> {
  let summary = "Autotile for VPU2.x";
  let constructor = "vpux::edsl::createAutoTileVPUXPass()";
  let options = [
    Option<"totalBuffer", "total-buffer", "uint64_t",
           /*default=*/"60000",
           "Total buffer in CMX">,
    Option<"minCount", "min-count", "uint32_t",
           /*default=*/"3",
           "Minimal outer loop count for the tiling">,
    Option<"maxCount", "max-count", "uint32_t",
           /*default=*/"1",
           "Maximal outer loop count for the tiling">,
    Option<"minInnerBuffer", "min-inner-buffer", "uint64_t",
           /*default=*/"3",
           "Minimal buffer used by inner loop">,
    Option<"outputIndicesOnly", "output-indices-only", "bool",
           /*default=*/"false",
           "The outer loop indices can be only the indices used by output tensors">,
    Option<"accIndicesOnly", "acc-indices-only", "bool",
           /*default=*/"false",
           "The inner loop indices can be only the accumulation">,
    Option<"noNegativeIndex", "no-negative-index", "bool",
           /*default=*/"false",
           "Negative items are not allowed in the memory access in the inner loop">,
    Option<"cacheWidth", "cache-width", "uint32_t",
           /*default=*/"64",
           "Cache width of VPUX">,
    Option<"vectorWidth", "vector-width", "uint32_t",
           /*default=*/"16",
           "Maximal vector width in bytes">,
    Option<"processingTags", "processing-tags", "std::string",
           /*default=*/"",
           "Tags for processing">,
    Option<"outerTags", "outer-tags", "std::string",
           /*default=*/"",
           "Outer loop tags">,
    Option<"innerTags", "inner-tags", "std::string",
           /*default=*/"",
           "Inner loop tags">,
  ];
}

def ShavePatterns : FunctionPass<"vpux-edsl-shave-patterns"> {
  let summary = "Optimize patterns for SHAVE ISA";
  let constructor = "vpux::edsl::createShavePatternsPass()";
}

def SinkScalar : FunctionPass<"vpux-edsl-sink-scalar"> {
  let summary = "Sink dependent scalars into outermost loops";
  let constructor = "vpux::edsl::createSinkScalarPass()";
}

def ShavePipeline : Pass<"vpux-edsl-shave-pipeline", "mlir::ModuleOp"> {
  let summary = "SHAVE pipeline pass";
  let constructor = "vpux::edsl::createShavePipelinePass()";
  let dependentDialects = [
    "vpux::VPUIP::VPUIPDialect",
    "mlir::vector::VectorDialect"
  ];
}

#endif // VPUX_COMPILER_EDSL_PASSES
