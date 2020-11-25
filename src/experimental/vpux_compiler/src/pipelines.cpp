//
// Copyright 2020 Intel Corporation.
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

#include "vpux/compiler/pipelines.hpp"

#include "vpux/compiler/conversion/passes.hpp"
#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"

using namespace vpux;

void vpux::buildReferenceModePipeline(mlir::OpPassManager& pm, uint32_t maxUPAShaves, Logger log) {
    pm.addPass(IE::createAdjustPrecisionForVPUPass(log));
    pm.addPass(createConvertIE2VPUIPPass(maxUPAShaves, log));
    pm.addPass(VPUIP::createRemoveExtraDMAPass(log));
    pm.addPass(VPUIP::createAssignTensorOffsetsDDRPass(log));
    pm.addPass(VPUIP::createAddLinearSchedulingPass(log));
}

void vpux::registerAllPipelines() {
    mlir::PassPipelineRegistration<> referenceMode(
            "reference-mode", "Compile IE Network in Reference mode (SW only execution)", [](mlir::OpPassManager& pm) {
                buildReferenceModePipeline(pm);
            });
}
