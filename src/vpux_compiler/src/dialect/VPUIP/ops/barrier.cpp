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

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include "vpux/utils/core/format.hpp"

using namespace vpux;

//
// ConfigureBarrierOp
//

void vpux::VPUIP::ConfigureBarrierOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, int64_t id) {
    build(builder, state, vpux::VPUIP::BarrierType::get(builder.getContext()), id, mlir::ValueRange{},
          mlir::ValueRange{});
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::ConfigureBarrierOp::serialize(VPUIP::BlobWriter& writer) {
    const auto barrier = writer.createBarrier(this->barrier(), this->id());

    MVCNN::BarrierConfigurationTaskBuilder subBuilder(writer);
    subBuilder.add_target(barrier);
    const auto subTask = subBuilder.Finish();

    MVCNN::ControllerTaskBuilder builder(writer);
    builder.add_task_type(MVCNN::ControllerSubTask_BarrierConfigurationTask);
    builder.add_task(subTask.Union());

    return {builder.Finish().Union(), MVCNN::SpecificTask_ControllerTask};
}
