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

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include "vpux/utils/core/format.hpp"

using namespace vpux;

//
// ConfigureBarrierOp
//

void vpux::VPUIP::ConfigureBarrierOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, uint32_t id) {
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
