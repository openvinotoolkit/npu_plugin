//
// Copyright 2020 Intel Corporation.
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

#include "vpux/compiler/dialect/VPUIPRegMapped/ops.hpp"

#include "vpux/utils/core/format.hpp"

using namespace vpux;

//
// EmptyOp
//

// VPUIPRegMapped::BlobWriter::SpecificTask vpux::VPUIPRegMapped::EmptyOp::serialize(VPUIPRegMapped::BlobWriter& writer)
// {
void vpux::VPUIPRegMapped::EmptyOp::serialize(std::vector<char>& buffer) {
    /*
    MVCNN::EmptyTaskBuilder subBuilder(writer);
    const auto subTask = subBuilder.Finish();

    MVCNN::ControllerTaskBuilder builder(writer);
    builder.add_task_type(MVCNN::ControllerSubTask_EmptyTask);
    builder.add_task(subTask.Union());

    return {builder.Finish().Union(), MVCNN::SpecificTask_ControllerTask};
    */

    (void)buffer;
}
