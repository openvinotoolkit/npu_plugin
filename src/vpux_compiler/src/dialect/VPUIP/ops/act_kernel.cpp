//
// Copyright 2021 Intel Corporation.
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

using namespace vpux;

//
// ActivationKernelOp
//

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::ActKernelOp::serialize(VPUIP::BlobWriter& writer) {
    // TODO: Update this code to handle the changes that have been made to the graphfile schema.
    /* auto text = */ writer.getKernelData(getOperation(), kernelText());
    /* auto data = */ writer.getKernelData(getOperation(), kernelData());

    MVCNN::ActKernelBuilder kernelBuilder(writer);
    kernelBuilder.add_type(MVCNN::ActKernelType_KERNEL);
    // kernelBuilder.add_kernelData(text);
    // kernelBuilder.add_kernelEntry(entryOffset());
    auto kernel = kernelBuilder.Finish();

    SmallVector<flatbuffers::Offset<MVCNN::ActKernelInvocation>, 16> invocationOffsets;

    for (auto invocationAttr : invocations()) {
        auto invocationArrayAttr = invocationAttr.dyn_cast<mlir::ArrayAttr>();
        VPUX_THROW_UNLESS(invocationArrayAttr, "Expected an array attribute");
        auto args = parseUIntArrayAttr<uint32_t>(invocationArrayAttr);
        /* auto argsVector = */ writer.createVector(args);
        MVCNN::ActKernelInvocationBuilder invocationBuilder(writer);
        // invocationBuilder.add_dataSection(data);
        // invocationBuilder.add_kernelArgs(argsVector);
        invocationOffsets.push_back(invocationBuilder.Finish());
    }
    auto invocationsVector = writer.createVector(invocationOffsets);

    MVCNN::ActKernelTaskBuilder builder(writer);
    builder.add_kernel(kernel);
    builder.add_invocations(invocationsVector);

    return {builder.Finish().Union(), MVCNN::SpecificTask_ActKernelTask};
}
