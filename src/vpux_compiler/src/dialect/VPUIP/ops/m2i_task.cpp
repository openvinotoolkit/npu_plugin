//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/graph-schema/utils.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"

using namespace vpux;

//
// M2ITaskOp::serialize
//

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::M2ITaskOp::serialize(VPUIP::BlobWriter& writer) {
    const auto getRawFP16 = [](auto val) {
        const auto valFP16 = float16(val);
        return valFP16.to_bits();
    };

    const auto getVecFP16 = [&](auto range) {
        return writer.createVector(range | transformed(getRawFP16));
    };

    VPUIP::BlobWriter::Vector<uint16_t> serializedCoefs;

    if (norm().hasValue()) {
        const auto coefs = parseFPArrayAttr<double>(norm().getValue());
        serializedCoefs = getVecFP16(coefs);
    }

    const auto getTensorCb = [this, &writer](mlir::Value val) {
        return writer.getTensorRef(val);
    };
    const auto inputs = writer.createVector(getInputs() | transformed(getTensorCb));
    const auto outputs = writer.createVector(getOutputs() | transformed(getTensorCb));

    MVCNN::M2ITaskBuilder builder(writer);
    builder.add_src(inputs);
    builder.add_dst(outputs);
    builder.add_do_csc(do_csc());
    builder.add_do_norm(do_norm());
    builder.add_in_fmt(convertM2iColor2MVCNN(inFmt()));
    builder.add_out_fmt(convertM2iColor2MVCNN(outFmt()));

    if (norm().hasValue()) {
        builder.add_norm_coefs(serializedCoefs);
    }

    return {builder.Finish().Union(), MVCNN::SpecificTask_M2ITask};
}
