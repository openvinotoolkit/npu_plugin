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

#include "vpux/compiler/dialect/EMU/ops.hpp"

#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/subspaces.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

//
// verifyOp
//

mlir::LogicalResult vpux::EMU::verifyOp(SplitUPAOp /*op*/) {
    // TODO::Add checks

    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::EMU::SplitUPAOp::serialize(EMU::BlobWriter& writer) {
    MVCNN::SplitParamsBuilder builder(writer);
    const auto paramsOff = builder.Finish();
    builder.add_axis(checked_cast<uint32_t>(axisAttr().getValue().getSExtValue()));
    builder.add_num_splits(checked_cast<uint32_t>(num_splitsAttr().getValue().getSExtValue()));

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_SplitParams});
}
