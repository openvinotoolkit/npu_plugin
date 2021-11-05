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

#include "vpux/compiler/utils/subspaces.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

EMU::BlobWriter::SpecificTask vpux::EMU::RegionYoloUPAOp::serialize(EMU::BlobWriter& writer) {
    EMU::BlobWriter::Vector<int32_t> serializedMask;
    if (mask().hasValue()) {
        serializedMask = writer.createVector(parseIntArrayAttr<int32_t>(mask().getValue()));
    }

    MVCNN::RegionYOLOParamsBuilder builder(writer);
    builder.add_coords(checked_cast<int32_t>(coords()));
    builder.add_classes(checked_cast<int32_t>(classes()));
    builder.add_num(checked_cast<int32_t>(regions()));
    builder.add_do_softmax(do_softmax().getValueOr(false));
    if (mask().hasValue()) {
        builder.add_mask(serializedMask);
    }

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_RegionYOLOParams});
}
