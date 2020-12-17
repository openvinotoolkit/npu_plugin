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

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::PoolingUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto kernel = VPUIP::BlobWriter::createOrder3(this->kernel());
    const auto strides = VPUIP::BlobWriter::createOrder3(this->strides());
    const auto padsBegin = VPUIP::BlobWriter::createOrder3(this->padsBegin());
    const auto padsEnd = VPUIP::BlobWriter::createOrder3(this->padsEnd());

    VPUIP::BlobWriter::String type;
    switch (this->type()) {
    case VPUIP::PoolLayerType::MAX:
        type = writer.createString("max");
        break;
    case VPUIP::PoolLayerType::AVG:
        type = writer.createString("avg");
        break;
    default:
        VPUX_THROW("Unsupported PoolLayerType {0}", this->type());
    }

    const auto excludePad = writer.createString(this->excludePad() ? "true" : "false");

    MVCNN::PoolingParamsBuilder builder(writer);
    builder.add_pool_method(type);
    builder.add_kernel(&kernel);
    builder.add_strides(&strides);
    builder.add_pads_begin(&padsBegin);
    builder.add_pads_end(&padsEnd);
    builder.add_exclude_pad(excludePad);
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(getOperation(), {paramsOff.Union(), MVCNN::SoftwareLayerParams_PoolingParams},
                                     maxShaves(), isTrailingSWLayer());
}
