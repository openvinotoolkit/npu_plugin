//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <vpux/compiler/dialect/VPUIP/graph-schema/blob_reader.hpp>
#include <vpux/compiler/dialect/VPUIP/graph-schema/import.hpp>

namespace vpux {
namespace VPUIP {

mlir::OwningModuleRef importBlob(mlir::MLIRContext* ctx, const std::vector<char>& blob, Logger log) {
    return BlobReader(ctx, blob, log).read();
}

}  // namespace VPUIP
}  // namespace vpux
