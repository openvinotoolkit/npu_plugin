//
// Copyright 2021 Intel Corporation.
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

#include <vpux/compiler/frontend/VPUIP.hpp>

#include <vpux/compiler/dialect/VPUIP/blob_reader.hpp>

namespace vpux {
namespace VPUIP {

mlir::OwningModuleRef importBlob(mlir::MLIRContext* ctx, const std::vector<char>& blob, Logger log) {
    return BlobReader(ctx, blob, log).read();
}

}  // namespace VPUIP
}  // namespace vpux
