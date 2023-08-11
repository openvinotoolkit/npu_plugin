//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/ELF/import.hpp"
#include "vpux/compiler/dialect/ELF/elf_importer.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/logging.hpp"

using namespace vpux;

mlir::OwningOpRef<mlir::ModuleOp> vpux::ELF::importELF(mlir::MLIRContext* ctx, const std::string& elfFileName,
                                                       Logger log) {
    return vpux::ELF::ElfImporter(ctx, elfFileName, log).read();
}
