//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FormatVariadic.h>

#include <vpux_elf/types/symbol_entry.hpp>
#include <vpux_elf/utils/error.hpp>
#include <vpux_loader/vpux_loader.hpp>

#include <cstdint>
#include <ctime>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#ifndef VPUX_ELF_UNIT_NAME
#define VPUX_ELF_UNIT_NAME Loader
#endif
#include <vpux_elf/utils/log.hpp>

using namespace elf;

namespace {

llvm::cl::opt<bool> verbose("v", llvm::cl::desc("Set verbosity"));

}  // namespace

int main(int argc, char* argv[]) {
    llvm::cl::ParseCommandLineOptions(argc, argv);

    if (verbose.getValue()) {
        Logger::setGlobalLevel(LogLevel::DEBUG);
    }

    VPUXLoader loader();
    VPUX_ELF_LOG(LogLevel::DEBUG, "loaderTest : DEBUG : Just checking if we can use VPUXLoader headers");
    VPUX_ELF_LOG(LogLevel::INFO, "loaderTest : INFO : Just checking if we can use VPUXLoader headers");

    return 0;
}
