//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/tools/options.hpp"
#include "vpux/compiler/utils/passes.hpp"
#include "vpux/utils/core/error.hpp"

#include <llvm/Support/CommandLine.h>
#include <mlir/Pass/PassOptions.h>

using namespace vpux;

// Use PassPipelineOptions to have access to a convenient API like "createFromString" method
struct InitCompilerPassOptions : mlir::PassPipelineOptions<InitCompilerPassOptions> {
    vpux::StrOption arch{*this, "vpu-arch", llvm::cl::desc("VPU architecture to compile for")};

    // This options are only needed to avoid an error message from "Pass-Options-Parser": "no such option <option_name>"
    vpux::StrOption compilationModeOpt{*this, "compilation-mode", ::llvm::cl::desc(""), ::llvm::cl::init("DefaultHW")};
    vpux::IntOption numberOfDPUGroupsOpt{*this, "num-of-dpu-groups", ::llvm::cl::desc("")};
    vpux::IntOption numberOfDMAPortsOpt{*this, "num-of-dma-ports", ::llvm::cl::desc("")};
    vpux::BoolOption allowCustomValues{*this, "allow-custom-values", ::llvm::cl::desc("")};
};

//
// parseArchKind
//

vpux::VPU::ArchKind vpux::parseArchKind(int argc, char* argv[]) {
    static llvm::cl::OptionCategory vpuxOptOptions("VPUX Options");

    // Please use this option to test pipelines only (DefaultHW, ReferenceSW, etc.)
    // This option allows us to avoid ambiguity here when the parameters contradict each other:
    // "vpux-opt --init-compiler="vpu-arch=VPUX30XX compilation-mode=ReferenceSW" --default-hw-mode"
    // Instead you should only pass arch version:
    // "vpux-opt --vpu-arch=VPUX30XX --default-hw-mode"
    static llvm::cl::opt<std::string> archOpt("vpu-arch", llvm::cl::desc("VPU architecture to compile for"),
                                              llvm::cl::init(""), llvm::cl::cat(vpuxOptOptions));

    // To test the appropriate passes and pipelines depending on the type of device, we need to get a value for arch
    // kind. Most of lit-tests use command line that contains following option: --init-compiler="vpu-arch=VPUX37XX" To
    // avoid duplicating command line options like this: "vpux-opt --vpu-arch=VPUX37XX
    // --init-compiler="vpu-arch=VPUX37XX".." let's parse init-compiler option to get arch kind. There is still code
    // duplication, but in one place that does not affect developers:

    static llvm::cl::opt<std::string> initCompiler("init-compiler",
                                                   llvm::cl::desc("Duplicated option for VPU::InitCompilerPass"),
                                                   llvm::cl::init(""), llvm::cl::cat(vpuxOptOptions));

    // skip unknown options here, like split-input-file, show-dialects, etc.
    llvm::cl::list<std::string> sink(llvm::cl::Sink);

    // At this moment output of such command line will be reduced: "vpux-opt --help",
    // since this is the first call of the parser here and not all options are registered
    llvm::cl::ParseCommandLineOptions(argc, argv, "Register passes and pipelines depending on device type", nullptr,
                                      nullptr,
                                      /*LongOptionsUseDoubleDash=*/true);

    VPUX_THROW_WHEN(!archOpt.empty() && !initCompiler.empty(),
                    "ArchKind value is ambiguous. Only one option can be used at a time, either \"vpu-arch\" or "
                    "\"init-compiler\"");
    VPUX_THROW_WHEN(archOpt.empty() && initCompiler.empty(), "Can't get ArchKind value");

    const auto getArchFromString = [](vpux::StringRef archOptStr) {
        auto archKind = vpux::VPU::symbolizeEnum<vpux::VPU::ArchKind>(archOptStr);
        VPUX_THROW_UNLESS(archKind.hasValue(), "Unknown VPU architecture : '{0}'", archOpt.getValue());
        return archKind.getValue();
    };

    auto arch = vpux::VPU::ArchKind::UNKNOWN;
    if (!archOpt.empty()) {
        arch = getArchFromString(archOpt);
    } else {
        const auto options = InitCompilerPassOptions::createFromString(initCompiler);
        arch = getArchFromString(options->arch);
    }

    initCompiler.removeArgument();
    sink.removeArgument();
    llvm::cl::ResetAllOptionOccurrences();

    return arch;
}
