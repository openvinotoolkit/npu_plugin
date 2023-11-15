//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/IMD/executor.hpp"

#include "vpux/IMD/parsed_config.hpp"
#include "vpux/IMD/platform_helpers.hpp"
#include "vpux/al/config/common.hpp"
#include "vpux/al/config/compiler.hpp"
#include "vpux/utils/IE/blob.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux/utils/core/scope_exit.hpp"

#include <blob_factory.hpp>
#include <openvino/util/file_util.hpp>

#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Program.h>

#include <fstream>

using namespace vpux;
using namespace InferenceEngine;
using InferenceEngine::VPUXConfigParams::VPUXPlatform;

//
// getMoviToolsPath
//

std::string vpux::IMD::ExecutorImpl::getMoviToolsPath(const Config& config) {
    if (config.has<IMD::MV_TOOLS_PATH>()) {
        return printToString("{0}/linux64/bin", config.get<IMD::MV_TOOLS_PATH>());
    } else {
        const auto* rootDir = std::getenv("MV_TOOLS_DIR");
        const auto* version = std::getenv("MV_TOOLS_VERSION");

        if (rootDir != nullptr && version != nullptr) {
            return printToString("{0}/{1}/linux64/bin", rootDir, version);
        } else {
            VPUX_THROW("Can't locate MOVI tools directory, please provide VPUX_IMD_MV_TOOLS_PATH config option or "
                       "MV_TOOLS_DIR/MV_TOOLS_VERSION env vars");
        }
    }
}

//
// isValidElfSignature
//

bool vpux::IMD::ExecutorImpl::isValidElfSignature(StringRef filePath) {
    std::ifstream in(std::string(filePath), std::ios_base::binary);

    VPUX_THROW_UNLESS(in.is_open(), "Could not open {0}", filePath);

    char buffer[4];
    in.read(buffer, 4);

    if (!in || buffer[0] != 0x7f || buffer[1] != 0x45 || buffer[2] != 0x4c || buffer[3] != 0x46) {
        return false;
    }

    return true;
}

//
// getSimicsPath
//

std::string vpux::IMD::ExecutorImpl::getSimicsPath(const Config& /* config */) {
    VPUX_THROW("Can't locate simics directory");
}

//
// setMoviSimRunArgs
//

void vpux::IMD::ExecutorImpl::setMoviSimRunArgs(VPUXPlatform platform, const Config& config) {
    const auto appName = getAppName(platform);
    const auto pathToTools = getMoviToolsPath(config);

    _app.runProgram = printToString("{0}/moviSim", pathToTools);
    _app.elfFile = printToString("{0}/vpux/simulator/{1}", ov::util::get_ov_lib_path(), appName);

    if (platform == VPUXPlatform::VPU3720) {
        // For some reason, -cv:3720xx doesn't work, while -cv:3700xx works OK for VPU3720
        _app.chipsetArg = "-cv:3700xx";
        _app.imdElfArg = printToString("-l:LRT:{0}", _app.elfFile);
    } else {
        _app.chipsetArg = "-cv:ma2490";
        _app.imdElfArg = printToString("-l:LRT0:{0}", _app.elfFile);
    }

    _app.runArgs = {_app.runProgram, _app.chipsetArg, "-nodasm", "-q", _app.imdElfArg, "-simLevel:fast"};
}

//
// setMoviDebugRunArgs
//

void vpux::IMD::ExecutorImpl::setMoviDebugRunArgs(VPUXPlatform platform, const Config& config) {
    const auto appName = getAppName(platform);
    const auto pathToTools = getMoviToolsPath(config);

    const auto* vpuElfPlatform = std::getenv("NPU_ELF_PLATFORM");
    const auto* vpuFirmwareDir = std::getenv("NPU_FIRMWARE_SOURCES_PATH");
    const auto* srvIP = std::getenv("NPU_SRV_IP");
    const auto* srvPort = std::getenv("NPU_SRV_PORT");

    if (vpuFirmwareDir == nullptr) {
        VPUX_THROW("Can't locate vpu firmware directory, please provide NPU_FIRMWARE_SOURCES_PATH env var");
    }
    if (vpuElfPlatform == nullptr) {
        vpuElfPlatform = "silicon";
        _log.warning("'NPU_ELF_PLATFORM' env variable is unset, using the default value: 'silicon'");
    } else {
        auto vpuElfPlatformStr = std::string(vpuElfPlatform);
        if (vpuElfPlatformStr != "silicon" && vpuElfPlatformStr != "fpga")
            VPUX_THROW("Unsupported value for NPU_ELF_PLATFORM env var, expected: 'silicon' or 'fpga', got '{0}'",
                       vpuElfPlatformStr);
    }

    _app.runProgram = printToString("{0}/moviDebug2", pathToTools);
    _app.elfFile = printToString("{0}/vpux/{1}/{2}", ov::util::get_ov_lib_path(), vpuElfPlatform, appName);
    _app.imdElfArg = printToString("-D:elf={0}", _app.elfFile);

    static std::string default_targetArg;

    switch (platform) {
    case VPUXPlatform::VPU3720:
        _app.chipsetArg = "-cv:3700xx";
        default_targetArg = "-D:default_target=LRT";
        break;
    case VPUXPlatform::VPU3700:
        _app.chipsetArg = "-cv:ma2490";
        default_targetArg = "-D:default_target=LRT0";
        break;
    default:
        VPUX_THROW("Platform '{0}' is not supported", platform);
        break;
    }

    _app.runArgs = {_app.runProgram, _app.imdElfArg, _app.chipsetArg, default_targetArg, "--no-uart"};

    if (srvIP != nullptr) {
        static auto srvIPArg = printToString("-srvIP:{0}", srvIP);
        _app.runArgs.append({srvIPArg});
    } else {
        _log.warning("'NPU_SRV_IP' env variable is unset, moviDebug2 will try to connect to localhost");
    }

    if (srvPort != nullptr) {
        static auto srvPortArg = printToString("-serverPort:{0}", srvPort);
        _app.runArgs.append({srvPortArg});
    } else {
        _log.warning("'NPU_SRV_PORT' env variable is unset, moviDebug2 will try to connect to 30000 or 30001 port");
    }

    // Debug scripts
    switch (platform) {
    case VPUXPlatform::VPU3720:
    case VPUXPlatform::VPU3700:
        static auto default_mdbg2Arg =
                printToString("{0}/build/buildSupport/scripts/debug/default_mdbg2.scr", vpuFirmwareDir);
        static auto default_pipe_mdbg2Arg =
                printToString("{0}/build/buildSupport/scripts/debug/default_pipe_mdbg2.scr", vpuFirmwareDir);
        static auto default_run_mdbg2Arg =
                printToString("{0}/build/buildSupport/scripts/debug/default_run_mdbg2.scr", vpuFirmwareDir);

        _app.runArgs.append({"--init", default_mdbg2Arg});
        _app.runArgs.append({"--init", default_pipe_mdbg2Arg});
        _app.runArgs.append({"--script", default_run_mdbg2Arg});
        break;
    default:
        VPUX_THROW("Platform '{0}' is not supported", platform);
        break;
    }

    _app.runArgs.append({"-D:run_opt=runw", "-D:exit_opt=exit"});
}

//
// setSimicsRunArgs
//

void vpux::IMD::ExecutorImpl::setSimicsRunArgs(const VPUXPlatform platform, const Config& config) {
    const auto appName = getAppName(platform);
    const auto simicsDir = getSimicsPath(config);

    _app.runProgram = printToString("{0}/simics", simicsDir);
    _app.elfFile = printToString("{0}/vpux/simics/{1}", ov::util::get_ov_lib_path(), appName);

    static auto binaryFile = "$binary=" + _app.elfFile;

    _app.runArgs = {_app.runProgram,
                    "-batch-mode",
                    "-no-win",
                    "-e",
                    binaryFile,
                    "-e",
                    "$NPU_GEN=4",
                    "-e",
                    "$NPU_GENSKU=4000",
                    "-e",
                    "run-command-file \"%simics%/targets/vpu/vpu.simics\""};
}

//
// parseAppConfig
//

void vpux::IMD::ExecutorImpl::parseAppConfig(VPUXPlatform platform, const Config& config) {
    VPUX_THROW_UNLESS(platformSupported(platform), "Platform '{0}' is not supported", platform);

    const auto mode = config.get<IMD::LAUNCH_MODE>();

    switch (mode) {
    case IMD::LaunchMode::Simulator: {
        setMoviSimRunArgs(platform, config);
        break;
    }
    case IMD::LaunchMode::MoviDebug: {
        setMoviDebugRunArgs(platform, config);
        break;
    }
    default:
        VPUX_THROW("Unsupported launch mode '{0}'", mode);
    }

    VPUX_THROW_UNLESS(isValidElfSignature(_app.elfFile),
                      "Elf signature check failed for {0}. Please fetch the file using `git lfs pull`, then rebuild "
                      "the project or the `npu_imd_backend_copy_app` cmake target.",
                      _app.elfFile);

    _app.timeoutSec = config.get<IMD::MV_RUN_TIMEOUT>().count();
}

//
// Base interface API implementation
//

vpux::IMD::ExecutorImpl::ExecutorImpl(VPUXPlatform platform, const NetworkDescription::Ptr& network,
                                      const Config& config)
        : _network(network), _log("InferenceManagerDemo", config.get<LOG_LEVEL>()) {
    parseAppConfig(platform, config);
}
