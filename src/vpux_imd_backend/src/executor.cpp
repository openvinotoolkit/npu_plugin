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
// parseAppConfig
//

void vpux::IMD::ExecutorImpl::parseAppConfig(VPUXPlatform platform, const Config& config) {
    // Check if platform is supported and get elf file name
    const auto appName = getAppName(platform);

    // Path to MOVI tools dir
    std::string pathToTools;

    if (config.has<IMD::MV_TOOLS_PATH>()) {
        pathToTools = printToString("{0}/linux64/bin", config.get<IMD::MV_TOOLS_PATH>());
    } else {
        const auto* rootDir = std::getenv("MV_TOOLS_DIR");
        const auto* version = std::getenv("MV_TOOLS_VERSION");

        if (rootDir != nullptr && version != nullptr) {
            pathToTools = printToString("{0}/{1}/linux64/bin", rootDir, version);
        } else {
            VPUX_THROW("Can't locate MOVI tools directory, please provide VPUX_IMD_MV_TOOLS_PATH config option or "
                       "MV_TOOLS_DIR/MV_TOOLS_VERSION env vars");
        }
    }

    // Run program configuration

    const auto mode = config.get<IMD::LAUNCH_MODE>();

    switch (mode) {
    case IMD::LaunchMode::MoviSim: {
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

        break;
    }
    case IMD::LaunchMode::MoviDebug: {
        const auto* vpuElfPlatform = std::getenv("VPU_ELF_PLATFORM");
        const auto* vpuFirmwareDir = std::getenv("VPU_FIRMWARE_SOURCES_PATH");
        const auto* srvIP = std::getenv("VPU_SRV_IP");
        const auto* srvPort = std::getenv("VPU_SRV_PORT");

        if (vpuFirmwareDir == nullptr) {
            VPUX_THROW("Can't locate vpu firmware directory, please provide VPU_FIRMWARE_SOURCES_PATH env var");
        }
        if (vpuElfPlatform == nullptr) {
            vpuElfPlatform = "silicon";
            _log.warning("'VPU_ELF_PLATFORM' env variable is unset, using the default value: 'silicon'");
        } else {
            auto vpuElfPlatformStr = std::string(vpuElfPlatform);
            if (vpuElfPlatformStr != "silicon" && vpuElfPlatformStr != "fpga")
                VPUX_THROW("Unsupported value for VPU_ELF_PLATFORM env var, expected: 'silicon' or 'fpga', got '{0}'",
                           vpuElfPlatformStr);
        }

        _app.runProgram = printToString("{0}/moviDebug2", pathToTools);
        _app.elfFile = printToString("{0}/vpux/{1}/{2}", ov::util::get_ov_lib_path(), vpuElfPlatform, appName);
        _app.imdElfArg = printToString("-D:elf={0}", _app.elfFile);

        static auto default_mdbg2Arg =
                printToString("{0}/build/buildSupport/scripts/debug/default_mdbg2.scr", vpuFirmwareDir);
        static auto default_pipe_mdbg2Arg =
                printToString("{0}/build/buildSupport/scripts/debug/default_pipe_mdbg2.scr", vpuFirmwareDir);
        static auto default_run_mdbg2Arg =
                printToString("{0}/build/buildSupport/scripts/debug/default_run_mdbg2.scr", vpuFirmwareDir);
        static std::string default_targetArg;

        if (platform == VPUXPlatform::VPU3720) {
            _app.chipsetArg = "-cv:3700xx";
            default_targetArg = "-D:default_target=LRT";
        } else {
            _app.chipsetArg = "-cv:ma2490";
            default_targetArg = "-D:default_target=LRT0";
        }

        _app.runArgs = {_app.runProgram, _app.imdElfArg, _app.chipsetArg, default_targetArg, "--no-uart"};

        if (srvIP != nullptr) {
            static auto srvIPArg = printToString("-srvIP:{0}", srvIP);
            _app.runArgs.append({srvIPArg});
        } else {
            _log.warning("'VPU_SRV_IP' env variable is unset, moviDebug2 will try to connect to localhost");
        }

        if (srvPort != nullptr) {
            static auto srvPortArg = printToString("-serverPort:{0}", srvPort);
            _app.runArgs.append({srvPortArg});
        } else {
            _log.warning("'VPU_SRV_PORT' env variable is unset, moviDebug2 will try to connect to 30000 or 30001 port");
        }

        // Additional script needed for VPU3720 silicon
        if (std::string(vpuElfPlatform) == "silicon" && platform == VPUXPlatform::VPU3720) {
            static auto vpux37xx_resetArg =
                    printToString("{0}/build/buildSupport/scripts/debug/mdkTcl/commands/SoC/372x/vpux37xx_reset.tcl",
                                  vpuFirmwareDir);  // Check for the real filepath
            _app.runArgs.append({"--init", vpux37xx_resetArg});
        }

        // Common debug scripts
        _app.runArgs.append({"--init", default_mdbg2Arg});
        _app.runArgs.append({"--init", default_pipe_mdbg2Arg});
        _app.runArgs.append({"--script", default_run_mdbg2Arg});
        _app.runArgs.append({"-D:run_opt=runw", "-D:exit_opt=exit"});

        break;
    }
    default:
        VPUX_THROW("Unsupported launch mode '{0}'", mode);
    }

    _app.timeoutSec = config.get<IMD::MV_RUN_TIMEOUT>().count();
}

//
// createTempWorkDir
//

SmallString vpux::IMD::ExecutorImpl::createTempWorkDir() {
    _log.trace("Create unique temporary working directory...");

    SmallString workDir;
    const auto errc = llvm::sys::fs::createUniqueDirectory("vpux-IMD", workDir);
    VPUX_THROW_WHEN(errc, "Failed to create temporary working directory : {0}", errc.message());

    _log.nest().trace("{0}", workDir);

    return workDir;
}

//
// storeNetworkBlob
//

void vpux::IMD::ExecutorImpl::storeNetworkBlob(StringRef workDir) {
    _log.trace("Store the network blob...");

    const auto& compiledBlob = _network->getCompiledNetwork();

    const std::string fileName = "test.blob";
    const auto modelFilePath = printToString("{0}/{1}", workDir, fileName);
    std::ofstream file(modelFilePath, std::ios::binary);
    VPUX_THROW_UNLESS(file.is_open(), "Can't open file '{0}' for write", modelFilePath);
    file.write(compiledBlob.data(), compiledBlob.size());

    _log.nest().trace("{0}", modelFilePath);
}

//
// storeNetworkInputs
//

void vpux::IMD::ExecutorImpl::storeNetworkInputs(StringRef workDir, const BlobMap& inputs) {
    _log.trace("Store the network inputs...");

    const auto& deviceInputsInfo = _network->getDeviceInputsInfo();

    for (const auto& p : inputs | indexed) {
        const auto& blobName = p.value().first;
        const auto ind = p.index();

        const auto& devInfo = deviceInputsInfo.at(blobName);

        const auto& userBlob = as<MemoryBlob>(p.value().second);
        VPUX_THROW_UNLESS(userBlob != nullptr, "Got non MemoryBlob");

        const auto devBlob = toLayout(toPrecision(userBlob, devInfo->getPrecision()), devInfo->getLayout());

        const auto mem = devBlob->rmap();
        const auto ptr = mem.as<const char*>();
        VPUX_THROW_UNLESS(ptr != nullptr, "Blob was not allocated");

        const auto inputFilePath = printToString("{0}/input-{1}.bin", workDir, ind);
        std::ofstream file(inputFilePath, std::ios_base::binary | std::ios_base::out);
        VPUX_THROW_UNLESS(file.is_open(), "Can't open file '{0}' for write", inputFilePath);
        file.write(ptr, devBlob->byteSize());

        _log.nest().trace("{0} - {1}", blobName, inputFilePath);
    }
}

//
// runApp
//

void vpux::IMD::ExecutorImpl::runApp(StringRef workDir) {
    _log.trace("Run the application...");

    SmallString curPath;
    auto errc = llvm::sys::fs::current_path(curPath);
    VPUX_THROW_WHEN(errc, "Failed to get current path : {0}", errc.message());

    VPUX_SCOPE_EXIT {
        _log.nest().trace("Restore current working directory '{0}'...", curPath);
        errc = llvm::sys::fs::set_current_path(curPath);

        if (errc) {
            _log.error("Failed to restore current path : {0}", errc.message());
        }
    };

    _log.nest().trace("Change current working directory to the new temporary folder '{0}'...", workDir);
    errc = llvm::sys::fs::set_current_path(workDir);
    VPUX_THROW_WHEN(errc, "Failed to change current path : {0}", errc.message());

    _log.nest().trace("{0}", _app.runArgs);

    std::string errMsg;
    const auto procErr = llvm::sys::ExecuteAndWait(_app.runProgram, makeArrayRef(_app.runArgs), /*Env=*/None,
                                                   /*Redirects=*/{}, checked_cast<uint32_t>(_app.timeoutSec),
                                                   /*MemoryLimit=*/0, &errMsg);
    VPUX_THROW_WHEN(procErr != 0, "Failed to run InferenceManagerDemo : {0}", errMsg);
}

//
// loadNetworkOutputs
//

void vpux::IMD::ExecutorImpl::loadNetworkOutputs(StringRef workDir, const BlobMap& outputs) {
    _log.trace("Load the network outputs...");

    const auto& deviceOutputsInfo = _network->getDeviceOutputsInfo();

    for (const auto& p : outputs | indexed) {
        const auto& blobName = p.value().first;
        const auto ind = p.index();

        const auto& devInfo = deviceOutputsInfo.at(blobName);

        const auto devBlob = as<MemoryBlob>(make_blob_with_precision(devInfo->getTensorDesc()));
        devBlob->allocate();

        const auto mem = devBlob->wmap();
        const auto ptr = mem.as<char*>();
        VPUX_THROW_UNLESS(ptr != nullptr, "Blob was not allocated");

        const auto outputFilePath = printToString("{0}/output-{1}.bin", workDir, ind);
        std::ifstream file(outputFilePath, std::ios_base::binary | std::ios_base::ate);
        VPUX_THROW_UNLESS(file.is_open(), "Can't open file '{0}' for READ", outputFilePath);

        const auto fileSize = static_cast<size_t>(file.tellg());
        file.seekg(0, std::ios_base::beg);
        VPUX_THROW_UNLESS(fileSize == devBlob->byteSize(),
                          "File '{0}' contains {1} bytes, but {2} expected for blob {3}", outputFilePath, fileSize,
                          devBlob->byteSize(), blobName);

        file.read(ptr, static_cast<std::streamsize>(devBlob->byteSize()));

        const auto& userBlob = as<MemoryBlob>(p.value().second);
        VPUX_THROW_UNLESS(userBlob != nullptr, "Got non MemoryBlob");

        cvtBlobLayout(toPrecision(devBlob, userBlob->getTensorDesc().getPrecision()), userBlob);

        _log.nest().trace("{0} - {1}", blobName, outputFilePath);
    }
}

//
// Base interface API implementation
//

vpux::IMD::ExecutorImpl::ExecutorImpl(VPUXPlatform platform, const NetworkDescription::Ptr& network,
                                      const Config& config)
        : _network(network), _log("InferenceManagerDemo", config.get<LOG_LEVEL>()) {
    _use_elf = config.get<USE_ELF_COMPILER_BACKEND>();
    parseAppConfig(platform, config);
}

void vpux::IMD::ExecutorImpl::setup(const ParamMap&) {
}

Executor::Ptr vpux::IMD::ExecutorImpl::clone() const {
    return std::make_shared<IMD::ExecutorImpl>(*this);
}

void vpux::IMD::ExecutorImpl::push(const BlobMap& inputs) {
    // Just store the inputs internally, the actual execution will be performed in the pull method
    _inputs = inputs;
}

void vpux::IMD::ExecutorImpl::push(const BlobMap&, const PreprocMap&) {
    VPUX_THROW("Pre-processing is not supported in VPUX IMD backend");
}

void vpux::IMD::ExecutorImpl::pull(BlobMap& outputs) {
    _log.info("Run inference using InferenceManagerDemo application...");
    _log = _log.nest();
    VPUX_SCOPE_EXIT {
        _log = _log.unnest();
    };

    const auto workDir = createTempWorkDir();
    VPUX_SCOPE_EXIT {
        _log.trace("Remove the temporary working directory '{0}'...", workDir);
        const auto errc = llvm::sys::fs::remove_directories(workDir);

        if (errc) {
            _log.error("Failed to remove temporary working directory : {0}", errc.message());
        }
    };

    storeNetworkBlob(workDir.str());
    storeNetworkInputs(workDir.str(), _inputs);
    runApp(workDir.str());
    loadNetworkOutputs(workDir.str(), outputs);
}

bool vpux::IMD::ExecutorImpl::isPreProcessingSupported(const PreprocMap&) const {
    return false;
}

std::map<std::string, InferenceEngineProfileInfo> vpux::IMD::ExecutorImpl::getLayerStatistics() {
    return {};
}

Parameter vpux::IMD::ExecutorImpl::getParameter(const std::string&) const {
    return {};
}
