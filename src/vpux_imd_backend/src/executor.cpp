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

#include "vpux/IMD/executor.hpp"

#include "vpux/IMD/parsed_config.hpp"
#include "vpux/IMD/platform_helpers.hpp"
#include "vpux/al/config/common.hpp"
#include "vpux/utils/IE/blob.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux/utils/core/scope_exit.hpp"

#include <blob_factory.hpp>
#include <openvino/util/file_util.hpp>

#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Program.h>

#include <fstream>

using namespace vpux;
using namespace InferenceEngine;
using namespace InferenceEngine::VPUXConfigParams;

//
// parseAppConfig
//

void vpux::IMD::ExecutorImpl::parseAppConfig(VPUXPlatform platform, const Config& config) {
    // InferenceManagerDemo application ELF file

    if (platformSupported(platform)) {
        auto chipset_value = getChipsetName(platform);  // If above check has passed, platform is present

        _app.elfFile =
                llvm::formatv("{0}/vpux/IMD/{1}/InferenceManagerDemo.elf", ov::util::get_ov_lib_path(), chipset_value)
                        .str();
    } else {
        VPUX_THROW("Unsupported VPU platform '{0}'", platform);
    }

    // Path to MOVI tools dir

    std::string pathToTools;

    if (config.has<IMD::MV_TOOLS_PATH>()) {
        pathToTools = llvm::formatv("{0}/linux64/bin", config.get<IMD::MV_TOOLS_PATH>()).str();
    } else {
        const auto* rootDir = std::getenv("MV_TOOLS_DIR");
        const auto* version = std::getenv("MV_TOOLS_VERSION");

        if (rootDir != nullptr && version != nullptr) {
            pathToTools = llvm::formatv("{0}/{1}/linux64/bin", rootDir, version).str();
        } else {
            VPUX_THROW("Can't locate MOVI tools directory, please provide VPUX_IMD_MV_TOOLS_PATH config option or "
                       "MV_TOOLS_DIR/MV_TOOLS_VERSION env vars");
        }
    }

    // Run program configuration

    const auto mode = config.get<IMD::LAUNCH_MODE>();

    switch (mode) {
    case IMD::LaunchMode::MoviSim: {
        _app.runProgram = llvm::formatv("{0}/moviSim", pathToTools).str();

        if (platformSupported(platform)) {
            if (platform == decltype(platform)::VPU3720) {
                // For some reason, -cv:3720xx doesn't work, while -cv:3700xx works OK for MTL
                _app.chipsetVersion = "-cv:3700xx";
                _app.imdElf = "-l:LRT:./InferenceManagerDemo.elf";
            } else {
                _app.chipsetVersion = "-cv:ma2490";
                // The only of the arch ids from config, that doesn't cause exceptions
                _app.imdElf = "-l:LRT0:./InferenceManagerDemo.elf";
            }

            _app.runArgs = {_app.runProgram, _app.chipsetVersion, "-nodasm", "-q", _app.imdElf};
        } else {
            VPUX_THROW("Unsupported VPU platform '{0}'", platform);
        }

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
// copyAppFile
//

void vpux::IMD::ExecutorImpl::copyAppFile(StringRef workDir) {
    _log.trace("Copy the application ELF file...");

    const auto elfFilePath = llvm::formatv("{0}/InferenceManagerDemo.elf", workDir).str();

    const auto errc = llvm::sys::fs::copy_file(_app.elfFile, elfFilePath);
    VPUX_THROW_WHEN(errc, "Failed to copy the application ELF file : {0}", errc.message());

    _log.nest().trace("{0}", elfFilePath);
}

//
// storeNetworkBlob
//

void vpux::IMD::ExecutorImpl::storeNetworkBlob(StringRef workDir) {
    _log.trace("Store the network blob...");

    const auto& compiledBlob = _network->getCompiledNetwork();

    const auto modeFilePath = llvm::formatv("{0}/test.blob", workDir).str();
    std::ofstream file(modeFilePath, std::ios::binary);
    VPUX_THROW_UNLESS(file.is_open(), "Can't open file '{0}' for write", modeFilePath);
    file.write(compiledBlob.data(), compiledBlob.size());

    _log.nest().trace("{0}", modeFilePath);
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

        const auto inputFilePath = llvm::formatv("{0}/input-{1}.bin", workDir, ind).str();
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

        const auto outputFilePath = llvm::formatv("{0}/output-{1}.bin", workDir, ind).str();
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

    copyAppFile(workDir.str());
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
