//
// Copyright 2019-2020 Intel Corporation.
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

#include "mcm_adapter.hpp"

#ifdef ENABLE_MCM_COMPILER

#include <net_pass.h>
#include <sys/stat.h>

#include <frontend_mcm.hpp>
#include <ie_icnn_network.hpp>
#include <ie_util_internal.hpp>

#include "include/mcm/compiler/compilation_unit.hpp"

#if defined(_WIN32)
#define mkdir(dir, mode) _mkdir(dir)
#endif

using namespace InferenceEngine;
using namespace vpu;

static std::string getMcmLogLevel(LogLevel lvl) {
    switch (lvl) {
    case LogLevel::None:
        return "Silent";

    case LogLevel::Fatal:
    case LogLevel::Error:
        return "Error";

    case LogLevel::Warning:
        return "Warning";

    case LogLevel::Info:
        return "Info";

    case LogLevel::Debug:
    case LogLevel::Trace:
        return "Debug";

    default:
        return "Silent";
    }
}

void MCMAdapter::compileNetwork(
    InferenceEngine::ICNNNetwork& network, const MCMConfig& config, std::vector<char>& blob) {
    auto unit = std::make_shared<mv::CompilationUnit>(network.getName());

    if (unit == nullptr) {
        THROW_IE_EXCEPTION << "CompilationUnit have not been created.";
    }
    bool ti_proc_ok =
        !InferenceEngine::NetPass::CombineRNNSeq(network) ? InferenceEngine::NetPass::UnrollTI(network) : true;
    if (!ti_proc_ok)
        THROW_IE_EXCEPTION << "Plugin doesn't support Tensor Iterator in pure form. "
                              "None TI optimization pattern has been applied successfully";

    blob.clear();
    Logger::Ptr _logger = std::make_shared<Logger>("compileMCM", config.logLevel(), consoleOutput());
    mv::OpModel& modelMcm = unit->model();

    auto frontEnd = std::make_shared<FrontEndMcm>(modelMcm, config);

    const auto& targetName = config.mcmTargetDesciptor();
    const auto targetPath =
        getIELibraryPath() + "/" + config.mcmTargetDesciptorPath() + "/" + config.mcmTargetDesciptor() + ".json";

    const auto& compDescName = config.mcmCompilationDesciptor();
    const auto compDescPath = getIELibraryPath() + "/" + config.mcmCompilationDesciptorPath() + "/" +
                              config.mcmCompilationDesciptor() + ".json";

    const auto& resultsPath = config.mcmCompilationResultsPath();
    auto resultNames = config.mcmCompilationResults();
    resultNames = (resultNames == "") ? network.getName() : resultNames;

    mkdir(resultsPath.c_str(), 0755);
    mkdir((resultsPath + "/" + compDescName).c_str(), 0755);
    mkdir((resultsPath + "/" + compDescName + "/" + targetName).c_str(), 0755);

    std::string resultsFullName = resultsPath + "/" + compDescName + "/" + targetName + "/" + resultNames;

    if (!unit->loadTargetDescriptor(targetPath)) {
        THROW_IE_EXCEPTION << "Target description loading failed! Path: " << targetPath;
    }

    if (!unit->loadCompilationDescriptor(compDescPath)) {
        THROW_IE_EXCEPTION << "Compilation description loading failed! Path: " << compDescPath;
    };
    auto& compDesc = unit->compilationDescriptor();

    // parse model after loading config
    frontEnd->buildInitialModel(network);
    if (config.mcmParseOnly()) {
        return;
    }

    _logger->info("Path for results: %s (%s)", resultsFullName, std::strerror(errno));

    if (config.mcmGenerateBlob()) {
        //-----------------------------------------------------------------------
        // Note: There are different passes in different Compilation Descriptors
        // Just now we try two of them
        //-----------------------------------------------------------------------
        bool isBlobFileSet = false;

        if (compDesc.validPass("GenerateBlobKmb")) {
            // validPass returns true here but 'setPassArg' attempt causes
            // 'Trying to set arguments for a non-existent pass' error
            // so we use try catch
            try {
                compDesc.setPassArg("GenerateBlobKmb", "output", resultsFullName + ".blob");
                isBlobFileSet = true;
            } catch (...) {
            }
        }
        if (!isBlobFileSet) {
            VPU_THROW_EXCEPTION << "Can't set mcmCompiler arguments for blob generation!";
        }
    }

    if (config.mcmGenerateDOT()) {
        VPU_THROW_EXCEPTION << "Not implemented";
    }

    compDesc.setPassArg("GlobalConfigParams", "verbose", getMcmLogLevel(config.mcmLogLevel()));

    IE_ASSERT(unit->initialize());

    try {
        auto result = unit->run();

        if (config.mcmGenerateJSON()) {
            std::fstream file_out(resultsFullName + ".json", std::fstream::out);
            file_out << result.toString() << std::endl;
            file_out.close();
        }
    } catch (const std::runtime_error& re) {
        VPU_THROW_EXCEPTION << "Caught std::runtime_error during unit run: " << re.what();
    } catch (...) {
        VPU_THROW_EXCEPTION << "Unknown exception during unit run";
    }

    if (config.mcmGenerateDOT()) {
        rename("original_model.dot", (resultsFullName + "_original.dot").c_str());
        rename("adapt_model.dot", (resultsFullName + "_adapt.dot").c_str());
        rename("final_model.dot", (resultsFullName + ".dot").c_str());
    }

    if (config.mcmGenerateBlob()) {
        auto memBlob = unit->getBlob();
        if (memBlob == nullptr) {
            std::ifstream blobFile(resultsFullName + ".blob", std::ios::binary);
            if (blobFile) {
                std::ostringstream blobContentStream;
                blobContentStream << blobFile.rdbuf();
                const std::string& blobContentString = blobContentStream.str();
                std::copy(blobContentString.begin(), blobContentString.end(), std::back_inserter(blob));
            } else {
                VPU_THROW_EXCEPTION << "Can not open blob file " << resultsFullName + ".blob"
                                    << ". It was not created by mcmCompiler!";
            }
        } else {
            std::copy(memBlob->begin(), memBlob->end(), std::back_inserter(blob));
        }

        if (blob.empty()) {
            VPU_THROW_EXCEPTION << "Blob file " << resultsFullName + ".blob"
                                << " created by mcmCompiler is empty!";
        }
    }
}

std::set<std::string> MCMAdapter::getSupportedLayers(InferenceEngine::ICNNNetwork& network, const MCMConfig& config) {
    std::shared_ptr<mv::CompilationUnit> tmpCompiler = std::make_shared<mv::CompilationUnit>(network.getName());
    if (tmpCompiler == nullptr) {
        THROW_IE_EXCEPTION << "CompilationUnit have not been created.\n"
                           << "Supported format: FP32 and FP16.";
    }

    auto frontEnd = std::make_shared<FrontEndMcm>(tmpCompiler->model(), config);

    return frontEnd->checkSupportedLayers(network);
}

bool vpu::MCMAdapter::isMCMCompilerAvailable() {
    std::shared_ptr<mv::CompilationUnit> tmpCompiler = std::make_shared<mv::CompilationUnit>("testModel");
    return tmpCompiler != nullptr;
}

#else

void vpu::MCMAdapter::compileNetwork(
    InferenceEngine::ICNNNetwork& network, const MCMConfig& config, std::vector<char>& blob) {
    THROW_IE_EXCEPTION << "Compiler is disabled";
}

std::set<std::string> vpu::MCMAdapter::getSupportedLayers(
    InferenceEngine::ICNNNetwork& network, const MCMConfig& config) {
    THROW_IE_EXCEPTION << "Compiler is disabled";
}

bool vpu::MCMAdapter::isMCMCompilerAvailable() { return false; }

#endif  // ENABLE_MCM_COMPILER
