//
// INTEL CONFIDENTIAL
// Copyright 2019 Intel Corporation.
//
// The source code contained or described herein and all documents
// related to the source code ("Material") are owned by Intel Corporation
// or its suppliers or licensors. Title to the Material remains with
// Intel Corporation or its suppliers and licensors. The Material may
// contain trade secrets and proprietary and confidential information
// of Intel Corporation and its suppliers and licensors, and is protected
// by worldwide copyright and trade secret laws and treaty provisions.
// No part of the Material may be used, copied, reproduced, modified,
// published, uploaded, posted, transmitted, distributed, or disclosed
// in any way without Intel's prior express written permission.
//
// No license under any patent, copyright, trade secret or other
// intellectual property right is granted to or conferred upon you by
// disclosure or delivery of the Materials, either expressly, by implication,
// inducement, estoppel or otherwise. Any license under such intellectual
// property rights must be express and approved by Intel in writing.
//
// Include any supplier copyright notices as supplier requires Intel to use.
//
// Include supplier trademarks or logos as supplier requires Intel to use,
// preceded by an asterisk. An asterisked footnote can be added as follows:
// *Third Party trademarks are the property of their respective owners.
//
// Unless otherwise agreed by Intel in writing, you may not remove or alter
// this notice or any other notice embedded in Materials by Intel or Intel's
// suppliers or licensors in any way.
//

#include "kmb_parser.hpp"

#include <precision_utils.h>
#include <sys/stat.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <climits>
#include <cstring>
#include <description_buffer.hpp>
#include <details/caseless.hpp>
#include <fstream>
#include <graph_tools.hpp>
#include <ie_util_internal.hpp>
#include <iomanip>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <streambuf>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <vpu/kmb_plugin_config.hpp>
#include <vpu/utils/error.hpp>

#if defined(_WIN32)
#define mkdir(dir, mode) _mkdir(dir)
#endif

#ifdef ENABLE_MCM_COMPILER
#include <include/mcm/target/kmb/runtime_model/runtime_model.hpp>

using namespace InferenceEngine;

namespace vpu {

namespace KmbPlugin {

namespace {

std::string getMcmLogLevel(LogLevel lvl) {
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

}  // namespace

void compileMcm(ie::ICNNNetwork& network, const KmbConfig& config, mv::CompilationUnit& unit, std::vector<char>& blob) {
    blob.clear();
    Logger::Ptr _logger = std::make_shared<Logger>("compileMCM", config.logLevel(), consoleOutput());
    mv::OpModel& modelMcm = unit.model();

    auto frontEnd = std::make_shared<FrontEndMcm>(modelMcm, config);
    frontEnd->buildInitialModel(network);

    if (config.mcmParseOnly()) {
        return;
    }

    const auto& targetName = config.mcmTargetDesciptor();
    const auto targetPath =
        getIELibraryPath() + "/" + config.mcmTargetDesciptorPath() + "/" + config.mcmTargetDesciptor() + ".json";

    const auto& compDescName = config.mcmCompilationDesciptor();
    const auto compDescPath = getIELibraryPath() + "/" + config.mcmCompilationDesciptorPath() + "/" +
                              config.mcmCompilationDesciptor() + ".json";

    const auto& resultsPath = config.mcmCompilationResultsPath();
    auto resultNames = config.mcmCompilationResults();
    resultNames = (resultNames == "") ? network.getName() : resultNames;

    /*IE_ASSERT(*/
    mkdir(resultsPath.c_str(), 0755);
    mkdir((resultsPath + "/" + compDescName).c_str(), 0755);
    mkdir((resultsPath + "/" + compDescName + "/" + targetName).c_str(), 0755);

    std::string resultsFullName = resultsPath + "/" + compDescName + "/" + targetName + "/" + resultNames;

    IE_ASSERT(unit.loadTargetDescriptor(targetPath));
    IE_ASSERT(unit.loadCompilationDescriptor(compDescPath));
    auto& compDesc = unit.compilationDescriptor();

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
        if (compDesc.validPass("GenerateDot")) {
            //            try {
            //                //--------------------------------------------------------------------------
            //                // Setting scope to control-model disables compute model details in dot-file
            //                //--------------------------------------------------------------------------
            //                compDesc.setPassArg("GenerateDot", "scope", std::string("ControlModel"));
            //                compDesc.setPassArg("GenerateDot", "content", std::string("full"));
            //                compDesc.setPassArg("GenerateDot", "html", true);
            //            } catch (...) {
            //                VPU_THROW_EXCEPTION << "Can't set mcmCompiler arguments for *.dot generation!";
            //            }
        }
    }

    compDesc.setPassArg("GlobalConfigParams", "verbose", getMcmLogLevel(config.mcmLogLevel()));

    IE_ASSERT(unit.initialize());

    // C++ exception if fails
    try {
        auto result = unit.run();

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
        auto memBlob = unit.getBlob();
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

std::set<std::string> getSupportedLayersMcm(ie::ICNNNetwork& network, mv::OpModel& pCompiler, const KmbConfig& config) {
    auto frontEnd = std::make_shared<FrontEndMcm>(pCompiler, config);

    return frontEnd->checkSupportedLayers(network);
}

}  // namespace KmbPlugin

}  // namespace vpu
#endif
