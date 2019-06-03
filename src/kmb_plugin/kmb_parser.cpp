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

#include <climits>
#include <cstring>

#include <string>
#include <memory>
#include <list>
#include <vector>
#include <array>
#include <unordered_set>
#include <set>
#include <unordered_map>
#include <fstream>
#include <utility>
#include <algorithm>
#include <map>
#include <streambuf>
#include <tuple>
#include <sstream>
#include <iomanip>
#include <atomic>
#include <sys/stat.h>

#include <precision_utils.h>
#include <details/caseless.hpp>
#include <graph_tools.hpp>
#include <description_buffer.hpp>
#include <xml_parse_utils.h>

#include <vpu/parsed_config.hpp>
#include <vpu/compile_env.hpp>
#include <vpu/frontend/stage_builder.hpp>
#include "kmb_parser.hpp"

#if defined(_WIN32)
#define mkdir(dir, mode) _mkdir(dir)
#endif

namespace vpu {

namespace KmbPlugin {

//
// CompileEnv
//

namespace  {

thread_local CompileEnv g_compileEnv;

}  // namespace

void compileMcm(
        const ie::ICNNNetwork& network,
        const KmbConfig& config,
        mv::CompilationUnit& unit,
        const Logger::Ptr& log) {
    CompileEnv::init(Platform::MYRIAD_X, config.compileConfig, log);
    AutoScope autoDeinit([] {
        CompileEnv::free();
    });

    mv::OpModel& modelMcm = unit.model();

    auto frontEnd = std::make_shared<FrontEndMcm>(modelMcm);
    frontEnd->buildInitialModel(network);

    auto parsedConfig = config.getParsedConfig();

    if (parsedConfig[VPU_KMB_CONFIG_KEY(MCM_PARSING_ONLY)] == "YES") {
        return;
    }

    std::string targetName = parsedConfig[VPU_KMB_CONFIG_KEY(MCM_TARGET_DESCRIPTOR)];
    std::string targetPath = mv::utils::projectRootPath()
                             + "/" + parsedConfig[VPU_KMB_CONFIG_KEY(MCM_TARGET_DESCRIPTOR_PATH)]
                             + "/" + parsedConfig[VPU_KMB_CONFIG_KEY(MCM_TARGET_DESCRIPTOR)] + ".json";
    std::string compDescName = parsedConfig[VPU_KMB_CONFIG_KEY(MCM_COMPILATION_DESCRIPTOR)];

    std::string compDescPath = mv::utils::projectRootPath()
                             + "/" + parsedConfig[VPU_KMB_CONFIG_KEY(MCM_COMPILATION_DESCRIPTOR_PATH)]
                             + "/" + parsedConfig[VPU_KMB_CONFIG_KEY(MCM_COMPILATION_DESCRIPTOR)] + ".json";

    std::string resultsPath = parsedConfig[VPU_KMB_CONFIG_KEY(MCM_COMPILATION_RESULTS_PATH)];
    std::string resultNames = parsedConfig[VPU_KMB_CONFIG_KEY(MCM_COMPILATION_RESULTS)];
    resultNames = (resultNames == "") ? network.getName() : resultNames;

    /*IE_ASSERT(*/
    mkdir(resultsPath.c_str(), 0755);
    mkdir((resultsPath + "/" + compDescName).c_str(), 0755);
    mkdir((resultsPath + "/" + compDescName + "/" + targetName).c_str(), 0755);

    std::string resultsFullName = resultsPath + "/" + compDescName + "/" + targetName + "/" + resultNames;

    IE_ASSERT(unit.loadCompilationDescriptor(compDescPath));
    auto &compDesc = unit.compilationDescriptor();

    std::cout << "Path for results:" << resultsFullName << "(" << std::strerror(errno) << ")" << std::endl;

    if (parsedConfig[VPU_KMB_CONFIG_KEY(MCM_GENERATE_BLOB)] == "YES") {
        try {
            //-----------------------------------------------------------------------
            // Note: MCM compiler does not support this attributyes for Keem Bay (yet)
            //-----------------------------------------------------------------------

//            compDesc.setPassArg("GenerateBlob", "fileName", resultsFullName + ".blob");
//            compDesc.setPassArg("GenerateBlob", "enableFileOutput", true);
//            compDesc.setPassArg("GenerateBlob", "enableRAMOutput", false);
        } catch (...) {
            VPU_THROW_EXCEPTION << "Can't set mcmCompiler arguments for blob generation!";
        }
    }

    if (parsedConfig[VPU_KMB_CONFIG_KEY(MCM_GENERATE_DOT)] == "YES") {
        try {
            //--------------------------------------------------------------------------
            // Setting scope to control-model disables compute model details in dot-file
            //--------------------------------------------------------------------------
            // compDesc.setPassArg("GenerateDot", "scope", std::string("ControlModel"));
            compDesc.setPassArg("GenerateDot", "content", std::string("full"));
            compDesc.setPassArg("GenerateDot", "html", true);

            //-------------------------------------------------------------------------
            // MCM compiler dumps only final model, if we setup output file's name here
            //-------------------------------------------------------------------------
            // compDesc.setPassArg("GenerateDot", "output", "layers_" + testGetName() + ".dot");
        } catch (...) {
            VPU_THROW_EXCEPTION << "Can't set mcmCompiler arguments for *.dot generation!";
        }
    }

    IE_ASSERT(unit.loadTargetDescriptor(targetPath));
    IE_ASSERT(unit.initialize());

    // C++ exception if fails
    auto result = unit.run();

    if (parsedConfig[VPU_KMB_CONFIG_KEY(MCM_GENERATE_JSON)] == "YES") {
        std::fstream file_out(resultsFullName + ".json", std::fstream::out);
        file_out << result.stringifyPretty() << std::endl;
        file_out.close();
    }

    if (parsedConfig[VPU_KMB_CONFIG_KEY(MCM_GENERATE_DOT)] == "YES") {
        rename("original_model.dot", (resultsFullName + "_original.dot").c_str());
        rename("adapt_model.dot"   , (resultsFullName +    "_adapt.dot").c_str());
        rename("final_model.dot"   , (resultsFullName +          ".dot").c_str());
    }

    //-----------------------------------------------------------------------
    // Note: MCM compiler generates blob only in blob.bin (yet)
    //-----------------------------------------------------------------------
    if (parsedConfig[VPU_KMB_CONFIG_KEY(MCM_GENERATE_BLOB)] == "YES") {
        rename("blob.bin"   , (resultsFullName +          ".blob").c_str());
    }
}

//
// getSupportedLayers
//

std::set<std::string> getSupportedLayersMcm(
        const ie::ICNNNetwork& network,
        mv::OpModel& pCompiler,
        const CompilationConfig& config,
        const Logger::Ptr& log) {
    CompileEnv::init(Platform::MYRIAD_X, config, log);

    AutoScope autoDeinit([] {
        CompileEnv::free();
    });

    auto frontEnd = std::make_shared<FrontEndMcm>(pCompiler);

    return frontEnd->checkSupportedLayers(network);
}

}  // namespace KmbPlugin

}  // namespace vpu
