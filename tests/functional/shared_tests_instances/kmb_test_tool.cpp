// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kmb_test_tool.hpp"

#include <iostream>
#include <fstream>
#include <iomanip>

#include <vpu/utils/error.hpp>

using namespace InferenceEngine;

namespace LayerTestsUtils {

KmbTestTool::KmbTestTool(const KmbTestEnvConfig& envCfg): envConfig(envCfg),
    DEVICE_NAME(envConfig.IE_KMB_TESTS_DEVICE_NAME.empty() ? "VPUX" : envConfig.IE_KMB_TESTS_DEVICE_NAME) {
}

void KmbTestTool::exportNetwork(ExecutableNetwork& exeNet, const std::string& fsName) {
    if (envConfig.IE_KMB_TESTS_DUMP_PATH.empty()) {
        std::cout << "IE_KMB_TESTS_DUMP_PATH is not set. Skip blob export" << std::endl;
        return;
    }

    const auto fileName = vpu::formatString("%v/%v", envConfig.IE_KMB_TESTS_DUMP_PATH, fsName);
    std::cout << "Exporting nn into " << (envConfig.IE_KMB_TESTS_RAW_EXPORT ? "" : "not ") << "raw file " << fileName
        << ", device " << DEVICE_NAME << std::endl;

    if (envConfig.IE_KMB_TESTS_RAW_EXPORT) {
        exeNet.Export(fileName);
    } else {
        std::ofstream file(fileName, std::ios_base::out | std::ios_base::binary);
        if (!file.is_open())
            THROW_IE_EXCEPTION << "exportNetwork(). Can't open file " << fileName;

        exeNet.Export(file);
    }
    std::cout << "Exported nn into file " << fileName << std::endl;
}

ExecutableNetwork KmbTestTool::importNetwork(const std::shared_ptr<InferenceEngine::Core>& core, const std::string& fsName) {
    IE_ASSERT(!envConfig.IE_KMB_TESTS_DUMP_PATH.empty());

    const auto fileName = vpu::formatString("%v/%v", envConfig.IE_KMB_TESTS_DUMP_PATH, fsName);
    std::cout << "Importing nn from " << (envConfig.IE_KMB_TESTS_RAW_EXPORT ? "" : "not ") << "raw file " << fileName
        << ", device " << DEVICE_NAME << std::endl;

    if (envConfig.IE_KMB_TESTS_RAW_EXPORT) {
        return core->ImportNetwork(fileName, DEVICE_NAME);
    } else {
        std::ifstream file(fileName, std::ios_base::in | std::ios_base::binary);
        if (!file.is_open())
            THROW_IE_EXCEPTION << "importNetwork(). Can't open file " << fileName;

        return core->ImportNetwork(file, DEVICE_NAME);
    }
}

unsigned long int FNV_hash(const std::string &str)
{
  const unsigned char* p = reinterpret_cast<const unsigned char *>(str.c_str());
  unsigned long int h = 2166136261UL;

  for(size_t i = 0; i < str.size(); i++)
    h = (h * 16777619) ^ p[i];

  return h;
}

std::string cleanName(std::string name) {
    std::replace_if(
        name.begin(), name.end(),
        [](char c) {
            return !std::isalnum(c);
        },
        '_');
    return name;
}

std::string filesysName(const testing::TestInfo* testInfo) {

    constexpr char ext[] = ".net";
    constexpr size_t maxFileNameLen = 256, extLen = sizeof(ext) / sizeof(ext[0]),
        maxNoExtLen = maxFileNameLen - extLen, maxNoExtShortenedLen = maxNoExtLen - 20 - 1;
    const auto testName = vpu::formatString("%v_%v", testInfo->test_case_name(), testInfo->name());
    auto fnameNoExt = (testName.size() < maxNoExtLen) ? testName : vpu::formatString("%v_%v", testName.substr(0, maxNoExtShortenedLen), FNV_hash(testName));

    return cleanName(fnameNoExt).append(ext);
}

}  // namespace LayerTestsUtils
